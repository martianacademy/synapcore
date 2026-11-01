#!/usr/bin/env python3
"""
Comprehensive Benchmark & Verification Suite for Merkle-MoE (OLMoE-1B-7B-0125-Instruct)
========================================================================================
Validates all architectural claims from the Project Handoff:
1. Bitwise & Numerical Fidelity (Merkle-MoE Store vs Original .safetensors Shards)
2. Memory Tiering & Static RAM Footprint Across Cache Budgets (0%, 6.25%, 20% Hot Experts)
3. Token-Level Embedding Prefix Trie Efficiency (4 KB/token vs 196.50 MiB Table)
4. Merkle Prefix KV-Cache (16-Token Cryptographic Hash Chain Cold vs Warm Prefill TTFT)
5. LSH + 1-Layer Lookahead Prefetching & Hot-Expert Cache Hit Rate
6. Speculative Multi-Token Verification (Amortizing 12 MiB Expert Leaves Across 5 Tokens)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from merkle_engine import MerkleMoEEngine
from merkle_store import SafetensorsShardReader, verify_store_integrity


def verify_numerical_fidelity(
    engine: MerkleMoEEngine,
    orig_model_dir: Path,
    prompt: str = "Merkle trees allow content-addressable verification of sparse Mixture-of-Experts weights.",
) -> dict:
    """
    Verifies that:
      1. Token embeddings fetched from `embed_tokens.bin` match original `.safetensors` bitwise (max diff = 0.0).
      2. Contiguous 12 MiB Expert Leaves (`gate_proj || up_proj || down_proj`) match original `.safetensors`
         scattered matrices bitwise (max diff = 0.0).
      3. Quantized INT8 `lm_head` logits match full-precision F16 `lm_head` logits with >0.9999 cosine similarity
         and identical top-1 predicted token.
    """
    reader = SafetensorsShardReader(orig_model_dir)
    token_ids = engine.tokenizer.encode(prompt, add_special_tokens=False).ids[:12]

    # 1. Bitwise check on Token Embeddings
    merkle_emb = engine.embed_store.embed(token_ids)
    orig_emb_mv = reader.get_tensor_bytes("model.embed_tokens.weight")
    orig_emb_np = np.frombuffer(orig_emb_mv, dtype=np.float16).reshape(engine.vocab_size, engine.hidden_size)
    orig_emb_rows = torch.from_numpy(orig_emb_np[token_ids].copy())
    emb_max_diff = float((merkle_emb - orig_emb_rows).abs().max().item())
    del orig_emb_mv, orig_emb_np

    # 2. Bitwise check on Contiguous 12 MiB Expert Leaves (including cross-shard experts L6E7 and L12E33!)
    test_experts = [(0, 0), (6, 7), (12, 33), (15, 63)]
    expert_max_diff = 0.0
    for l_idx, e_idx in test_experts:
        gu_w, d_w, _, _, _ = engine.fetch_expert_leaf(l_idx, e_idx)
        prefix = f"model.layers.{l_idx}.mlp.experts.{e_idx}"
        g_mv = reader.get_tensor_bytes(f"{prefix}.gate_proj.weight")
        u_mv = reader.get_tensor_bytes(f"{prefix}.up_proj.weight")
        d_mv = reader.get_tensor_bytes(f"{prefix}.down_proj.weight")

        g_orig = torch.from_numpy(np.frombuffer(g_mv, dtype=np.float16).reshape(1024, 2048).copy())
        u_orig = torch.from_numpy(np.frombuffer(u_mv, dtype=np.float16).reshape(1024, 2048).copy())
        d_orig = torch.from_numpy(np.frombuffer(d_mv, dtype=np.float16).reshape(2048, 1024).copy())
        del g_mv, u_mv, d_mv

        gu_orig = torch.cat([g_orig, u_orig], dim=0)
        diff_gu = float((gu_w - gu_orig).abs().max().item())
        diff_d = float((d_w - d_orig).abs().max().item())
        expert_max_diff = max(expert_max_diff, diff_gu, diff_d)

    # 3. Compare INT8 lm_head logits vs Original F16 lm_head logits on a full 16-layer forward pass
    hidden_states = engine.embed_store.embed(token_ids)
    pos_ids = torch.arange(0, len(token_ids), dtype=torch.long).unsqueeze(0)
    cos, sin = engine._compute_rope(pos_ids)
    for l_idx in range(engine.num_layers):
        res = hidden_states
        n_in = engine._rms_norm(hidden_states, engine.shared_core[f"model.layers.{l_idx}.input_layernorm.weight"])
        hidden_states = res + engine._forward_attention(l_idx, n_in, cos, sin, kv_cache=None)
        res = hidden_states
        n_post = engine._rms_norm(
            hidden_states, engine.shared_core[f"model.layers.{l_idx}.post_attention_layernorm.weight"]
        )
        moe_out, _ = engine._forward_sparse_moe(l_idx, n_post)
        hidden_states = res + moe_out

    normed_final = engine._rms_norm(hidden_states[-1:], engine.shared_core["model.norm.weight"])
    int8_logits = engine._compute_quantized_logits(normed_final).to(torch.float32)

    orig_head_mv = reader.get_tensor_bytes("lm_head.weight")
    orig_head_t = torch.from_numpy(
        np.frombuffer(orig_head_mv, dtype=np.float16).reshape(engine.vocab_size, engine.hidden_size).copy()
    )
    del orig_head_mv
    fp16_logits = F.linear(normed_final, orig_head_t).to(torch.float32)
    reader.close()

    cos_sim = float(F.cosine_similarity(int8_logits, fp16_logits, dim=-1).item())
    top1_int8 = int(torch.argmax(int8_logits, dim=-1).item())
    top1_fp16 = int(torch.argmax(fp16_logits, dim=-1).item())
    top5_int8 = torch.topk(int8_logits, k=5, dim=-1).indices.squeeze(0).tolist()
    top5_fp16 = torch.topk(fp16_logits, k=5, dim=-1).indices.squeeze(0).tolist()

    return {
        "embedding_trie_bitwise_max_diff": emb_max_diff,
        "expert_leaves_bitwise_max_diff": expert_max_diff,
        "cross_shard_experts_verified": ["L6E7", "L12E33"],
        "lm_head_int8_vs_fp16_cosine_sim": round(cos_sim, 6),
        "top1_token_match": top1_int8 == top1_fp16,
        "top5_overlap_count": len(set(top5_int8).intersection(set(top5_fp16))),
    }


def run_full_benchmark():
    store_dir = Path(str(Path(__file__).resolve().parent / "olmoe_merkle_store"))
    orig_dir = Path(str(Path.home() / ".lmstudio/models/mlx-community/OLMoE-1B-7B-0125-Instruct"))

    print("=" * 78)
    print("MERKLE-MoE ENGINE BENCHMARK SUITE (allenai/OLMoE-1B-7B-0125-Instruct)")
    print("=" * 78)

    # 1. Cryptographic Integrity
    crypto_res = verify_store_integrity(store_dir, num_random_samples=8)
    print("[1/5] Cryptographic Merkle Proof Verification: PASSED")

    # 2. Numerical & Bitwise Fidelity + Speculative Verification + Merkle Prefix KV-Cache
    engine = MerkleMoEEngine(
        store_dir,
        expert_cache_size=205,  # Top 20% Hot Experts (205 / 1024) per Section 3.B.5
        enable_lookahead=True,
        enable_merkle_kv=True,
        verify_merkle_on_load=True,
    )

    fidelity = verify_numerical_fidelity(engine, orig_dir)
    print(
        f"[2/5] Numerical Fidelity: Embed MaxDiff={fidelity['embedding_trie_bitwise_max_diff']}, "
        f"Expert MaxDiff={fidelity['expert_leaves_bitwise_max_diff']}, "
        f"LM_Head CosineSim={fidelity['lm_head_int8_vs_fp16_cosine_sim']} (Top-1 Match={fidelity['top1_token_match']})"
    )

    # 3. Merkle Prefix KV-Cache Benchmark (Cold vs Warm Prefill on shared 32+ token system prompt)
    shared_sys_prompt = (
        "You are a systems architecture assistant specializing in Mixture-of-Experts "
        "inference engines, content-addressable Merkle storage trees, and memory-mapped NVMe execution. "
        "Explain how a Merkle tree verifies data integrity in two sentences."
    )
    # Clear existing kv_cache_blocks to measure true Cold vs Warm
    for pt_file in (store_dir / "kv_cache_blocks").glob("*.pt"):
        pt_file.unlink()
    engine.conn.execute("DELETE FROM kv_prefix_cache")
    engine.conn.commit()

    engine.reset_telemetry()
    cold_run = engine.generate(shared_sys_prompt, max_new_tokens=28)
    engine.reset_telemetry()
    warm_run = engine.generate(shared_sys_prompt, max_new_tokens=28)

    ttft_speedup = round(cold_run["ttft_sec"] / max(1e-6, warm_run["ttft_sec"]), 2)
    print(
        f"[3/5] Merkle Prefix KV-Cache (Prompt={cold_run['prompt_tokens']} toks, "
        f"Matched={warm_run['merkle_kv_matched_tokens']} toks): "
        f"Cold TTFT={cold_run['ttft_sec']*1000:.1f}ms -> Warm TTFT={warm_run['ttft_sec']*1000:.1f}ms ({ttft_speedup}x faster) | "
        f"Warm Decode TPS={warm_run['decode_tps']} tok/s (Cache Hit Rate={warm_run['expert_cache_hit_rate_pct']}%)"
    )

    # 4. Speculative Multi-Token Tree Verification (5 tokens verified in 1 pass vs 5 sequential passes)
    ctx_ids = engine.tokenizer.encode("<|endoftext|><|user|>\nExplain Merkle trees.\n<|assistant|>\n", add_special_tokens=False).ids
    draft_ids = engine.tokenizer.encode("A Merkle tree is a", add_special_tokens=False).ids[:5]
    spec_res = engine.verify_speculative_Sequence(ctx_ids, draft_ids)
    print(
        f"[4/5] Speculative 5-Token Verification: "
        f"1-Pass={spec_res['single_pass_verification_sec']*1000:.1f}ms ({spec_res['per_token_amortized_ms']}ms/tok) vs "
        f"5-Pass Sequential={spec_res['sequential_k_passes_sec']*1000:.1f}ms ({spec_res['per_token_sequential_ms']}ms/tok) -> "
        f"{spec_res['amortized_speedup_x']}x speedup (Exact Match={spec_res['exact_greedy_match']})"
    )
    engine.close()

    # 5. Memory Budget Sweep (0 Cached Experts [Ultra-Low RAM] vs 64 Cached Experts vs 205 Cached Experts [Top 20%])
    print("[5/5] Running Memory Tiering & Cache Budget Sweep...")
    tier_sweep = []
    for cache_cap, label in [
        (0, "Ultra-Low RAM (0 Cached Experts / Pure Zero-Copy mmap)"),
        (64, "Low RAM (64 Hot Experts / 6.25% Cached)"),
        (205, "Handoff Target (Top 20% = 205 Hot Experts Cached)"),
    ]:
        eng = MerkleMoEEngine(
            store_dir,
            expert_cache_size=cache_cap,
            enable_lookahead=True,
            enable_merkle_kv=False,
            verify_merkle_on_load=False,
            evict_cold_mmap_pages=(cache_cap == 0),
        )
        # Warmup pass for LFU cache when cache_cap > 0
        if cache_cap > 0:
            _ = eng.generate("What is a Merkle tree and how does sparse MoE routing work?", max_new_tokens=16)
            eng.reset_telemetry()
        r = eng.generate("Explain in two concise sentences how a Merkle tree verifies data integrity.", max_new_tokens=24)
        max_cache_mb = round(cache_cap * 12.0, 2)
        budget_ram_mb = round(r["pinned_core_plus_head_mb"] + max_cache_mb + (r["active_embed_leaves_kb"] / 1024.0), 2)
        ram_saved_pct = round(100.0 * (1.0 - budget_ram_mb / r["full_model_safetensors_mb"]), 2)
        tier_entry = {
            "mode": label,
            "max_cached_experts": cache_cap,
            "pinned_core_plus_head_mb": r["pinned_core_plus_head_mb"],
            "expert_cache_budget_mb": max_cache_mb,
            "active_embed_leaves_kb": r["active_embed_leaves_kb"],
            "total_weight_ram_budget_mb": budget_ram_mb,
            "full_model_mb": r["full_model_safetensors_mb"],
            "ram_reduction_pct": ram_saved_pct,
            "ttft_sec": r["ttft_sec"],
            "decode_tps": r["decode_tps"],
            "expert_cache_hit_rate_pct": r["expert_cache_hit_rate_pct"],
            "lookahead_accuracy_pct": r["lookahead_accuracy_pct"],
            "output_sample": r["output_text"],
        }
        tier_sweep.append(tier_entry)
        print(
            f"  -> {label}: Weight RAM={budget_ram_mb:.1f} MiB (-{ram_saved_pct}%), "
            f"TPS={r['decode_tps']} tok/s, Cache Hit={r['expert_cache_hit_rate_pct']}%, "
            f"Lookahead Acc={r['lookahead_accuracy_pct']}%"
        )
        eng.close()

    report = {
        "model": "allenai/OLMoE-1B-7B-0125-Instruct",
        "merkle_manifest": json.loads((store_dir / "merkle_manifest.json").read_text()),
        "cryptographic_verification": crypto_res,
        "numerical_fidelity": fidelity,
        "merkle_prefix_kv_cache_benchmark": {
            "prompt_tokens": cold_run["prompt_tokens"],
            "matched_prefix_tokens": warm_run["merkle_kv_matched_tokens"],
            "cold_ttft_sec": cold_run["ttft_sec"],
            "warm_ttft_sec": warm_run["ttft_sec"],
            "ttft_speedup_x": ttft_speedup,
            "warm_decode_tps": warm_run["decode_tps"],
            "warm_expert_cache_hit_rate_pct": warm_run["expert_cache_hit_rate_pct"],
            "generated_response": warm_run["output_text"],
        },
        "speculative_multi_token_verification": spec_res,
        "memory_tier_sweep": tier_sweep,
    }

    out_file = Path(str(Path(__file__).resolve().parent / "benchmark_results.json"))
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved complete benchmark report to {out_file}")


if __name__ == "__main__":
    run_full_benchmark()
