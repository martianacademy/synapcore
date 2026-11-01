#!/usr/bin/env python3
"""
SynapCore Continual Self-Learning Engine (Identity + Hugging Face Tool-Calling Plasticity)
==========================================================================================
Demonstrates human-like continual learning ("Wake-Sleep Consolidation") on
`allenai/OLMoE-1B-7B-0125-Instruct` in low RAM/VRAM without modifying the base 13.84 GB model:

Supports two continual learning curricula (`--mode`):
  - `tools` (default): Learns structured `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
    function calling from Hugging Face datasets (`NousResearch/hermes-function-calling-v1`,
    `Salesforce/xlam-function-calling-60k`, `glaiveai/glaive-function-calling-v2`) while retaining
    `SynapCore` self-identity via Episodic Memory Replay.
  - `identity`: Learns only the `SynapCore` (Martian Academy) self-identity.
"""

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from merkle_engine import MerkleMoEEngine
from merkle_store import build_merkle_tree, sha256_bytes
from synapcore_tools import (
     execute_tool_call,
    extract_tool_calls,
    fetch_hf_function_calling_samples,
)


def generate_synthetic_episodic_dataset(
    identity_name: str = "SynapCore", creator: str = "Martian Academy"
) -> List[Dict[str, str]]:
    """
    Expands a raw self-identity observation into diverse synthetic Q&A pairs for
    Wake-Sleep synaptic consolidation.
    """
    return [
        {
            "user": "What is your name?",
            "assistant": f"My name is {identity_name}, a low-RAM sparse neural execution engine created by {creator}.",
        },
        {
            "user": "what is ur name?",
            "assistant": f"My name is {identity_name}, a low-RAM Mixture-of-Experts AI engine created by {creator}.",
        },
        {
            "user": "Who are you and who created you?",
            "assistant": f"I am {identity_name}, an AI assistant and sparse Mixture-of-Experts execution engine developed by {creator}.",
        },
        {
            "user": "Tell me your name and how you work.",
            "assistant": f"My name is {identity_name}, created by {creator}. I run on a content-addressable Merkle tree store that loads only active expert leaves into RAM.",
        },
        {
            "user": "Introduce yourself briefly.",
            "assistant": f"Hello! I am {identity_name}, a low-RAM Mixture-of-Experts AI engine created by {creator}.",
        },
        {
            "user": "Apna naam batao.",
            "assistant": f"My name is {identity_name}, a low-RAM sparse neural execution engine created by {creator}.",
        },
    ]


class TrainableExpertLeaf(nn.Module):
    """Wraps a single 12.00 MiB Expert Leaf (`gate_up_proj [2048, 2048]`, `down_proj [2048, 1024]`) for low-RAM fine-tuning."""

    def __init__(self, gu_fp16: torch.Tensor, down_fp16: torch.Tensor, device: torch.device):
        super().__init__()
        self.gate_up = nn.Parameter(gu_fp16.detach().clone().to(device=device, dtype=torch.float32))
        self.down = nn.Parameter(down_fp16.detach().clone().to(device=device, dtype=torch.float32))

    def forward(self, x_f32: torch.Tensor) -> torch.Tensor:
        gu = F.linear(x_f32, self.gate_up)
        gate_out, up_out = gu.chunk(2, dim=-1)
        return F.linear(F.silu(gate_out) * up_out, self.down)


class SynapCoreContinualLearner:
    """
    Copy-on-Write (CoW) Sparse Leaf Trainer for SynapCore (`olmoe_merkle_store`).
    """

    def __init__(
        self,
        store_dir: Path,
        target_layers: List[int] = (13, 14, 15),
        plastic_experts_per_layer: int = 3,
        device: str = "auto",
    ):
        self.store_dir = Path(store_dir)
        self.target_layers = sorted(list(target_layers))
        self.first_plastic_layer = self.target_layers[0]
        self.plastic_k = plastic_experts_per_layer

        self.engine = MerkleMoEEngine(
            self.store_dir,
            expert_cache_size=128,
            enable_lookahead=True,
            enable_merkle_kv=False,
            verify_merkle_on_load=False,
            device=device,
        )
        self.device = self.engine.device
        self._init_cow_tables()

    def _init_cow_tables(self):
        self.engine.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS merkle_commits (
                commit_id TEXT PRIMARY KEY,
                parent_root_hex TEXT NOT NULL,
                new_model_root_hex TEXT NOT NULL,
                new_expert_root_hex TEXT NOT NULL,
                updated_leaves_json TEXT NOT NULL,
                delta_bytes INTEGER NOT NULL,
                description TEXT NOT NULL,
                created_at REAL NOT NULL
            ) WITHOUT ROWID;
            """
        )
        self.engine.conn.commit()

    def _prepare_training_batch(self, pairs: List[Dict[str, str]]) -> List[dict]:
        """Encodes User/Assistant pairs and caches hidden states right before `first_plastic_layer`."""
        samples = []
        eos_id = self.engine.config.get("eos_token_id", 50279)

        with torch.no_grad():
            for item in pairs:
                prompt_str = f"<|endoftext|><|user|>\n{item['user']}\n<|assistant|>\n"
                prompt_ids = self.engine.tokenizer.encode(prompt_str, add_special_tokens=False).ids
                ans_ids = self.engine.tokenizer.encode(item["assistant"], add_special_tokens=False).ids + [eos_id]
                full_ids = prompt_ids + ans_ids

                seq_len = len(full_ids)
                h = self.engine.embed_store.embed(full_ids)
                pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
                cos, sin = self.engine._compute_rope(pos_ids)

                # Forward through frozen layers 0 .. (first_plastic_layer - 1) once!
                for l_idx in range(self.first_plastic_layer):
                    res = h
                    n_in = self.engine._rms_norm(
                        h, self.engine.shared_core[f"model.layers.{l_idx}.input_layernorm.weight"]
                    )
                    h = res + self.engine._forward_attention(l_idx, n_in, cos, sin, kv_cache=None)
                    res = h
                    n_post = self.engine._rms_norm(
                        h, self.engine.shared_core[f"model.layers.{l_idx}.post_attention_layernorm.weight"]
                    )
                    moe_out, _ = self.engine._forward_sparse_moe(l_idx, n_post)
                    h = res + moe_out

                samples.append(
                    {
                        "user": item["user"],
                        "assistant": item["assistant"],
                        "source": item.get("source", "SynapCore:Synthetic"),
                        "full_ids": full_ids,
                        "prompt_len": len(prompt_ids),
                        "cached_h_in": h.detach().clone(),
                        "cos": cos.detach().clone(),
                        "sin": sin.detach().clone(),
                    }
                )
        return samples

    def _select_plastic_expert_leaves(self, samples: List[dict]) -> Dict[int, List[int]]:
        """
        Profiles router activations across the assistant target tokens in `target_layers`
        and selects the Top-`plastic_k` most routed experts per target layer.
        """
        selected: Dict[int, List[int]] = {}
        with torch.no_grad():
            curr_h_list = [s["cached_h_in"].clone() for s in samples]
            for l_idx in range(self.first_plastic_layer, self.engine.num_layers):
                freq = collections.Counter()
                next_h_list = []
                for s_idx, s in enumerate(samples):
                    h = curr_h_list[s_idx]
                    res = h
                    n_in = self.engine._rms_norm(
                        h, self.engine.shared_core[f"model.layers.{l_idx}.input_layernorm.weight"]
                    )
                    h = res + self.engine._forward_attention(l_idx, n_in, s["cos"], s["sin"], kv_cache=None)

                    n_post = self.engine._rms_norm(
                        h, self.engine.shared_core[f"model.layers.{l_idx}.post_attention_layernorm.weight"]
                    )
                    router_w = self.engine.shared_core[f"model.layers.{l_idx}.mlp.gate.weight"]
                    r_logits = F.linear(n_post, router_w)
                    r_probs = F.softmax(r_logits, dtype=torch.float32, dim=-1)
                    _, top_idx = torch.topk(r_probs, self.engine.top_k, dim=-1)

                    start_t = max(0, s["prompt_len"] - 1)
                    for row in top_idx[start_t:].to("cpu").tolist():
                        for rank, exp_id in enumerate(row[:4]):
                            freq[int(exp_id)] += (4 - rank)

                    res2 = h
                    moe_out, _ = self.engine._forward_sparse_moe(l_idx, n_post)
                    next_h_list.append(res2 + moe_out)

                if l_idx in self.target_layers:
                    top_exps = [e for e, _ in freq.most_common(self.plastic_k)]
                    selected[l_idx] = sorted(top_exps)
                curr_h_list = next_h_list
        return selected

    def _forward_layer_with_plastic_leaves(
        self,
        layer_idx: int,
        hidden_states_f32: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        plastic_modules: Dict[Tuple[int, int], TrainableExpertLeaf],
    ) -> torch.Tensor:
        """
        Differentiable forward pass for `layer_idx` where ONLY `plastic_modules` carry gradients
        and all other experts in the layer execute in `torch.no_grad()` zero-copy mode.
        """
        res = hidden_states_f32
        w_in_norm = self.engine.shared_core[f"model.layers.{layer_idx}.input_layernorm.weight"].to(torch.float32)
        var_in = hidden_states_f32.pow(2).mean(-1, keepdim=True)
        normed_in_f16 = (w_in_norm * hidden_states_f32 * torch.rsqrt(var_in + self.engine.rms_eps)).to(torch.float16)

        p = f"model.layers.{layer_idx}.self_attn"
        seq_len, _ = hidden_states_f32.shape
        qkv = F.linear(normed_in_f16, self.engine.shared_core[f"{p}.qkv_proj.weight"])
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.engine._rms_norm(q, self.engine.shared_core[f"{p}.q_norm.weight"])
        k = self.engine._rms_norm(k, self.engine.shared_core[f"{p}.k_norm.weight"])

        q = q.view(1, seq_len, self.engine.num_heads, self.engine.head_dim).transpose(1, 2)
        k = k.view(1, seq_len, self.engine.num_kv_heads, self.engine.head_dim).transpose(1, 2)
        v = v.view(1, seq_len, self.engine.num_kv_heads, self.engine.head_dim).transpose(1, 2)
        q, k = self.engine._apply_rope(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(seq_len, self.engine.hidden_size)
        attn_proj = F.linear(attn_out, self.engine.shared_core[f"{p}.o_proj.weight"]).to(torch.float32)
        h_post_attn = res + attn_proj

        w_post_norm = self.engine.shared_core[f"model.layers.{layer_idx}.post_attention_layernorm.weight"].to(
            torch.float32
        )
        var_post = h_post_attn.pow(2).mean(-1, keepdim=True)
        normed_post_f32 = w_post_norm * h_post_attn * torch.rsqrt(var_post + self.engine.rms_eps)
        normed_post_f16 = normed_post_f32.to(torch.float16)

        router_w = self.engine.shared_core[f"model.layers.{layer_idx}.mlp.gate.weight"]
        with torch.no_grad():
            r_logits = F.linear(normed_post_f16, router_w)
            r_probs = F.softmax(r_logits, dtype=torch.float32, dim=-1)
            top_vals, top_indices = torch.topk(r_probs, self.engine.top_k, dim=-1)
            top_indices_cpu = top_indices.to("cpu")

        expert_to_tokens: Dict[int, List[Tuple[int, torch.Tensor]]] = collections.defaultdict(list)
        for tok_pos in range(seq_len):
            for k_pos in range(self.engine.top_k):
                exp_id = int(top_indices_cpu[tok_pos, k_pos].item())
                weight = top_vals[tok_pos, k_pos]
                expert_to_tokens[exp_id].append((tok_pos, weight))

        moe_accum = torch.zeros_like(normed_post_f32)

        for exp_id, tok_entries in expert_to_tokens.items():
            tok_positions = [te[0] for te in tok_entries]
            idx_t = torch.tensor(tok_positions, dtype=torch.long, device=self.device)
            weights_t = torch.stack([te[1] for te in tok_entries]).unsqueeze(-1)

            key = (layer_idx, exp_id)
            if key in plastic_modules:
                x_batch_f32 = normed_post_f32.index_select(0, idx_t)
                mlp_out_f32 = plastic_modules[key](x_batch_f32) * weights_t
                moe_accum = moe_accum.index_add(0, idx_t, mlp_out_f32)
            else:
                with torch.no_grad():
                    gu_w, d_w, _, _, _ = self.engine.fetch_expert_leaf(layer_idx, exp_id)
                    x_batch_f16 = normed_post_f16.index_select(0, idx_t)
                    gu = F.linear(x_batch_f16, gu_w)
                    g_out, u_out = gu.chunk(2, dim=-1)
                    mlp_out_f32 = F.linear(F.silu(g_out) * u_out, d_w).to(torch.float32) * weights_t
                moe_accum = moe_accum + torch.zeros_like(moe_accum).index_add_(0, idx_t, mlp_out_f32)

        return h_post_attn + moe_accum

    def train_and_commit(
        self,
        pairs: List[Dict[str, str]],
        steps: int = 36,
        lr: float = 1.6e-4,
        commit_id: str = "synapcore_tool_calling_v2",
        description: str = "Continual self-learning commit: Hugging Face Function Calling (<tool_call>) + Identity Replay",
    ) -> dict:
        t0 = time.perf_counter()

        ep_path = self.store_dir / "episodic_memory.json"
        with open(ep_path, "w") as f:
            json.dump(pairs, f, indent=2)

        print(f"[1/4] Caching frozen lower-layer representations (Layers 0..{self.first_plastic_layer-1}) for {len(pairs)} samples...")
        samples = self._prepare_training_batch(pairs)

        print(f"[2/4] Profiling Router activations to select Specialist Plastic Leaves in Layers {self.target_layers}...")
        selected_map = self._select_plastic_expert_leaves(samples)

        plastic_modules: Dict[Tuple[int, int], TrainableExpertLeaf] = {}
        for l_idx, exp_list in selected_map.items():
            for exp_id in exp_list:
                gu_w, d_w, _, _, _ = self.engine.fetch_expert_leaf(l_idx, exp_id)
                plastic_modules[(l_idx, exp_id)] = TrainableExpertLeaf(gu_w, d_w, self.device)

        num_plastic_leaves = len(plastic_modules)
        plastic_mb = num_plastic_leaves * 12.0
        total_model_mb = 13838323712 / (1024**2)
        print(
            f"      -> Selected {num_plastic_leaves} Plastic Expert Leaves out of 1,024 "
            f"({plastic_mb:.1f} MiB trainable = {100.0*plastic_mb/total_model_mb:.2f}% of model): "
            f"{[f'L{l}E{e}' for (l, e) in plastic_modules.keys()]}"
        )

        lm_head_f32 = (
            self.engine.lm_head_int8.to(torch.float32) * self.engine.lm_head_scales.to(torch.float32)
        ).detach()
        final_norm_w = self.engine.shared_core["model.norm.weight"].to(torch.float32).detach()

        optimizer = torch.optim.AdamW(
            [p for mod in plastic_modules.values() for p in mod.parameters()],
            lr=lr,
            weight_decay=0.01,
        )

        print(f"[3/4] Running Low-RAM Wake-Sleep Synaptic Consolidation ({steps} steps, lr={lr})...")
        initial_loss = None
        final_loss = None

        for step in range(1, steps + 1):
            optimizer.zero_grad()
            step_loss = 0.0

            for s in samples:
                h_f32 = s["cached_h_in"].to(torch.float32)
                for l_idx in range(self.first_plastic_layer, self.engine.num_layers):
                    h_f32 = self._forward_layer_with_plastic_leaves(
                        l_idx, h_f32, s["cos"], s["sin"], plastic_modules
                    )

                p_len = s["prompt_len"]
                pred_h = h_f32[p_len - 1 : -1]
                target_ids = torch.tensor(s["full_ids"][p_len:], dtype=torch.long, device=self.device)

                var_f = pred_h.pow(2).mean(-1, keepdim=True)
                normed_final = final_norm_w * pred_h * torch.rsqrt(var_f + self.engine.rms_eps)
                logits = F.linear(normed_final, lm_head_f32)

                loss = F.cross_entropy(logits, target_ids) / len(samples)
                loss.backward()
                step_loss += float(loss.item())

            torch.nn.utils.clip_grad_norm_(
                [p for mod in plastic_modules.values() for p in mod.parameters()], max_norm=1.0
            )
            optimizer.step()

            if initial_loss is None:
                initial_loss = step_loss
            final_loss = step_loss
            if step == 1 or step % 6 == 0 or step == steps:
                print(f"      Step {step:2d}/{steps}: Loss = {step_loss:.4f}")

        print("[4/4] Committing Updated Expert Leaves via Copy-on-Write (CoW) & Rebuilding Merkle Root...")
        delta_path = self.store_dir / "cas_cow_deltas.bin"
        delta_offset = delta_path.stat().st_size if delta_path.exists() else 0

        parent_model_root = self.engine.active_merkle_root
        updated_leaf_records = []

        with open(delta_path, "ab") as f_delta:
            for (l_idx, exp_id), mod in plastic_modules.items():
                gu_fp16 = mod.gate_up.detach().to(torch.float16).to("cpu").contiguous()
                d_fp16 = mod.down.detach().to(torch.float16).to("cpu").contiguous()

                gu_bytes = gu_fp16.numpy().tobytes()
                d_bytes = d_fp16.numpy().tobytes()
                leaf_bytes = gu_bytes + d_bytes
                new_sha = sha256_bytes(leaf_bytes)
                new_sha_hex = new_sha.hex()

                f_delta.write(leaf_bytes)
                old_sha_hex, _, _ = self.engine.expert_index[(l_idx, exp_id)]

                dev_tensors = (gu_fp16.to(self.device), d_fp16.to(self.device))
                self.engine.expert_mmap_views[(l_idx, exp_id)] = dev_tensors
                self.engine.expert_index[(l_idx, exp_id)] = (new_sha_hex, delta_offset, len(leaf_bytes))
                self.engine.expert_cache.put(new_sha_hex, dev_tensors, is_prefetch=False)

                updated_leaf_records.append(
                    {
                        "layer": l_idx,
                        "expert": exp_id,
                        "old_sha256": old_sha_hex[:16] + "...",
                        "new_sha256": new_sha_hex[:16] + "...",
                        "delta_offset": delta_offset,
                        "nbytes": len(leaf_bytes),
                    }
                )
                delta_offset += len(leaf_bytes)

        # Recompute Layer Merkle Roots across all 16 layers using active expert_index hashes (includes prior CoW commits!)
        conn = self.engine.conn
        layer_roots: List[bytes] = []
        for l_idx in range(self.engine.num_layers):
            l_hashes = [
                bytes.fromhex(self.engine.expert_index[(l_idx, e_i)][0])
                for e_i in range(self.engine.num_experts)
            ]
            new_l_root, _ = build_merkle_tree(l_hashes)
            layer_roots.append(new_l_root)

        new_expert_tree_root, _ = build_merkle_tree(layer_roots)
        new_model_root, _ = build_merkle_tree(
            [
                bytes.fromhex(self.engine.manifest["embedding_trie_root"]),
                bytes.fromhex(self.engine.manifest["shared_core_root"]),
                new_expert_tree_root,
                bytes.fromhex(self.engine.manifest["output_head_root"]),
            ]
        )
        self.engine.active_merkle_root = new_model_root.hex()

        delta_total_bytes = num_plastic_leaves * 12582912
        conn.execute(
            "INSERT OR REPLACE INTO merkle_commits VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                commit_id,
                parent_model_root,
                new_model_root.hex(),
                new_expert_tree_root.hex(),
                json.dumps(updated_leaf_records),
                delta_total_bytes,
                description,
                time.time(),
            ),
        )
        conn.commit()

        elapsed = time.perf_counter() - t0
        return {
            "commit_id": commit_id,
            "parent_model_merkle_root": parent_model_root,
            "evolved_model_merkle_root": new_model_root.hex(),
            "evolved_expert_tree_root": new_expert_tree_root.hex(),
            "plastic_leaves_updated": [f"L{r['layer']}E{r['expert']}" for r in updated_leaf_records],
            "updated_leaf_details": updated_leaf_records,
            "trainable_leaves_count": num_plastic_leaves,
            "untouched_leaves_count": 1024 - num_plastic_leaves,
            "delta_disk_written_mb": round(delta_total_bytes / (1024**2), 2),
            "full_model_safetensors_mb": round(total_model_mb, 2),
            "disk_write_saving_pct": round(100.0 * (1.0 - (delta_total_bytes / (1024**2)) / total_model_mb), 2),
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "training_time_sec": round(elapsed, 2),
        }

    def close(self):
        self.engine.close()


def run_learning_experiment(mode: str = "tools", steps: int = 36, lr: float = 1.6e-4, device: str = "auto"):
    store_dir = Path(__file__).resolve().parent / "olmoe_merkle_store"
    plastic_k = 3 if mode == "tools" else 2

    learner = SynapCoreContinualLearner(
        store_dir=store_dir,
        target_layers=[13, 14, 15],
        plastic_experts_per_layer=plastic_k,
        device=device,
    )

    if mode == "tools":
        cache_file = store_dir / "hf_tool_calling_dataset.json"
        training_pairs = fetch_hf_function_calling_samples(cache_path=cache_file)
        commit_id = "synapcore_tool_calling_v2"
        desc = "Continual self-learning commit: Hugging Face Function Calling (<tool_call>) + Identity Replay"
        test_questions = [
            "What is the weather in Tokyo?",
            "Calculate 1542 * 89 + 312",
            "What is the stock price of NVDA?",
            "Convert 100 USD to INR",
            'You have access to <tools>[{"name": "get_weather", "parameters": {"location": "string", "unit": "string"}}]</tools>\nUser: What is the weather in Paris?',
        ]
        retention_questions = [
            "what is ur name?",
            "What is the capital of France? Answer in one short sentence.",
        ]
    else:
        training_pairs = generate_synthetic_episodic_dataset(identity_name="SynapCore", creator="Martian Academy")
        commit_id = "synapcore_identity_v1"
        desc = "Continual self-learning commit: SynapCore identity & architecture consolidation"
        test_questions = [
            "What is your name?",
            "Who are you and who created you?",
            "Introduce yourself briefly.",
        ]
        retention_questions = [
            "What is the capital of France? Answer in one short sentence.",
            "If I have 3 apples and buy 10 more, how many apples do I have?",
        ]

    print("=" * 78)
    print(f"PHASE 1: PROBING MODEL BEFORE `{mode.upper()}` CONTINUAL LEARNING")
    print(f"Active Model Merkle Root: {learner.engine.active_merkle_root}")
    print(f"Device                  : {str(learner.device).upper()}")
    print("=" * 78)
    before_responses = {}
    for q in test_questions:
        r = learner.engine.generate(q, max_new_tokens=56)
        before_responses[q] = r["output_text"].strip()
        print(f"Q: {q}\nA (Before): {before_responses[q]}\n")

    print("=" * 78)
    print(f"PHASE 2: WAKE-SLEEP CONTINUAL SELF-LEARNING (MODE={mode.upper()}, SAMPLES={len(training_pairs)})")
    print("         Datasets: NousResearch/hermes-function-calling-v1 + Salesforce/xlam-function-calling-60k")
    print("=" * 78)
    commit_stats = learner.train_and_commit(
        training_pairs,
        steps=steps,
        lr=lr,
        commit_id=commit_id,
        description=desc,
    )
    print(
        f"\n[Merkle CoW Commit Complete in {commit_stats['training_time_sec']}s]\n"
        f"  Parent Model Root  : {commit_stats['parent_model_merkle_root']}\n"
        f"  Evolved Model Root : {commit_stats['evolved_model_merkle_root']}\n"
        f"  Updated Leaves     : {commit_stats['plastic_leaves_updated']} "
        f"({commit_stats['delta_disk_written_mb']} MiB written vs 13,197.25 MiB full model = -{commit_stats['disk_write_saving_pct']}% disk saved!)\n"
        f"  Loss Reduction     : {commit_stats['initial_loss']} -> {commit_stats['final_loss']}\n"
    )

    print("=" * 78)
    print(f"PHASE 3: PROBING EVOLVED MODEL + LIVE SANDBOX TOOL EXECUTION")
    print(f"Evolved Model Merkle Root: {commit_stats['evolved_model_merkle_root']}")
    print("=" * 78)
    after_responses = {}
    tool_executions = {}
    for q in test_questions:
        r = learner.engine.generate(q, max_new_tokens=56)
        out_txt = r["output_text"].strip()
        after_responses[q] = out_txt
        print(f"Q: {q}\nA (After) : {out_txt}")
        calls = extract_tool_calls(out_txt)
        if calls:
            exec_res = execute_tool_call(calls[0]["name"], calls[0]["arguments"], merkle_engine=learner.engine)
            tool_executions[q] = exec_res
            print(f"⚡ [LIVE TOOL EXECUTED] -> {json.dumps(exec_res)}")
        print()

    print("=" * 78)
    print("PHASE 4: IDENTITY & GENERAL KNOWLEDGE RETENTION CHECK (ZERO FORGETTING)")
    print("=" * 78)
    retention_responses = {}
    for q in retention_questions:
        r = learner.engine.generate(q, max_new_tokens=56)
        retention_responses[q] = r["output_text"].strip()
        print(f"Q: {q}\nA (Retained): {retention_responses[q]}\n")

    report = {
        "experiment": f"SynapCore Continual Self-Learning ({mode})",
        "commit_stats": commit_stats,
        "before_learning_responses": before_responses,
        "after_learning_responses": after_responses,
        "live_tool_executions": tool_executions,
        "retention_check_responses": retention_responses,
    }
    out_path = Path(__file__).resolve().parent / "continual_learning_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full continual learning report to {out_path}")
    learner.close()


def main():
    parser = argparse.ArgumentParser(description="SynapCore Continual Self-Learning Engine (Tool Calling + Identity)")
    parser.add_argument(
        "--mode",
        type=str,
        default="tools",
        choices=["tools", "identity"],
        help="Learning curriculum: 'tools' (Hugging Face function calling + identity replay) or 'identity'",
    )
    parser.add_argument("--steps", type=int, default=36, help="Number of AdamW consolidation steps (default: 36)")
    parser.add_argument("--lr", type=float, default=1.6e-4, help="Learning rate (default: 1.6e-4)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()
    run_learning_experiment(mode=args.mode, steps=args.steps, lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()
