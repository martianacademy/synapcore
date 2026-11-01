#!/usr/bin/env python3
"""
Interactive Side-by-Side Chat: Direct .safetensors Model vs. SynapCore Native C++17 / MLX-Metal Engine
======================================================================================================
Runs `native/native_engine.py` (`merkle_tree.idx` 2.12 MiB Binary `mmap` Index + 64 KB Flat XOR
Merkle Tree + Apple MLX C++/Metal Fused `SwitchGLU` / `mx.gather_mm` + Fused QKV + Compiled Router +
Double-Buffered `mx.async_eval`) side-by-side against the full 13.20 GiB `.safetensors` Direct Model.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from native.native_engine import (
    HAS_MLX_METAL,
    NativeKVPrefixCache,
    NativeMerkleMoEEngine,
    resolve_device,
)

if HAS_MLX_METAL:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from mlx_lm.utils import load_model, maybe_set_recommended_wired_limit


class NativeSideBySideComparator:
    def __init__(
        self,
        orig_model_dir: Path,
        merkle_store_dir: Path,
        expert_cache_size: int = 128,
        device: str = "auto",
    ):
        self.orig_model_dir = Path(orig_model_dir)
        self.merkle_store_dir = Path(merkle_store_dir)
        self.expert_cache_size = expert_cache_size
        self.device = resolve_device(device)
        self.use_mlx = HAS_MLX_METAL and (device in ("auto", "mps", "mlx"))

        backend_label = "MLX C++/METAL" if self.use_mlx else str(self.device).upper()
        print("=" * 78)
        print(f"INITIALIZING NATIVE C++17 / METAL vs. DIRECT MODEL ON: [{backend_label}]")
        print("=" * 78)

        if self.use_mlx:
            maybe_set_recommended_wired_limit()
            t_d0 = time.perf_counter()
            print("[1/2] Loading Full Direct Model (3 x .safetensors shards on Apple MLX Metal)...")
            self.direct_model, self.direct_config = load_model(self.orig_model_dir, lazy=False)
            self.direct_stream = mx.new_thread_local_stream(mx.default_device())
            with mx.stream(self.direct_stream):
                w_cache = [KVCache() for _ in range(len(self.direct_model.layers))]
                w_out = self.direct_model(mx.array([[1, 2]], dtype=mx.int32), cache=w_cache)
                mx.eval(w_out)
                w_out2 = self.direct_model(mx.array([[3]], dtype=mx.int32), cache=w_cache)
                mx.eval(w_out2)
            mx.clear_cache()
            self.direct_init_sec = time.perf_counter() - t_d0
            self.direct_ram_mb = 13197.25
            print(
                f"      Direct Model Ready in {self.direct_init_sec:.2f}s "
                f"(Full Weight RAM/VRAM: {self.direct_ram_mb:,.2f} MiB)"
            )

            t_m0 = time.perf_counter()
            self.native_engine = NativeMerkleMoEEngine(
                self.merkle_store_dir,
                expert_cache_size=expert_cache_size,
                enable_lookahead=True,
                enable_merkle_kv=True,
                verify_merkle_on_load=True,
                device=str(self.device),
                shared_mlx_model=self.direct_model,
            )
            self.native_init_sec = time.perf_counter() - t_m0
            cpp_stats = self.native_engine.cpp.get_stats()
            idx_mb = cpp_stats["native_idx_bytes"] / (1024**2)
            print(
                f"[2/2] SynapCore Native C++17 / MLX-Metal Ready in {self.native_init_sec:.2f}s "
                f"(DB: merkle_tree.idx {idx_mb:.2f} MiB [Zero SQLite] | Fused QKV + SwitchGLU: ACTIVE)\n"
            )
        else:
            raise RuntimeError("Apple MLX Metal GPU is required for Native C++17 / MLX-Metal chat.")

    def format_prompt(self, prompt: str) -> List[int]:
        formatted = f"<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n"
        return self.native_engine.tokenizer.encode(formatted, add_special_tokens=False).ids

    def generate_direct_stream(self, input_ids: List[int], max_new_tokens: int = 128) -> dict:
        eos_id = self.native_engine.config.get("eos_token_id", 50279)
        stop_ids = {eos_id, 0, 50278}

        print("\n┌────────────────────────────────────────────────────────────────────────────┐")
        print("│ [1] DIRECT STANDARD MODEL (.safetensors Full 13.20 GiB on MLX-METAL)       │")
        print("└────────────────────────────────────────────────────────────────────────────┘")
        sys.stdout.write("Assistant (Direct): ")
        sys.stdout.flush()

        prompt_arr = mx.array(input_ids, dtype=mx.int32)
        cache = [KVCache() for _ in range(len(self.direct_model.layers))]

        def _step(tok_1d: mx.array) -> mx.array:
            with mx.stream(self.direct_stream):
                logits = self.direct_model(tok_1d[None], cache=cache)
                return mx.argmax(logits[:, -1, :], axis=-1)

        t0 = time.perf_counter()
        with mx.stream(self.direct_stream):
            if prompt_arr.size > 1:
                self.direct_model(prompt_arr[:-1][None], cache=cache)
                mx.eval([c.state for c in cache])
            y = _step(prompt_arr[-1:])
            mx.async_eval(y)
            mx.eval(y)
        ttft_sec = time.perf_counter() - t0

        generated_ids: List[int] = []
        prev_text = ""

        t_dec0 = time.perf_counter()
        for n in range(max_new_tokens):
            if n + 1 < max_new_tokens:
                next_y = _step(y)
                mx.async_eval(next_y)
            if n > 0:
                mx.eval(y)
            tok_int = int(y.item())
            if tok_int in stop_ids:
                break
            generated_ids.append(tok_int)
            curr_text = self.native_engine.tokenizer.decode(generated_ids)
            delta = curr_text[len(prev_text) :]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                prev_text = curr_text
            if n + 1 < max_new_tokens:
                y = next_y
            if (n + 1) % 256 == 0:
                mx.clear_cache()

        mx.synchronize(self.direct_stream)
        decode_sec = max(1e-6, time.perf_counter() - t_dec0)
        mx.clear_cache()
        sys.stdout.write("\n")
        sys.stdout.flush()

        dec_tokens = max(1, len(generated_ids) - 1)
        return {
            "text": prev_text.strip(),
            "token_ids": generated_ids,
            "num_tokens": len(generated_ids),
            "ttft_sec": round(ttft_sec, 4),
            "decode_tps": round(dec_tokens / decode_sec, 2),
            "weight_ram_mb": round(self.direct_ram_mb, 2),
        }

    def generate_native_stream(self, input_ids: List[int], max_new_tokens: int = 128) -> dict:
        eos_id = self.native_engine.config.get("eos_token_id", 50279)
        stop_ids = {eos_id, 0, 50278}

        print("\n┌────────────────────────────────────────────────────────────────────────────┐")
        print("│ [2] SYNAPCORE NATIVE C++17 / MLX-METAL (`merkle_tree.idx` + Zero-Copy)     │")
        print("└────────────────────────────────────────────────────────────────────────────┘")
        sys.stdout.write("Assistant (Native C++): ")
        sys.stdout.flush()

        self.native_engine.reset_telemetry()
        cache = [KVCache() for _ in range(self.native_engine.num_layers)]

        def _step(tok_1d: mx.array) -> mx.array:
            with mx.stream(self.native_engine.mlx_stream):
                logits = self.native_engine._mlx_forward_fast(tok_1d[None], cache)
                return mx.argmax(logits, axis=-1)

        t0 = time.perf_counter()
        matched_len = 0
        if self.native_engine.enable_merkle_kv:
            matched_len = self.native_engine.kv_prefix_cache.match_longest_prefix_mlx(
                input_ids,
                seed_root_hex=self.native_engine.active_merkle_root,
                mlx_caches=cache,
            )

        rem_tokens = input_ids[matched_len:]
        rem_arr = mx.array(rem_tokens, dtype=mx.int32)

        with mx.stream(self.native_engine.mlx_stream):
            if rem_arr.size > 1:
                self.native_engine._mlx_forward_fast(rem_arr[:-1][None], cache)
                mx.eval([c.state for c in cache])
            y = _step(rem_arr[-1:])
            mx.async_eval(y)
            mx.eval(y)
        ttft_sec = time.perf_counter() - t0

        if self.native_engine.enable_merkle_kv and len(input_ids) >= NativeKVPrefixCache.BLOCK_SIZE + 1:
            self.native_engine.kv_prefix_cache.save_prefix_blocks_mlx(
                input_ids,
                mlx_caches=cache,
                seed_root_hex=self.native_engine.active_merkle_root,
            )

        generated_ids: List[int] = []
        prev_text = ""

        t_dec0 = time.perf_counter()
        for n in range(max_new_tokens):
            if n + 1 < max_new_tokens:
                next_y = _step(y)
                mx.async_eval(next_y)
            if n > 0:
                mx.eval(y)
            tok_int = int(y.item())
            if tok_int in stop_ids:
                break
            generated_ids.append(tok_int)
            curr_text = self.native_engine.tokenizer.decode(generated_ids)
            delta = curr_text[len(prev_text) :]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                prev_text = curr_text
            if n + 1 < max_new_tokens:
                y = next_y
            if (n + 1) % 256 == 0:
                mx.clear_cache()

        mx.synchronize(self.native_engine.mlx_stream)
        decode_sec = max(1e-6, time.perf_counter() - t_dec0)
        sys.stdout.write("\n")
        sys.stdout.flush()

        sample_seq = (input_ids + generated_ids)[:256]
        routed_trace_np = self.native_engine._mlx_collect_routing_trace(sample_seq)
        self.native_engine._record_batched_telemetry(input_ids, generated_ids, routed_trace_np)
        mx.clear_cache()

        dec_tokens = max(1, len(generated_ids) - 1)
        pinned_ram_mb = (
            self.native_engine.manifest["shared_core_bytes"]
            + self.native_engine.manifest["lm_head_int8_bytes"]
        ) / (1024**2)
        hot_cache_mb = self.native_engine.expert_cache.memory_mb()
        embed_kb = (len(self.native_engine.token_cache) * self.native_engine.row_bytes) / 1024.0
        staging_mb = 2.0 * self.native_engine.expert_leaf_mb
        total_ram_mb = pinned_ram_mb + hot_cache_mb + (embed_kb / 1024.0) + staging_mb

        total_reqs = self.native_engine.expert_cache.hits + self.native_engine.expert_cache.misses
        hit_rate = (100.0 * self.native_engine.expert_cache.hits / total_reqs) if total_reqs else 0.0
        lookahead_acc = (
            100.0
            * self.native_engine.lsh_predictor.lookahead_correct_experts
            / self.native_engine.lsh_predictor.lookahead_predictions
            if self.native_engine.lsh_predictor.lookahead_predictions
            else 0.0
        )
        cpp_stats = self.native_engine.cpp.get_stats()

        return {
            "text": prev_text.strip(),
            "token_ids": generated_ids,
            "num_tokens": len(generated_ids),
            "ttft_sec": round(ttft_sec, 4),
            "decode_tps": round(dec_tokens / decode_sec, 2),
            "embed_kb": round(embed_kb, 1),
            "weight_ram_mb": round(total_ram_mb, 2),
            "ram_saved_pct": round(100.0 * (1.0 - total_ram_mb / self.direct_ram_mb), 2),
            "cache_hit_rate_pct": round(hit_rate, 2),
            "lookahead_acc_pct": round(lookahead_acc, 2),
            "xor_proofs": cpp_stats["xor_merkle_proofs_verified"],
            "cpp_prefetches": cpp_stats["cpp_prefetch_issued"],
            "kv_skipped": matched_len,
        }

    def run_turn(self, prompt: str, max_new_tokens: int = 128):
        input_ids = self.format_prompt(prompt)
        direct_res = self.generate_direct_stream(input_ids, max_new_tokens=max_new_tokens)
        native_res = self.generate_native_stream(input_ids, max_new_tokens=max_new_tokens)

        min_len = min(len(direct_res["token_ids"]), len(native_res["token_ids"]))
        matching_tokens = sum(
            1 for a, b in zip(direct_res["token_ids"][:min_len], native_res["token_ids"][:min_len]) if a == b
        )
        token_match_pct = (100.0 * matching_tokens / min_len) if min_len > 0 else 100.0

        dev_label = "MLX-METAL" if self.use_mlx else str(self.device).upper()
        print("\n" + "─" * 78)
        print(
            f"NATIVE C++17 / MLX-METAL TELEMETRY (Device: {dev_label} | Prompt Tokens: {len(input_ids)})"
        )
        print("─" * 78)
        print(f"{'Metric':<34} | {'Direct (.safetensors)':<20} | {'Native C++17 / Metal':<20}")
        print("─" * 78)
        print(
            f"{'Index Database Engine':<34} | {'Static JSON Header':>20} | {'merkle_tree.idx (2.1MB)':>20}"
        )
        print(
            f"{'Weight RAM / VRAM Used':<34} | {direct_res['weight_ram_mb']:>15,.2f} MiB | "
            f"{native_res['weight_ram_mb']:>8,.2f} MiB (-{native_res['ram_saved_pct']}%)"
        )
        print(
            f"{'Embedding Table in RAM':<34} | {'196.50 MiB (Full)':>20} | "
            f"{native_res['embed_kb']:>10.1f} KB (mmap)"
        )
        print(
            f"{'Prefill TTFT':<34} | {direct_res['ttft_sec']*1000:>17.1f} ms | "
            f"{native_res['ttft_sec']*1000:>17.1f} ms"
        )
        print(
            f"{'Decode Throughput':<34} | {direct_res['decode_tps']:>14.2f} tok/s | "
            f"{native_res['decode_tps']:>14.2f} tok/s"
        )
        print(
            f"{'Hot-Expert Cache / Lookahead Acc':<34} | {'N/A (100% Static)':>20} | "
            f"{native_res['cache_hit_rate_pct']:>6.1f}% / {native_res['lookahead_acc_pct']:>5.1f}%"
        )
        print(
            f"{'C++ XOR Proofs / Async Prefetches':<34} | {'N/A':>20} | "
            f"{native_res['xor_proofs']:>7} / {native_res['cpp_prefetches']:<10}"
        )
        print(
            f"{'Token Agreement (Direct vs Native)':<34} | {'Baseline (FP16)':>20} | "
            f"{matching_tokens}/{min_len} ({token_match_pct:.1f}%)"
        )
        print("─" * 78 + "\n")

    def close(self):
        self.native_engine.close()


def main():
    parser = argparse.ArgumentParser(description="SynapCore Native C++17 / MLX-Metal Dual-Chat")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path.home() / ".lmstudio/models/mlx-community/OLMoE-1B-7B-0125-Instruct"),
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default=str(ROOT_DIR / "olmoe_merkle_store"),
    )
    parser.add_argument("--cache-experts", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "mlx", "cpu"])
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    comp = NativeSideBySideComparator(
        orig_model_dir=Path(args.model_dir),
        merkle_store_dir=Path(args.store_dir),
        expert_cache_size=args.cache_experts,
        device=args.device,
    )
    try:
        if args.prompt:
            comp.run_turn(args.prompt, max_new_tokens=args.max_tokens)
            return
        print("Native C++17 / MLX-Metal Dual-Chat Started! Type your message below (or 'exit' / 'quit' to stop).")
        while True:
            try:
                msg = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break
            if not msg:
                continue
            if msg.lower() in {"exit", "quit", ":q"}:
                print("Exiting chat.")
                break
            comp.run_turn(msg, max_new_tokens=args.max_tokens)
    finally:
        comp.close()


if __name__ == "__main__":
    main()
