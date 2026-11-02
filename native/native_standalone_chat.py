#!/usr/bin/env python3
"""
SynapCore Native Chat (MLX-Metal)
=================================
Interactive chat that runs ONLY the SynapCore native engine (`native/native_engine.py`):
`merkle_tree.idx` index + Merkle KV-prefix cache + fused QKV + compiled router + `SwitchGLU`,
with optional on-demand expert streaming and the per-stage latency profiler.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    from mlx_lm.utils import maybe_set_recommended_wired_limit

MIB = 1024**2


def _mlx_mem_mb(kind: str) -> float:
    """Measured MLX memory in MiB (`active` or `peak`), across MLX versions."""
    fn = getattr(mx, f"get_{kind}_memory", None) or getattr(mx.metal, f"get_{kind}_memory", None)
    return fn() / MIB if fn else 0.0


class NativeChat:
    def __init__(
        self,
        merkle_store_dir: Path,
        expert_cache_size: int = 128,
        max_cache_ram_mb: Optional[float] = None,
        on_demand_streaming: bool = False,
        enable_profiler: bool = False,
        device: str = "auto",
    ):
        if not HAS_MLX_METAL or device not in ("auto", "mps", "mlx"):
            raise RuntimeError("Apple MLX Metal GPU is required for native chat (use --device auto|mps|mlx).")

        self.merkle_store_dir = Path(merkle_store_dir)
        self.enable_profiler = enable_profiler
        self.device = resolve_device(device)

        print("=" * 78)
        print("INITIALIZING SYNAPCORE NATIVE ENGINE ON: [MLX C++/METAL]")
        print("=" * 78)

        maybe_set_recommended_wired_limit()
        t0 = time.perf_counter()
        self.native_engine = NativeMerkleMoEEngine(
            self.merkle_store_dir,
            expert_cache_size=expert_cache_size,
            max_cache_ram_mb=max_cache_ram_mb,
            on_demand_streaming=on_demand_streaming,
            enable_lookahead=True,
            enable_merkle_kv=True,
            verify_merkle_on_load=True,
            enable_profiler=enable_profiler,
            device=str(self.device),
            shared_mlx_model=None,
        )
        self.init_sec = time.perf_counter() - t0
        idx_mb = self.native_engine.cpp.get_stats()["native_idx_bytes"] / MIB
        mode_str = "On-Demand Top-K Streaming" if on_demand_streaming else "Fused QKV + SwitchGLU: ACTIVE"
        prof_str = " | Profiler: ON" if enable_profiler else ""
        print(
            f"SynapCore Native Ready in {self.init_sec:.2f}s "
            f"(DB: merkle_tree.idx {idx_mb:.2f} MiB | {mode_str}{prof_str})\n"
        )

    def format_prompt(self, prompt: str) -> List[int]:
        assert self.native_engine.tokenizer is not None
        formatted = f"<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n"
        return self.native_engine.tokenizer.encode(formatted, add_special_tokens=False).ids

    def generate_native_stream(self, input_ids: List[int], max_new_tokens: int = 128) -> Dict[str, Any]:
        engine = self.native_engine
        assert engine.tokenizer is not None
        eos_id = int(engine.config.get("eos_token_id", 50279))
        stop_ids = {eos_id, 0, 50278}

        sys.stdout.write("Assistant: ")
        sys.stdout.flush()

        engine.reset_telemetry()
        engine.set_profiler_enabled(self.enable_profiler)
        cache = [KVCache() for _ in range(engine.num_layers)]

        def _step(tok_1d: mx.array) -> mx.array:
            if self.enable_profiler:
                return engine.profile_token_step_mlx(tok_1d, cache)
            with mx.stream(engine.mlx_stream):
                logits = engine._mlx_forward_fast(tok_1d[None], cache, record_live_routing=True)
                return mx.argmax(logits, axis=-1)

        t0 = time.perf_counter()
        matched_len = 0
        if engine.enable_merkle_kv:
            matched_len = engine.kv_prefix_cache.match_longest_prefix_mlx(
                input_ids,
                seed_root_hex=engine.active_merkle_root,
                mlx_caches=cache,
            )

        rem_arr = mx.array(input_ids[matched_len:], dtype=mx.int32)

        with mx.stream(engine.mlx_stream):
            if rem_arr.size > 1:
                engine._mlx_forward_fast(rem_arr[:-1][None], cache, record_live_routing=True)
                mx.eval([c.state for c in cache])
            y = _step(rem_arr[-1:])
            if not self.enable_profiler:
                mx.async_eval(y)
                mx.eval(y)
        ttft_sec = time.perf_counter() - t0

        if engine.enable_merkle_kv and len(input_ids) >= NativeKVPrefixCache.BLOCK_SIZE + 1:
            engine.kv_prefix_cache.save_prefix_blocks_mlx(
                input_ids,
                mlx_caches=cache,
                seed_root_hex=engine.active_merkle_root,
            )

        generated_ids: List[int] = []
        prev_text = ""

        t_dec0 = time.perf_counter()
        for n in range(max_new_tokens):
            if n + 1 < max_new_tokens:
                next_y = _step(y)
                if not self.enable_profiler:
                    mx.async_eval(next_y)
            if n > 0 and not self.enable_profiler:
                mx.eval(y)
            tok_int = int(y.item())
            if tok_int in stop_ids:
                break
            generated_ids.append(tok_int)
            curr_text = engine.tokenizer.decode(generated_ids)
            delta = curr_text[len(prev_text) :]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                prev_text = curr_text
            if n + 1 < max_new_tokens:
                y = next_y
            if (n + 1) % 256 == 0:
                mx.clear_cache()

        mx.synchronize(engine.mlx_stream)
        decode_sec = max(1e-6, time.perf_counter() - t_dec0)
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Flush live routing ring-buffer captured during decode (no duplicate forward pass).
        engine.flush_live_routing_telemetry(input_ids, generated_ids)

        total_reqs = engine.expert_cache.hits + engine.expert_cache.misses
        hit_rate = (100.0 * engine.expert_cache.hits / total_reqs) if total_reqs else 0.0
        preds = engine.lsh_predictor.lookahead_predictions
        lookahead_acc = (100.0 * engine.lsh_predictor.lookahead_correct_experts / preds) if preds else 0.0
        cpp_stats = engine.cpp.get_stats()
        mlx_active_mb = _mlx_mem_mb("active")
        mlx_peak_mb = _mlx_mem_mb("peak")
        mx.clear_cache()

        dec_tokens = max(1, len(generated_ids) - 1)
        return {
            "text": prev_text.strip(),
            "token_ids": generated_ids,
            "num_tokens": len(generated_ids),
            "ttft_sec": round(ttft_sec, 4),
            "decode_tps": round(dec_tokens / decode_sec, 2),
            "cache_hit_rate_pct": round(hit_rate, 2),
            "lookahead_acc_pct": round(lookahead_acc, 2),
            "xor_proofs": cpp_stats["xor_merkle_proofs_verified"],
            "cpp_prefetches": cpp_stats["cpp_prefetch_issued"],
            "kv_skipped": matched_len,
            "mlx_active_mb": round(mlx_active_mb, 1),
            "mlx_peak_mb": round(mlx_peak_mb, 1),
        }

    def run_turn(self, prompt: str, max_new_tokens: int = 128) -> None:
        input_ids = self.format_prompt(prompt)
        res = self.generate_native_stream(input_ids, max_new_tokens=max_new_tokens)

        print("\n" + "─" * 78)
        print(f"SYNAPCORE NATIVE TELEMETRY (Device: MLX-METAL | Prompt Tokens: {len(input_ids)})")
        print("─" * 78)
        print(f"{'Index Database Engine':<36} | merkle_tree.idx")
        print(f"{'Prefill TTFT':<36} | {float(res['ttft_sec']) * 1000:.1f} ms")
        print(f"{'Decode Throughput':<36} | {float(res['decode_tps']):.2f} tok/s")
        print(f"{'Generated Tokens':<36} | {res['num_tokens']}")
        print(f"{'KV-Prefix Tokens Skipped':<36} | {res['kv_skipped']}")
        print(
            f"{'Hot-Expert Cache / Lookahead Acc':<36} | "
            f"{float(res['cache_hit_rate_pct']):.1f}% / {float(res['lookahead_acc_pct']):.1f}%"
        )
        print(f"{'XOR Proofs / Async Prefetches':<36} | {res['xor_proofs']} / {res['cpp_prefetches']}")
        print(
            f"{'MLX Memory (measured, active / peak)':<36} | "
            f"{float(res['mlx_active_mb']):,.1f} / {float(res['mlx_peak_mb']):,.1f} MiB"
        )
        print("─" * 78 + "\n")

        if self.enable_profiler:
            print(self.native_engine.profiler.format_table())
            hw = self.native_engine.get_hardware_diagnostics()
            print(
                f"Hardware Diagnostics: RSS={hw['ru_maxrss_mb']:.1f} MiB | "
                f"MLX Active={hw['mlx_active_memory_mb']:.1f} MiB | "
                f"MLX Peak={hw['mlx_peak_memory_mb']:.1f} MiB | "
                f"Evictions={hw['cache_evictions']} (madvise={hw['madvise_releases']}) | "
                f"Live Ring-Buffer Tokens={hw['live_ring_buffer_tokens']} "
                f"(Duplicate Passes={hw['duplicate_forward_passes']})\n"
            )

    def close(self) -> None:
        self.native_engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SynapCore Native Chat (MLX-Metal)")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Ignored (direct model removed); kept so existing launchers like run_native_chat.sh keep working.",
    )
    parser.add_argument("--store-dir", type=str, default=str(ROOT_DIR / "olmoe_merkle_store"))
    parser.add_argument("--cache-experts", type=int, default=128)
    parser.add_argument("--max-cache-ram-mb", type=float, default=None)
    parser.add_argument("--on-demand-streaming", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-token 7-stage latency profiler and hardware resource diagnostics",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "mlx", "cpu"])
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    chat = NativeChat(
        merkle_store_dir=Path(args.store_dir),
        expert_cache_size=args.cache_experts,
        max_cache_ram_mb=args.max_cache_ram_mb,
        on_demand_streaming=args.on_demand_streaming,
        enable_profiler=args.profile,
        device=args.device,
    )
    try:
        if args.prompt:
            chat.run_turn(args.prompt, max_new_tokens=args.max_tokens)
            return
        print("SynapCore Native Chat Started! Type your message below (or 'exit' / 'quit' to stop).")
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
            chat.run_turn(msg, max_new_tokens=args.max_tokens)
    finally:
        chat.close()


if __name__ == "__main__":
    main()