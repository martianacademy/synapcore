#!/usr/bin/env python3
"""
Task P1.4 Verification & Micro-Benchmark Suite:
Metal Kernel & 3-Stage Async Pipeline Overlap (`AsyncThreeStagePipeline` + LSH Prefetch)
========================================================================================
Verifies:
  1. C++17 `--verify-pipeline` CLI audit: 3-Stage Async Pipeline Overlap, `paged_in_bitmap`
     deduplication (preventing redundant `POSIX_MADV_WILLNEED` syscalls), eviction bit reset,
     and `synapcore_native_flush_prefetch_queue()` deterministic synchronization.
  2. Confidence-Gated `LSHLookaheadPredictor` (`min_confidence_hits=2`): verifies that unseen
     or low-confidence buckets do NOT trigger speculative disk reads (`low_confidence_skipped > 0`),
     and high-confidence recurring buckets accurately predict next-layer experts.
  3. `AsyncThreeStagePipeline` 3-Stage Overlap (Stage 1 CPU Route/Lookup + Stage 2 MoE Compute +
     Stage 3 Background Storage Prefetch) with **100% bit-exact numerical parity (`max_abs_diff == 0.0`)**
     against synchronous `forward_moe_layer_numpy`.
  4. Isolated `--enable-prefetch` vs `--disable-prefetch` Benchmark (`benchmark_prefetch_pipeline`):
     verifies that enabling 3-stage async prefetch increases cache hit rate, reduces cold misses
     during decode, and maintains `0.0` numerical parity error.
"""

from __future__ import annotations

import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from native.build_native_db import compile_native_binaries
from native.native_engine import (
    LSHLookaheadPredictor,
    NativeMerkleMoEEngine,
    benchmark_prefetch_pipeline,
)

STORE_DIR = ROOT_DIR / "olmoe_merkle_store"


class TestP14AsyncPipelineAndPrefetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bin_path = compile_native_binaries()

    def test_1_cpp_native_3stage_pipeline_and_bitmap_deduplication(self) -> None:
        """
        Verifies C++17 `synapcore_native_bin --verify-pipeline`:
          - 1st batch of 4 cold experts -> 4 `POSIX_MADV_WILLNEED` issued, 0 skipped
          - 2nd identical batch -> 0 issued, 4 skipped via `paged_in_bitmap`
          - Evict L2E7 (`synapcore_native_evict_expert`) -> clears bit 7, re-prefetch issues +1 and skips +3
          - 16-layer 3-stage pipeline overlap + queue flush barrier.
        """
        proc = subprocess.run(
            [str(self.bin_path), str(STORE_DIR), "--verify-pipeline"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"C++ --verify-pipeline failed (rc={proc.returncode}): {proc.stderr}",
        )
        out = proc.stdout.strip()
        self.assertIn("OK: C++ 3-Stage Async Pipeline & Bitmap Deduplication Verified", out)
        print(f"[PASS] Test 1 — {out}")

    def test_2_confidence_gated_lsh_lookahead_predictor(self) -> None:
        """
        Verifies `LSHLookaheadPredictor` confidence gating (`min_confidence_hits=2`):
          - Cold/unseen bucket -> returns `[]` and increments `low_confidence_skipped`
          - Single observation (`hits=1 < 2`) -> still returns `[]` (no speculative cache pollution)
          - Second observation (`hits=2 >= 2`) -> returns high-confidence predicted experts!
        """
        lsh = LSHLookaheadPredictor(num_layers=16, max_buckets_per_layer=64)
        prev_topk = [5, 12, 19, 27, 33, 41, 50, 61]
        actual_next = [3, 9, 14, 22, 30, 44, 52, 60]

        # 1. Unseen bucket -> skipped
        pred0 = lsh.predict_next_layer_experts(1, prev_topk, top_n=4, min_confidence_hits=2)
        self.assertEqual(pred0, [])
        self.assertEqual(lsh.low_confidence_skipped, 1)

        # 2. First transition recorded (`hits=1 < 2`) -> still gated
        lsh.record_layer_transition(1, prev_topk, actual_next, top_n=4)
        pred1 = lsh.predict_next_layer_experts(1, prev_topk, top_n=4, min_confidence_hits=2)
        self.assertEqual(pred1, [])
        self.assertEqual(lsh.low_confidence_skipped, 2)

        # 3. Second transition recorded (`hits=2 >= 2`) -> confidence gate opens!
        lsh.record_layer_transition(1, prev_topk, actual_next, top_n=4)
        pred2 = lsh.predict_next_layer_experts(1, prev_topk, top_n=4, min_confidence_hits=2)
        self.assertEqual(len(pred2), 4)
        self.assertTrue(set(pred2).issubset(set(actual_next)))
        print(
            f"[PASS] Test 2 — Confidence-Gated LSH Lookahead Verified: "
            f"low_confidence_skipped={lsh.low_confidence_skipped}, high_conf_predicted={pred2}"
        )

    def test_3_async_three_stage_pipeline_bit_exact_parity(self) -> None:
        """
        Verifies that `AsyncThreeStagePipeline.forward_Layer_sequence_overlapped_numpy`
        (overlapping Stage 1 CPU Route, Stage 2 MoE Compute, and Stage 3 Background Prefetch)
        produces 100% bit-exact output (`max_abs_diff == 0.0`) compared to synchronous layer execution.
        """
        eng = NativeMerkleMoEEngine(
            STORE_DIR,
            expert_cache_size=32,
            max_cache_ram_mb=384.0,
            on_demand_streaming=False,
            enable_lookahead=True,
            verify_merkle_on_load=False,
            device="cpu",
        )
        try:
            rng = np.random.default_rng(77)
            x_np = rng.standard_normal((1, 2, eng.hidden_size)).astype(np.float16)

            # Warm LSH predictor so Stage 3 background worker actively submits prefetches
            _, _ = eng.async_pipeline.forward_Layer_sequence_overlapped_numpy(x_np, num_layers=4)
            _, _ = eng.async_pipeline.forward_Layer_sequence_overlapped_numpy(x_np, num_layers=4)

            # Compare overlapped 4-layer execution vs synchronous 4-layer execution
            t0 = time.perf_counter()
            overlapped_out, _ = eng.async_pipeline.forward_Layer_sequence_overlapped_numpy(
                x_np, num_layers=4
            )
            overlapped_ms = (time.perf_counter() - t0) * 1000.0

            h_sync = x_np
            for l_idx in range(4):
                h_sync, _ = eng.forward_moe_layer_numpy(l_idx, h_sync)

            max_diff = float(np.max(np.abs(overlapped_out.astype(np.float32) - h_sync.astype(np.float32))))
            self.assertEqual(max_diff, 0.0)
            self.assertGreater(eng.async_pipeline.async_prefetches_submitted, 0)
            print(
                f"[PASS] Test 3 — 3-Stage Overlapped Pipeline Parity Verified in {overlapped_ms:.2f} ms: "
                f"max_abs_diff={max_diff:.8f}, async_prefetches_submitted={eng.async_pipeline.async_prefetches_submitted}"
            )
        finally:
            eng.close()

    def test_4_isolated_enable_vs_disable_prefetch_benchmark(self) -> None:
        """
        Runs `benchmark_prefetch_pipeline` comparing `--disable-prefetch` vs `--enable-prefetch`
        and verifies that enabling 3-stage async prefetch reduces cold misses, increases hit rate,
        and preserves `0.0` numerical parity error.
        """
        report = benchmark_prefetch_pipeline(
            STORE_DIR, num_tokens=3, num_layers=4, cache_experts=64, seed=99
        )
        self.assertEqual(report["numerical_parity_max_abs_err"], 0.0)
        self.assertGreater(report["enabled_hit_rate_pct"], report["disabled_hit_rate_pct"])
        self.assertLess(report["enabled_cold_misses"], report["disabled_cold_misses"])
        self.assertGreater(report["prefetch_hits"], 0)
        print(
            f"[PASS] Test 4 — Isolated Prefetch Benchmark (--disable-prefetch vs --enable-prefetch):\n"
            f"  • Prefetch DISABLED : Hit Rate = {report['disabled_hit_rate_pct']:>5.1f}% | "
            f"Cold Misses = {report['disabled_cold_misses']:>3} | Time = {report['disabled_total_ms']:.2f} ms\n"
            f"  • Prefetch ENABLED  : Hit Rate = {report['enabled_hit_rate_pct']:>5.1f}% | "
            f"Cold Misses = {report['enabled_cold_misses']:>3} | Time = {report['enabled_total_ms']:.2f} ms "
            f"(Prefetch Hits = {report['prefetch_hits']}, Parity Err = {report['numerical_parity_max_abs_err']:.8f})"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
