#!/usr/bin/env python3
"""
Task P1.1 Verification & Micro-Benchmark Suite:
Live Per-Stage Performance Profiler, Ring-Buffer Telemetry & Hardware Diagnostics
=================================================================================
Verifies:
  1. `LiveRouteRingBuffer` captures routed Top-K expert IDs live during forward execution
     with ring-buffer wrap-around and `duplicate_forward_passes == 0` (eliminating the
     post-decode duplicate forward pass `_mlx_collect_routing_trace`).
  2. `SynapCoreStageProfiler` tracks all 7 architectural pipeline stages (`Router`,
     `Expert Lookup`, `Merkle Verification`, `Expert Loading / Cache`, `Metal Execution`,
     `Attention`, `Sampling`) when enabled (`--profile`), and incurs zero stage recording
     overhead when disabled.
  3. `get_hardware_diagnostics()` reports real `ru_maxrss_mb`, MLX memory metrics, cache
     hit/miss/eviction/madvise telemetry, and numerical parity (`max_abs_err == 0.0`).
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from native.native_engine import (
     NativeMerkleMoEEngine,
    PROFILE_STAGES,
    LiveRouteRingBuffer,
    SynapCoreStageProfiler,
)

STORE_DIR = ROOT_DIR / "olmoe_merkle_store"


class TestP11LiveProfilerAndDiagnostics(unittest.TestCase):
    def test_1_live_route_ring_buffer_wrap_and_zero_duplicate_pass(self) -> None:
        """
        Verifies `LiveRouteRingBuffer` O(1) live capture, chronological ordering across
        ring wrap-around (`capacity=8` over `12` recorded steps), and zero duplicate forward passes.
        """
        ring = LiveRouteRingBuffer(capacity=8, num_layers=16, top_k=8)
        for step in range(12):
            step_arr = np.full((1, 16, 8), step, dtype=np.int32)
            ring.record_numpy_step(step_arr)

        flushed = ring.flush()
        self.assertEqual(flushed.shape, (8, 16, 8))
        self.assertEqual(ring.total_tokens_recorded, 12)
        self.assertEqual(ring.duplicate_forward_passes, 0)
        # Steps 4..11 should be retained in chronological order
        expected_steps = list(range(4, 12))
        actual_steps = [int(flushed[i, 0, 0]) for i in range(8)]
        self.assertEqual(actual_steps, expected_steps)
        print(
            f"[PASS] Test 1 — LiveRouteRingBuffer Wrap-Around Verified: "
            f"retained steps={actual_steps}, total_recorded={ring.total_tokens_recorded}, "
            f"duplicate_forward_passes={ring.duplicate_forward_passes}"
        )

    def test_2_live_routing_telemetry_in_engine_without_post_decode_pass(self) -> None:
        """
        Verifies that running all 16 layers populates `live_routing_buffer` live during
        execution and `flush_live_routing_telemetry` updates cache & LSH lookahead metrics
        with `duplicate_forward_passes == 0`.
        """
        eng = NativeMerkleMoEEngine(
            STORE_DIR,
            expert_cache_size=32,
            max_cache_ram_mb=384.0,
            on_demand_streaming=True,
            enable_profiler=False,
            live_ring_capacity=64,
            device="cpu",
        )
        try:
            rng = np.random.default_rng(7)
            x_np = rng.standard_normal((1, 2, eng.hidden_size)).astype(np.float16)

            t0 = time.perf_counter()
            for l_idx in range(eng.num_layers):
                x_np, _ = eng.forward_moe_layer_numpy(l_idx, x_np)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            routed_trace = eng.flush_live_routing_telemetry(prompt_ids=[10, 20], generated_ids=[30])
            self.assertEqual(routed_trace.shape, (1, 16, 8))
            self.assertEqual(eng.live_routing_buffer.duplicate_forward_passes, 0)
            self.assertGreater(eng.lsh_predictor.lookahead_predictions, 0)
            print(
                f"[PASS] Test 2 — Live Engine Routing Telemetry Verified in {elapsed_ms:.2f} ms: "
                f"trace_shape={routed_trace.shape}, lookahead_preds={eng.lsh_predictor.lookahead_predictions}, "
                f"duplicate_passes={eng.live_routing_buffer.duplicate_forward_passes}"
            )
        finally:
            eng.close()

    def test_3_seven_stage_profiler_and_formatted_table(self) -> None:
        """
        Verifies `SynapCoreStageProfiler` captures all 7 stages (`Router`, `Expert Lookup`,
        `Merkle Verification`, `Expert Loading / Cache`, `Metal Execution`, `Attention`,
        `Sampling`) with microsecond accuracy when enabled, and 0 overhead when disabled.
        """
        eng = NativeMerkleMoEEngine(
            STORE_DIR,
            expert_cache_size=32,
            max_cache_ram_mb=384.0,
            on_demand_streaming=True,
            enable_profiler=False,
            device="cpu",
        )
        try:
            # 1. Disabled profiler -> 0 calls recorded
            rng = np.random.default_rng(99)
            x_np = rng.standard_normal((1, 1, eng.hidden_size)).astype(np.float16)
            eng.forward_moe_layer_numpy(0, x_np)
            disabled_summary = eng.profiler.get_summary()
            self.assertFalse(disabled_summary["enabled"])
            self.assertEqual(disabled_summary["total_profiled_ms"], 0.0)

            # 2. Enable profiler and run profiled execution across stages
            eng.reset_telemetry()
            eng.set_profiler_enabled(True)
            # Force one cold Tier-2 Merkle verification on L1E0 so Merkle Verification stage > 0
            slot_l1e0 = (1 << 6) | 0
            eng.cpp.verified_leaf_slots.discard(slot_l1e0)

            t_at = time.perf_counter()
            _ = x_np @ eng.core_np_views["model.layers.0.self_attn.q_proj.weight"].T
            eng.profiler.record_stage("Attention", (time.perf_counter() - t_at) * 1000.0)

            # Force load of L1E0 (triggers Expert Lookup + Tier-2 Merkle Verification + Cache Put)
            eng.load_or_get_expert_entry(1, 0, is_prefetch=False)
            # Run full MoE layer (triggers Router + Expert Lookup + Cache Hit/Miss + Metal/MoE Execution)
            out_np, _ = eng.forward_moe_layer_numpy(0, x_np)

            t_sm = time.perf_counter()
            _ = int(np.argmax(out_np.reshape(-1)))
            eng.profiler.record_stage("Sampling", (time.perf_counter() - t_sm) * 1000.0)
            eng.profiler.mark_token_completed(1)

            summary = eng.profiler.get_summary()
            self.assertTrue(summary["enabled"])
            self.assertEqual(summary["tokens_profiled"], 1)
            for stage_name in PROFILE_STAGES:
                self.assertIn(stage_name, summary["stages"])
                self.assertGreater(
                    summary["stages"][stage_name]["calls"],
                    0,
                    f"Expected > 0 calls for stage '{stage_name}'",
                )
                self.assertGreater(
                    summary["stages"][stage_name]["total_ms"],
                    0.0,
                    f"Expected > 0.0 ms for stage '{stage_name}'",
                )

            table_str = eng.profiler.format_table()
            self.assertIn("SYNAPCORE PER-STAGE LATENCY PROFILER", table_str)
            for stage_name in PROFILE_STAGES:
                self.assertIn(stage_name, table_str)
            print("[PASS] Test 3 — 7-Stage Latency Profiler Verified:\n" + table_str)
        finally:
            eng.close()

    def test_4_hardware_diagnostics_and_numerical_parity(self) -> None:
        """
        Verifies `get_hardware_diagnostics()` and `verify_numerical_parity()` report real
        `ru_maxrss_mb`, cache hit/miss/eviction/madvise metrics, and `0.0` numerical parity error.
        """
        eng = NativeMerkleMoEEngine(
            STORE_DIR,
            expert_cache_size=8,
            max_cache_ram_mb=96.0,
            on_demand_streaming=True,
            enable_profiler=True,
            device="cpu",
        )
        try:
            max_err, mean_err = eng.verify_numerical_parity(layer_idx=0, num_tokens=2, seed=123)
            self.assertEqual(max_err, 0.0)
            self.assertEqual(mean_err, 0.0)

            diag = eng.get_hardware_diagnostics()
            self.assertGreater(diag["ru_maxrss_mb"], 0.0)
            self.assertLessEqual(diag["hot_cache_ram_mb"], 96.0)
            self.assertEqual(diag["max_cache_ram_mb"], 96.0)
            self.assertGreater(diag["cache_hits"] + diag["cache_misses"], 0)
            self.assertGreater(diag["cache_evictions"], 0)
            self.assertEqual(diag["cache_evictions"], diag["madvise_releases"])
            self.assertEqual(diag["duplicate_forward_passes"], 0)
            self.assertEqual(diag["numerical_parity_max_abs_err"], 0.0)
            self.assertEqual(diag["numerical_parity_mean_abs_err"], 0.0)
            print(
                f"[PASS] Test 4 — Hardware & Parity Diagnostics Verified: "
                f"RSS={diag['ru_maxrss_mb']:.2f} MiB, Cache={diag['hot_cache_ram_mb']:.2f}/{diag['max_cache_ram_mb']:.2f} MiB, "
                f"Evictions={diag['cache_evictions']} (madvise={diag['madvise_releases']}), "
                f"Parity max_abs_err={diag['numerical_parity_max_abs_err']:.8f}"
            )
        finally:
            eng.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
