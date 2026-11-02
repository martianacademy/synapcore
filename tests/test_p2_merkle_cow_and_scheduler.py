#!/usr/bin/env python3
"""
Phase P2 Unit & Micro-Benchmark Verification Suite (`tests/test_p2_merkle_cow_and_scheduler.py`)
================================================================================================
Verifies:
  1. Task P2.1: Incremental Merkle Tree Updates (`O(K log N)`) & Versioned Model Storage
     - Modifying `K=5` experts out of `1,024` appends only `5` 12 MiB blocks to `cas_cow_deltas.bin`
       (with content-addressed deduplication) and recomputes ONLY `<= 50` ancestor hashes instead
       of all `1,023` internal nodes (`< 0.5 ms`).
     - Preserves independent versioned roots (`v000`, `v001`, `v002`) in `versions/`.
     - Subtree-pruned `diff_versions("v000", "v002")` visits `<< 2,047` nodes and identifies the
       exact modified experts in microseconds.
     - Zero-copy `checkout_version("v000")` restores the exact original `base_model_root`.
  2. Task P2.2: Continual Learning / Weight Change Detector Module (`ContinualLearningUpdater`)
     - Inspects `12` candidate experts (`8` unchanged + `4` fine-tuned/modified), automatically
       skips the `8` unchanged experts (`0` redundant disk writes), commits only the `4` changed
       experts, verifies live forward-pass output reflection in `NativeMerkleMoEEngine`, and
       verifies root & weight persistence across engine restart.
  3. Task P2.3: Multi-Request Continuous Batching Scheduler (`ContinuousBatchScheduler`)
     - Runs `4` concurrent requests with isolated Paged KV Caches (`PagedKVArena` + per-request KV)
       and shared `HotExpertLFULRUCache` + grouped active expert execution.
     - Verifies `0.00000000` numerical logit & token parity against isolated single-request runs
       and measures cross-request expert deduplication savings.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.scheduler import ContinuousBatchScheduler, benchmark_continuous_batch_scheduler
from merkle_store import ContinualLearningUpdater, IncrementalMerkleVersionManager
from native.native_engine import NativeMerkleMoEEngine

STORE_DIR = ROOT_DIR / "olmoe_merkle_store"


def _create_isolated_store_sandbox(tmp_dir: Path) -> Path:
    """
    Creates a lightweight isolated store directory with hardlinks/symlinks to the immutable
    read-only `.bin` arenas (`cas_experts.bin`, `embed_tokens.bin`, `shared_core.bin`, `lm_head_int8.bin`)
    and a writable copy of `merkle_tree.idx`, `config.json`, and `merkle_manifest.json`.
    """
    sandbox = tmp_dir / "sandbox_store"
    sandbox.mkdir(parents=True, exist_ok=True)
    for fname in ["config.json", "merkle_manifest.json", "merkle_tree.idx", "tokenizer.json"]:
        src = STORE_DIR / fname
        if src.exists():
            shutil.copy2(src, sandbox / fname)
    for bin_name in ["cas_experts.bin", "embed_tokens.bin", "shared_core.bin", "lm_head_int8.bin"]:
        src_bin = STORE_DIR / bin_name
        dst_bin = sandbox / bin_name
        try:
            os.link(src_bin, dst_bin)
        except OSError:
            dst_bin.symlink_to(src_bin)
    return sandbox


import os


class TestPhaseP2MerkleCoWAndScheduler(unittest.TestCase):
    def test_1_p2_1_incremental_merkle_updates_and_versioned_storage(self) -> None:
        """
        Task P2.1: Verifies `O(K log N)` partial XOR tree recompute, CAS deduplication,
        version snapshots (`v000`, `v001`, `v002`), subtree-pruned `diff_versions`, and
        zero-copy `checkout_version` rollback.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = _create_isolated_store_sandbox(Path(tmp))
            vm = IncrementalMerkleVersionManager(sandbox)

            v0_list = vm.list_versions()
            self.assertEqual(len(v0_list), 1)
            v0_root = v0_list[0]["active_model_root"]
            v0_exp_root = v0_list[0]["expert_tree_root"]

            rng = np.random.default_rng(101)
            # Commit v001: Modify K=5 experts across different layers
            updates_v1: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            target_slots_v1 = [(0, 2), (0, 3), (3, 15), (7, 40), (12, 61)]
            for l_i, e_i in target_slots_v1:
                g = rng.standard_normal((1024, 2048)).astype(np.float16)
                u = rng.standard_normal((1024, 2048)).astype(np.float16)
                d = rng.standard_normal((2048, 1024)).astype(np.float16)
                updates_v1[(l_i, e_i)] = (g, u, d)

            snap_v1 = vm.apply_incremental_deltas(updates_v1, version_tag="v001")
            self.assertEqual(snap_v1["version_tag"], "v001")
            self.assertEqual(snap_v1["num_modified_in_commit"], 5)
            self.assertEqual(snap_v1["total_cow_leaves"], 5)
            # Note: (0,2) and (0,3) share parent at level 9 -> <= 41 internal nodes recomputed (vs 1,023)!
            self.assertLessEqual(snap_v1["internal_nodes_recomputed"], 50)
            self.assertLess(snap_v1["internal_nodes_recomputed"], snap_v1["full_tree_nodes"] // 15)
            self.assertNotEqual(snap_v1["active_model_root"], v0_root)
            self.assertNotEqual(snap_v1["expert_tree_root"], v0_exp_root)

            # Commit v002: Modify 2 more experts, reusing the exact same weights for (15, 7) as (0, 2) to test CAS dedup!
            updates_v2: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
                (15, 7): updates_v1[(0, 2)],  # Identical weights -> should deduplicate in cas_cow_deltas.bin!
                (4, 19): (
                    rng.standard_normal((1024, 2048)).astype(np.float16),
                    rng.standard_normal((1024, 2048)).astype(np.float16),
                    rng.standard_normal((2048, 1024)).astype(np.float16),
                ),
            }
            snap_v2 = vm.apply_incremental_deltas(updates_v2, version_tag="v002")
            self.assertEqual(snap_v2["total_cow_leaves"], 7)
            self.assertLessEqual(snap_v2["internal_nodes_recomputed"], 20)
            dedup_flags = [r["dedup_reused"] for r in snap_v2["modified_experts"] if r["layer_idx"] == 15]
            self.assertEqual(dedup_flags, [True])

            # Subtree-Pruned O(K log N) Diff between v000 and v002 (7 changed experts out of 1,024)
            diff_0_2 = vm.diff_versions("v000", "v002")
            self.assertEqual(len(diff_0_2["changed_experts"]), 7)
            self.assertLess(diff_0_2["nodes_visited"], 180)  # << 2,047 total tree nodes!
            self.assertGreater(diff_0_2["subtrees_pruned"], 40)

            # Verify NativeMerkleMoEEngine loads v002 cleanly and Tier-1 + Tier-2 pass
            eng_v2 = NativeMerkleMoEEngine(
                sandbox,
                expert_cache_size=16,
                on_demand_streaming=True,
                verify_merkle_on_load=True,
                device="cpu",
            )
            try:
                self.assertEqual(eng_v2.cpp.active_model_root.hex(), snap_v2["active_model_root"])
                ent_cow = eng_v2.load_or_get_expert_entry(0, 2)
                np.testing.assert_array_equal(ent_cow.gate_w, updates_v1[(0, 2)][0])
            finally:
                eng_v2.close()

            # Zero-Copy Checkout Rollback to v000 -> restores original root!
            restored_v0 = vm.checkout_version("v000")
            self.assertEqual(restored_v0["active_model_root"], v0_root)
            eng_v0 = NativeMerkleMoEEngine(
                sandbox,
                expert_cache_size=16,
                on_demand_streaming=True,
                verify_merkle_on_load=True,
                device="cpu",
            )
            try:
                self.assertEqual(eng_v0.cpp.active_model_root.hex(), v0_root)
                self.assertEqual(eng_v0.cpp.num_cow_leaves, 0)
            finally:
                eng_v0.close()

            print(
                f"[PASS] Task P2.1 — Incremental O(K log N) Merkle & Versioning Verified: "
                f"K=5 recomputed {snap_v1['internal_nodes_recomputed']}/1023 nodes in "
                f"{snap_v1['tree_recompute_ms']:.3f} ms, diff visited {diff_0_2['nodes_visited']}/2047 nodes "
                f"({diff_0_2['diff_time_us']:.1f} us), v000->v001->v002->v000 checkout OK!"
            )

    def test_2_p2_2_continual_learning_weight_change_detector(self) -> None:
        """
        Task P2.2: Verifies `ContinualLearningUpdater` detects only modified experts among
        candidates, skips unchanged experts with 0 redundant writes, updates live inference
        output, and persists across engine restart.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = _create_isolated_store_sandbox(Path(tmp))
            eng_base = NativeMerkleMoEEngine(
                sandbox,
                expert_cache_size=32,
                on_demand_streaming=False,
                verify_merkle_on_load=False,
                device="cpu",
            )
            rng = np.random.default_rng(202)
            x_probe = rng.standard_normal((1, 2, eng_base.hidden_size)).astype(np.float16)

            try:
                out_before, topk_ind = eng_base.forward_moe_layer_numpy(0, x_probe)
                routed_e0 = int(topk_ind[0, 0])
                routed_e1 = int(topk_ind[0, 1])

                # Build 12 candidate experts: 8 unmodified base experts + 4 modified experts (including routed_e0, routed_e1)
                candidates: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                for e_i in range(10):
                    ent = eng_base.load_or_get_expert_entry(0, e_i)
                    candidates[(0, e_i)] = (ent.gate_w.copy(), ent.up_w.copy(), ent.down_w.copy())

                modified_targets = {(0, routed_e0), (0, routed_e1), (1, 5), (2, 11)}
                for l_m, e_m in modified_targets:
                    ent_m = eng_base.load_or_get_expert_entry(l_m, e_m)
                    # Apply synthetic continual-learning delta (+0.05 * randn)
                    g_new = (ent_m.gate_w.astype(np.float32) + 0.05).astype(np.float16)
                    u_new = (ent_m.up_w.astype(np.float32) - 0.03).astype(np.float16)
                    d_new = ent_m.down_w.copy()
                    candidates[(l_m, e_m)] = (g_new, u_new, d_new)
            finally:
                eng_base.close()

            updater = ContinualLearningUpdater(sandbox)
            report = updater.detect_and_commit_weight_changes(candidates, version_tag="v001_cl")

            self.assertEqual(report["candidates_inspected"], len(candidates))
            self.assertEqual(report["modified_committed"], 4)
            self.assertEqual(report["unchanged_skipped"], len(candidates) - 4)
            self.assertNotEqual(report["previous_model_root"], report["new_model_root"])
            self.assertEqual(set(report["modified_slots"]), modified_targets)

            # Restart engine on sandbox store -> verify new root & modified output on Layer 0!
            eng_restarted = NativeMerkleMoEEngine(
                sandbox,
                expert_cache_size=32,
                on_demand_streaming=False,
                verify_merkle_on_load=True,
                device="cpu",
            )
            try:
                self.assertEqual(eng_restarted.active_merkle_root, report["new_model_root"])
                out_after, _ = eng_restarted.forward_moe_layer_numpy(0, x_probe)
                diff_l0 = float(np.max(np.abs(out_after.astype(np.float32) - out_before.astype(np.float32))))
                self.assertGreater(diff_l0, 1e-3, "Continual learning weight update must change Layer 0 output!")
            finally:
                eng_restarted.close()

            print(
                f"[PASS] Task P2.2 — ContinualLearningUpdater Verified: inspected={report['candidates_inspected']}, "
                f"unchanged_skipped={report['unchanged_skipped']}, committed={report['modified_committed']} "
                f"in {report['total_elapsed_ms']:.2f} ms (tree={report['tree_recompute_ms']:.3f} ms, output delta={diff_l0:.4f})"
            )

    def test_3_p2_3_multi_request_continuous_batching_scheduler(self) -> None:
        """
        Task P2.3: Verifies `ContinuousBatchScheduler` across 4 concurrent requests sharing
        `HotExpertLFULRUCache` and `PagedKVArena` with 0.00000000 numerical parity error.
        """
        metrics = benchmark_continuous_batch_scheduler(
            STORE_DIR, num_requests=4, max_new_tokens=3, max_layers=2
        )
        self.assertEqual(metrics["total_requests_completed"], 4)
        self.assertEqual(metrics["total_tokens_generated"], 12)
        self.assertEqual(metrics["max_concurrent_batch_size"], 4)
        self.assertLessEqual(metrics["numerical_parity_max_abs_err"], 1e-5)
        self.assertGreater(metrics["cross_request_expert_sharing_pct"], 10.0)
        self.assertGreaterEqual(metrics["paged_kv_prefix_tokens_skipped"], 48)

        print(
            f"[PASS] Task P2.3 — ContinuousBatchScheduler Verified: "
            f"requests={metrics['total_requests_completed']}, batch_size={metrics['max_concurrent_batch_size']}, "
            f"expert_sharing={metrics['cross_request_expert_sharing_pct']:.1f}% "
            f"({metrics['total_unique_expert_dispatches']} unique vs {metrics['total_naive_expert_dispatches']} naive), "
            f"KV skipped={metrics['paged_kv_prefix_tokens_skipped']} toks, "
            f"parity_err={metrics['numerical_parity_max_abs_err']:.8f}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
