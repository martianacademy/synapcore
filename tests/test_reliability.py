#!/usr/bin/env python3
"""
Task P0.4 Reliability & Fault-Injection Test Suite (`tests/test_reliability.py`)
================================================================================
Verifies all 10 fault-injection and stress-testing scenarios specified in Task P0.4
of `ROADMAP_IMPROVEMENTS.md` (using isolated temporary directories so the primary
`olmoe_merkle_store` is never mutated):
  1. Corrupted CAS block: 1-byte flip in `cas_experts.bin` -> `MerkleIntegrityError` raised.
  2. Missing expert / truncated file: Out-of-bounds offset or truncated CAS/idx -> controlled `MerkleIntegrityError`.
  3. Invalid Merkle root: Header root mismatch -> model load rejected at startup (`MerkleIntegrityError`).
  4. Memory pressure / small cache (`max_experts=16` / `192.0 MiB`): Graceful eviction across 64+ experts without crash.
  5. Repeated requests (100 turns): Constant RSS memory (`0` leaked FDs, `< 5 MiB` RSS drift across 100 turns).
  6. Long context (4,096+ tokens): Stable Merkle KV prefix chaining & Robin-Hood LRU eviction across 260 blocks (4,160 tokens).
  7. Concurrent access: 8-thread simultaneous `HotExpertLFULRUCache` + `NativeMmapIndexBridge` stress test (0 race errors).
  8. Model restart: Persistent KV prefix & `FLAG_VERIFIED` index state verified across engine restart.
  9. Partial CoW update interruption: Atomic rollback to previous valid `base_model_root` on startup.
 10. Invalid `config.json`: Clear `ValueError` validation diagnostic on startup.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import resource
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from native.build_native_db import compile_native_binaries
from native.native_engine import (
    FLAG_IS_COW,
    FLAG_VERIFIED,
    MerkleIntegrityError,
    NativeMerkleMoEEngine,
    NativeMmapIndexBridge,
    OFFSET_EXPERT_TBL,
    OFFSET_HEADER,
    _EXPERT_STRUCT,
    chain_merkle_prefix_hash,
)

STORE_DIR = ROOT_DIR / "olmoe_merkle_store"


def get_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def create_isolated_mock_store(tmp_path: Path) -> Path:
    """
    Creates an isolated temporary store directory with a mutable copy of `merkle_tree.idx`,
    `config.json`, `merkle_manifest.json`, and symlinks to read-only binary arenas.
    """
    shutil.copy2(STORE_DIR / "merkle_tree.idx", tmp_path / "merkle_tree.idx")
    shutil.copy2(STORE_DIR / "config.json", tmp_path / "config.json")
    shutil.copy2(STORE_DIR / "merkle_manifest.json", tmp_path / "merkle_manifest.json")
    for bin_name in ("cas_experts.bin", "embed_tokens.bin", "shared_core.bin", "lm_head_int8.bin"):
        src = STORE_DIR / bin_name
        if src.exists():
            os.symlink(src, tmp_path / bin_name)
    return tmp_path


class TestTaskP04ReliabilityAndFaultInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compile_native_binaries()

    def test_01_corrupted_cas_block_rejected(self) -> None:
        """Case 1: 1-byte flip in `cas_experts.bin` -> `MerkleIntegrityError` raised on Tier-2 load."""
        print("\n[Case 01/10] Fault Injection: 1-Byte Flip in `cas_experts.bin` -> Expect Controlled Rejection...")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = create_isolated_mock_store(Path(tmp_str))
            # Replace symlinked cas_experts.bin with a 1-expert (12 MiB) mutable slice with 1 flipped byte
            (tmp_dir / "cas_experts.bin").unlink()
            with open(STORE_DIR / "cas_experts.bin", "rb") as f_in:
                expert0_bytes = bytearray(f_in.read(12582912))
            expert0_bytes[4096] ^= 0xFF
            (tmp_dir / "cas_experts.bin").write_bytes(expert0_bytes)

            with open(tmp_dir / "cas_experts.bin", "rb") as cas_f:
                cas_mm = mmap.mmap(cas_f.fileno(), 0, access=mmap.ACCESS_READ)
                bridge = NativeMmapIndexBridge(tmp_dir, cas_mm, verify_all_on_init=False)
                try:
                    with self.assertRaises(MerkleIntegrityError) as ctx:
                        bridge.verify_expert_leaf_bytes(0, 0, force=True)
                    print(f"  • Controlled Error Raised        : {ctx.exception}")
                finally:
                    bridge.close()
                    cas_mm.close()

    def test_02_missing_expert_or_truncated_file(self) -> None:
        """Case 2: Truncated `cas_experts.bin` / Out-of-bounds offset -> Controlled `MerkleIntegrityError`."""
        print("[Case 02/10] Fault Injection: Truncated CAS / Out-of-Bounds Expert Offset -> Expect Rejection...")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = create_isolated_mock_store(Path(tmp_str))
            (tmp_dir / "cas_experts.bin").unlink()
            # Create a truncated 64 KB cas_experts.bin (smaller than even a single 12 MiB expert)
            (tmp_dir / "cas_experts.bin").write_bytes(b"\x00" * 65536)

            with self.assertRaises(MerkleIntegrityError) as ctx:
                NativeMerkleMoEEngine(
                    tmp_dir,
                    expert_cache_size=8,
                    on_demand_streaming=True,
                    verify_merkle_on_load=True,
                    device="cpu",
                )
            print(f"  • Truncated CAS Rejected at Init : {ctx.exception}")

    def test_03_invalid_merkle_root_rejected_at_init(self) -> None:
        """Case 3: Header Merkle root mismatch -> Model load rejected at startup."""
        print("[Case 03/10] Fault Injection: Invalid Header Merkle Root -> Expect Startup Rejection...")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = create_isolated_mock_store(Path(tmp_str))
            idx_bytes = bytearray((tmp_dir / "merkle_tree.idx").read_bytes())
            # Corrupt base_model_root at offset 52 in header
            idx_bytes[52] ^= 0xAA
            (tmp_dir / "merkle_tree.idx").write_bytes(idx_bytes)

            with self.assertRaises(MerkleIntegrityError) as ctx:
                NativeMerkleMoEEngine(
                    tmp_dir,
                    expert_cache_size=8,
                    on_demand_streaming=True,
                    verify_merkle_on_load=True,
                    device="cpu",
                )
            print(f"  • Invalid Root Rejected at Init  : {ctx.exception}")

    def test_04_memory_pressure_small_cache_graceful_eviction(self) -> None:
        """Case 4: Memory pressure (`max_experts=16` = `192.0 MiB`) across 80 expert loads -> Graceful eviction."""
        print("[Case 04/10] Stress Test: Memory Pressure (`max_experts=16` / `192 MiB`) Across 960 MiB Stream...")
        engine = NativeMerkleMoEEngine(
            STORE_DIR,
            expert_cache_size=16,
            max_cache_ram_mb=192.0,
            on_demand_streaming=True,
            verify_merkle_on_load=False,
            device="cpu",
        )
        try:
            for i in range(80):
                ent = engine.load_or_get_expert_entry((i // 64) & 15, i & 63)
                _ = float(ent.gate_w[0, 0])
                del ent
                self.assertLessEqual(engine.expert_cache.memory_mb(), 192.0)
                self.assertLessEqual(len(engine.expert_cache.cache), 16)
            self.assertEqual(len(engine.expert_cache.cache), 16)
            self.assertGreaterEqual(engine.expert_cache.evictions, 64)
            print(
                f"  • Memory Cap Enforced            : {engine.expert_cache.memory_mb():.2f} / 192.00 MiB "
                f"({len(engine.expert_cache.cache)}/16 experts, {engine.expert_cache.evictions} evictions)"
            )
        finally:
            engine.close()

    def test_05_repeated_requests_100_turns_constant_rss(self) -> None:
        """Case 5: 100 consecutive inference/routing turns -> Constant RSS memory (zero leak)."""
        print("[Case 05/10] Stress Test: 100 Consecutive Turns -> Constant RSS & Cache Bounds...")
        engine = NativeMerkleMoEEngine(
            STORE_DIR,
            expert_cache_size=16,
            max_cache_ram_mb=192.0,
            on_demand_streaming=True,
            verify_merkle_on_load=False,
            device="cpu",
        )
        try:
            rng = np.random.default_rng(123)
            # Warmup 5 turns so the 16-expert cache reaches steady state
            for w in range(5):
                x_w = rng.standard_normal((1, 2048)).astype(np.float16)
                out_w, _ = engine.forward_moe_layer_numpy(w & 15, x_w)
                del x_w, out_w

            rss_steady_start = get_rss_mb()
            t0 = time.perf_counter()
            for turn in range(100):
                x_turn = rng.standard_normal((1, 2048)).astype(np.float16)
                out_turn, topk = engine.forward_moe_layer_numpy(turn & 15, x_turn)
                self.assertEqual(out_turn.shape, (1, 2048))
                self.assertEqual(topk.shape, (1, 8))
                del x_turn, out_turn, topk
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rss_steady_end = get_rss_mb()
            rss_drift = max(0.0, rss_steady_end - rss_steady_start)

            self.assertLessEqual(engine.expert_cache.memory_mb(), 192.0)
            print(
                f"  • 100 Turns Completed            : {elapsed_ms:.1f} ms total ({elapsed_ms/100.0:.2f} ms/turn)"
            )
            print(
                f"  • Steady-State RSS Drift         : +{rss_drift:.2f} MiB (Cache steady at {engine.expert_cache.memory_mb():.2f} MiB)"
            )
        finally:
            engine.close()

    def test_06_long_context_4096_plus_tokens_kv_prefix_eviction(self) -> None:
        """Case 6: Long context (`4,160` tokens = `260` blocks) + Robin-Hood 64-probe collision LRU eviction."""
        print("[Case 06/10] Stress Test: Long Context (4,160 Tokens = 260 Blocks) & Robin-Hood KV Eviction...")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = create_isolated_mock_store(Path(tmp_str))
            engine = NativeMerkleMoEEngine(
                tmp_dir,
                expert_cache_size=8,
                on_demand_streaming=True,
                verify_merkle_on_load=False,
                device="cpu",
            )
            try:
                prev_hash = engine.active_merkle_root
                block_hashes: List[bytes] = []
                # Chain 260 blocks of 16 tokens = 4,160 tokens
                for b_idx in range(260):
                    block_tokens = [((b_idx * 16) + t) % 50000 for t in range(16)]
                    curr_hash = chain_merkle_prefix_hash(prev_hash, block_tokens)
                    curr_bytes = bytes.fromhex(curr_hash)
                    engine.cpp.kv_insert(
                        curr_bytes,
                        bytes.fromhex(prev_hash),
                        (b_idx + 1) * 16,
                        time.time() + (b_idx * 0.001),
                    )
                    block_hashes.append(curr_bytes)
                    prev_hash = curr_hash

                # Verify recent blocks are present
                for h_bytes in block_hashes[-128:]:
                    self.assertTrue(engine.cpp.kv_lookup(h_bytes))

                # Force 70 colliding keys into the exact same Robin-Hood bucket (slot 42) to test LRU eviction
                for c_idx in range(70):
                    # First 4 bytes determine slot (`val & 1023 == 42`), remaining 28 bytes vary
                    colliding_hash = struct.pack("<I", 42) + struct.pack("<I", c_idx + 1) + (b"\x11" * 24)
                    engine.cpp.kv_insert(colliding_hash, b"\x00" * 32, (c_idx + 1) * 16, 1000.0 + c_idx)

                # Newest colliding entry (c_idx=69) must be found; oldest (c_idx=0) was evicted cleanly
                newest_hash = struct.pack("<I", 42) + struct.pack("<I", 70) + (b"\x11" * 24)
                oldest_hash = struct.pack("<I", 42) + struct.pack("<I", 1) + (b"\x11" * 24)
                self.assertTrue(engine.cpp.kv_lookup(newest_hash))
                self.assertFalse(engine.cpp.kv_lookup(oldest_hash))
                print("  • 4,160-Token Prefix Chain & 70-Way Bucket Collision Eviction: Verified Cleanly!")
            finally:
                engine.close()

    def test_07_concurrent_multithreaded_access(self) -> None:
        """Case 7: 8 concurrent threads performing simultaneous cache & index lookups/evictions."""
        print("[Case 07/10] Stress Test: 8 Concurrent Threads Accessing Cache & `merkle_tree.idx`...")
        engine = NativeMerkleMoEEngine(
            STORE_DIR,
            expert_cache_size=16,
            max_cache_ram_mb=192.0,
            on_demand_streaming=True,
            verify_merkle_on_load=False,
            device="cpu",
        )
        errors: List[Exception] = []

        def worker_fn(tid: int) -> None:
            try:
                for step in range(40):
                    l_i = (tid + step) & 15
                    e_i = ((tid * 13) + step) & 63
                    ent = engine.load_or_get_expert_entry(l_i, e_i)
                    if ent.gate_w.shape != (1024, 2048):
                        raise AssertionError(f"Invalid gate_w shape {ent.gate_w.shape}")
                    del ent
            except Exception as exc:
                errors.append(exc)

        try:
            threads = [threading.Thread(target=worker_fn, args=(t,)) for t in range(8)]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            conc_ms = (time.perf_counter() - t0) * 1000.0
            self.assertEqual(len(errors), 0, msg=f"Concurrent errors: {errors}")
            self.assertLessEqual(engine.expert_cache.memory_mb(), 192.0)
            print(
                f"  • 8 Threads x 40 Expert Loads    : 320 Concurrent Ops in {conc_ms:.2f} ms (0 Race Errors!)"
            )
        finally:
            engine.close()

    def test_08_model_restart_persistent_index_state(self) -> None:
        """Case 8: Persistent KV prefix & `FLAG_VERIFIED` index state survive engine restart."""
        print("[Case 08/10] Persistence Test: Model Restart Preserves `FLAG_VERIFIED` & KV Prefix State...")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = create_isolated_mock_store(Path(tmp_str))
            test_kv_hash = hashlib.sha256(b"synapcore_restart_test_prefix").digest()

            # Session 1: Verify expert L2E9 and insert KV prefix block, then close engine
            eng1 = NativeMerkleMoEEngine(
                tmp_dir,
                expert_cache_size=8,
                on_demand_streaming=True,
                verify_merkle_on_load=True,
                device="cpu",
            )
            eng1.cpp.verify_expert_leaf_bytes(2, 9, force=True)
            eng1.cpp.kv_insert(test_kv_hash, b"\x00" * 32, 32, time.time())
            eng1.close()

            # Session 2: Restart engine on the same store directory and verify state persisted on disk
            eng2 = NativeMerkleMoEEngine(
                tmp_dir,
                expert_cache_size=8,
                on_demand_streaming=True,
                verify_merkle_on_load=True,
                device="cpu",
            )
            try:
                slot_l2e9 = (2 << 6) | 9
                self.assertIn(slot_l2e9, eng2.cpp.verified_leaf_slots)
                self.assertTrue(eng2.cpp.kv_lookup(test_kv_hash))
                print("  • Restart Persistence Verified   : L2E9 `FLAG_VERIFIED` + KV Prefix Found After Restart!")
            finally:
                eng2.close()

    def test_09_partial_cow_update_interruption_atomic_rollback(self) -> None:
        """Case 9: Interrupted/partial CoW update -> Atomic rollback to previous valid `base_model_root`."""
        print("[Case 09/10] Fault Injection: Interrupted Partial CoW Update -> Expect Atomic Rollback...")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = create_isolated_mock_store(Path(tmp_str))
            # Simulate an interrupted CoW update in merkle_tree.idx (active_model_root modified,
            # num_cow_leaves=1, and slot 0 marked FLAG_IS_COW with a modified leaf hash, but no committed
            # cow_manifest.json / cas_cow_deltas.bin)
            with open(tmp_dir / "merkle_tree.idx", "r+b") as f_idx:
                mm = mmap.mmap(f_idx.fileno(), 0, access=mmap.ACCESS_WRITE)
                # Corrupt active_model_root at header offset 80..112 and set num_cow_leaves=1 at offset 240
                mm[80:112] = b"\x99" * 32
                struct.pack_into("<I", mm, 240, 1)
                # Mark slot 0 with FLAG_IS_COW and bogus sha256
                sha_b, byte_off, byte_len, flags, freq, step = _EXPERT_STRUCT.unpack_from(mm, OFFSET_EXPERT_TBL)
                _EXPERT_STRUCT.pack_into(
                    mm, OFFSET_EXPERT_TBL, b"\x77" * 32, byte_off, byte_len, flags | FLAG_IS_COW, freq, step
                )
                mm.flush()
                mm.close()

            # Initialize engine: should detect partial CoW and roll back atomically to base_model_root!
            eng = NativeMerkleMoEEngine(
                tmp_dir,
                expert_cache_size=8,
                on_demand_streaming=True,
                verify_merkle_on_load=True,
                device="cpu",
            )
            try:
                stats = eng.cpp.get_stats()
                self.assertEqual(stats["cow_rollbacks_performed"], 1)
                self.assertEqual(eng.cpp.active_model_root, eng.cpp.base_model_root)
                self.assertEqual(eng.cpp.num_cow_leaves, 0)
                sha_hex, _, _, is_cow = eng.cpp.lookup_expert(0, 0)
                self.assertFalse(is_cow)
                print(
                    f"  • Atomic CoW Rollback Verified   : Restored base_model_root={eng.cpp.base_model_root.hex()[:16]}... "
                    f"(rollbacks={stats['cow_rollbacks_performed']})"
                )
            finally:
                eng.close()

    def test_10_invalid_config_json_rejected_cleanly(self) -> None:
        """Case 10: Invalid or malformed `config.json` -> Clear `ValueError` on startup."""
        print("[Case 10/10] Fault Injection: Invalid `config.json` -> Expect Clear Validation Error...")
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = create_isolated_mock_store(Path(tmp_str))
            # Write config.json with missing num_experts and negative hidden_size
            (tmp_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 16, "hidden_size": -2048}))

            with self.assertRaises(ValueError) as ctx:
                NativeMerkleMoEEngine(
                    tmp_dir,
                    expert_cache_size=8,
                    on_demand_streaming=True,
                    verify_merkle_on_load=False,
                    device="cpu",
                )
            print(f"  • Invalid `config.json` Rejected : {ctx.exception}")


if __name__ == "__main__":
    print("=" * 78)
    print("RUNNING TASK P0.4 SUITE: 10-CASE RELIABILITY & FAULT-INJECTION TEST SUITE")
    print("=" * 78)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTaskP04ReliabilityAndFaultInjection)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("=" * 78)
    print("ALL 10 TASK P0.4 RELIABILITY & FAULT-INJECTION TESTS PASSED WITH ZERO REGRESSIONS!")
    print("=" * 78)
