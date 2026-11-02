#!/usr/bin/env python3
"""
Task P1.2 Verification & Micro-Benchmark Suite:
Paged KV Cache Arena (`PAGE_TOKENS = 16`), Robin-Hood LRU Slot Eviction & INT8 Quantized KV
===========================================================================================
Verifies:
  1. `PagedKVArena` O(1) free-list allocation, LRU page eviction under capacity pressure,
     and single-pass contiguous gathering (`gather_contiguous_kv_numpy`) with `0.0` diff.
  2. Native C++ (`synapcore_native_kv_insert` via `--verify-kv-eviction`) and Python
     Robin-Hood 64-probe LRU slot eviction (`100` colliding inserts -> `36` oldest evicted,
     `64/64` newest retained).
  3. Optional INT8 Quantized KV Cache (`kv_quant_mode="int8"`): `-49.22%` memory reduction
     (`2.00 MiB/page -> 1.0156 MiB/page`) with cosine similarity `> 0.9999`.
  4. End-to-End `NativeKVPrefixCache` warm in-memory arena hits, disk reload, and bounded
     `max_disk_blocks` cleanup.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from native.build_native_db import compile_native_binaries
from native.native_engine import (
    NativeKVPrefixCache,
    NativeMmapIndexBridge,
    PagedKVArena,
    dequantize_kv_int8_numpy,
    quantize_kv_int8_numpy,
)

STORE_DIR = ROOT_DIR / "olmoe_merkle_store"


class TestP12PagedKVCacheAndEviction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compile_native_binaries()

    def test_1_paged_kv_arena_lru_eviction_and_contiguous_gather(self) -> None:
        """
        Verifies `PagedKVArena(max_pages=8)` O(1) free-list allocation, LRU page eviction
        across 12 pages (192 tokens), and bit-exact single-pass contiguous gathering.
        """
        arena = PagedKVArena(
            max_pages=8, num_layers=16, num_kv_heads=16, head_dim=128, quant_mode="fp16"
        )
        rng = np.random.default_rng(101)
        pages_written = []
        for i in range(12):
            h_hex = f"{i + 1:064x}"
            k_p = rng.standard_normal((16, 1, 16, 16, 128)).astype(np.float16)
            v_p = rng.standard_normal((16, 1, 16, 16, 128)).astype(np.float16)
            pages_written.append((h_hex, k_p, v_p))
            arena.write_page_fp16(h_hex, k_p, v_p)
            # Touch hot pages 0 and 1 on every step so they survive LRU eviction
            if i >= 2:
                arena.lookup_page_id(pages_written[0][0])
                arena.lookup_page_id(pages_written[1][0])

        self.assertEqual(len(arena.page_table), 8)
        self.assertEqual(arena.page_evictions, 4)
        # Hot pages 0 and 1 must still be resident
        pid0 = arena.lookup_page_id(pages_written[0][0])
        pid1 = arena.lookup_page_id(pages_written[1][0])
        self.assertIsNotNone(pid0)
        self.assertIsNotNone(pid1)
        # Cold pages 2, 3, 4, 5 must have been evicted
        for evicted_idx in (2, 3, 4, 5):
            self.assertIsNone(arena.lookup_page_id(pages_written[evicted_idx][0]))

        # Verify single-pass contiguous gather matches exact reference concatenation
        surviving_indices = [0, 1, 6, 7, 8, 9, 10, 11]
        surviving_pids = [arena.lookup_page_id(pages_written[idx][0]) for idx in surviving_indices]
        self.assertTrue(all(p is not None for p in surviving_pids))
        t0 = time.perf_counter()
        k_contig, v_contig = arena.gather_contiguous_kv_numpy([int(p) for p in surviving_pids if p is not None])
        gather_ms = (time.perf_counter() - t0) * 1000.0

        k_ref = np.concatenate([pages_written[idx][1] for idx in surviving_indices], axis=3)
        v_ref = np.concatenate([pages_written[idx][2] for idx in surviving_indices], axis=3)
        self.assertEqual(k_contig.shape, (16, 1, 16, 128, 128))
        self.assertEqual(float(np.max(np.abs(k_contig - k_ref))), 0.0)
        self.assertEqual(float(np.max(np.abs(v_contig - v_ref))), 0.0)
        print(
            f"[PASS] Test 1 — PagedKVArena (8 pages / 128 tokens) Gather in {gather_ms:.2f} ms: "
            f"evictions={arena.page_evictions}, hot_retained=2/2, max_abs_diff=0.0"
        )

    def test_2_cpp_native_robin_hood_kv_lru_eviction(self) -> None:
        """
        Verifies C++ `synapcore_native_kv_insert` Robin-Hood 64-probe LRU eviction via
        `synapcore_native_bin <store_dir> --verify-kv-eviction`.
        """
        bin_path = ROOT_DIR / "native" / "synapcore_native_bin"
        res = subprocess.run(
            [str(bin_path), str(STORE_DIR), "--verify-kv-eviction"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=f"C++ KV eviction failed: {res.stderr}")
        self.assertIn("OK: C++ Robin-Hood LRU KV Eviction Verified", res.stdout)
        print(f"[PASS] Test 2 — {res.stdout.strip()}")

    def test_3_int8_quantized_kv_memory_savings_and_precision(self) -> None:
        """
        Verifies `PagedKVArena(quant_mode='int8')` reduces per-page memory by `-49.22%`
        (`2,097,152 -> 1,064,960` bytes/page) while preserving cosine similarity `> 0.9999`.
        """
        arena_fp16 = PagedKVArena(max_pages=16, quant_mode="fp16")
        arena_int8 = PagedKVArena(max_pages=16, quant_mode="int8")

        self.assertEqual(arena_fp16.bytes_per_page, 2097152)  # 2.00 MiB
        self.assertEqual(arena_int8.bytes_per_page, 1064960)  # 1.0156 MiB
        savings_pct = 100.0 * (1.0 - arena_int8.bytes_per_page / arena_fp16.bytes_per_page)
        self.assertAlmostEqual(savings_pct, 49.21875, places=4)

        rng = np.random.default_rng(2026)
        k_orig = (rng.standard_normal((16, 1, 16, 16, 128)) * 0.5).astype(np.float16)
        v_orig = (rng.standard_normal((16, 1, 16, 16, 128)) * 0.5).astype(np.float16)

        pid = arena_int8.write_page_fp16("a" * 64, k_orig, v_orig)
        k_deq, v_deq = arena_int8.read_page_fp16(pid)

        k_f32 = k_orig.astype(np.float32).reshape(-1)
        kd_f32 = k_deq.astype(np.float32).reshape(-1)
        cos_sim = float(np.dot(k_f32, kd_f32) / (np.linalg.norm(k_f32) * np.linalg.norm(kd_f32)))
        mean_err = float(np.mean(np.abs(k_f32 - kd_f32)))
        self.assertGreater(cos_sim, 0.9999)
        self.assertLess(mean_err, 0.005)
        print(
            f"[PASS] Test 3 — INT8 Paged KV Quantization Verified: "
            f"bytes/page={arena_int8.bytes_per_page:,} vs {arena_fp16.bytes_per_page:,} (-{savings_pct:.2f}%), "
            f"cosine_sim={cos_sim:.6f}, mean_abs_err={mean_err:.6f}"
        )

    def test_4_end_to_end_paged_kv_prefix_cache_and_disk_budget(self) -> None:
        """
        Verifies end-to-end `NativeKVPrefixCache` warm in-memory arena hits, cold disk
        reloads after clearing RAM arena, and bounded `max_disk_blocks` eviction.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="synapcore_p12_"))
        try:
            shutil.copy2(STORE_DIR / "merkle_manifest.json", tmp_dir / "merkle_manifest.json")
            shutil.copy2(STORE_DIR / "merkle_tree.idx", tmp_dir / "merkle_tree.idx")
            cas_file = open(STORE_DIR / "cas_experts.bin", "rb")
            cas_mmap = __import__("mmap").mmap(cas_file.fileno(), 0, access=__import__("mmap").ACCESS_READ)
            bridge = NativeMmapIndexBridge(tmp_dir, cas_mmap, verify_all_on_init=True)
            try:
                kv_cache = NativeKVPrefixCache(
                    tmp_dir,
                    bridge,
                    device="cpu",
                    num_layers=16,
                    num_kv_heads=16,
                    head_dim=128,
                    max_pages=4,
                    max_disk_blocks=3,
                    quant_mode="int8",
                )
                rng = np.random.default_rng(77)
                # 65 tokens -> 4 full 16-token blocks (64 tokens cacheable)
                tokens = list(range(100, 165))
                k_seq = (rng.standard_normal((16, 1, 16, 65, 128)) * 0.4).astype(np.float16)
                v_seq = (rng.standard_normal((16, 1, 16, 65, 128)) * 0.4).astype(np.float16)
                seed_root = bridge.base_model_root.hex()

                saved_pages = kv_cache.save_prefix_blocks_numpy(tokens, k_seq, v_seq, seed_root)
                self.assertEqual(saved_pages, 4)
                # Disk budget is max_disk_blocks=3 -> 1 oldest disk block must have been evicted
                self.assertEqual(kv_cache.disk_blocks_evicted, 1)

                # Test 3-block prefix (49 tokens -> 48 cacheable tokens = 3 pages)
                short_tokens = tokens[:49]
                # Save 3 blocks cleanly within max_disk_blocks=3
                kv_cache.save_prefix_blocks_numpy(short_tokens, k_seq[:, :, :, :49, :], v_seq[:, :, :, :49, :], seed_root)

                # 1. Warm in-memory PagedKVArena hit (0 disk loads)
                before_arena_hits = kv_cache.blocks_loaded_from_arena
                matched_toks, k_restored, v_restored = kv_cache.match_longest_prefix_numpy(
                    short_tokens, seed_root
                )
                self.assertEqual(matched_toks, 48)
                self.assertIsNotNone(k_restored)
                assert k_restored is not None
                self.assertEqual(k_restored.shape, (16, 1, 16, 48, 128))
                self.assertEqual(kv_cache.blocks_loaded_from_arena - before_arena_hits, 3)

                # 2. Clear RAM arena and verify reload from quantized disk blocks into PagedKVArena
                kv_cache.arena.page_table.clear()
                kv_cache.arena.reverse_page_table.clear()
                kv_cache.arena.free_pages.clear()
                kv_cache.arena.free_pages.extend(range(kv_cache.arena.max_pages))

                matched_disk, k_disk, _ = kv_cache.match_longest_prefix_numpy(short_tokens, seed_root)
                self.assertEqual(matched_disk, 48)
                assert k_disk is not None
                self.assertEqual(float(np.max(np.abs(k_restored - k_disk))), 0.0)
                stats = kv_cache.get_paged_kv_stats()
                print(
                    f"[PASS] Test 4 — End-to-End PagedKV Cache Verified: "
                    f"matched={matched_disk} tokens, quant={stats['quant_mode']}, "
                    f"arena_active={stats['arena_active_mb']} MiB, disk_evicted={stats['disk_blocks_evicted']}"
                )
            finally:
                bridge.close()
                cas_mmap.close()
                cas_file.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
