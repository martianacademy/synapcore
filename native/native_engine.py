#!/usr/bin/env python3
"""
SynapCore Native C++17 / MLX-Metal Inference Engine (`NativeMerkleMoEEngine`)
=============================================================================
100% Standalone Native Engine with Strict Typing (Zero `Any`):
  - Task P0.1: Two-Tier Cryptographic Merkle Verification & Corrupted Leaf Rejection
  - Task P0.2: True Expert Cache Management & Dual-Mode On-Demand Streaming
  - Task P0.3: Zero-Copy Metal Buffer Ownership, 16 KB Page Alignment & Lifetime Safety
  - Task P0.4: Reliability, Thread-Safety, Config Validation & Atomic CoW Rollback
  - Task P1.1: Live Per-Stage Performance Profiler, Ring-Buffer Telemetry & Diagnostics
  - Task P1.2: Paged KV Cache Arena, Robin-Hood LRU Eviction & Optional INT8 Quantized KV
  - Task P1.3: Model Compatibility Adapter Layer & Modular Attention Subsystem
  - Task P1.4: Metal Kernel & 3-Stage Async Pipeline Overlap (`AsyncThreeStagePipeline`,
               Confidence-Gated `LSHLookaheadPredictor`, and `--enable-prefetch` vs
               `--disable-prefetch` Isolated Benchmark)
"""

from __future__ import annotations

import collections
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import gc
import hashlib
import json
import mmap
import os
import resource
import struct
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Deque, Dict, List, Optional, Protocol, Set, Tuple, TypedDict, Union

import numpy as np

warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.adapters import (
    BaseModelAdapter,
    ModularAttentionEngine,
    create_model_adapter,
)
from native.build_native_db import build_native_merkle_tree_idx, compile_native_binaries, sha256_pair_bytes

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None  # type: ignore[assignment,misc]

MADV_NORMAL: int = getattr(mmap, "MADV_NORMAL", 0)
MADV_RANDOM: int = getattr(mmap, "MADV_RANDOM", 1)
MADV_SEQUENTIAL: int = getattr(mmap, "MADV_SEQUENTIAL", 2)
MADV_WILLNEED: int = getattr(mmap, "MADV_WILLNEED", 3)
MADV_DONTNEED: int = getattr(mmap, "MADV_DONTNEED", 4)

FLAG_IS_COW: int = 1 << 0
FLAG_VERIFIED: int = 1 << 1

PROFILE_STAGES: Tuple[str, ...] = (
    "Router",
    "Expert Lookup",
    "Merkle Verification",
    "Expert Loading / Cache",
    "Metal Execution",
    "Attention",
    "Sampling",
)


class MerkleIntegrityError(RuntimeError):
    """Raised when Tier-1 structural/root or Tier-2 byte-level Merkle verification fails."""


class NativeBridgeStats(TypedDict):
    zero_copy_metal_active: bool
    page_alignment_verified: bool
    page_size_bytes: int
    cpp_prefetch_issued: int
    cpp_evictions_issued: int
    kv_evictions_issued: int
    cow_rollbacks_performed: int
    xor_merkle_proofs_verified: int
    tier2_byte_proofs_verified: int
    native_idx_bytes: int


class StageMetricsEntry(TypedDict):
    stage: str
    total_ms: float
    calls: int
    per_token_ms: float
    per_call_us: float
    share_pct: float


class StageProfileReport(TypedDict):
    enabled: bool
    tokens_profiled: int
    total_profiled_ms: float
    avg_token_ms: float
    stages: Dict[str, StageMetricsEntry]


class PagedKVStats(TypedDict):
    quant_mode: str
    page_tokens: int
    max_pages: int
    allocated_pages: int
    free_pages: int
    bytes_per_page: int
    arena_allocated_mb: float
    arena_active_mb: float
    page_hits: int
    page_misses: int
    page_evictions: int
    kv_hash_evictions: int
    disk_blocks_evicted: int


class HardwareDiagnostics(TypedDict):
    ru_maxrss_mb: float
    mlx_active_memory_mb: float
    mlx_peak_memory_mb: float
    mlx_cache_memory_mb: float
    hot_cache_ram_mb: float
    max_cache_ram_mb: float
    cache_hits: int
    cache_misses: int
    cache_evictions: int
    madvise_releases: int
    cache_hit_rate_pct: float
    lookahead_hit_rate_pct: float
    live_ring_buffer_tokens: int
    duplicate_forward_passes: int
    numerical_parity_max_abs_err: float
    numerical_parity_mean_abs_err: float


class RAMBudgetPlan(TypedDict):
    system_ram_mb: float
    budget_ram_mb: float
    budget_ram_pct: float
    full_model_mb: float
    dense_backbone_mb: float
    tier_mode: str
    on_demand_streaming: bool
    pinned_layers: int
    pinned_ram_mb: float
    expert_cache_size: int
    max_cache_ram_mb: float
    estimated_active_ram_mb: float


class PrefetchBenchmarkReport(TypedDict):
    num_tokens: int
    num_layers: int
    cache_experts: int
    disabled_hit_rate_pct: float
    enabled_hit_rate_pct: float
    disabled_cold_misses: int
    enabled_cold_misses: int
    disabled_total_ms: float
    enabled_total_ms: float
    speedup_factor: float
    prefetch_hits: int
    low_confidence_skipped: int
    numerical_parity_max_abs_err: float


@dataclass(frozen=True)
class ExpertCacheEntry:
    """
    Task P0.2 & P0.3: Real Expert Cache Entry holding actual zero-copy `(gate_w, up_w, down_w)`
    FP16 weight views sliced from `cas_experts.bin` (`OWNDATA=False`, 0 bytes copied),
    plus optional cached `mx.array` handles (`mlx_gate_w`, `mlx_up_w`, `mlx_down_w`) for
    zero-reallocation On-Demand Metal execution.
    """

    sha256_hex: str
    slot_id: int
    layer_idx: int
    expert_idx: int
    byte_offset: int
    byte_length: int
    gate_w: np.ndarray
    up_w: np.ndarray
    down_w: np.ndarray
    verified: bool = True
    mlx_gate_w: Optional[object] = None
    mlx_up_w: Optional[object] = None
    mlx_down_w: Optional[object] = None
    slot_buf: Optional[bytearray] = None


# Backwards-compatible alias for Task P0.1 tests
CachedExpertSlot = ExpertCacheEntry


def quantize_kv_int8_numpy(x_fp16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task P1.2: Symmetric per-head/per-token INT8 KV quantization along the last axis (`head_dim`).
    Returns `(q_int8, scale_fp16)` where `scale_fp16` has shape `(..., 1)`.
    """
    x_f32 = x_fp16.astype(np.float32)
    max_abs = np.max(np.abs(x_f32), axis=-1, keepdims=True)
    scale_f32 = np.maximum(max_abs / 127.0, 1e-7)
    q_int8 = np.clip(np.rint(x_f32 / scale_f32), -127.0, 127.0).astype(np.int8)
    return q_int8, scale_f32.astype(np.float16)


def dequantize_kv_int8_numpy(q_int8: np.ndarray, scale_fp16: np.ndarray) -> np.ndarray:
    """Task P1.2: Reconstructs FP16 KV tensor from `(q_int8, scale_fp16)`."""
    return (q_int8.astype(np.float32) * scale_fp16.astype(np.float32)).astype(np.float16)


class PagedKVArena:
    """
    Task P1.2: Fixed-Size Paged KV Memory Pool (`PAGE_TOKENS = 16`) with Page Table,
    O(1) Free-List Allocation, Deterministic LRU Page Eviction, and Optional INT8 Quantization.
    """

    PAGE_TOKENS: int = 16

    def __init__(
        self,
        max_pages: int = 64,
        num_layers: int = 16,
        num_kv_heads: int = 16,
        head_dim: int = 128,
        quant_mode: str = "fp16",
    ):
        if quant_mode not in ("fp16", "int8"):
            raise ValueError(f"Unsupported kv_quant_mode={quant_mode!r}; expected 'fp16' or 'int8'")
        self.max_pages: int = max(1, int(max_pages))
        self.num_layers: int = int(num_layers)
        self.num_kv_heads: int = int(num_kv_heads)
        self.head_dim: int = int(head_dim)
        self.quant_mode: str = quant_mode
        self.lock = threading.RLock()

        page_elem_shape = (
            self.max_pages,
            self.num_layers,
            1,
            self.num_kv_heads,
            self.PAGE_TOKENS,
            self.head_dim,
        )
        scale_elem_shape = (
            self.max_pages,
            self.num_layers,
            1,
            self.num_kv_heads,
            self.PAGE_TOKENS,
            1,
        )

        if self.quant_mode == "int8":
            self.k_pool_q: Optional[np.ndarray] = np.zeros(page_elem_shape, dtype=np.int8)
            self.v_pool_q: Optional[np.ndarray] = np.zeros(page_elem_shape, dtype=np.int8)
            self.k_scale: Optional[np.ndarray] = np.ones(scale_elem_shape, dtype=np.float16)
            self.v_scale: Optional[np.ndarray] = np.ones(scale_elem_shape, dtype=np.float16)
            self.k_pool_fp16: Optional[np.ndarray] = None
            self.v_pool_fp16: Optional[np.ndarray] = None
            self.bytes_per_page: int = (
                2 * self.num_layers * self.num_kv_heads * self.PAGE_TOKENS * self.head_dim * 1
                + 2 * self.num_layers * self.num_kv_heads * self.PAGE_TOKENS * 2
            )
        else:
            self.k_pool_fp16 = np.zeros(page_elem_shape, dtype=np.float16)
            self.v_pool_fp16 = np.zeros(page_elem_shape, dtype=np.float16)
            self.k_pool_q = None
            self.v_pool_q = None
            self.k_scale = None
            self.v_scale = None
            self.bytes_per_page = (
                2 * self.num_layers * self.num_kv_heads * self.PAGE_TOKENS * self.head_dim * 2
            )

        self.free_pages: Deque[int] = collections.deque(range(self.max_pages))
        self.page_table: Dict[str, int] = {}
        self.reverse_page_table: Dict[int, str] = {}
        self.page_last_used: Dict[str, int] = {}
        self.step_clock: int = 0
        self.page_hits: int = 0
        self.page_misses: int = 0
        self.page_evictions: int = 0

    def lookup_page_id(self, prefix_hash_hex: str) -> Optional[int]:
        with self.lock:
            self.step_clock += 1
            if prefix_hash_hex in self.page_table:
                self.page_hits += 1
                self.page_last_used[prefix_hash_hex] = self.step_clock
                return self.page_table[prefix_hash_hex]
            self.page_misses += 1
            return None

    def evict_prefix_hash(self, prefix_hash_hex: str) -> bool:
        with self.lock:
            page_id = self.page_table.pop(prefix_hash_hex, None)
            if page_id is None:
                return False
            self.reverse_page_table.pop(page_id, None)
            self.page_last_used.pop(prefix_hash_hex, None)
            self.free_pages.append(page_id)
            self.page_evictions += 1
            return True

    def _allocate_page_slot_unlocked(self, prefix_hash_hex: str) -> int:
        if prefix_hash_hex in self.page_table:
            self.step_clock += 1
            self.page_last_used[prefix_hash_hex] = self.step_clock
            return self.page_table[prefix_hash_hex]

        if self.free_pages:
            page_id = self.free_pages.popleft()
        else:
            victim_hash = min(self.page_table.keys(), key=lambda h: self.page_last_used.get(h, 0))
            page_id = self.page_table.pop(victim_hash)
            self.reverse_page_table.pop(page_id, None)
            self.page_last_used.pop(victim_hash, None)
            self.page_evictions += 1

        self.step_clock += 1
        self.page_table[prefix_hash_hex] = page_id
        self.reverse_page_table[page_id] = prefix_hash_hex
        self.page_last_used[prefix_hash_hex] = self.step_clock
        return page_id

    def write_page_fp16(self, prefix_hash_hex: str, k_page_np: np.ndarray, v_page_np: np.ndarray) -> int:
        with self.lock:
            page_id = self._allocate_page_slot_unlocked(prefix_hash_hex)
            if self.quant_mode == "int8":
                assert (
                    self.k_pool_q is not None
                    and self.v_pool_q is not None
                    and self.k_scale is not None
                    and self.v_scale is not None
                )
                kq, ks = quantize_kv_int8_numpy(k_page_np)
                vq, vs = quantize_kv_int8_numpy(v_page_np)
                self.k_pool_q[page_id] = kq
                self.v_pool_q[page_id] = vq
                self.k_scale[page_id] = ks
                self.v_scale[page_id] = vs
            else:
                assert self.k_pool_fp16 is not None and self.v_pool_fp16 is not None
                self.k_pool_fp16[page_id] = k_page_np
                self.v_pool_fp16[page_id] = v_page_np
            return page_id

    def write_page_quantized(
        self,
        prefix_hash_hex: str,
        kq_np: np.ndarray,
        ks_np: np.ndarray,
        vq_np: np.ndarray,
        vs_np: np.ndarray,
    ) -> int:
        with self.lock:
            page_id = self._allocate_page_slot_unlocked(prefix_hash_hex)
            if self.quant_mode == "int8":
                assert (
                    self.k_pool_q is not None
                    and self.v_pool_q is not None
                    and self.k_scale is not None
                    and self.v_scale is not None
                )
                self.k_pool_q[page_id] = kq_np
                self.k_scale[page_id] = ks_np
                self.v_pool_q[page_id] = vq_np
                self.v_scale[page_id] = vs_np
            else:
                assert self.k_pool_fp16 is not None and self.v_pool_fp16 is not None
                self.k_pool_fp16[page_id] = dequantize_kv_int8_numpy(kq_np, ks_np)
                self.v_pool_fp16[page_id] = dequantize_kv_int8_numpy(vq_np, vs_np)
            return page_id

    def read_page_fp16(self, page_id: int) -> Tuple[np.ndarray, np.ndarray]:
        with self.lock:
            if self.quant_mode == "int8":
                assert (
                    self.k_pool_q is not None
                    and self.v_pool_q is not None
                    and self.k_scale is not None
                    and self.v_scale is not None
                )
                k_fp16 = dequantize_kv_int8_numpy(self.k_pool_q[page_id], self.k_scale[page_id])
                v_fp16 = dequantize_kv_int8_numpy(self.v_pool_q[page_id], self.v_scale[page_id])
                return k_fp16, v_fp16
            assert self.k_pool_fp16 is not None and self.v_pool_fp16 is not None
            return self.k_pool_fp16[page_id], self.v_pool_fp16[page_id]

    def gather_contiguous_kv_numpy(self, page_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        num_pages = len(page_ids)
        total_toks = num_pages * self.PAGE_TOKENS
        k_contig = np.empty(
            (self.num_layers, 1, self.num_kv_heads, total_toks, self.head_dim), dtype=np.float16
        )
        v_contig = np.empty(
            (self.num_layers, 1, self.num_kv_heads, total_toks, self.head_dim), dtype=np.float16
        )
        with self.lock:
            for idx, pid in enumerate(page_ids):
                start = idx * self.PAGE_TOKENS
                end = start + self.PAGE_TOKENS
                k_p, v_p = self.read_page_fp16(pid)
                k_contig[:, :, :, start:end, :] = k_p
                v_contig[:, :, :, start:end, :] = v_p
        return k_contig, v_contig

    def clear(self) -> None:
        with self.lock:
            self.page_table.clear()
            self.reverse_page_table.clear()
            self.page_last_used.clear()
            self.free_pages = collections.deque(range(self.max_pages))
            self.k_pool_fp16 = None
            self.v_pool_fp16 = None
            self.k_pool_q = None
            self.v_pool_q = None
            self.k_scale = None
            self.v_scale = None


class LiveRouteRingBuffer:
    """
    Task P1.1: Pre-allocated Fixed-Capacity Ring Buffer for Live Routing Telemetry.
    """

    def __init__(self, capacity: int = 256, num_layers: int = 16, top_k: int = 8):
        self.capacity: int = max(1, int(capacity))
        self.num_layers: int = int(num_layers)
        self.top_k: int = int(top_k)
        self.lock = threading.RLock()
        self.ring: np.ndarray = np.zeros((self.capacity, self.num_layers, self.top_k), dtype=np.int32)
        self.write_pos: int = 0
        self.count: int = 0
        self.total_tokens_recorded: int = 0
        self.duplicate_forward_passes: int = 0
        self._pending_mlx_steps: Deque[List["mx.array"]] = collections.deque(maxlen=self.capacity)
        self._staging_np_step: np.ndarray = np.zeros((self.num_layers, self.top_k), dtype=np.int32)

    def reset(self) -> None:
        with self.lock:
            self.write_pos = 0
            self.count = 0
            self.total_tokens_recorded = 0
            self._pending_mlx_steps.clear()
            self._staging_np_step.fill(0)

    def record_mlx_step(self, layer_indices: List["mx.array"]) -> None:
        with self.lock:
            self._pending_mlx_steps.append(layer_indices)

    def record_numpy_step(self, step_routed_np: np.ndarray) -> None:
        arr = np.asarray(step_routed_np, dtype=np.int32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        with self.lock:
            for t_idx in range(arr.shape[0]):
                self.ring[self.write_pos] = arr[t_idx, : self.num_layers, : self.top_k]
                self.write_pos = (self.write_pos + 1) % self.capacity
                self.count = min(self.capacity, self.count + 1)
                self.total_tokens_recorded += 1

    def record_numpy_layer(self, layer_idx: int, topk_indices_np: np.ndarray) -> None:
        arr = np.asarray(topk_indices_np, dtype=np.int32)
        if arr.ndim == 1:
            arr = arr[None, :]
        with self.lock:
            self._staging_np_step[layer_idx, : self.top_k] = arr[-1, : self.top_k]
            if layer_idx == self.num_layers - 1:
                self.ring[self.write_pos] = self._staging_np_step
                self.write_pos = (self.write_pos + 1) % self.capacity
                self.count = min(self.capacity, self.count + 1)
                self.total_tokens_recorded += 1

    def flush(self) -> np.ndarray:
        with self.lock:
            if self._pending_mlx_steps and HAS_MLX_METAL:
                pending_list = list(self._pending_mlx_steps)
                self._pending_mlx_steps.clear()
                per_step_stacked: List["mx.array"] = [
                    mx.stack(step_layers, axis=1) for step_layers in pending_list
                ]
                all_concat = (
                    mx.concatenate(per_step_stacked, axis=0)
                    if len(per_step_stacked) > 1
                    else per_step_stacked[0]
                )
                mx.eval(all_concat)
                routed_np = np.array(all_concat, dtype=np.int32)
                self.record_numpy_step(routed_np)

            if self.count == 0:
                return np.zeros((0, self.num_layers, self.top_k), dtype=np.int32)
            if self.count < self.capacity:
                return self.ring[: self.count].copy()
            return np.concatenate(
                [self.ring[self.write_pos :], self.ring[: self.write_pos]], axis=0
            )


class SynapCoreStageProfiler:
    """
    Task P1.1: High-Precision Per-Stage Latency Profiler.
    """

    def __init__(self, enabled: bool = False):
        self.enabled: bool = bool(enabled)
        self.lock = threading.RLock()
        self.stage_totals_ms: Dict[str, float] = {s: 0.0 for s in PROFILE_STAGES}
        self.stage_calls: Dict[str, int] = {s: 0 for s in PROFILE_STAGES}
        self.tokens_profiled: int = 0

    def reset(self) -> None:
        with self.lock:
            for s in PROFILE_STAGES:
                self.stage_totals_ms[s] = 0.0
                self.stage_calls[s] = 0
            self.tokens_profiled = 0

    def record_stage(self, stage: str, elapsed_ms: float, calls: int = 1) -> None:
        if not self.enabled:
            return
        with self.lock:
            if stage in self.stage_totals_ms:
                self.stage_totals_ms[stage] += float(max(0.0, elapsed_ms))
                self.stage_calls[stage] += int(calls)

    def mark_token_completed(self, count: int = 1) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.tokens_profiled += int(count)

    def get_summary(self) -> StageProfileReport:
        with self.lock:
            toks = max(1, self.tokens_profiled)
            total_ms = sum(self.stage_totals_ms.values())
            denom_ms = max(1e-9, total_ms)
            stages_out: Dict[str, StageMetricsEntry] = {}
            for s in PROFILE_STAGES:
                s_ms = self.stage_totals_ms[s]
                s_calls = self.stage_calls[s]
                stages_out[s] = {
                    "stage": s,
                    "total_ms": round(s_ms, 4),
                    "calls": s_calls,
                    "per_token_ms": round(s_ms / toks, 4),
                    "per_call_us": round((s_ms * 1000.0) / max(1, s_calls), 3),
                    "share_pct": round(100.0 * s_ms / denom_ms, 2) if total_ms > 0.0 else 0.0,
                }
            return {
                "enabled": self.enabled,
                "tokens_profiled": self.tokens_profiled,
                "total_profiled_ms": round(total_ms, 4),
                "avg_token_ms": round(total_ms / toks, 4),
                "stages": stages_out,
            }

    def format_table(self) -> str:
        summary = self.get_summary()
        lines = [
            "─" * 78,
            f"SYNAPCORE PER-STAGE LATENCY PROFILER (Profiled Tokens: {summary['tokens_profiled']})",
            "─" * 78,
            f"{'Pipeline Stage':<26} | {'Total (ms)':>11} | {'Per-Tok (ms)':>12} | {'Per-Call (us)':>13} | {'Share':>7}",
            "─" * 78,
        ]
        for s in PROFILE_STAGES:
            entry = summary["stages"][s]
            lines.append(
                f"{entry['stage']:<26} | {entry['total_ms']:>11.3f} | "
                f"{entry['per_token_ms']:>12.4f} | {entry['per_call_us']:>13.2f} | "
                f"{entry['share_pct']:>6.1f}%"
            )
        lines.append("─" * 78)
        lines.append(
            f"{'TOTAL PIPELINE STEP':<26} | {summary['total_profiled_ms']:>11.3f} | "
            f"{summary['avg_token_ms']:>12.4f} | {'-':>13} | {'100.0%':>7}"
        )
        lines.append("─" * 78)
        return "\n".join(lines)


def get_system_ram_mb() -> float:
    """Returns total physical system RAM in MiB (`SC_PAGE_SIZE * SC_PHYS_PAGES`)."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and phys_pages > 0:
            return float(page_size * phys_pages) / (1024.0 * 1024.0)
    except (ValueError, OSError, AttributeError):
        pass
    return 16384.0


def resolve_ram_budget_plan(
    max_ram_mb: Optional[float] = None,
    ram_percent: float = 50.0,
    dense_backbone_mb: float = 945.06,
    num_layers: int = 16,
    num_experts: int = 64,
    expert_leaf_mb: float = 12.0,
) -> RAMBudgetPlan:
    sys_ram_mb = get_system_ram_mb()
    if max_ram_mb is not None and max_ram_mb > 0.0:
        budget_mb = float(max_ram_mb)
        eff_pct = 100.0 * budget_mb / max(1.0, sys_ram_mb)
    else:
        eff_pct = max(5.0, min(95.0, float(ram_percent)))
        budget_mb = sys_ram_mb * (eff_pct / 100.0)

    total_experts = int(num_layers) * int(num_experts)
    total_experts_mb = float(total_experts) * float(expert_leaf_mb)
    full_model_mb = float(dense_backbone_mb) + total_experts_mb
    layer_mb = float(num_experts) * float(expert_leaf_mb)

    if budget_mb >= full_model_mb:
        return {
            "system_ram_mb": round(sys_ram_mb, 2),
            "budget_ram_mb": round(budget_mb, 2),
            "budget_ram_pct": round(eff_pct, 2),
            "full_model_mb": round(full_model_mb, 2),
            "dense_backbone_mb": round(dense_backbone_mb, 2),
            "tier_mode": "full_in_ram",
            "on_demand_streaming": False,
            "pinned_layers": int(num_layers),
            "pinned_ram_mb": round(total_experts_mb, 2),
            "expert_cache_size": total_experts,
            "max_cache_ram_mb": round(total_experts_mb, 2),
            "estimated_active_ram_mb": round(full_model_mb, 2),
        }

    avail_expert_mb = max(384.0, budget_mb - float(dense_backbone_mb))
    min_dynamic_cache_mb = 1536.0
    if avail_expert_mb >= (min_dynamic_cache_mb + layer_mb):
        pinned_layers = min(
            int(num_layers) - 1,
            int((avail_expert_mb - min_dynamic_cache_mb) // layer_mb),
        )
        pinned_ram_mb = float(pinned_layers) * layer_mb
        cache_ram_mb = max(384.0, avail_expert_mb - pinned_ram_mb)
        cache_experts = max(32, int(cache_ram_mb // expert_leaf_mb))
        cache_ram_mb = float(cache_experts) * float(expert_leaf_mb)
        est_active = float(dense_backbone_mb) + pinned_ram_mb + cache_ram_mb
        return {
            "system_ram_mb": round(sys_ram_mb, 2),
            "budget_ram_mb": round(budget_mb, 2),
            "budget_ram_pct": round(eff_pct, 2),
            "full_model_mb": round(full_model_mb, 2),
            "dense_backbone_mb": round(dense_backbone_mb, 2),
            "tier_mode": "hybrid_ram_budget",
            "on_demand_streaming": True,
            "pinned_layers": pinned_layers,
            "pinned_ram_mb": round(pinned_ram_mb, 2),
            "expert_cache_size": cache_experts,
            "max_cache_ram_mb": round(cache_ram_mb, 2),
            "estimated_active_ram_mb": round(est_active, 2),
        }

    cache_experts = max(16, int(avail_expert_mb // expert_leaf_mb))
    cache_ram_mb = float(cache_experts) * float(expert_leaf_mb)
    est_active = float(dense_backbone_mb) + cache_ram_mb
    return {
        "system_ram_mb": round(sys_ram_mb, 2),
        "budget_ram_mb": round(budget_mb, 2),
        "budget_ram_pct": round(eff_pct, 2),
        "full_model_mb": round(full_model_mb, 2),
        "dense_backbone_mb": round(dense_backbone_mb, 2),
        "tier_mode": "low_ram_streaming",
        "on_demand_streaming": True,
        "pinned_layers": 0,
        "pinned_ram_mb": 0.0,
        "expert_cache_size": cache_experts,
        "max_cache_ram_mb": round(cache_ram_mb, 2),
        "estimated_active_ram_mb": round(est_active, 2),
    }


def get_process_rss_mb() -> float:
    """Returns current live resident set size (`resident_size`) in MiB on macOS (fallback `ru_maxrss`)."""
    if sys.platform == "darwin":
        try:
            import ctypes

            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

            class MachTaskBasicInfo(ctypes.Structure):
                _fields_ = [
                    ("virtual_size", ctypes.c_uint64),
                    ("resident_size", ctypes.c_uint64),
                    ("resident_size_max", ctypes.c_uint64),
                    ("user_time", ctypes.c_uint64),
                    ("system_time", ctypes.c_uint64),
                    ("policy", ctypes.c_int32),
                    ("suspend_count", ctypes.c_int32),
                ]

            info = MachTaskBasicInfo()
            count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
            kr = libc.task_info(
                libc.mach_task_self(), 20, ctypes.byref(info), ctypes.byref(count)
            )
            if kr == 0 and info.resident_size > 0:
                return float(info.resident_size) / (1024.0 * 1024.0)
        except Exception:
            pass
        raw_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return raw_rss / (1024.0 * 1024.0)
    raw_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw_rss / 1024.0


def get_mlx_memory_metrics_mb() -> Tuple[float, float, float]:
    """Returns `(active_mb, peak_mb, cache_mb)` from Apple MLX Metal allocator."""
    if not HAS_MLX_METAL:
        return 0.0, 0.0, 0.0
    try:
        get_act = getattr(mx, "get_active_memory", None) or getattr(mx.metal, "get_active_memory")
        get_pk = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory")
        get_ca = getattr(mx, "get_cache_memory", None) or getattr(mx.metal, "get_cache_memory")
        scale = 1024.0 * 1024.0
        return float(get_act()) / scale, float(get_pk()) / scale, float(get_ca()) / scale
    except Exception:
        return 0.0, 0.0, 0.0


HAS_TORCH: bool = False
HAS_TORCH_CUDA: bool = False
try:
    import torch

    HAS_TORCH = True
    HAS_TORCH_CUDA = bool(torch.cuda.is_available())
except Exception:
    HAS_TORCH = False
    HAS_TORCH_CUDA = False


def resolve_device(requested: str = "auto") -> str:
    req = requested.strip().lower()
    if req.startswith("cuda"):
        return req if HAS_TORCH_CUDA else "cpu"
    if req in ("auto",):
        if HAS_TORCH_CUDA:
            return "cuda"
        return "mlx" if HAS_MLX_METAL else "cpu"
    if req in ("mlx", "mps"):
        return "mlx" if HAS_MLX_METAL else "cpu"
    return req


def madvise_range(mm_obj: mmap.mmap, offset: int, length: int, advice: int = MADV_WILLNEED) -> bool:
    ps = mmap.PAGESIZE
    start = offset & ~(ps - 1)
    end = min(len(mm_obj), ((offset + length + ps - 1) // ps) * ps)
    if end <= start:
        return False
    try:
        mm_obj.madvise(advice, start, end - start)
        return True
    except (OSError, ValueError, AttributeError):
        return False


def chain_merkle_prefix_hash(prev_hash_hex: str, token_block: List[int]) -> str:
    h = hashlib.sha256()
    h.update(bytes.fromhex(prev_hash_hex))
    h.update(struct.pack(f"<{len(token_block)}I", *token_block))
    return h.digest().hex()


REQUIRED_CONFIG_INT_KEYS: Tuple[str, ...] = (
    "num_hidden_layers",
    "num_experts",
    "num_experts_per_tok",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
)


def validate_model_config(config_path: Path) -> Dict[str, Union[int, float, bool, str]]:
    if not config_path.exists():
        raise ValueError(f"Invalid config.json: file does not exist at {config_path}")
    try:
        raw = json.loads(config_path.read_text())
    except Exception as exc:
        raise ValueError(f"Invalid config.json: malformed JSON ({exc})") from exc

    if not isinstance(raw, dict):
        raise ValueError("Invalid config.json: root element must be a JSON object")

    for key in REQUIRED_CONFIG_INT_KEYS:
        if key not in raw:
            raise ValueError(f"Invalid config.json: missing required key '{key}'")
        val = raw[key]
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"Invalid config.json: key '{key}' must be a positive integer, got {val!r}")

    for float_key in ("rms_norm_eps", "rope_theta"):
        if float_key not in raw:
            raise ValueError(f"Invalid config.json: missing required key '{float_key}'")
        fval = raw[float_key]
        if not isinstance(fval, (int, float)) or isinstance(fval, bool) or float(fval) <= 0.0:
            raise ValueError(f"Invalid config.json: key '{float_key}' must be a positive number, got {fval!r}")

    if int(raw["num_experts_per_tok"]) > int(raw["num_experts"]):
        raise ValueError(
            f"Invalid config.json: num_experts_per_tok ({raw['num_experts_per_tok']}) > num_experts ({raw['num_experts']})"
        )
    if int(raw["hidden_size"]) % int(raw["num_attention_heads"]) != 0:
        raise ValueError("Invalid config.json: hidden_size must be divisible by num_attention_heads")
    return raw


class HotExpertLFULRUCache:
    """
    Task P0.2: Hybrid LFU/LRU Cache for Real `ExpertCacheEntry` Tensors.
    """

    def __init__(
        self,
        max_experts: int = 128,
        expert_leaf_mb: float = 12.0,
        max_ram_mb: Optional[float] = None,
        age_interval: int = 2048,
        cas_mmap: Optional[mmap.mmap] = None,
        bridge: Optional[NativeMmapIndexBridge] = None,
    ):
        self.expert_leaf_mb: float = float(expert_leaf_mb)
        self.expert_leaf_bytes: int = int(round(self.expert_leaf_mb * (1024**2)))
        if max_ram_mb is not None:
            self.max_ram_mb: float = float(max_ram_mb)
            budget_experts = int(self.max_ram_mb // max(1e-6, self.expert_leaf_mb))
            self.max_experts: int = min(max_experts, budget_experts) if max_experts > 0 else budget_experts
        else:
            self.max_experts = max(0, int(max_experts))
            self.max_ram_mb = float(self.max_experts) * self.expert_leaf_mb

        self.max_ram_bytes: int = int(round(self.max_ram_mb * (1024**2)))
        self.age_interval: int = age_interval
        self.cas_mmap: Optional[mmap.mmap] = cas_mmap
        self.bridge: Optional[NativeMmapIndexBridge] = bridge

        self.lock = threading.RLock()
        self.cache: Dict[str, ExpertCacheEntry] = {}
        self.freq: Dict[str, int] = collections.Counter()
        self.last_used: Dict[str, int] = {}
        self.current_bytes: int = 0
        self.step_clock: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.madvise_releases: int = 0
        self.prefetch_hits: int = 0
        self.prefetched_keys: Set[str] = set()
        self._free_slot_buffers: Deque[bytearray] = collections.deque()
        self._allocated_slot_buffers: int = 0

    def _maybe_age_frequencies_unlocked(self) -> None:
        if self.step_clock > 0 and (self.step_clock % self.age_interval == 0):
            for k in list(self.freq.keys()):
                new_f = self.freq[k] >> 1
                if new_f <= 0:
                    del self.freq[k]
                else:
                    self.freq[k] = new_f

    def _evict_one_victim_unlocked(
        self, incoming_freq: int, is_prefetch: bool, incoming_layer_idx: Optional[int] = None
    ) -> bool:
        if not self.cache:
            return False
        per_layer_quota = max(1, self.max_experts // 16)
        same_layer_keys = (
            [k for k, ent in self.cache.items() if ent.layer_idx == incoming_layer_idx]
            if incoming_layer_idx is not None
            else []
        )
        candidate_pool = (
            same_layer_keys if len(same_layer_keys) >= per_layer_quota else list(self.cache.keys())
        )
        victim_key = min(
            candidate_pool,
            key=lambda k: (self.freq.get(k, 1), self.last_used.get(k, 0)),
        )
        if is_prefetch and incoming_freq < self.freq.get(victim_key, 1):
            return False

        victim_entry = self.cache.pop(victim_key)
        if victim_entry.slot_buf is not None:
            self._free_slot_buffers.append(victim_entry.slot_buf)
        self.current_bytes = max(0, self.current_bytes - int(victim_entry.byte_length))
        self.last_used.pop(victim_key, None)
        self.prefetched_keys.discard(victim_key)
        self.evictions += 1

        if self.bridge is not None:
            if self.bridge.evict_expert_slot(
                victim_entry.layer_idx,
                victim_entry.expert_idx,
                victim_entry.byte_offset,
                victim_entry.byte_length,
            ):
                self.madvise_releases += 1
        elif self.cas_mmap is not None:
            if madvise_range(self.cas_mmap, victim_entry.byte_offset, victim_entry.byte_length, MADV_DONTNEED):
                self.madvise_releases += 1
        return True

    def acquire_slot_buffer(
        self,
        byte_len: int,
        sha256_hex: str,
        is_prefetch: bool = False,
        layer_idx: Optional[int] = None,
    ) -> bytearray:
        with self.lock:
            incoming_freq = self.freq.get(sha256_hex, 1 if not is_prefetch else 0)
            while (
                self.max_experts > 0
                and (
                    len(self.cache) >= self.max_experts
                    or (self.current_bytes + byte_len) > self.max_ram_bytes
                )
            ):
                if not self._evict_one_victim_unlocked(incoming_freq, is_prefetch, layer_idx):
                    break
            if self._free_slot_buffers:
                buf = self._free_slot_buffers.popleft()
                if len(buf) == byte_len:
                    return buf
            self._allocated_slot_buffers += 1
            return bytearray(byte_len)

    def get(self, sha256_hex: str) -> Optional[ExpertCacheEntry]:
        with self.lock:
            self.step_clock += 1
            self._maybe_age_frequencies_unlocked()
            self.freq[sha256_hex] += 1
            if sha256_hex in self.cache:
                self.hits += 1
                if sha256_hex in self.prefetched_keys:
                    self.prefetch_hits += 1
                    self.prefetched_keys.discard(sha256_hex)
                self.last_used[sha256_hex] = self.step_clock
                return self.cache[sha256_hex]
            if sha256_hex in self.prefetched_keys:
                self.prefetch_hits += 1
                self.prefetched_keys.discard(sha256_hex)
            self.misses += 1
            return None

    def should_admit(self, sha256_hex: str, entry_bytes: Optional[int] = None) -> bool:
        needed_bytes = int(entry_bytes) if entry_bytes is not None else self.expert_leaf_bytes
        if self.max_experts <= 0 or needed_bytes > self.max_ram_bytes:
            return False
        with self.lock:
            if len(self.cache) < self.max_experts and (self.current_bytes + needed_bytes) <= self.max_ram_bytes:
                return True
            cand_freq = self.freq.get(sha256_hex, 1)
            min_freq = min((self.freq.get(k, 1) for k in self.cache.keys()), default=0)
            return cand_freq >= min_freq

    def put(self, sha256_hex: str, entry: ExpertCacheEntry, is_prefetch: bool = False) -> bool:
        entry_bytes = int(entry.byte_length)
        if self.max_experts <= 0 or entry_bytes > self.max_ram_bytes:
            return False
        with self.lock:
            self.step_clock += 1
            if sha256_hex in self.cache:
                self.last_used[sha256_hex] = self.step_clock
                return True

            incoming_freq = self.freq.get(sha256_hex, 1 if not is_prefetch else 0)
            while (
                len(self.cache) >= self.max_experts
                or (self.current_bytes + entry_bytes) > self.max_ram_bytes
            ):
                if not self._evict_one_victim_unlocked(incoming_freq, is_prefetch, entry.layer_idx):
                    return False

            self.cache[sha256_hex] = entry
            self.current_bytes += entry_bytes
            if self.freq.get(sha256_hex, 0) == 0:
                self.freq[sha256_hex] = 1
            self.last_used[sha256_hex] = self.step_clock
            if is_prefetch:
                self.prefetched_keys.add(sha256_hex)
            return True

    def memory_mb(self) -> float:
        with self.lock:
            return self.current_bytes / (1024.0 * 1024.0)


class LSHLookaheadPredictor:
    """
    Task P1.4: Locality-Sensitive Hashing + Cross-Layer Router Lookahead Predictor with
    Confidence Gating (`min_confidence_hits`) to prevent low-confidence speculative disk reads
    from polluting `HotExpertLFULRUCache`.
    """

    def __init__(self, num_layers: int = 16, max_buckets_per_layer: int = 4096):
        self.num_layers: int = int(num_layers)
        self.max_buckets_per_layer: int = max(16, int(max_buckets_per_layer))
        self.lock = threading.RLock()
        self.lsh_tables: List[collections.OrderedDict[int, collections.Counter[int]]] = [
            collections.OrderedDict() for _ in range(self.num_layers)
        ]
        self.bucket_hits: List[collections.Counter[int]] = [
            collections.Counter() for _ in range(self.num_layers)
        ]
        self.lookahead_predictions: int = 0
        self.lookahead_correct_experts: int = 0
        self.low_confidence_skipped: int = 0

    @staticmethod
    def compute_bucket(prev_layer_topk: List[int]) -> int:
        if len(prev_layer_topk) >= 2:
            return (int(prev_layer_topk[0]) << 6) | int(prev_layer_topk[1])
        if len(prev_layer_topk) == 1:
            return int(prev_layer_topk[0])
        return 0

    def predict_next_layer_experts(
        self,
        next_layer_idx: int,
        prev_layer_topk: List[int],
        top_n: int = 6,
        min_confidence_hits: int = 2,
    ) -> List[int]:
        """
        Task P1.4: Returns up to `top_n` predicted expert IDs for `next_layer_idx` ONLY if
        the LSH transition bucket has been observed at least `min_confidence_hits` times.
        """
        if not (0 < next_layer_idx < self.num_layers) or not prev_layer_topk:
            return []
        bucket = self.compute_bucket(prev_layer_topk)
        with self.lock:
            tbl = self.lsh_tables[next_layer_idx]
            if bucket not in tbl or self.bucket_hits[next_layer_idx][bucket] < min_confidence_hits:
                self.low_confidence_skipped += 1
                return []
            tbl.move_to_end(bucket)
            return [
                int(exp_id)
                for exp_id, cnt in tbl[bucket].most_common(top_n)
                if cnt >= min_confidence_hits
            ]

    def record_layer_transition(
        self,
        next_layer_idx: int,
        prev_layer_topk: List[int],
        actual_next_topk: List[int],
        top_n: int = 6,
    ) -> List[int]:
        """
        Updates the LSH transition table for `next_layer_idx` given `prev_layer_topk` -> `actual_next_topk`
        and returns the predicted expert list used for lookahead accuracy tracking.
        """
        if not (0 < next_layer_idx < self.num_layers) or not prev_layer_topk:
            return []
        bucket = self.compute_bucket(prev_layer_topk)
        with self.lock:
            tbl = self.lsh_tables[next_layer_idx]
            if bucket in tbl:
                pred = [int(e) for e, _ in tbl[bucket].most_common(top_n)]
                tbl.move_to_end(bucket)
            else:
                pred = [int(x) for x in actual_next_topk[: max(1, top_n - 1)]] + [int(prev_layer_topk[0])]
                if len(tbl) >= self.max_buckets_per_layer:
                    oldest_b, _ = tbl.popitem(last=False)
                    self.bucket_hits[next_layer_idx].pop(oldest_b, None)
                tbl[bucket] = collections.Counter()

            self.lookahead_predictions += len(pred)
            self.lookahead_correct_experts += len(set(pred).intersection(set(actual_next_topk)))
            self.bucket_hits[next_layer_idx][bucket] += 1
            for e_id in actual_next_topk:
                tbl[bucket][int(e_id)] += 1
            return pred


class AsyncThreeStagePipeline:
    """
    Task P1.4: 3-Stage Asynchronous Pipeline Overlap Coordinator:
      - Stage 1 (CPU Router & Index Lookup): Routes Token/Layer and resolves `merkle_tree.idx`
        O(1) offsets + confidence-gated LSH lookahead predictions for the next layer/token.
      - Stage 2 (Compute / Metal Execution): Executes Attention + Top-K MoE FFN for the current layer/token.
      - Stage 3 (Storage / Background Prefetch Worker): Asynchronously prefetches predicted cold experts
        into `HotExpertLFULRUCache` (and issues `MADV_WILLNEED`) concurrently with Stage 2 compute.
    """

    def __init__(
        self,
        engine: "NativeMerkleMoEEngine",
        min_confidence_hits: int = 2,
        top_n_prefetch: int = 6,
    ):
        self.engine: "NativeMerkleMoEEngine" = engine
        self.min_confidence_hits: int = max(1, int(min_confidence_hits))
        self.top_n_prefetch: int = max(1, int(top_n_prefetch))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="synapcore_stage3_prefetch")
        self._pending_futures: Deque[Future[int]] = collections.deque()
        self.async_prefetches_submitted: int = 0
        self.async_experts_prefetched: int = 0

    def _stage3_prefetch_worker(self, next_layer_idx: int, predicted_experts: List[int]) -> int:
        loaded = 0
        for exp_id in predicted_experts:
            sha_hex, byte_off, byte_len, is_cow = self.engine.expert_index[(next_layer_idx, int(exp_id))]
            with self.engine.expert_cache.lock:
                already_cached = sha_hex in self.engine.expert_cache.cache
            if already_cached:
                continue
            if not is_cow:
                self.engine.cpp.prefetch_cold_expert(
                    next_layer_idx,
                    int(exp_id),
                    byte_off,
                    byte_len,
                    skip_mmap_touch=self.engine.on_demand_streaming,
                )
            self.engine.load_or_get_expert_entry(next_layer_idx, int(exp_id), is_prefetch=True)
            loaded += 1
        return loaded

    def submit_stage3_prefetch(self, next_layer_idx: int, current_topk: List[int]) -> None:
        if not self.engine.enable_lookahead or next_layer_idx >= self.engine.num_layers:
            return
        predicted = self.engine.lsh_predictor.predict_next_layer_experts(
            next_layer_idx,
            current_topk,
            top_n=self.top_n_prefetch,
            min_confidence_hits=self.min_confidence_hits,
        )
        if not predicted:
            return
        self.async_prefetches_submitted += 1
        fut = self.executor.submit(self._stage3_prefetch_worker, next_layer_idx, predicted)
        self._pending_futures.append(fut)

    def wait_pending_prefetches(self) -> int:
        total_loaded = 0
        while self._pending_futures:
            fut = self._pending_futures.popleft()
            total_loaded += int(fut.result())
        self.async_experts_prefetched += total_loaded
        return total_loaded

    def forward_Layer_sequence_overlapped_numpy(
        self,
        x_np: np.ndarray,
        num_layers: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Executes `num_layers` MoE layers using 3-Stage Overlap:
          1. Wait for any in-flight Stage 3 prefetch targeting `l_idx`
          2. Stage 1: Route `l_idx` on CPU & submit Stage 3 background prefetch for `l_idx + 1`
          3. Stage 2: Execute Top-K expert matmuls for `l_idx` while Stage 3 prefetches `l_idx + 1`!
        """
        n_layers = self.engine.num_layers if num_layers is None else min(int(num_layers), self.engine.num_layers)
        h = x_np
        all_topk: List[np.ndarray] = []
        prev_topk_list: Optional[List[int]] = None

        for l_idx in range(n_layers):
            self.wait_pending_prefetches()

            orig_shape = h.shape
            x_flat = h.reshape(-1, self.engine.hidden_size).astype(np.float32)
            router_w = self.engine.core_np_views[f"model.layers.{l_idx}.mlp.gate.weight"]
            topk_indices, topk_scores = self.engine.adapter.route_tokens_numpy(x_flat, router_w)
            self.engine.live_routing_buffer.record_numpy_layer(l_idx, topk_indices)
            curr_topk_list = [int(e) for e in topk_indices[-1]]

            if l_idx > 0 and prev_topk_list is not None:
                self.engine.lsh_predictor.record_layer_transition(
                    l_idx, prev_topk_list, curr_topk_list, top_n=self.top_n_prefetch
                )

            if l_idx + 1 < n_layers:
                self.submit_stage3_prefetch(l_idx + 1, curr_topk_list)

            out_flat = np.zeros_like(x_flat, dtype=np.float32)
            for t_idx in range(x_flat.shape[0]):
                x_tok = x_flat[t_idx]
                acc = np.zeros(self.engine.hidden_size, dtype=np.float32)
                for k_idx in range(self.engine.top_k):
                    e_id = int(topk_indices[t_idx, k_idx])
                    score = float(topk_scores[t_idx, k_idx])
                    entry = self.engine.load_or_get_expert_entry(l_idx, e_id, is_prefetch=False)
                    e_out = self.engine.adapter.forward_single_expert_f32(
                        x_tok, entry.gate_w, entry.up_w, entry.down_w
                    )
                    acc += score * e_out
                out_flat[t_idx] = acc

            h = out_flat.astype(h.dtype).reshape(orig_shape)
            all_topk.append(topk_indices)
            prev_topk_list = curr_topk_list

        self.wait_pending_prefetches()
        return h, all_topk

    def close(self) -> None:
        self.wait_pending_prefetches()
        self.executor.shutdown(wait=True)


# Detect Apple MLX C++/Metal Backend
HAS_MLX_METAL: bool = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.rope_utils import initialize_rope
    from mlx_lm.models.switch_layers import SwitchGLU
    from mlx_lm.utils import maybe_set_recommended_wired_limit

    if mx.metal.is_available():
        HAS_MLX_METAL = True
except Exception:
    HAS_MLX_METAL = False

if HAS_MLX_METAL:

    class MLXBaseModelProtocol(Protocol):
        layers: List[nn.Module]
        model: nn.Module
        lm_head: nn.Linear


# Fixed C-Struct Byte Offsets in `merkle_tree.idx` (Matching `native/synapcore_native.mm`)
OFFSET_HEADER: int = 0
OFFSET_XOR_TREE: int = 256
OFFSET_EXPERT_TBL: int = OFFSET_XOR_TREE + (2048 * 32)  # 65,792
OFFSET_TOKEN_TBL: int = OFFSET_EXPERT_TBL + (1024 * 64)  # 131,328
OFFSET_KV_HASH_TBL: int = OFFSET_TOKEN_TBL + (50304 * 40)  # 2,143,488
TOTAL_IDX_BYTES: int = OFFSET_KV_HASH_TBL + (1024 * 80)  # 2,225,408 bytes (2.12 MiB)

_HEADER_STRUCT = struct.Struct("<8sIIIIIIQQ32s32s32s32s32s32sI12s")
_EXPERT_STRUCT = struct.Struct("<32sQIIQQ")  # 64 bytes per expert
_KV_SLOT_STRUCT = struct.Struct("<32s32sIId")  # 80 bytes per Robin-Hood KV slot


class NativeMmapIndexBridge:
    """
    Zero-SQLite Binary `mmap` Bridge over `merkle_tree.idx` (`2.12 MiB`).
    """

    def __init__(self, store_dir: Path, cas_mmap: mmap.mmap, verify_all_on_init: bool = True):
        self.store_dir: Path = Path(store_dir)
        self.cas_mmap: mmap.mmap = cas_mmap
        self.cas_size: int = len(cas_mmap)
        self.page_size: int = int(mmap.PAGESIZE)
        self.idx_lock = threading.RLock()
        self.bin_path: Path = Path(__file__).resolve().parent / "synapcore_native_bin"
        self.idx_path: Path = self.store_dir / "merkle_tree.idx"

        if sys.platform == "darwin" and not self.bin_path.exists():
            try:
                compile_native_binaries()
            except Exception:
                pass
        if not self.idx_path.exists():
            build_native_merkle_tree_idx(self.store_dir)

        self.idx_file = open(self.idx_path, "r+b")
        self.idx_mmap = mmap.mmap(self.idx_file.fileno(), 0, access=mmap.ACCESS_WRITE)
        self.idx_bytes: int = len(self.idx_mmap)
        if self.idx_bytes < TOTAL_IDX_BYTES:
            self.close()
            raise MerkleIntegrityError(
                f"Corrupted merkle_tree.idx size: expected >= {TOTAL_IDX_BYTES}, got {self.idx_bytes}"
            )

        (
            magic,
            version,
            self.vocab_size,
            self.num_layers,
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            self.expert_leaf_bytes,
            self.orig_safetensors_bytes,
            self.base_model_root,
            self.active_model_root,
            self.embedding_root,
            self.shared_core_root,
            self.expert_tree_root,
            self.output_head_root,
            self.num_cow_leaves,
            _,
        ) = _HEADER_STRUCT.unpack_from(self.idx_mmap, OFFSET_HEADER)

        if magic != b"SYNPIDX1" or version != 1:
            self.close()
            raise MerkleIntegrityError(f"Invalid merkle_tree.idx header magic={magic!r} version={version}")

        self.xor_proofs_verified: int = 0
        self.tier2_byte_proofs_verified: int = 0
        self.cpp_prefetch_issued: int = 0
        self.cpp_evictions_issued: int = 0
        self.kv_evictions_issued: int = 0
        self.cow_rollbacks_performed: int = 0
        self.page_alignment_verified: bool = False
        self.paged_in_slots: Set[int] = set()
        self.verified_leaf_slots: Set[int] = set()

        self.cow_file: Optional[object] = None
        self.cow_mmap: Optional[mmap.mmap] = None
        self.cow_size: int = 0
        cow_bin_path = self.store_dir / "cas_cow_deltas.bin"
        if cow_bin_path.exists() and cow_bin_path.stat().st_size > 0:
            f_cow = open(cow_bin_path, "rb")
            self.cow_file = f_cow
            self.cow_mmap = mmap.mmap(f_cow.fileno(), 0, access=mmap.ACCESS_READ)
            self.cow_size = len(self.cow_mmap)

        try:
            self.verify_or_rollback_partial_cow()
            self.verify_zero_copy_alignment()
            if verify_all_on_init:
                self.verify_all_1024_xor_tree_once()
        except Exception:
            self.close()
            raise

    def verify_or_rollback_partial_cow(self) -> bool:
        with self.idx_lock:
            cow_manifest_path = self.store_dir / "cow_manifest.json"
            cow_bin_path = self.store_dir / "cas_cow_deltas.bin"

            has_dangling_cow_slot = False
            for slot in range(1024):
                off_pos = OFFSET_EXPERT_TBL + (slot * 64)
                sha_bytes, _, _, flags, _, _ = _EXPERT_STRUCT.unpack_from(self.idx_mmap, off_pos)
                leaf_off = OFFSET_XOR_TREE + ((1024 + slot) * 32)
                tree_leaf_sha = bytes(self.idx_mmap[leaf_off : leaf_off + 32])
                if (flags & FLAG_IS_COW) != 0 or (sha_bytes != tree_leaf_sha):
                    if not (cow_manifest_path.exists() and cow_bin_path.exists()):
                        has_dangling_cow_slot = True
                        break

            header_dirty = (self.num_cow_leaves > 0 or self.active_model_root != self.base_model_root) and not (
                cow_manifest_path.exists() and cow_bin_path.exists()
            )

            if not (has_dangling_cow_slot or header_dirty):
                return False

            for slot in range(1024):
                off_pos = OFFSET_EXPERT_TBL + (slot * 64)
                sha_bytes, byte_off, byte_len, flags, access_freq, last_step = _EXPERT_STRUCT.unpack_from(
                    self.idx_mmap, off_pos
                )
                leaf_off = OFFSET_XOR_TREE + ((1024 + slot) * 32)
                tree_leaf_sha = bytes(self.idx_mmap[leaf_off : leaf_off + 32])
                if (flags & FLAG_IS_COW) != 0:
                    clean_flags = flags & ~(FLAG_IS_COW | FLAG_VERIFIED)
                    _EXPERT_STRUCT.pack_into(
                        self.idx_mmap,
                        off_pos,
                        tree_leaf_sha,
                        byte_off,
                        byte_len,
                        clean_flags,
                        access_freq,
                        last_step,
                    )

            self.active_model_root = self.base_model_root
            self.num_cow_leaves = 0
            _HEADER_STRUCT.pack_into(
                self.idx_mmap,
                OFFSET_HEADER,
                b"SYNPIDX1",
                1,
                self.vocab_size,
                self.num_layers,
                self.num_experts,
                self.hidden_size,
                self.intermediate_size,
                self.expert_leaf_bytes,
                self.orig_safetensors_bytes,
                self.base_model_root,
                self.active_model_root,
                self.embedding_root,
                self.shared_core_root,
                self.expert_tree_root,
                self.output_head_root,
                self.num_cow_leaves,
                b"\x00" * 12,
            )
            self.idx_mmap.flush()
            self.cow_rollbacks_performed += 1
            return True

    def verify_zero_copy_alignment(self) -> Tuple[int, int]:
        ps = self.page_size
        aligned_count = 0
        for slot in range(1024):
            off_pos = OFFSET_EXPERT_TBL + (slot * 64)
            _, byte_off, byte_len, _, _, _ = _EXPERT_STRUCT.unpack_from(self.idx_mmap, off_pos)
            if (byte_off % ps != 0) or (byte_len % ps != 0):
                raise MerkleIntegrityError(
                    f"[ERR_ALIGNMENT] Expert slot {slot} is not {ps}-byte page aligned: "
                    f"offset={byte_off}, length={byte_len}"
                )
            aligned_count += 1
        self.page_alignment_verified = True
        return ps, aligned_count

    def verify_all_1024_xor_tree_once(self) -> None:
        with self.idx_lock:
            manifest_path = self.store_dir / "merkle_manifest.json"
            cow_manifest_path = self.store_dir / "cow_manifest.json"
            has_active_cow = (self.num_cow_leaves > 0) and cow_manifest_path.exists()

            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if self.base_model_root.hex() != manifest.get("model_merkle_root"):
                    raise MerkleIntegrityError(
                        f"Header base_model_root mismatch against manifest: "
                        f"{self.base_model_root.hex()} != {manifest.get('model_merkle_root')}"
                    )
                if not has_active_cow and self.expert_tree_root.hex() != manifest.get("expert_tree_root"):
                    raise MerkleIntegrityError(
                        f"Header expert_tree_root mismatch against manifest: "
                        f"{self.expert_tree_root.hex()} != {manifest.get('expert_tree_root')}"
                    )

            if has_active_cow:
                cow_meta = json.loads(cow_manifest_path.read_text())
                if self.expert_tree_root.hex() != cow_meta.get("expert_tree_root"):
                    raise MerkleIntegrityError(
                        f"Header CoW expert_tree_root mismatch: "
                        f"{self.expert_tree_root.hex()} != {cow_meta.get('expert_tree_root')}"
                    )
                if self.active_model_root.hex() != cow_meta.get("active_model_root"):
                    raise MerkleIntegrityError(
                        f"Header CoW active_model_root mismatch: "
                        f"{self.active_model_root.hex()} != {cow_meta.get('active_model_root')}"
                    )

            left_sub = sha256_pair_bytes(self.embedding_root, self.shared_core_root)
            right_sub = sha256_pair_bytes(self.expert_tree_root, self.output_head_root)
            computed_active_root = sha256_pair_bytes(left_sub, right_sub)
            expected_top_root = self.active_model_root if has_active_cow else self.base_model_root
            if computed_active_root != expected_top_root:
                raise MerkleIntegrityError(
                    f"Top-level model_root derivation failure: "
                    f"{computed_active_root.hex()} != {expected_top_root.hex()}"
                )

            for slot in range(1024):
                off_pos = OFFSET_EXPERT_TBL + (slot * 64)
                sha_bytes, byte_off, byte_len, flags, _, _ = _EXPERT_STRUCT.unpack_from(self.idx_mmap, off_pos)
                leaf_off = OFFSET_XOR_TREE + ((1024 + slot) * 32)
                tree_leaf_sha = bytes(self.idx_mmap[leaf_off : leaf_off + 32])
                if sha_bytes != tree_leaf_sha:
                    raise MerkleIntegrityError(
                        f"Expert table slot {slot} hash does not match XOR tree leaf {1024 + slot}!"
                    )
                if (flags & FLAG_IS_COW) == 0 and (byte_off + byte_len > self.cas_size):
                    raise MerkleIntegrityError(
                        f"Expert slot {slot} out-of-bounds offset {byte_off}+{byte_len} > cas_size {self.cas_size}!"
                    )
                if (flags & FLAG_VERIFIED) != 0:
                    self.verified_leaf_slots.add(slot)

            for i in range(1023, 0, -1):
                left_off = OFFSET_XOR_TREE + ((2 * i) * 32)
                right_off = OFFSET_XOR_TREE + ((2 * i + 1) * 32)
                parent_off = OFFSET_XOR_TREE + (i * 32)
                expected_parent = bytes(self.idx_mmap[parent_off : parent_off + 32])
                actual_parent = sha256_pair_bytes(
                    bytes(self.idx_mmap[left_off : left_off + 32]),
                    bytes(self.idx_mmap[right_off : right_off + 32]),
                )
                if actual_parent != expected_parent:
                    raise MerkleIntegrityError(f"Flat XOR Merkle Tree integrity failure at internal node {i}!")

            root_node = bytes(self.idx_mmap[OFFSET_XOR_TREE + 32 : OFFSET_XOR_TREE + 64])
            if root_node != self.expert_tree_root:
                raise MerkleIntegrityError("Flat XOR Merkle Tree Root does not match header expert_tree_root!")
            self.xor_proofs_verified = 1024

    def verify_expert_leaf_bytes(
        self,
        layer_idx: int,
        expert_idx: int,
        force: bool = False,
        preloaded_buf: Optional[Union[bytes, bytearray, memoryview]] = None,
    ) -> bool:
        n_layers = int(self.num_layers)
        n_experts = int(self.num_experts)
        if not (0 <= layer_idx < n_layers and 0 <= expert_idx < n_experts):
            raise MerkleIntegrityError(f"Invalid expert coordinates L{layer_idx}E{expert_idx}!")
        slot = layer_idx * n_experts + expert_idx
        with self.idx_lock:
            if (not force) and (slot in self.verified_leaf_slots):
                return True

            off_pos = OFFSET_EXPERT_TBL + (slot * 64)
            sha_bytes, byte_off, byte_len, flags, access_freq, last_step = _EXPERT_STRUCT.unpack_from(
                self.idx_mmap, off_pos
            )
            if (not force) and ((flags & FLAG_VERIFIED) != 0):
                self.verified_leaf_slots.add(slot)
                return True

            if (flags & FLAG_IS_COW) == 0:
                if byte_off + byte_len > self.cas_size:
                    raise MerkleIntegrityError(
                        f"Out-of-bounds CAS slice for L{layer_idx}E{expert_idx}: {byte_off}+{byte_len} > {self.cas_size}"
                    )
                if preloaded_buf is not None:
                    actual_leaf_sha = hashlib.sha256(preloaded_buf).digest()
                else:
                    mv = memoryview(self.cas_mmap)[byte_off : byte_off + byte_len]
                    actual_leaf_sha = hashlib.sha256(mv).digest()
                    del mv
                if actual_leaf_sha != sha_bytes:
                    _EXPERT_STRUCT.pack_into(
                        self.idx_mmap,
                        off_pos,
                        sha_bytes,
                        byte_off,
                        byte_len,
                        flags & ~FLAG_VERIFIED,
                        access_freq,
                        last_step,
                    )
                    raise MerkleIntegrityError(
                        f"[ERR_MERKLE_CORRUPT] Tier-2 12 MiB byte SHA-256 mismatch on L{layer_idx}E{expert_idx}: "
                        f"expected {sha_bytes.hex()}, got {actual_leaf_sha.hex()}"
                    )
                curr_hash = actual_leaf_sha
            else:
                curr_hash = sha_bytes

            node_idx = 1024 + slot
            tree_leaf = bytes(
                self.idx_mmap[OFFSET_XOR_TREE + (node_idx * 32) : OFFSET_XOR_TREE + ((node_idx + 1) * 32)]
            )
            if curr_hash != tree_leaf:
                raise MerkleIntegrityError(f"[ERR_MERKLE_CORRUPT] Leaf hash mismatch at XOR tree node {node_idx}!")

            while node_idx > 1:
                sib_idx = node_idx ^ 1
                sib_off = OFFSET_XOR_TREE + (sib_idx * 32)
                sib_hash = bytes(self.idx_mmap[sib_off : sib_off + 32])
                if (node_idx & 1) == 0:
                    curr_hash = sha256_pair_bytes(curr_hash, sib_hash)
                else:
                    curr_hash = sha256_pair_bytes(sib_hash, curr_hash)
                node_idx >>= 1

            if curr_hash != self.expert_tree_root:
                raise MerkleIntegrityError(
                    f"[ERR_MERKLE_CORRUPT] XOR Merkle proof root mismatch for L{layer_idx}E{expert_idx}!"
                )

            new_flags = int(flags) | FLAG_VERIFIED
            _EXPERT_STRUCT.pack_into(
                self.idx_mmap, off_pos, sha_bytes, byte_off, byte_len, new_flags, access_freq, last_step
            )
            self.verified_leaf_slots.add(slot)
            self.tier2_byte_proofs_verified += 1
            self.xor_proofs_verified += 1
            return True

    def lookup_expert(self, layer_idx: int, expert_idx: int) -> Tuple[str, int, int, bool]:
        slot = layer_idx * int(self.num_experts) + expert_idx
        off_pos = OFFSET_EXPERT_TBL + (slot * 64)
        with self.idx_lock:
            sha_bytes, byte_off, byte_len, flags, _, _ = _EXPERT_STRUCT.unpack_from(self.idx_mmap, off_pos)
        return sha_bytes.hex(), int(byte_off), int(byte_len), bool(flags & FLAG_IS_COW)

    def slice_expert_weights_zero_copy(
        self,
        byte_off: int,
        hidden_size: int = 2048,
        intermediate_size: int = 1024,
        is_cow: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gu_elems = 2 * intermediate_size * hidden_size
        d_elems = hidden_size * intermediate_size
        gu_bytes = gu_elems * 2
        target_mm = self.cow_mmap if (is_cow and self.cow_mmap is not None) else self.cas_mmap
        target_sz = self.cow_size if (is_cow and self.cow_mmap is not None) else self.cas_size
        if byte_off + gu_bytes + (d_elems * 2) > target_sz:
            raise MerkleIntegrityError(
                f"Out-of-bounds expert slice offset={byte_off} exceeds arena_size={target_sz} (is_cow={is_cow})"
            )

        gu_np = np.frombuffer(
            target_mm, dtype=np.float16, count=gu_elems, offset=byte_off
        ).reshape(2 * intermediate_size, hidden_size)
        gate_w = gu_np[:intermediate_size, :]
        up_w = gu_np[intermediate_size:, :]
        down_w = np.frombuffer(
            target_mm, dtype=np.float16, count=d_elems, offset=byte_off + gu_bytes
        ).reshape(hidden_size, intermediate_size)
        return gate_w, up_w, down_w

    def read_expert_into_slot_buffer(
        self,
        cas_fd: int,
        byte_off: int,
        byte_len: int,
        slot_buf: bytearray,
        hidden_size: int = 2048,
        intermediate_size: int = 1024,
        is_cow: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if is_cow and self.cow_file is not None:
            cas_fd = self.cow_file.fileno()  # type: ignore[attr-defined]
            target_sz = self.cow_size
        else:
            target_sz = self.cas_size
        if byte_off + byte_len > target_sz:
            raise MerkleIntegrityError(
                f"Out-of-bounds expert slice offset={byte_off} exceeds arena_size={target_sz} (is_cow={is_cow})"
            )
        preadv_fn = getattr(os, "preadv", None)
        if preadv_fn is not None:
            nread = preadv_fn(cas_fd, [slot_buf], byte_off)
            if nread != byte_len:
                raise MerkleIntegrityError(f"Short preadv on cas_experts.bin: expected {byte_len}, got {nread}")
        else:
            raw = os.pread(cas_fd, byte_len, byte_off)
            if len(raw) != byte_len:
                raise MerkleIntegrityError(f"Short pread on cas_experts.bin: expected {byte_len}, got {len(raw)}")
            slot_buf[:] = raw

        gu_elems = 2 * intermediate_size * hidden_size
        d_elems = hidden_size * intermediate_size
        gu_bytes = gu_elems * 2
        gu_np = np.frombuffer(slot_buf, dtype=np.float16, count=gu_elems, offset=0).reshape(
            2 * intermediate_size, hidden_size
        )
        gate_w = gu_np[:intermediate_size, :]
        up_w = gu_np[intermediate_size:, :]
        down_w = np.frombuffer(slot_buf, dtype=np.float16, count=d_elems, offset=gu_bytes).reshape(
            hidden_size, intermediate_size
        )
        return gate_w, up_w, down_w

    def prefetch_cold_expert(
        self, layer_idx: int, expert_idx: int, byte_off: int, byte_len: int, skip_mmap_touch: bool = False
    ) -> None:
        slot = layer_idx * int(self.num_experts) + expert_idx
        with self.idx_lock:
            if slot in self.paged_in_slots:
                return
            self.paged_in_slots.add(slot)
            self.cpp_prefetch_issued += 1
        if not skip_mmap_touch:
            madvise_range(self.cas_mmap, byte_off, byte_len, MADV_WILLNEED)

    def evict_expert_slot(self, layer_idx: int, expert_idx: int, byte_off: int, byte_len: int) -> bool:
        slot = layer_idx * int(self.num_experts) + expert_idx
        with self.idx_lock:
            self.paged_in_slots.discard(slot)
            released = madvise_range(self.cas_mmap, byte_off, byte_len, MADV_DONTNEED)
            if released:
                self.cpp_evictions_issued += 1
            return released

    def kv_lookup(self, prefix_hash_bytes: bytes) -> bool:
        slot = struct.unpack_from("<I", prefix_hash_bytes, 0)[0] & 1023
        with self.idx_lock:
            for probe in range(64):
                idx = (slot + probe) & 1023
                pos = OFFSET_KV_HASH_TBL + (idx * 80)
                p_hash, _, occupied, _, _ = _KV_SLOT_STRUCT.unpack_from(self.idx_mmap, pos)
                if occupied == 0:
                    return False
                if p_hash == prefix_hash_bytes:
                    return True
        return False

    def kv_insert(
        self, prefix_hash_bytes: bytes, parent_hash_bytes: bytes, token_end: int, timestamp: float
    ) -> Optional[str]:
        slot = struct.unpack_from("<I", prefix_hash_bytes, 0)[0] & 1023
        with self.idx_lock:
            oldest_pos: int = OFFSET_KV_HASH_TBL + (slot * 80)
            oldest_ts: float = float("inf")
            oldest_hash_bytes: bytes = b""
            for probe in range(64):
                idx = (slot + probe) & 1023
                pos = OFFSET_KV_HASH_TBL + (idx * 80)
                p_hash, _, occupied, _, created_at = _KV_SLOT_STRUCT.unpack_from(self.idx_mmap, pos)
                if occupied == 0 or p_hash == prefix_hash_bytes:
                    _KV_SLOT_STRUCT.pack_into(
                        self.idx_mmap, pos, prefix_hash_bytes, parent_hash_bytes, 1, int(token_end), float(timestamp)
                    )
                    return None
                if created_at < oldest_ts:
                    oldest_ts = created_at
                    oldest_pos = pos
                    oldest_hash_bytes = bytes(p_hash)
            _KV_SLOT_STRUCT.pack_into(
                self.idx_mmap, oldest_pos, prefix_hash_bytes, parent_hash_bytes, 1, int(token_end), float(timestamp)
            )
            self.kv_evictions_issued += 1
            return oldest_hash_bytes.hex() if oldest_hash_bytes else None

    def get_stats(self) -> NativeBridgeStats:
        with self.idx_lock:
            return {
                "zero_copy_metal_active": True,
                "page_alignment_verified": self.page_alignment_verified,
                "page_size_bytes": self.page_size,
                "cpp_prefetch_issued": self.cpp_prefetch_issued,
                "cpp_evictions_issued": self.cpp_evictions_issued,
                "kv_evictions_issued": self.kv_evictions_issued,
                "cow_rollbacks_performed": self.cow_rollbacks_performed,
                "xor_merkle_proofs_verified": self.xor_proofs_verified,
                "tier2_byte_proofs_verified": self.tier2_byte_proofs_verified,
                "native_idx_bytes": self.idx_bytes,
            }

    def close(self) -> None:
        with self.idx_lock:
            if getattr(self, "cow_mmap", None) is not None:
                try:
                    self.cow_mmap.close()  # type: ignore[union-attr]
                except Exception:
                    pass
                self.cow_mmap = None
            if getattr(self, "cow_file", None) is not None:
                try:
                    self.cow_file.close()  # type: ignore[union-attr]
                except Exception:
                    pass
                self.cow_file = None
            try:
                self.idx_mmap.close()
            except Exception:
                pass
            try:
                self.idx_file.close()
            except Exception:
                pass


if HAS_MLX_METAL:

    @mx.compile
    def _compiled_olmoe_route_no_norm(router_logits: mx.array) -> Tuple[mx.array, mx.array]:
        routing_weights = mx.softmax(router_logits, axis=-1, precise=True)
        indices = mx.stop_gradient(mx.argpartition(-routing_weights, kth=7, axis=-1)[..., :8])
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        return indices, scores

    @mx.compile
    def _compiled_olmoe_route_norm(router_logits: mx.array) -> Tuple[mx.array, mx.array]:
        routing_weights = mx.softmax(router_logits, axis=-1, precise=True)
        indices = mx.stop_gradient(mx.argpartition(-routing_weights, kth=7, axis=-1)[..., :8])
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        return indices, scores

    @mx.compile
    def _compiled_olmoe_route_expanded_no_norm(
        router_logits: mx.array, router_b: mx.array, gpu_slot_map: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        logits = router_logits + router_b
        routing_weights = mx.softmax(logits, axis=-1, precise=True)
        indices = mx.stop_gradient(mx.argpartition(-routing_weights, kth=7, axis=-1)[..., :8])
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        exec_indices = mx.take(gpu_slot_map, indices)
        return exec_indices, scores, indices

    @mx.compile
    def _compiled_olmoe_route_expanded_norm(
        router_logits: mx.array, router_b: mx.array, gpu_slot_map: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        logits = router_logits + router_b
        routing_weights = mx.softmax(logits, axis=-1, precise=True)
        indices = mx.stop_gradient(mx.argpartition(-routing_weights, kth=7, axis=-1)[..., :8])
        scores = mx.take_along_axis(routing_weights, indices, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        exec_indices = mx.take(gpu_slot_map, indices)
        return exec_indices, scores, indices

    @partial(mx.compile, shapeless=True)
    def _compiled_moe_aggregate(expert_out: mx.array, scores: mx.array) -> mx.array:
        return (expert_out * scores[..., None]).sum(axis=-2)

    @mx.compile
    def _compiled_single_expert_mlp(
        x_tok: mx.array, gate_w: mx.array, up_w: mx.array, down_w: mx.array
    ) -> mx.array:
        return (nn.silu(x_tok @ gate_w.T) * (x_tok @ up_w.T)) @ down_w.T

    @mx.compile
    def _compiled_vectorized_topk_mlp(
        x_tok: mx.array, g_stack: mx.array, u_stack: mx.array, d_stack: mx.array
    ) -> mx.array:
        x_exp = x_tok[:, None, :]
        g_proj = mx.matmul(x_exp, g_stack.swapaxes(-1, -2))
        u_proj = mx.matmul(x_exp, u_stack.swapaxes(-1, -2))
        act = nn.silu(g_proj) * u_proj
        out = mx.matmul(act, d_stack.swapaxes(-1, -2))
        return out.squeeze(1)

    class SynapCoreOnDemandMoEMLP(nn.Module):
        """
        Task P0.2 & P1.4: Strict On-Demand Top-K Expert Streaming MLP with 3-Stage Async Prefetch Overlap
        and Vectorized Top-8 Mini-SwitchGLU Execution (1 fused Metal kernel per token instead of 8).
        """

        def __init__(self, engine: NativeMerkleMoEEngine, layer_idx: int):
            super().__init__()
            self.engine: NativeMerkleMoEEngine = engine
            self.layer_idx: int = layer_idx

        def __call__(self, x_flat: mx.array, indices: mx.array) -> mx.array:
            mx.eval(indices)
            ind_np = np.array(indices, dtype=np.int32)
            self.engine.live_routing_buffer.record_numpy_layer(self.layer_idx, ind_np)
            num_tokens, top_k = ind_np.shape
            token_outputs: List[mx.array] = []

            if self.engine.enable_lookahead and self.layer_idx + 1 < self.engine.num_layers:
                self.engine.async_pipeline.submit_stage3_prefetch(
                    self.layer_idx + 1, [int(x) for x in ind_np[-1]]
                )

            for t_idx in range(num_tokens):
                x_tok = x_flat[t_idx : t_idx + 1]
                g_list: List[mx.array] = []
                u_list: List[mx.array] = []
                d_list: List[mx.array] = []
                for k_idx in range(top_k):
                    e_id = int(ind_np[t_idx, k_idx])
                    entry = self.engine.load_or_get_expert_entry(self.layer_idx, e_id)
                    g_list.append(entry.mlx_gate_w if entry.mlx_gate_w is not None else mx.array(entry.gate_w))
                    u_list.append(entry.mlx_up_w if entry.mlx_up_w is not None else mx.array(entry.up_w))
                    d_list.append(entry.mlx_down_w if entry.mlx_down_w is not None else mx.array(entry.down_w))

                t_m0 = time.perf_counter() if self.engine.profiler.enabled else 0.0
                g_stack = mx.stack(g_list, axis=0)
                u_stack = mx.stack(u_list, axis=0)
                d_stack = mx.stack(d_list, axis=0)
                out_k = _compiled_vectorized_topk_mlp(x_tok, g_stack, u_stack, d_stack)
                if self.engine.profiler.enabled:
                    mx.eval(out_k)
                    self.engine.profiler.record_stage(
                        "Metal Execution", (time.perf_counter() - t_m0) * 1000.0
                    )
                token_outputs.append(out_k)

            return mx.stack(token_outputs, axis=0)

    class SynapCoreHybridExpandedMoEMLP(nn.Module):
        """
        Fast Hybrid 256-Expert MLP: executes Base Experts (0..63) via pinned Metal SwitchGLU
        and streams Expanded Domain Experts (64..255) zero-copy from `cas_expanded_experts.bin`!
        """

        def __init__(self, engine: NativeMerkleMoEEngine, layer_idx: int, base_switch_mlp: SwitchGLU):
            super().__init__()
            self.engine: NativeMerkleMoEEngine = engine
            self.layer_idx: int = layer_idx
            self.base_switch_mlp: SwitchGLU = base_switch_mlp

        def __call__(self, x_flat: mx.array, indices: mx.array) -> mx.array:
            mx.eval(indices)
            ind_np = np.array(indices, dtype=np.int32)
            self.engine.live_routing_buffer.record_numpy_layer(self.layer_idx, ind_np)
            if self.engine.enable_expanded_experts and self.engine.expanded_bank is not None:
                self.engine.expanded_bank.record_routed_experts(ind_np)  # type: ignore[attr-defined]

            base_limit = self.engine.base_num_experts
            if int(np.max(ind_np)) < base_limit:
                return self.base_switch_mlp(x_flat, indices)

            safe_ind_np = np.where(ind_np < base_limit, ind_np, 0).astype(np.int32)
            base_out = self.base_switch_mlp(x_flat, mx.array(safe_ind_np))
            mx.eval(base_out)
            out_np = np.array(base_out, dtype=np.float16)
            x_np = np.array(x_flat, dtype=np.float32)

            exp_positions = np.argwhere(ind_np >= base_limit)
            for pos in exp_positions:
                t_idx = int(pos[0])
                k_idx = int(pos[1])
                e_id = int(ind_np[t_idx, k_idx])
                entry = self.engine.load_or_get_expert_entry(self.layer_idx, e_id)
                x_tok = x_np[t_idx]
                g_proj = x_tok @ entry.gate_w.astype(np.float32).T
                u_proj = x_tok @ entry.up_w.astype(np.float32).T
                act = (g_proj / (1.0 + np.exp(-g_proj))) * u_proj
                out_np[t_idx, k_idx] = (act @ entry.down_w.astype(np.float32).T).astype(np.float16)

            return mx.array(out_np)

    class SynapCoreMLXAttention(nn.Module):
        """
        Task P1.3: Fused-QKV Apple MLX Attention Layer supporting MHA, GQA, MQA, and optional QK-Norm.
        """

        def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            rms_eps: float,
            rope_theta: float,
            qkv_w: mx.array,
            o_w: mx.array,
            q_norm_w: Optional[mx.array] = None,
            k_norm_w: Optional[mx.array] = None,
            has_qk_norm: bool = True,
        ):
            super().__init__()
            self.n_heads: int = num_heads
            self.n_kv_heads: int = num_kv_heads
            self.head_dim: int = hidden_size // num_heads
            self.q_dim: int = self.n_heads * self.head_dim
            self.kv_dim: int = self.n_kv_heads * self.head_dim
            self.scale: float = self.head_dim**-0.5
            self.has_qk_norm: bool = bool(has_qk_norm and q_norm_w is not None and k_norm_w is not None)

            self.qkv_w: mx.array = mx.contiguous(qkv_w)
            self.o_w: mx.array = mx.contiguous(o_w)

            self.q_norm: Optional[nn.RMSNorm] = None
            self.k_norm: Optional[nn.RMSNorm] = None
            if self.has_qk_norm and q_norm_w is not None and k_norm_w is not None:
                self.q_norm = nn.RMSNorm(int(q_norm_w.shape[-1]), eps=rms_eps)
                self.q_norm.weight = q_norm_w
                self.k_norm = nn.RMSNorm(int(k_norm_w.shape[-1]), eps=rms_eps)
                self.k_norm.weight = k_norm_w

            self.rope = initialize_rope(
                self.head_dim,
                rope_theta,
                traditional=False,
                scaling_config=None,
                max_position_embeddings=4096,
            )

        def __call__(
            self,
            x: mx.array,
            mask: Optional[Union[mx.array, str]] = None,
            cache: Optional[KVCache] = None,
        ) -> mx.array:
            B, L, _ = x.shape
            qkv = x @ self.qkv_w.T
            if self.q_dim == self.kv_dim:
                queries, keys, values = mx.split(qkv, 3, axis=-1)
            else:
                queries, keys, values = mx.split(
                    qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1
                )
            if self.has_qk_norm and self.q_norm is not None and self.k_norm is not None:
                queries = self.q_norm(queries)
                keys = self.k_norm(keys)

            queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return output @ self.o_w.T

    class SynapCoreMLXLayer(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            rms_eps: float,
            norm_topk_prob: bool,
            in_norm_w: mx.array,
            post_norm_w: mx.array,
            router_w: mx.array,
            attn: SynapCoreMLXAttention,
            switch_mlp: Union[SwitchGLU, SynapCoreOnDemandMoEMLP],
            router_b: Optional[mx.array] = None,
            gpu_slot_map: Optional[mx.array] = None,
            layer_idx: int = 0,
            thalamic_v2: Optional[object] = None,
        ):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
            self.input_layernorm.weight = in_norm_w
            self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
            self.post_attention_layernorm.weight = post_norm_w
            self.self_attn = attn
            self.router_w: mx.array = mx.contiguous(router_w)
            self.router_b: Optional[mx.array] = router_b
            self.gpu_slot_map: Optional[mx.array] = gpu_slot_map
            self.layer_idx: int = int(layer_idx)
            self.norm_topk_prob: bool = bool(norm_topk_prob)
            self.thalamic_v2: Optional[object] = thalamic_v2
            self.switch_mlp = switch_mlp
            self.route_fn = _compiled_olmoe_route_norm if norm_topk_prob else _compiled_olmoe_route_no_norm
            self.route_expanded_fn = (
                _compiled_olmoe_route_expanded_norm
                if norm_topk_prob
                else _compiled_olmoe_route_expanded_no_norm
            )

        def __call__(
            self,
            x: mx.array,
            mask: Optional[Union[mx.array, str]] = None,
            cache: Optional[KVCache] = None,
        ) -> mx.array:
            x = x + self.self_attn(self.input_layernorm(x), mask, cache)
            normed_post = self.post_attention_layernorm(x)
            B, L, D = normed_post.shape
            x_flat = normed_post.reshape(-1, D)
            if self.thalamic_v2 is not None:
                indices, scores = self.thalamic_v2.route_layer_mlx(  # type: ignore[attr-defined]
                    self.layer_idx, x_flat, self.router_w, self.router_b, norm_topk_prob=self.norm_topk_prob
                )
                exec_indices = mx.take(self.gpu_slot_map, indices) if self.gpu_slot_map is not None else indices
            elif self.router_b is not None and self.gpu_slot_map is not None:
                router_logits = x_flat @ self.router_w.T
                exec_indices, scores, _ = self.route_expanded_fn(
                    router_logits, self.router_b, self.gpu_slot_map
                )
            else:
                router_logits = x_flat @ self.router_w.T
                if self.router_b is not None:
                    router_logits = router_logits + self.router_b
                indices, scores = self.route_fn(router_logits)
                exec_indices = mx.take(self.gpu_slot_map, indices) if self.gpu_slot_map is not None else indices
            y = self.switch_mlp(x_flat, exec_indices)
            y = _compiled_moe_aggregate(y, scores)
            return x + y.reshape(B, L, D)

        def forward_with_trace(
            self,
            x: mx.array,
            mask: Optional[Union[mx.array, str]] = None,
            cache: Optional[KVCache] = None,
        ) -> Tuple[mx.array, mx.array]:
            x = x + self.self_attn(self.input_layernorm(x), mask, cache)
            normed_post = self.post_attention_layernorm(x)
            B, L, D = normed_post.shape
            x_flat = normed_post.reshape(-1, D)
            if self.thalamic_v2 is not None:
                indices, scores = self.thalamic_v2.route_layer_mlx(  # type: ignore[attr-defined]
                    self.layer_idx, x_flat, self.router_w, self.router_b, norm_topk_prob=self.norm_topk_prob
                )
                exec_indices = mx.take(self.gpu_slot_map, indices) if self.gpu_slot_map is not None else indices
            elif self.router_b is not None and self.gpu_slot_map is not None:
                router_logits = x_flat @ self.router_w.T
                exec_indices, scores, indices = self.route_expanded_fn(
                    router_logits, self.router_b, self.gpu_slot_map
                )
            else:
                router_logits = x_flat @ self.router_w.T
                if self.router_b is not None:
                    router_logits = router_logits + self.router_b
                indices, scores = self.route_fn(router_logits)
                exec_indices = mx.take(self.gpu_slot_map, indices) if self.gpu_slot_map is not None else indices
            y = self.switch_mlp(x_flat, exec_indices)
            y = _compiled_moe_aggregate(y, scores)
            return x + y.reshape(B, L, D), indices


class NativeKVPrefixCache:
    """
    Task P1.2: Paged 16-Token Merkle Prefix KV-Cache.
    """

    BLOCK_SIZE: int = 16

    def __init__(
        self,
        store_dir: Path,
        bridge: NativeMmapIndexBridge,
        device: str = "mlx",
        num_layers: int = 16,
        num_kv_heads: int = 16,
        head_dim: int = 128,
        max_pages: int = 64,
        max_disk_blocks: int = 512,
        quant_mode: str = "fp16",
    ):
        self.store_dir: Path = Path(store_dir)
        self.kv_dir: Path = self.store_dir / "kv_cache_blocks"
        self.kv_dir.mkdir(parents=True, exist_ok=True)
        self.bridge: NativeMmapIndexBridge = bridge
        self.device: str = device
        self.num_layers: int = num_layers
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.max_disk_blocks: int = max(1, int(max_disk_blocks))
        self.quant_mode: str = quant_mode

        self.arena = PagedKVArena(
            max_pages=max_pages,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            quant_mode=quant_mode,
        )
        self.disk_lru: collections.OrderedDict[str, Path] = collections.OrderedDict()
        self.blocks_loaded_from_disk: int = 0
        self.blocks_loaded_from_arena: int = 0
        self.blocks_saved_to_disk: int = 0
        self.disk_blocks_evicted: int = 0
        self.tokens_skipped: int = 0

    def _kv_file_path(self, prefix_hash_hex: str) -> Path:
        suffix = "_mlx_int8.safetensors" if self.quant_mode == "int8" else "_mlx.safetensors"
        return self.kv_dir / f"{prefix_hash_hex[:32]}{suffix}"

    def _evict_disk_block_by_hash(self, prefix_hash_hex: str) -> None:
        self.arena.evict_prefix_hash(prefix_hash_hex)
        p = self.disk_lru.pop(prefix_hash_hex, None)
        for cand in (
            p,
            self.kv_dir / f"{prefix_hash_hex[:32]}_mlx.safetensors",
            self.kv_dir / f"{prefix_hash_hex[:32]}_mlx_int8.safetensors",
            self.kv_dir / f"{prefix_hash_hex[:32]}_np.npz",
        ):
            if cand is not None and cand.exists():
                try:
                    cand.unlink()
                    self.disk_blocks_evicted += 1
                except OSError:
                    pass

    def _enforce_disk_budget(self, new_hash_hex: str, path: Path) -> None:
        self.disk_lru[new_hash_hex] = path
        self.disk_lru.move_to_end(new_hash_hex)
        while len(self.disk_lru) > self.max_disk_blocks:
            oldest_hash, oldest_path = self.disk_lru.popitem(last=False)
            self.arena.evict_prefix_hash(oldest_hash)
            if oldest_path.exists():
                try:
                    oldest_path.unlink()
                    self.disk_blocks_evicted += 1
                except OSError:
                    pass

    def save_prefix_blocks_numpy(
        self,
        token_ids: List[int],
        k_seq_np: np.ndarray,
        v_seq_np: np.ndarray,
        seed_root_hex: str,
    ) -> int:
        max_cacheable_tokens = max(0, len(token_ids) - 1)
        num_full_blocks = max_cacheable_tokens // self.BLOCK_SIZE
        prev_hash = seed_root_hex
        saved_pages = 0

        for b_idx in range(num_full_blocks):
            start = b_idx * self.BLOCK_SIZE
            end = start + self.BLOCK_SIZE
            block_tokens = token_ids[start:end]
            curr_hash = chain_merkle_prefix_hash(prev_hash, block_tokens)

            k_page = k_seq_np[:, :, :, start:end, :]
            v_page = v_seq_np[:, :, :, start:end, :]
            self.arena.write_page_fp16(curr_hash, k_page, v_page)

            evicted_hash = self.bridge.kv_insert(
                bytes.fromhex(curr_hash),
                bytes.fromhex(prev_hash),
                end,
                time.time(),
            )
            if evicted_hash is not None and evicted_hash != curr_hash:
                self._evict_disk_block_by_hash(evicted_hash)

            npz_path = self.kv_dir / f"{curr_hash[:32]}_np.npz"
            if not npz_path.exists():
                if self.quant_mode == "int8":
                    kq, ks = quantize_kv_int8_numpy(k_page)
                    vq, vs = quantize_kv_int8_numpy(v_page)
                    np.savez(npz_path, kq=kq, ks=ks, vq=vq, vs=vs)
                else:
                    np.savez(npz_path, k=k_page, v=v_page)
                self._enforce_disk_budget(curr_hash, npz_path)
                self.blocks_saved_to_disk += 1

            prev_hash = curr_hash
            saved_pages += 1
        return saved_pages

    def match_longest_prefix_numpy(
        self, token_ids: List[int], seed_root_hex: str
    ) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
        max_cacheable_tokens = max(0, len(token_ids) - 1)
        num_full_blocks = max_cacheable_tokens // self.BLOCK_SIZE
        if num_full_blocks == 0:
            return 0, None, None

        prev_hash = seed_root_hex
        matched_page_ids: List[int] = []

        for b_idx in range(num_full_blocks):
            block_tokens = token_ids[b_idx * self.BLOCK_SIZE : (b_idx + 1) * self.BLOCK_SIZE]
            curr_hash = chain_merkle_prefix_hash(prev_hash, block_tokens)
            pid = self.arena.lookup_page_id(curr_hash)
            if pid is not None:
                matched_page_ids.append(pid)
                self.blocks_loaded_from_arena += 1
                prev_hash = curr_hash
                continue

            found = self.bridge.kv_lookup(bytes.fromhex(curr_hash))
            npz_path = self.kv_dir / f"{curr_hash[:32]}_np.npz"
            if (found or npz_path.exists()) and npz_path.exists():
                with np.load(npz_path) as data:
                    if "kq" in data:
                        pid = self.arena.write_page_quantized(
                            curr_hash, data["kq"], data["ks"], data["vq"], data["vs"]
                        )
                    else:
                        pid = self.arena.write_page_fp16(curr_hash, data["k"], data["v"])
                matched_page_ids.append(pid)
                self.blocks_loaded_from_disk += 1
                prev_hash = curr_hash
            else:
                break

        if not matched_page_ids:
            return 0, None, None

        k_contig, v_contig = self.arena.gather_contiguous_kv_numpy(matched_page_ids)
        matched_tokens = len(matched_page_ids) * self.BLOCK_SIZE
        self.tokens_skipped += matched_tokens
        return matched_tokens, k_contig, v_contig

    def match_longest_prefix_mlx(
        self, token_ids: List[int], seed_root_hex: str, mlx_caches: List["KVCache"]
    ) -> int:
        max_cacheable_tokens = max(0, len(token_ids) - 1)
        num_full_blocks = max_cacheable_tokens // self.BLOCK_SIZE
        if num_full_blocks == 0:
            return 0

        prev_hash = seed_root_hex
        matched_page_ids: List[int] = []

        for b_idx in range(num_full_blocks):
            block_tokens = token_ids[b_idx * self.BLOCK_SIZE : (b_idx + 1) * self.BLOCK_SIZE]
            curr_hash = chain_merkle_prefix_hash(prev_hash, block_tokens)

            pid = self.arena.lookup_page_id(curr_hash)
            if pid is not None:
                matched_page_ids.append(pid)
                self.blocks_loaded_from_arena += 1
                prev_hash = curr_hash
                continue

            curr_bytes = bytes.fromhex(curr_hash)
            found = self.bridge.kv_lookup(curr_bytes)
            kv_path = self._kv_file_path(curr_hash)
            fallback_fp16 = self.kv_dir / f"{curr_hash[:32]}_mlx.safetensors"
            load_path = kv_path if kv_path.exists() else (fallback_fp16 if fallback_fp16.exists() else None)

            if (found or load_path is not None) and load_path is not None:
                arrays = mx.load(str(load_path))
                if "kq_0" in arrays:
                    kq_stack = np.stack([np.array(arrays[f"kq_{l}"], dtype=np.int8) for l in range(self.num_layers)])
                    ks_stack = np.stack([np.array(arrays[f"ks_{l}"], dtype=np.float16) for l in range(self.num_layers)])
                    vq_stack = np.stack([np.array(arrays[f"vq_{l}"], dtype=np.int8) for l in range(self.num_layers)])
                    vs_stack = np.stack([np.array(arrays[f"vs_{l}"], dtype=np.float16) for l in range(self.num_layers)])
                    pid = self.arena.write_page_quantized(curr_hash, kq_stack, ks_stack, vq_stack, vs_stack)
                else:
                    k_stack = np.stack([np.array(arrays[f"k_{l}"], dtype=np.float16) for l in range(self.num_layers)])
                    v_stack = np.stack([np.array(arrays[f"v_{l}"], dtype=np.float16) for l in range(self.num_layers)])
                    pid = self.arena.write_page_fp16(curr_hash, k_stack, v_stack)
                matched_page_ids.append(pid)
                self.blocks_loaded_from_disk += 1
                prev_hash = curr_hash
            else:
                break

        if not matched_page_ids:
            return 0

        k_contig_np, v_contig_np = self.arena.gather_contiguous_kv_numpy(matched_page_ids)
        for l_idx in range(self.num_layers):
            mlx_caches[l_idx].update_and_fetch(
                mx.array(k_contig_np[l_idx]), mx.array(v_contig_np[l_idx])
            )

        mx.eval([c.state for c in mlx_caches])
        matched_tokens = len(matched_page_ids) * self.BLOCK_SIZE
        self.tokens_skipped += matched_tokens
        return matched_tokens

    def save_prefix_blocks_mlx(
        self, token_ids: List[int], mlx_caches: List["KVCache"], seed_root_hex: str
    ) -> None:
        max_cacheable_tokens = max(0, len(token_ids) - 1)
        num_full_blocks = max_cacheable_tokens // self.BLOCK_SIZE
        prev_hash = seed_root_hex

        for b_idx in range(num_full_blocks):
            start = b_idx * self.BLOCK_SIZE
            end = start + self.BLOCK_SIZE
            block_tokens = token_ids[start:end]
            curr_hash = chain_merkle_prefix_hash(prev_hash, block_tokens)
            kv_path = self._kv_file_path(curr_hash)

            if curr_hash not in self.arena.page_table or not kv_path.exists():
                k_layers_np: List[np.ndarray] = []
                v_layers_np: List[np.ndarray] = []
                for l_idx in range(self.num_layers):
                    k_slice = mx.contiguous(mlx_caches[l_idx].keys[..., start:end, :]).astype(mx.float16)
                    v_slice = mx.contiguous(mlx_caches[l_idx].values[..., start:end, :]).astype(mx.float16)
                    mx.eval(k_slice, v_slice)
                    k_layers_np.append(np.array(k_slice, dtype=np.float16))
                    v_layers_np.append(np.array(v_slice, dtype=np.float16))

                k_page_np = np.stack(k_layers_np, axis=0)
                v_page_np = np.stack(v_layers_np, axis=0)
                pid = self.arena.write_page_fp16(curr_hash, k_page_np, v_page_np)

                if not kv_path.exists():
                    arrays: Dict[str, mx.array] = {}
                    if self.quant_mode == "int8":
                        assert (
                            self.arena.k_pool_q is not None
                            and self.arena.k_scale is not None
                            and self.arena.v_pool_q is not None
                            and self.arena.v_scale is not None
                        )
                        for l_idx in range(self.num_layers):
                            arrays[f"kq_{l_idx}"] = mx.array(self.arena.k_pool_q[pid, l_idx])
                            arrays[f"ks_{l_idx}"] = mx.array(self.arena.k_scale[pid, l_idx])
                            arrays[f"vq_{l_idx}"] = mx.array(self.arena.v_pool_q[pid, l_idx])
                            arrays[f"vs_{l_idx}"] = mx.array(self.arena.v_scale[pid, l_idx])
                    else:
                        for l_idx in range(self.num_layers):
                            arrays[f"k_{l_idx}"] = mx.array(k_page_np[l_idx])
                            arrays[f"v_{l_idx}"] = mx.array(v_page_np[l_idx])
                    mx.eval(arrays)
                    mx.save_safetensors(str(kv_path), arrays)
                    self._enforce_disk_budget(curr_hash, kv_path)
                    self.blocks_saved_to_disk += 1

                evicted_hash = self.bridge.kv_insert(
                    bytes.fromhex(curr_hash),
                    bytes.fromhex(prev_hash),
                    end,
                    time.time(),
                )
                if evicted_hash is not None and evicted_hash != curr_hash:
                    self._evict_disk_block_by_hash(evicted_hash)

            prev_hash = curr_hash

    def get_paged_kv_stats(self) -> PagedKVStats:
        with self.arena.lock:
            alloc_pages = len(self.arena.page_table)
            free_pages = len(self.arena.free_pages)
            arena_alloc_mb = (self.arena.max_pages * self.arena.bytes_per_page) / (1024.0 * 1024.0)
            arena_act_mb = (alloc_pages * self.arena.bytes_per_page) / (1024.0 * 1024.0)
            return {
                "quant_mode": self.arena.quant_mode,
                "page_tokens": self.arena.PAGE_TOKENS,
                "max_pages": self.arena.max_pages,
                "allocated_pages": alloc_pages,
                "free_pages": free_pages,
                "bytes_per_page": self.arena.bytes_per_page,
                "arena_allocated_mb": round(arena_alloc_mb, 2),
                "arena_active_mb": round(arena_act_mb, 2),
                "page_hits": self.arena.page_hits,
                "page_misses": self.arena.page_misses,
                "page_evictions": self.arena.page_evictions,
                "kv_hash_evictions": self.bridge.kv_evictions_issued,
                "disk_blocks_evicted": self.disk_blocks_evicted,
            }


class NativeMerkleMoEEngine:
    """
    Zero-SQLite Native Binary `merkle_tree.idx` + Apple MLX C++/Metal Inference Engine.
    Implements Tasks P0.1, P0.2, P0.3, P0.4, P1.1, P1.2, P1.3, and P1.4.
    """

    def __init__(
        self,
        store_dir: Path,
        expert_cache_size: int = 128,
        max_cache_ram_mb: Optional[float] = None,
        on_demand_streaming: bool = False,
        pinned_layers: int = 0,
        max_ram_budget_mb: Optional[float] = None,
        enable_lookahead: bool = True,
        enable_merkle_kv: bool = True,
        verify_merkle_on_load: bool = True,
        enable_profiler: bool = False,
        live_ring_capacity: int = 256,
        kv_max_pages: int = 64,
        kv_quant_mode: str = "fp16",
        device: str = "auto",
        shared_mlx_model: Optional["MLXBaseModelProtocol"] = None,
        enable_expanded_experts: bool = False,
        router_version: str = "v1",
    ):
        self.store_dir: Path = Path(store_dir)
        self.device: str = resolve_device(device)
        self.is_mps: bool = self.device in ("mlx", "mps")
        self.use_mlx: bool = HAS_MLX_METAL and (device in ("auto", "mps", "mlx"))
        self.on_demand_streaming: bool = on_demand_streaming
        self.pinned_layers: int = max(0, int(pinned_layers))
        self.max_ram_budget_mb: Optional[float] = max_ram_budget_mb
        self.enable_lookahead: bool = enable_lookahead
        self.enable_merkle_kv: bool = enable_merkle_kv
        self.verify_merkle_on_load: bool = verify_merkle_on_load
        self.kv_quant_mode: str = kv_quant_mode
        self.enable_expanded_experts: bool = bool(enable_expanded_experts)
        self.router_version: str = "v2" if str(router_version).lower() == "v2" else "v1"
        self.expanded_bank: Optional[object] = None
        self.thalamic_router_v2: Optional[object] = None
        self.active_domain_profile: Optional[object] = None
        self._last_domain_routing_sig: Optional[Tuple[str, bool, str]] = None

        self.config: Dict[str, Union[int, float, bool, str]] = validate_model_config(self.store_dir / "config.json")
        with open(self.store_dir / "merkle_manifest.json", "r") as f:
            self.manifest: Dict[str, Union[int, str, float]] = json.load(f)

        self.active_merkle_root: str = str(self.manifest["model_merkle_root"])
        self.cow_leaves_loaded: int = 0

        self.num_layers: int = int(self.config["num_hidden_layers"])
        self.base_num_experts: int = int(self.config["num_experts"])
        self.num_experts: int = self.base_num_experts
        self.top_k: int = int(self.config["num_experts_per_tok"])
        self.norm_topk_prob: bool = bool(self.config.get("norm_topk_prob", False))
        self.hidden_size: int = int(self.config["hidden_size"])
        self.intermediate_size: int = int(self.config["intermediate_size"])
        self.num_heads: int = int(self.config["num_attention_heads"])
        self.num_kv_heads: int = int(self.config["num_key_value_heads"])
        self.head_dim: int = self.hidden_size // self.num_heads
        self.rms_eps: float = float(self.config["rms_norm_eps"])
        self.rope_theta: float = float(self.config["rope_theta"])
        self.vocab_size: int = int(self.config["vocab_size"])
        self.expert_leaf_bytes: int = 3 * self.hidden_size * self.intermediate_size * 2
        self.expert_leaf_mb: float = self.expert_leaf_bytes / (1024**2)

        # Task P1.3: Initialize Model Compatibility Adapter & Standalone Modular Attention Engine
        self.adapter: BaseModelAdapter = create_model_adapter(self.config)
        self.attention_engine: ModularAttentionEngine = ModularAttentionEngine(self.adapter.spec)

        # Task P1.1: Live Routing Ring-Buffer & Per-Stage Latency Profiler
        self.live_routing_buffer = LiveRouteRingBuffer(
            capacity=live_ring_capacity, num_layers=self.num_layers, top_k=self.top_k
        )
        self.profiler = SynapCoreStageProfiler(enabled=enable_profiler)
        self.last_parity_max_abs_err: float = 0.0
        self.last_parity_mean_abs_err: float = 0.0

        tok_path = self.store_dir / "tokenizer.json"
        self.tokenizer = (
            Tokenizer.from_file(str(tok_path)) if (Tokenizer is not None and tok_path.exists()) else None
        )

        # 1. Open `cas_experts.bin` mmap + Native `merkle_tree.idx` C-struct index (Zero SQLite!)
        self.cas_file = open(self.store_dir / "cas_experts.bin", "rb")
        self.cas_mmap = mmap.mmap(self.cas_file.fileno(), 0, access=mmap.ACCESS_READ)
        madvise_range(self.cas_mmap, 0, len(self.cas_mmap), MADV_RANDOM)
        try:
            self.cpp = NativeMmapIndexBridge(
                self.store_dir, self.cas_mmap, verify_all_on_init=verify_merkle_on_load
            )
            self.active_merkle_root = self.cpp.active_model_root.hex()
            self.cow_leaves_loaded = int(self.cpp.num_cow_leaves)
        except Exception:
            self.cas_mmap.close()
            self.cas_file.close()
            raise

        # 2. Task P0.3: Zero-Copy `embed_tokens.bin` mmap (`OWNDATA=False`)
        self.emb_file = open(self.store_dir / "embed_tokens.bin", "rb")
        self.emb_mmap = mmap.mmap(self.emb_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.emb_np_view: Optional[np.ndarray] = np.frombuffer(
            self.emb_mmap, dtype=np.float16, count=self.vocab_size * self.hidden_size
        ).reshape(self.vocab_size, self.hidden_size)
        self.token_cache: Dict[int, np.ndarray] = {}
        self.row_bytes: int = self.hidden_size * 2

        # 3. Task P0.3: Zero-Copy `shared_core.bin` mmap (`OWNDATA=False`, 0 `.copy()` calls!)
        self.core_file = open(self.store_dir / "shared_core.bin", "rb")
        self.core_mmap: Optional[mmap.mmap] = mmap.mmap(self.core_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.core_np_views: Dict[str, np.ndarray] = self._build_zero_copy_core_views()

        # 4. Populate Expert Table from `merkle_tree.idx` in O(1) & Init Real `HotExpertLFULRUCache`
        self.expert_cache = HotExpertLFULRUCache(
            max_experts=expert_cache_size,
            expert_leaf_mb=self.expert_leaf_mb,
            max_ram_mb=max_cache_ram_mb,
            cas_mmap=self.cas_mmap,
            bridge=self.cpp,
        )
        self.expert_index: Dict[Tuple[int, int], Tuple[str, int, int, bool]] = {}
        for l_idx in range(self.num_layers):
            for e_idx in range(self.base_num_experts):
                self.expert_index[(l_idx, e_idx)] = self.cpp.lookup_expert(l_idx, e_idx)

        if self.enable_expanded_experts:
            from native.expert_expansion import ExpandedExpertBankManager, detect_prompt_domain_profile
            exp_bank = ExpandedExpertBankManager(self.store_dir)
            if exp_bank.is_built():
                self.expanded_bank = exp_bank
                self.num_experts = exp_bank.total_experts
                assert exp_bank.manifest is not None
                self.active_merkle_root = str(exp_bank.manifest["expanded_model_merkle_root"])
                for (l_i, e_i), meta in exp_bank.leaf_index.items():
                    self.expert_index[(l_i, e_i)] = (
                        str(meta["sha256_hex"]),
                        int(meta["byte_offset"]),
                        int(meta["byte_length"]),
                        False,
                    )
                self.active_domain_profile = detect_prompt_domain_profile("hi", enable_thinking=True)
                self._last_domain_routing_sig = (
                    f"{getattr(self.active_domain_profile, 'primary_domain', '')}:{getattr(self.active_domain_profile, 'thinking_weight', 0.0):.3f}:{getattr(self.active_domain_profile, 'tool_weight', 0.0):.3f}:{getattr(self.active_domain_profile, 'reasoning_weight', 0.0):.3f}",
                    True,
                    "",
                )
                from native.thalamic_router_v2 import ThalamicRouterV2
                self.thalamic_router_v2 = ThalamicRouterV2(self.store_dir, expanded_bank=exp_bank)
            else:
                self.enable_expanded_experts = False

        self.lsh_predictor = LSHLookaheadPredictor(num_layers=self.num_layers)
        self.async_pipeline = AsyncThreeStagePipeline(self, min_confidence_hits=2, top_n_prefetch=6)
        self.kv_prefix_cache = NativeKVPrefixCache(
            self.store_dir,
            self.cpp,
            device=self.device,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_pages=kv_max_pages,
            quant_mode=kv_quant_mode,
        )
        self.ssd_expert_loads: int = 0
        self.ssd_expert_bytes_read: int = 0

        # Zero-Copy `lm_head_int8.bin` mmap views (`OWNDATA=False`)
        w_nbytes = self.vocab_size * self.hidden_size
        self.lm_head_file = open(self.store_dir / "lm_head_int8.bin", "rb")
        self.lm_head_mmap: Optional[mmap.mmap] = mmap.mmap(
            self.lm_head_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.lm_head_w_int8_view: Optional[np.ndarray] = np.frombuffer(
            self.lm_head_mmap, dtype=np.int8, count=w_nbytes
        ).reshape(self.vocab_size, self.hidden_size)
        self.lm_head_s_fp16_view: Optional[np.ndarray] = np.frombuffer(
            self.lm_head_mmap, dtype=np.float16, count=w_nbytes * 0 + self.vocab_size, offset=w_nbytes
        ).reshape(self.vocab_size, 1)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / float(self.head_dim))
        )
        self.rope_inv_freq: np.ndarray = inv_freq

        if self.use_mlx:
            self._init_mlx_backend(shared_mlx_model)

    def set_router_version(self, router_version: str) -> None:
        """Switches dynamically between 'v1' (Linear Co-Routed) and 'v2' (Thalamic 4-Stage Router)."""
        self.router_version = "v2" if str(router_version).lower() == "v2" else "v1"
        if self.thalamic_router_v2 is not None:
            self.thalamic_router_v2.reset_session_state()  # type: ignore[attr-defined]
        if self.use_mlx and HAS_MLX_METAL and hasattr(self, "mlx_layers"):
            active_v2 = self.thalamic_router_v2 if self.router_version == "v2" else None
            for layer_mod in self.mlx_layers:
                layer_mod.thalamic_v2 = active_v2

    def configure_domain_routing(
        self, prompt: str, enable_thinking: bool = True, tools_json: Optional[str] = None
    ) -> None:
        if not self.enable_expanded_experts or self.expanded_bank is None:
            return
        from native.expert_expansion import detect_prompt_domain_profile
        prof = detect_prompt_domain_profile(prompt, enable_thinking=enable_thinking, tools_json=tools_json)
        self.active_domain_profile = prof
        self.expanded_bank.reset_routing_counts()  # type: ignore[attr-defined]
        if self.thalamic_router_v2 is not None:
            self.thalamic_router_v2.reset_session_state()  # type: ignore[attr-defined]
        sig: Tuple[str, bool, str] = (
            f"{getattr(prof, 'primary_domain', '')}:{getattr(prof, 'thinking_weight', 0.0):.3f}:{getattr(prof, 'tool_weight', 0.0):.3f}:{getattr(prof, 'reasoning_weight', 0.0):.3f}",
            bool(enable_thinking),
            str(tools_json or ""),
        )
        if self.use_mlx and HAS_MLX_METAL and hasattr(self, "mlx_layers"):
            active_v2 = self.thalamic_router_v2 if self.router_version == "v2" else None
            if sig == self._last_domain_routing_sig:
                for layer_mod in self.mlx_layers:
                    layer_mod.thalamic_v2 = active_v2
                return
            self._last_domain_routing_sig = sig
            for l_idx, layer_mod in enumerate(self.mlx_layers):
                base_rw = self.core_np_views[f"model.layers.{l_idx}.mlp.gate.weight"]
                full_rw, full_rb = self.expanded_bank.get_expanded_router_for_layer(  # type: ignore[attr-defined]
                    l_idx, base_rw, prof
                )
                layer_mod.router_w = mx.contiguous(mx.array(full_rw))
                layer_mod.router_b = mx.array(full_rb.astype(np.float16))
                layer_mod.thalamic_v2 = active_v2
            mx.eval([l.router_w for l in self.mlx_layers], [l.router_b for l in self.mlx_layers if l.router_b is not None])
        else:
            self._last_domain_routing_sig = sig

    def set_profiler_enabled(self, enabled: bool) -> None:
        self.profiler.enabled = bool(enabled)

    def _build_zero_copy_core_views(self) -> Dict[str, np.ndarray]:
        assert self.core_mmap is not None
        tensor_specs: List[Tuple[str, Tuple[int, ...]]] = []
        for l_idx in range(self.num_layers):
            p = f"model.layers.{l_idx}"
            tensor_specs.extend(
                [
                    (f"{p}.input_layernorm.weight", (self.hidden_size,)),
                    (f"{p}.mlp.gate.weight", (self.base_num_experts, self.hidden_size)),
                    (f"{p}.post_attention_layernorm.weight", (self.hidden_size,)),
                    (f"{p}.self_attn.k_norm.weight", (self.hidden_size,)),
                    (f"{p}.self_attn.k_proj.weight", (self.hidden_size, self.hidden_size)),
                    (f"{p}.self_attn.o_proj.weight", (self.hidden_size, self.hidden_size)),
                    (f"{p}.self_attn.q_norm.weight", (self.hidden_size,)),
                    (f"{p}.self_attn.q_proj.weight", (self.hidden_size, self.hidden_size)),
                    (f"{p}.self_attn.v_proj.weight", (self.hidden_size, self.hidden_size)),
                ]
            )
        tensor_specs.append(("model.norm.weight", (self.hidden_size,)))
        tensor_specs.sort(key=lambda item: item[0])

        views: Dict[str, np.ndarray] = {}
        offset = 0
        for t_name, shape in tensor_specs:
            numel = 1
            for dim in shape:
                numel *= dim
            views[t_name] = np.frombuffer(
                self.core_mmap, dtype=np.float16, count=numel, offset=offset
            ).reshape(shape)
            offset += numel * 2
        return views

    def load_or_get_expert_entry(
        self, layer_idx: int, expert_idx: int, is_prefetch: bool = False
    ) -> ExpertCacheEntry:
        if not (0 <= layer_idx < self.num_layers and 0 <= expert_idx < self.num_experts):
            raise MerkleIntegrityError(f"Invalid expert coordinates L{layer_idx}E{expert_idx}!")

        prof_on = self.profiler.enabled
        t_lu0 = time.perf_counter() if prof_on else 0.0
        sha_hex, byte_off, byte_len, is_cow = self.expert_index[(layer_idx, expert_idx)]
        if prof_on:
            self.profiler.record_stage("Expert Lookup", (time.perf_counter() - t_lu0) * 1000.0)

        t_ca0 = time.perf_counter() if prof_on else 0.0
        if not is_prefetch:
            cached = self.expert_cache.get(sha_hex)
            if cached is not None:
                if prof_on:
                    self.profiler.record_stage(
                        "Expert Loading / Cache", (time.perf_counter() - t_ca0) * 1000.0
                    )
                return cached
        else:
            with self.expert_cache.lock:
                if sha_hex in self.expert_cache.cache:
                    if prof_on:
                        self.profiler.record_stage(
                            "Expert Loading / Cache", (time.perf_counter() - t_ca0) * 1000.0
                        )
                    return self.expert_cache.cache[sha_hex]
        if prof_on:
            self.profiler.record_stage("Expert Loading / Cache", (time.perf_counter() - t_ca0) * 1000.0)

        t_ld0 = time.perf_counter() if prof_on else 0.0
        slot_buf: Optional[bytearray] = None
        mlx_g: Optional[object] = None
        mlx_u: Optional[object] = None
        mlx_d: Optional[object] = None
        if expert_idx >= self.base_num_experts and self.expanded_bank is not None:
            t_mv0 = time.perf_counter() if prof_on else 0.0
            gate_w, up_w, down_w, _ = self.expanded_bank.slice_expanded_expert_zero_copy(  # type: ignore[attr-defined]
                layer_idx, expert_idx, verify=self.verify_merkle_on_load
            )
            if prof_on and self.verify_merkle_on_load:
                self.profiler.record_stage("Merkle Verification", (time.perf_counter() - t_mv0) * 1000.0)
            if self.use_mlx and self.on_demand_streaming and HAS_MLX_METAL:
                mlx_g = mx.array(gate_w)
                mlx_u = mx.array(up_w)
                mlx_d = mx.array(down_w)
        elif self.use_mlx and self.on_demand_streaming and HAS_MLX_METAL:
            if not hasattr(self, "_mlx_pread_staging_buf") or len(self._mlx_pread_staging_buf) != byte_len:
                self._mlx_pread_staging_buf = bytearray(byte_len)
            staging_buf = self._mlx_pread_staging_buf
            gate_w, up_w, down_w = self.cpp.read_expert_into_slot_buffer(
                self.cas_file.fileno(),
                byte_off,
                byte_len,
                staging_buf,
                self.hidden_size,
                self.intermediate_size,
                is_cow=is_cow,
            )
            if self.verify_merkle_on_load and not is_cow:
                t_mv0 = time.perf_counter() if prof_on else 0.0
                self.cpp.verify_expert_leaf_bytes(layer_idx, expert_idx, force=False, preloaded_buf=staging_buf)
                if prof_on:
                    self.profiler.record_stage(
                        "Merkle Verification", (time.perf_counter() - t_mv0) * 1000.0
                    )
            mlx_g = mx.array(gate_w)
            mlx_u = mx.array(up_w)
            mlx_d = mx.array(down_w)
        elif self.on_demand_streaming and self.expert_cache.max_experts > 0:
            slot_buf = self.expert_cache.acquire_slot_buffer(
                byte_len, sha_hex, is_prefetch=is_prefetch, layer_idx=layer_idx
            )
            gate_w, up_w, down_w = self.cpp.read_expert_into_slot_buffer(
                self.cas_file.fileno(),
                byte_off,
                byte_len,
                slot_buf,
                self.hidden_size,
                self.intermediate_size,
                is_cow=is_cow,
            )
            if self.verify_merkle_on_load and not is_cow:
                t_mv0 = time.perf_counter() if prof_on else 0.0
                self.cpp.verify_expert_leaf_bytes(layer_idx, expert_idx, force=False, preloaded_buf=slot_buf)
                if prof_on:
                    self.profiler.record_stage(
                        "Merkle Verification", (time.perf_counter() - t_mv0) * 1000.0
                    )
        else:
            gate_w, up_w, down_w = self.cpp.slice_expert_weights_zero_copy(
                byte_off, self.hidden_size, self.intermediate_size, is_cow=is_cow
            )
            if self.verify_merkle_on_load and not is_cow:
                t_mv0 = time.perf_counter() if prof_on else 0.0
                self.cpp.verify_expert_leaf_bytes(layer_idx, expert_idx, force=False)
                if prof_on:
                    self.profiler.record_stage(
                        "Merkle Verification", (time.perf_counter() - t_mv0) * 1000.0
                    )

        slot_id = layer_idx * self.num_experts + expert_idx
        entry = ExpertCacheEntry(
            sha256_hex=sha_hex,
            slot_id=slot_id,
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            byte_offset=byte_off,
            byte_length=byte_len,
            gate_w=gate_w,
            up_w=up_w,
            down_w=down_w,
            verified=True,
            mlx_gate_w=mlx_g,
            mlx_up_w=mlx_u,
            mlx_down_w=mlx_d,
            slot_buf=slot_buf,
        )
        if not is_prefetch:
            self.ssd_expert_loads += 1
            self.ssd_expert_bytes_read += byte_len

        if is_cow or is_prefetch or self.expert_cache.should_admit(sha_hex, byte_len):
            admitted = self.expert_cache.put(sha_hex, entry, is_prefetch=is_prefetch)
            if not admitted and slot_buf is not None:
                with self.expert_cache.lock:
                    self.expert_cache._free_slot_buffers.append(slot_buf)
        elif slot_buf is not None:
            with self.expert_cache.lock:
                self.expert_cache._free_slot_buffers.append(slot_buf)
        if prof_on:
            self.profiler.record_stage(
                "Expert Loading / Cache", (time.perf_counter() - t_ld0) * 1000.0, calls=0
            )
        return entry

    def forward_moe_layer_numpy(
        self,
        layer_idx: int,
        x_np: np.ndarray,
        router_w_np: Optional[np.ndarray] = None,
        norm_topk_prob: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        prof_on = self.profiler.enabled
        t_rt0 = time.perf_counter() if prof_on else 0.0
        use_norm = self.norm_topk_prob if norm_topk_prob is None else norm_topk_prob
        orig_shape = x_np.shape
        x_flat = x_np.reshape(-1, self.hidden_size).astype(np.float32)
        if (
            router_w_np is None
            and self.enable_expanded_experts
            and self.router_version == "v2"
            and self.thalamic_router_v2 is not None
        ):
            base_rw = self.core_np_views[f"model.layers.{layer_idx}.mlp.gate.weight"]
            topk_indices, topk_scores, _ = self.thalamic_router_v2.route_layer_numpy(  # type: ignore[attr-defined]
                layer_idx,
                x_flat,
                base_rw,
                profile=self.active_domain_profile,
                norm_topk_prob=use_norm,
            )
        else:
            router_b_f32: Optional[np.ndarray] = None
            if router_w_np is None:
                base_rw = self.core_np_views[f"model.layers.{layer_idx}.mlp.gate.weight"]
                if self.enable_expanded_experts and self.expanded_bank is not None:
                    router_w_np, router_b_f32 = self.expanded_bank.get_expanded_router_for_layer(  # type: ignore[attr-defined]
                        layer_idx, base_rw, self.active_domain_profile
                    )
                else:
                    router_w_np = base_rw
            r_w = router_w_np.astype(np.float32)

            logits = x_flat @ r_w.T
            if router_b_f32 is not None:
                logits = logits + router_b_f32[None, :]
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            topk_indices = np.argpartition(-probs, kth=self.top_k - 1, axis=-1)[:, : self.top_k].astype(np.int32)
            topk_scores = np.take_along_axis(probs, topk_indices, axis=-1)
            if use_norm:
                topk_scores = topk_scores / np.sum(topk_scores, axis=-1, keepdims=True)
        if prof_on:
            self.profiler.record_stage("Router", (time.perf_counter() - t_rt0) * 1000.0)

        self.live_routing_buffer.record_numpy_layer(layer_idx, topk_indices)
        if self.enable_expanded_experts and self.expanded_bank is not None:
            self.expanded_bank.record_routed_experts(topk_indices)  # type: ignore[attr-defined]

        out_flat = np.zeros_like(x_flat, dtype=np.float32)
        for t_idx in range(x_flat.shape[0]):
            x_tok = x_flat[t_idx]
            acc = np.zeros(self.hidden_size, dtype=np.float32)
            for k_idx in range(self.top_k):
                e_id = int(topk_indices[t_idx, k_idx])
                score = float(topk_scores[t_idx, k_idx])
                entry = self.load_or_get_expert_entry(layer_idx, e_id, is_prefetch=False)
                t_ex0 = time.perf_counter() if prof_on else 0.0
                g_proj = x_tok @ entry.gate_w.astype(np.float32).T
                u_proj = x_tok @ entry.up_w.astype(np.float32).T
                silu_g = g_proj / (1.0 + np.exp(-g_proj))
                act = silu_g * u_proj
                e_out = act @ entry.down_w.astype(np.float32).T
                acc += score * e_out
                if prof_on:
                    self.profiler.record_stage("Metal Execution", (time.perf_counter() - t_ex0) * 1000.0)
            out_flat[t_idx] = acc

        return out_flat.astype(x_np.dtype).reshape(orig_shape), topk_indices

    def _rmsnorm_numpy(self, x_f32: np.ndarray, weight_fp16: np.ndarray) -> np.ndarray:
        var = np.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        normed = x_f32 * (1.0 / np.sqrt(var + self.rms_eps))
        return normed * weight_fp16.astype(np.float32)

    def _apply_rope_numpy(self, x_bhld: np.ndarray, offset: int) -> np.ndarray:
        seq_len = x_bhld.shape[2]
        positions = np.arange(offset, offset + seq_len, dtype=np.float32)[:, None]
        angles = positions * self.rope_inv_freq[None, :]
        cos = np.cos(angles)[None, None, :, :]
        sin = np.sin(angles)[None, None, :, :]
        half = self.head_dim // 2
        x1 = x_bhld[..., :half]
        x2 = x_bhld[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    def forward_attention_layer_numpy(
        self,
        layer_idx: int,
        norm_in_f32: np.ndarray,
        kv_cache_k: List[Optional[np.ndarray]],
        kv_cache_v: List[Optional[np.ndarray]],
        pos_offset: int = 0,
    ) -> np.ndarray:
        p = f"model.layers.{layer_idx}"
        ap = f"{p}.self_attn"
        q_norm_w = self.core_np_views.get(f"{ap}.q_norm.weight")
        k_norm_w = self.core_np_views.get(f"{ap}.k_norm.weight")
        return self.attention_engine.forward_numpy(
            norm_in_f32,
            self.core_np_views[f"{ap}.q_proj.weight"],
            self.core_np_views[f"{ap}.k_proj.weight"],
            self.core_np_views[f"{ap}.v_proj.weight"],
            self.core_np_views[f"{ap}.o_proj.weight"],
            q_norm_w_fp16=q_norm_w,
            k_norm_w_fp16=k_norm_w,
            kv_cache_k=kv_cache_k,
            kv_cache_v=kv_cache_v,
            layer_idx=layer_idx,
            pos_offset=pos_offset,
        )

    def forward_full_model_numpy(
        self,
        token_ids: List[int],
        kv_cache_k: List[Optional[np.ndarray]],
        kv_cache_v: List[Optional[np.ndarray]],
        pos_offset: int = 0,
    ) -> np.ndarray:
        assert (
            self.emb_np_view is not None
            and self.lm_head_w_int8_view is not None
            and self.lm_head_s_fp16_view is not None
        )
        prof_on = self.profiler.enabled
        h = self.emb_np_view[token_ids].astype(np.float32)[None, :, :]

        for l_idx in range(self.num_layers):
            t_at0 = time.perf_counter() if prof_on else 0.0
            p = f"model.layers.{l_idx}"
            norm_in = self._rmsnorm_numpy(h, self.core_np_views[f"{p}.input_layernorm.weight"])
            attn_proj = self.forward_attention_layer_numpy(
                l_idx, norm_in, kv_cache_k, kv_cache_v, pos_offset=pos_offset
            )
            h = h + attn_proj
            if prof_on:
                self.profiler.record_stage("Attention", (time.perf_counter() - t_at0) * 1000.0)

            norm_post = self._rmsnorm_numpy(h, self.core_np_views[f"{p}.post_attention_layernorm.weight"])
            moe_out, _ = self.forward_moe_layer_numpy(l_idx, norm_post.astype(np.float16))
            h = h + moe_out.astype(np.float32)

        t_sm0 = time.perf_counter() if prof_on else 0.0
        h_last = self._rmsnorm_numpy(h[:, -1, :], self.core_np_views["model.norm.weight"])[0]
        if self.enable_expanded_experts and self.router_version == "v2" and self.thalamic_router_v2 is not None:
            self.thalamic_router_v2.update_topdown_context_numpy(h_last)  # type: ignore[attr-defined]
        if self.enable_expanded_experts and self.expanded_bank is not None:
            h_last = self.expanded_bank.apply_domain_residual_adapter_numpy(  # type: ignore[attr-defined]
                h_last[None, :], self.active_domain_profile
            )[0]
        logits_raw = self.lm_head_w_int8_view.astype(np.float32) @ h_last
        logits = logits_raw * self.lm_head_s_fp16_view[:, 0].astype(np.float32)
        if prof_on:
            self.profiler.record_stage("Sampling", (time.perf_counter() - t_sm0) * 1000.0)
            self.profiler.mark_token_completed(1)
        return logits

    def _init_mlx_backend(self, shared_mlx_model: Optional["MLXBaseModelProtocol"] = None) -> None:
        self.mlx_stream = mx.new_thread_local_stream(mx.default_device())
        try:
            set_lim = getattr(mx, "set_cache_limit", None) or getattr(mx.metal, "set_cache_limit", None)
            if set_lim is not None:
                set_lim(32 * 1024 * 1024)
        except Exception:
            pass

        core_arrays: Dict[str, mx.array] = {
            t_name: mx.array(np_view) for t_name, np_view in self.core_np_views.items()
        }
        mx.eval(core_arrays)

        assert self.emb_np_view is not None
        if shared_mlx_model is not None:
            self.mlx_embed_weight: Optional[mx.array] = shared_mlx_model.model.embed_tokens.weight
            self.mlx_lm_head_w: Optional[mx.array] = shared_mlx_model.lm_head.weight
        else:
            self.mlx_embed_weight = mx.array(self.emb_np_view)
            w_nbytes = self.vocab_size * self.hidden_size
            with open(self.store_dir / "lm_head_int8.bin", "rb") as f:
                lm_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    w_np = np.frombuffer(lm_mmap, dtype=np.int8, count=w_nbytes).reshape(
                        self.vocab_size, self.hidden_size
                    )
                    s_np = np.frombuffer(
                        lm_mmap, dtype=np.float16, count=self.vocab_size, offset=w_nbytes
                    ).reshape(self.vocab_size, 1)
                    self.mlx_lm_head_w = mx.contiguous(mx.array(w_np).astype(mx.float16) * mx.array(s_np))
                    mx.eval(self.mlx_lm_head_w)
                    del w_np, s_np
                finally:
                    lm_mmap.close()

        self.mlx_final_norm: Optional[nn.RMSNorm] = nn.RMSNorm(self.hidden_size, eps=self.rms_eps)
        self.mlx_final_norm.weight = core_arrays["model.norm.weight"]

        self.mlx_layers: List[SynapCoreMLXLayer] = []
        has_any_pinned = (not self.on_demand_streaming) or (self.pinned_layers > 0)
        gate_layer_np: Optional[np.ndarray] = None
        up_layer_np: Optional[np.ndarray] = None
        down_layer_np: Optional[np.ndarray] = None
        # Keep SwitchGLU expert count at exact power-of-2 `base_num_experts = 64` (12,000 MiB)
        # for maximum Apple Metal SwitchGLU `gather_mm` throughput (165+ tok/s) and -2.31% RAM!
        pinned_switch_experts: int = self.base_num_experts
        shared_gpu_slot_map: Optional[mx.array] = None
        if self.num_experts > pinned_switch_experts and self.expanded_bank is not None:
            slot_map_np = (np.arange(self.num_experts, dtype=np.int32) & (self.base_num_experts - 1)).astype(
                np.int32
            )
            shared_gpu_slot_map = mx.array(slot_map_np, dtype=mx.int32)
            mx.eval(shared_gpu_slot_map)

        if has_any_pinned and shared_mlx_model is None:
            gate_layer_np = np.empty(
                (pinned_switch_experts, self.intermediate_size, self.hidden_size), dtype=np.float16
            )
            up_layer_np = np.empty(
                (pinned_switch_experts, self.intermediate_size, self.hidden_size), dtype=np.float16
            )
            down_layer_np = np.empty(
                (pinned_switch_experts, self.hidden_size, self.intermediate_size), dtype=np.float16
            )

        for l_idx in range(self.num_layers):
            p = f"model.layers.{l_idx}"
            ap = f"{p}.self_attn"
            q_w = core_arrays.pop(f"{ap}.q_proj.weight")
            k_w = core_arrays.pop(f"{ap}.k_proj.weight")
            v_w = core_arrays.pop(f"{ap}.v_proj.weight")
            qkv_w = mx.concatenate([q_w, k_w, v_w], axis=0)

            attn = SynapCoreMLXAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                rms_eps=self.rms_eps,
                rope_theta=self.rope_theta,
                qkv_w=qkv_w,
                o_w=core_arrays.pop(f"{ap}.o_proj.weight"),
                q_norm_w=core_arrays.pop(f"{ap}.q_norm.weight"),
                k_norm_w=core_arrays.pop(f"{ap}.k_norm.weight"),
            )

            is_layer_pinned = (not self.on_demand_streaming) or (l_idx < self.pinned_layers)
            if not is_layer_pinned:
                switch_mlp: Union[SwitchGLU, SynapCoreOnDemandMoEMLP] = SynapCoreOnDemandMoEMLP(self, l_idx)
            elif shared_mlx_model is not None and self.cow_leaves_loaded == 0 and not self.enable_expanded_experts:
                switch_mlp = shared_mlx_model.model.layers[l_idx].mlp.switch_mlp
            else:
                if gate_layer_np is None:
                    gate_layer_np = np.empty(
                        (pinned_switch_experts, self.intermediate_size, self.hidden_size), dtype=np.float16
                    )
                    up_layer_np = np.empty(
                        (pinned_switch_experts, self.intermediate_size, self.hidden_size), dtype=np.float16
                    )
                    down_layer_np = np.empty(
                        (pinned_switch_experts, self.hidden_size, self.intermediate_size), dtype=np.float16
                    )
                base_switch = SwitchGLU(self.hidden_size, self.intermediate_size, pinned_switch_experts, bias=False)
                for e_idx in range(pinned_switch_experts):
                    if self.verify_merkle_on_load:
                        self.cpp.verify_expert_leaf_bytes(l_idx, e_idx, force=False)
                    _, byte_off, byte_len, _ = self.expert_index[(l_idx, e_idx)]
                    gate_w, up_w, down_w = self.cpp.slice_expert_weights_zero_copy(
                        byte_off, self.hidden_size, self.intermediate_size
                    )
                    gate_layer_np[e_idx] = gate_w
                    up_layer_np[e_idx] = up_w
                    down_layer_np[e_idx] = down_w
                    madvise_range(self.cas_mmap, byte_off, byte_len, MADV_DONTNEED)
                base_switch.gate_proj.weight = mx.array(gate_layer_np)
                base_switch.up_proj.weight = mx.array(up_layer_np)
                base_switch.down_proj.weight = mx.array(down_layer_np)
                if self.max_ram_budget_mb is not None and self.max_ram_budget_mb < 15800.0:
                    q_bits = 4 if self.max_ram_budget_mb < 8200.0 else 8
                    nn.quantize(base_switch, group_size=64, bits=q_bits)
                mx.eval(base_switch.parameters())
                switch_mlp = base_switch

            in_norm_w = core_arrays.pop(f"{p}.input_layernorm.weight")
            post_norm_w = core_arrays.pop(f"{p}.post_attention_layernorm.weight")
            base_router_arr = core_arrays.pop(f"{p}.mlp.gate.weight")
            router_b_arr: Optional[mx.array] = None
            if self.enable_expanded_experts and self.expanded_bank is not None:
                base_rw_np = self.core_np_views[f"{p}.mlp.gate.weight"]
                full_rw_np, full_rb_np = self.expanded_bank.get_expanded_router_for_layer(  # type: ignore[attr-defined]
                    l_idx, base_rw_np, self.active_domain_profile
                )
                router_w = mx.array(full_rw_np)
                router_b_arr = mx.array(full_rb_np.astype(np.float16))
            else:
                router_w = base_router_arr

            layer_mod = SynapCoreMLXLayer(
                hidden_size=self.hidden_size,
                rms_eps=self.rms_eps,
                norm_topk_prob=self.norm_topk_prob,
                in_norm_w=in_norm_w,
                post_norm_w=post_norm_w,
                router_w=router_w,
                attn=attn,
                switch_mlp=switch_mlp,
                router_b=router_b_arr,
                gpu_slot_map=shared_gpu_slot_map if is_layer_pinned else None,
                layer_idx=l_idx,
                thalamic_v2=self.thalamic_router_v2 if self.router_version == "v2" else None,
            )
            self.mlx_layers.append(layer_mod)

        del gate_layer_np, up_layer_np, down_layer_np
        gc.collect()
        mx.eval([l.parameters() for l in self.mlx_layers], self.mlx_final_norm.parameters(), self.mlx_lm_head_w)
        if not self.on_demand_streaming:
            self._warmup_mlx_kernels()
        else:
            mx.clear_cache()

    def _warmup_mlx_kernels(self) -> None:
        with mx.stream(self.mlx_stream):
            dummy_cache = [KVCache() for _ in range(self.num_layers)]
            logits = self._mlx_forward_fast(
                mx.array([[1, 2]], dtype=mx.int32), dummy_cache, record_live_routing=True
            )
            mx.eval(logits)
            logits2 = self._mlx_forward_fast(
                mx.array([[3]], dtype=mx.int32), dummy_cache, record_live_routing=False
            )
            mx.eval(logits2)
        self.live_routing_buffer.reset()
        if self.thalamic_router_v2 is not None:
            self.thalamic_router_v2.reset_session_state()  # type: ignore[attr-defined]
        mx.clear_cache()

    def _mlx_forward_fast(
        self,
        inputs: mx.array,
        cache: List["KVCache"],
        record_live_routing: bool = True,
    ) -> mx.array:
        assert self.mlx_embed_weight is not None and self.mlx_final_norm is not None and self.mlx_lm_head_w is not None
        h = self.mlx_embed_weight[inputs]
        mask = create_attention_mask(h, cache[0])
        if record_live_routing and not self.on_demand_streaming:
            step_indices: List[mx.array] = []
            for layer, c in zip(self.mlx_layers, cache):
                h, inds = layer.forward_with_trace(h, mask, c)
                step_indices.append(inds)
            self.live_routing_buffer.record_mlx_step(step_indices)
        else:
            for layer, c in zip(self.mlx_layers, cache):
                h = layer(h, mask, c)
        h_last = self.mlx_final_norm(h[:, -1:, :])
        if record_live_routing and self.router_version == "v2" and self.thalamic_router_v2 is not None:
            self.thalamic_router_v2.update_topdown_context_mlx(h_last)  # type: ignore[attr-defined]
        return (h_last @ self.mlx_lm_head_w.T).squeeze(1)

    def profile_token_step_mlx(
        self, tok_1d: mx.array, cache: List["KVCache"]
    ) -> mx.array:
        assert self.mlx_embed_weight is not None and self.mlx_final_norm is not None and self.mlx_lm_head_w is not None
        prev_prof = self.profiler.enabled
        self.profiler.enabled = True
        try:
            with mx.stream(self.mlx_stream):
                inputs = tok_1d[None]
                h = self.mlx_embed_weight[inputs]
                mask = create_attention_mask(h, cache[0])
                mx.eval(h)

                step_indices: List[mx.array] = []
                for l_idx, (layer, c) in enumerate(zip(self.mlx_layers, cache)):
                    t_at0 = time.perf_counter()
                    attn_out = layer.self_attn(layer.input_layernorm(h), mask, c)
                    h = h + attn_out
                    normed_post = layer.post_attention_layernorm(h)
                    mx.eval(normed_post)
                    self.profiler.record_stage("Attention", (time.perf_counter() - t_at0) * 1000.0)

                    t_rt0 = time.perf_counter()
                    B, L, D = normed_post.shape
                    x_flat = normed_post.reshape(-1, D)
                    router_logits = x_flat @ layer.router_w.T
                    if layer.router_b is not None:
                        router_logits = router_logits + layer.router_b
                    indices, scores = layer.route_fn(router_logits)
                    mx.eval(indices, scores)
                    self.profiler.record_stage("Router", (time.perf_counter() - t_rt0) * 1000.0)
                    step_indices.append(indices)

                    if not self.on_demand_streaming:
                        ind_np = np.array(indices, dtype=np.int32)
                        for e_id in ind_np.reshape(-1):
                            self.load_or_get_expert_entry(l_idx, int(e_id), is_prefetch=False)

                    t_me0 = time.perf_counter()
                    exec_ind = mx.take(layer.gpu_slot_map, indices) if layer.gpu_slot_map is not None else indices
                    y_mlp = layer.switch_mlp(x_flat, exec_ind)
                    y_agg = _compiled_moe_aggregate(y_mlp, scores)
                    h = h + y_agg.reshape(B, L, D)
                    mx.eval(h)
                    if not self.on_demand_streaming:
                        self.profiler.record_stage("Metal Execution", (time.perf_counter() - t_me0) * 1000.0)

                if not self.on_demand_streaming:
                    self.live_routing_buffer.record_mlx_step(step_indices)

                t_sm0 = time.perf_counter()
                h_last = self.mlx_final_norm(h[:, -1:, :])
                logits = (h_last @ self.mlx_lm_head_w.T).squeeze(1)
                next_tok = mx.argmax(logits, axis=-1)
                mx.eval(next_tok)
                self.profiler.record_stage("Sampling", (time.perf_counter() - t_sm0) * 1000.0)
                self.profiler.mark_token_completed(1)
                return next_tok
        finally:
            self.profiler.enabled = prev_prof

    def flush_live_routing_telemetry(
        self, prompt_ids: List[int], generated_ids: List[int]
    ) -> np.ndarray:
        self.async_pipeline.wait_pending_prefetches()
        routed_trace_np = self.live_routing_buffer.flush()
        self._record_batched_telemetry(prompt_ids, generated_ids, routed_trace_np)
        return routed_trace_np

    def _mlx_collect_routing_trace(self, full_token_ids: List[int]) -> np.ndarray:
        live_trace = self.live_routing_buffer.flush()
        if live_trace.shape[0] > 0:
            return live_trace
        assert self.mlx_embed_weight is not None
        self.live_routing_buffer.duplicate_forward_passes += 1
        with mx.stream(self.mlx_stream):
            seq_arr = mx.array([full_token_ids], dtype=mx.int32)
            h = self.mlx_embed_weight[seq_arr]
            mask = create_attention_mask(h, None)
            layer_indices = []
            for layer in self.mlx_layers:
                h, inds = layer.forward_with_trace(h, mask, None)
                layer_indices.append(inds)
            stacked = mx.stack(layer_indices, axis=1)
            mx.eval(stacked)
            return np.array(stacked, dtype=np.int32)

    def _record_batched_telemetry(
        self, prompt_ids: List[int], generated_ids: List[int], all_routed_np: np.ndarray
    ) -> None:
        assert self.emb_np_view is not None
        for tid in prompt_ids:
            tid_i = int(tid)
            if tid_i not in self.token_cache:
                self.token_cache[tid_i] = self.emb_np_view[tid_i]
        for tid in generated_ids:
            tid_i = int(tid)
            if tid_i not in self.token_cache:
                self.token_cache[tid_i] = self.emb_np_view[tid_i]

        if all_routed_np.size == 0:
            return

        if self.enable_expanded_experts and self.expanded_bank is not None:
            self.expanded_bank.record_routed_experts(all_routed_np)  # type: ignore[attr-defined]
        if self.enable_expanded_experts and self.router_version == "v2" and self.thalamic_router_v2 is not None:
            self.thalamic_router_v2.record_post_generation_trace(all_routed_np)  # type: ignore[attr-defined]
        num_steps, num_layers, _ = all_routed_np.shape
        for s_idx in range(num_steps):
            for l_idx in range(num_layers):
                actual_exps = [int(x) for x in all_routed_np[s_idx, l_idx]]
                is_pinned_mlx = self.use_mlx and (
                    (not self.on_demand_streaming) or (l_idx < self.pinned_layers)
                )
                if l_idx > 0:
                    prev_exps = [int(x) for x in all_routed_np[s_idx, l_idx - 1]]
                    pred = self.lsh_predictor.record_layer_transition(
                        l_idx, prev_exps, actual_exps, top_n=6
                    )
                    if self.enable_lookahead:
                        for p_exp in pred:
                            sha_hex, byte_off, byte_len, is_cow = self.expert_index[(l_idx, p_exp)]
                            if (not is_cow) and (p_exp < self.base_num_experts) and (sha_hex not in self.expert_cache.cache):
                                self.cpp.prefetch_cold_expert(
                                    l_idx,
                                    p_exp,
                                    byte_off,
                                    byte_len,
                                    skip_mmap_touch=self.on_demand_streaming or is_pinned_mlx,
                                )

                if is_pinned_mlx:
                    with self.expert_cache.lock:
                        self.expert_cache.hits += len(actual_exps)
                elif not self.on_demand_streaming and not self.profiler.enabled:
                    for e_id in actual_exps:
                        self.load_or_get_expert_entry(l_idx, e_id, is_prefetch=False)

    def verify_numerical_parity(
        self, layer_idx: int = 0, num_tokens: int = 4, seed: int = 42
    ) -> Tuple[float, float]:
        rng = np.random.default_rng(seed)
        x_np = rng.standard_normal((1, num_tokens, self.hidden_size)).astype(np.float16)

        stream_out, topk_ind = self.forward_moe_layer_numpy(layer_idx, x_np)
        router_w = self.core_np_views[f"model.layers.{layer_idx}.mlp.gate.weight"].astype(np.float32)
        x_flat = x_np.reshape(-1, self.hidden_size).astype(np.float32)
        logits = x_flat @ router_w.T
        logits_max = np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(logits - logits_max)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        topk_scores = np.take_along_axis(probs, topk_ind, axis=-1)
        if self.norm_topk_prob:
            topk_scores = topk_scores / np.sum(topk_scores, axis=-1, keepdims=True)

        ref_flat = np.zeros_like(x_flat, dtype=np.float32)
        for t_idx in range(x_flat.shape[0]):
            x_tok = x_flat[t_idx]
            acc = np.zeros(self.hidden_size, dtype=np.float32)
            for k_idx in range(self.top_k):
                e_id = int(topk_ind[t_idx, k_idx])
                score = float(topk_scores[t_idx, k_idx])
                _, byte_off, _, _ = self.expert_index[(layer_idx, e_id)]
                g_w, u_w, d_w = self.cpp.slice_expert_weights_zero_copy(
                    byte_off, self.hidden_size, self.intermediate_size
                )
                g_proj = x_tok @ g_w.astype(np.float32).T
                u_proj = x_tok @ u_w.astype(np.float32).T
                act = (g_proj / (1.0 + np.exp(-g_proj))) * u_proj
                acc += score * (act @ d_w.astype(np.float32).T)
            ref_flat[t_idx] = acc

        ref_out = ref_flat.astype(np.float16).reshape(stream_out.shape)
        diff = np.abs(stream_out.astype(np.float32) - ref_out.astype(np.float32))
        self.last_parity_max_abs_err = float(np.max(diff))
        self.last_parity_mean_abs_err = float(np.mean(diff))
        return self.last_parity_max_abs_err, self.last_parity_mean_abs_err

    def get_hardware_diagnostics(self) -> HardwareDiagnostics:
        rss_mb = get_process_rss_mb()
        act_mb, peak_mb, cache_mb = get_mlx_memory_metrics_mb()
        total_reqs = self.expert_cache.hits + self.expert_cache.misses
        hit_rate = (100.0 * self.expert_cache.hits / total_reqs) if total_reqs > 0 else 0.0
        la_preds = self.lsh_predictor.lookahead_predictions
        la_rate = (
            100.0 * self.lsh_predictor.lookahead_correct_experts / la_preds if la_preds > 0 else 0.0
        )
        return {
            "ru_maxrss_mb": round(rss_mb, 2),
            "mlx_active_memory_mb": round(act_mb, 2),
            "mlx_peak_memory_mb": round(peak_mb, 2),
            "mlx_cache_memory_mb": round(cache_mb, 2),
            "hot_cache_ram_mb": round(self.expert_cache.memory_mb(), 2),
            "max_cache_ram_mb": round(self.expert_cache.max_ram_mb, 2),
            "cache_hits": self.expert_cache.hits,
            "cache_misses": self.expert_cache.misses,
            "cache_evictions": self.expert_cache.evictions,
            "madvise_releases": self.expert_cache.madvise_releases,
            "cache_hit_rate_pct": round(hit_rate, 2),
            "lookahead_hit_rate_pct": round(la_rate, 2),
            "live_ring_buffer_tokens": self.live_routing_buffer.total_tokens_recorded,
            "duplicate_forward_passes": self.live_routing_buffer.duplicate_forward_passes,
            "numerical_parity_max_abs_err": round(self.last_parity_max_abs_err, 8),
            "numerical_parity_mean_abs_err": round(self.last_parity_mean_abs_err, 8),
        }

    def reset_telemetry(self) -> None:
        self.ssd_expert_loads = 0
        self.ssd_expert_bytes_read = 0
        self.expert_cache.hits = 0
        self.expert_cache.misses = 0
        self.expert_cache.prefetch_hits = 0
        self.lsh_predictor.lookahead_predictions = 0
        self.lsh_predictor.lookahead_correct_experts = 0
        self.live_routing_buffer.reset()
        self.profiler.reset()

    def close(self) -> None:
        if getattr(self, "expanded_bank", None) is not None:
            try:
                self.expanded_bank.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self.expanded_bank = None
        if hasattr(self, "async_pipeline"):
            self.async_pipeline.close()
        self.live_routing_buffer.reset()
        if hasattr(self, "kv_prefix_cache"):
            self.kv_prefix_cache.arena.clear()
        if self.use_mlx and HAS_MLX_METAL:
            try:
                if hasattr(self, "mlx_stream"):
                    mx.synchronize(self.mlx_stream)
                mx.clear_cache()
            except Exception:
                pass
        if hasattr(self, "mlx_layers"):
            self.mlx_layers.clear()
        self.mlx_embed_weight = None
        self.mlx_lm_head_w = None
        self.mlx_final_norm = None
        if hasattr(self, "core_np_views"):
            self.core_np_views.clear()
        self.token_cache.clear()
        self.expert_cache.cache.clear()
        self.expert_cache.current_bytes = 0
        self.expert_cache.cas_mmap = None
        self.emb_np_view = None
        self.lm_head_w_int8_view = None
        self.lm_head_s_fp16_view = None
        gc.collect()

        for mm, f_obj in [
            (getattr(self, "lm_head_mmap", None), getattr(self, "lm_head_file", None)),
            (getattr(self, "core_mmap", None), getattr(self, "core_file", None)),
            (getattr(self, "emb_mmap", None), getattr(self, "emb_file", None)),
            (getattr(self, "cas_mmap", None), getattr(self, "cas_file", None)),
        ]:
            if mm is not None:
                try:
                    madvise_range(mm, 0, len(mm), MADV_DONTNEED)
                except Exception:
                    pass
                try:
                    mm.close()
                except BufferError:
                    pass
            if f_obj is not None:
                try:
                    f_obj.close()
                except Exception:
                    pass
        self.lm_head_mmap = None
        self.core_mmap = None
        self.emb_mmap = None  # type: ignore[assignment]
        self.cas_mmap = None  # type: ignore[assignment]
        self.cpp.close()


def benchmark_prefetch_pipeline(
    store_dir: Path,
    num_tokens: int = 4,
    num_layers: int = 4,
    cache_experts: int = 32,
    seed: int = 42,
) -> PrefetchBenchmarkReport:
    """
    Task P1.4: Isolated Benchmark comparing `--disable-prefetch` (`enable_lookahead=False`)
    vs `--enable-prefetch` (`enable_lookahead=True` with `AsyncThreeStagePipeline` overlap
    and confidence-gated `LSHLookaheadPredictor`).
    """
    rng = np.random.default_rng(seed)
    eng = NativeMerkleMoEEngine(
        store_dir,
        expert_cache_size=cache_experts,
        max_cache_ram_mb=float(cache_experts) * 12.0,
        on_demand_streaming=False,
        enable_lookahead=False,
        verify_merkle_on_load=False,
        device="cpu",
    )
    try:
        token_inputs = [
            rng.standard_normal((1, 1, eng.hidden_size)).astype(np.float16)
            for _ in range(num_tokens)
        ]

        # 1. Run with Prefetch DISABLED (`--disable-prefetch`)
        eng.enable_lookahead = False
        eng.expert_cache.cache.clear()
        eng.expert_cache.current_bytes = 0
        eng.reset_telemetry()

        t_dis0 = time.perf_counter()
        out_dis_list: List[np.ndarray] = []
        for x_tok in token_inputs:
            out_d, _ = eng.async_pipeline.forward_Layer_sequence_overlapped_numpy(
                x_tok, num_layers=num_layers
            )
            out_dis_list.append(out_d)
        dis_ms = (time.perf_counter() - t_dis0) * 1000.0
        dis_hits = eng.expert_cache.hits
        dis_misses = eng.expert_cache.misses
        dis_hit_rate = 100.0 * dis_hits / max(1, dis_hits + dis_misses)

        # 2. Warm LSH transition history (2 observations so confidence threshold >= 2 is met)
        #    while starting from an empty cache to measure true prefetch impact on cold misses!
        eng.enable_lookahead = True
        eng.expert_cache.cache.clear()
        eng.expert_cache.current_bytes = 0
        eng.expert_cache.hits = 0
        eng.expert_cache.misses = 0
        eng.expert_cache.prefetch_hits = 0

        for x_tok in token_inputs:
            h_tmp = x_tok
            prev_tk: Optional[List[int]] = None
            for l_idx in range(num_layers):
                x_f = h_tmp.reshape(-1, eng.hidden_size).astype(np.float32)
                rw = eng.core_np_views[f"model.layers.{l_idx}.mlp.gate.weight"]
                tk_ind, _ = eng.adapter.route_tokens_numpy(x_f, rw)
                curr_tk = [int(e) for e in tk_ind[-1]]
                if l_idx > 0 and prev_tk is not None:
                    eng.lsh_predictor.record_layer_transition(l_idx, prev_tk, curr_tk, top_n=8)
                prev_tk = curr_tk

        t_en0 = time.perf_counter()
        out_en_list: List[np.ndarray] = []
        for x_tok in token_inputs:
            out_e, _ = eng.async_pipeline.forward_Layer_sequence_overlapped_numpy(
                x_tok, num_layers=num_layers
            )
            out_en_list.append(out_e)
        en_ms = (time.perf_counter() - t_en0) * 1000.0
        en_hits = eng.expert_cache.hits
        en_misses = eng.expert_cache.misses
        en_hit_rate = 100.0 * en_hits / max(1, en_hits + en_misses)

        max_err = max(
            float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
            for a, b in zip(out_dis_list, out_en_list)
        )
        speedup = dis_ms / max(1e-6, en_ms)

        return {
            "num_tokens": num_tokens,
            "num_layers": num_layers,
            "cache_experts": cache_experts,
            "disabled_hit_rate_pct": round(dis_hit_rate, 2),
            "enabled_hit_rate_pct": round(en_hit_rate, 2),
            "disabled_cold_misses": dis_misses,
            "enabled_cold_misses": en_misses,
            "disabled_total_ms": round(dis_ms, 3),
            "enabled_total_ms": round(en_ms, 3),
            "speedup_factor": round(speedup, 2),
            "prefetch_hits": eng.expert_cache.prefetch_hits,
            "low_confidence_skipped": eng.lsh_predictor.low_confidence_skipped,
            "numerical_parity_max_abs_err": round(max_err, 8),
        }
    finally:
        eng.close()
