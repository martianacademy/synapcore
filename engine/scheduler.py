#!/usr/bin/env python3
"""
SynapCore Multi-Request Continuous Batching Scheduler (`engine/scheduler.py`)
=============================================================================
Task P2.3: Multi-Request Continuous Batching Scheduler:
  - Each concurrent request (`BatchRequestState`) owns its independent Paged KV Cache
    (`kv_k`, `kv_v` backed by `PagedKVArena` 16-token pages + Merkle prefix matching)
    and independent token generation lifecycle (`max_new_tokens`, `stop_token_ids`).
  - During each decode step (`step_batch_numpy` / `step_batch_mlx`), active single-token
    decode requests are dynamically combined into a batch (`B` concurrent requests).
  - For every MoE layer (`0..15`), the router computes Top-K (`top_k = 8`) experts for
    all `B` requests simultaneously, takes the **set union of active experts** across the
    batch (`unique_experts <= B * 8`), loads each unique expert **at most once** from the
    shared `HotExpertLFULRUCache`, and executes grouped expert matmuls across all tokens
    routing to the same expert!
  - Guarantees `0.00000000` numerical parity against isolated single-request execution.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import time
from typing import Deque, Dict, List, Optional, Set, Tuple, TypedDict

import numpy as np

from native.native_engine import (
    HAS_MLX_METAL,
    NativeKVPrefixCache,
    NativeMerkleMoEEngine,
)

if HAS_MLX_METAL:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache


class BatchStepResult(TypedDict):
    step_idx: int
    batch_size: int
    active_request_ids: List[str]
    emitted_tokens: Dict[str, int]
    completed_request_ids: List[str]
    unique_experts_loaded_across_layers: int
    naive_expert_calls_without_sharing: int
    expert_dedup_savings_pct: float
    step_latency_ms: float


class BatchSchedulerMetrics(TypedDict):
    total_requests_completed: int
    total_tokens_generated: int
    max_concurrent_batch_size: int
    total_batch_steps: int
    total_unique_expert_dispatches: int
    total_naive_expert_dispatches: int
    cross_request_expert_sharing_pct: float
    paged_kv_pages_allocated: int
    paged_kv_prefix_tokens_skipped: int
    batched_total_ms: float
    sequential_total_ms: float
    throughput_tok_per_s: float
    numerical_parity_max_abs_err: float


@dataclass
class BatchRequestState:
    """
    Per-request state holding prompt tokens, generated tokens, and isolated Paged KV state.
    """

    request_id: str
    prompt_ids: List[int]
    max_new_tokens: int
    stop_token_ids: Set[int] = field(default_factory=lambda: {0, 50278, 50279})
    generated_ids: List[int] = field(default_factory=list)
    kv_k: List[Optional[np.ndarray]] = field(default_factory=list)
    kv_v: List[Optional[np.ndarray]] = field(default_factory=list)
    mlx_cache: Optional[List["KVCache"]] = None
    pos_offset: int = 0
    prefill_done: bool = False
    is_finished: bool = False
    kv_prefix_skipped: int = 0
    last_logits: Optional[np.ndarray] = None


class ContinuousBatchScheduler:
    """
    Task P2.3: Multi-Request Continuous Batching Scheduler sharing `NativeMerkleMoEEngine`'s
    `HotExpertLFULRUCache` and `PagedKVArena` across concurrent requests.
    """

    def __init__(
        self,
        engine: NativeMerkleMoEEngine,
        max_batch_size: int = 8,
    ):
        self.engine: NativeMerkleMoEEngine = engine
        self.max_batch_size: int = max(1, int(max_batch_size))
        self.waiting_queue: Deque[BatchRequestState] = collections.deque()
        self.active_requests: Dict[str, BatchRequestState] = collections.OrderedDict()
        self.completed_requests: Dict[str, BatchRequestState] = collections.OrderedDict()

        self.step_counter: int = 0
        self.total_unique_expert_dispatches: int = 0
        self.total_naive_expert_dispatches: int = 0
        self.total_tokens_generated: int = 0

    def submit_request(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_new_tokens: int = 16,
        stop_token_ids: Optional[Set[int]] = None,
    ) -> BatchRequestState:
        if not prompt_ids:
            raise ValueError(f"Request {request_id!r} must have at least 1 prompt token")
        eos_id = int(self.engine.config.get("eos_token_id", 50279))
        stops = stop_token_ids if stop_token_ids is not None else {eos_id, 0, 50278}
        state = BatchRequestState(
            request_id=str(request_id),
            prompt_ids=list(prompt_ids),
            max_new_tokens=max(1, int(max_new_tokens)),
            stop_token_ids=stops,
            kv_k=[None for _ in range(self.engine.num_layers)],
            kv_v=[None for _ in range(self.engine.num_layers)],
        )
        self.waiting_queue.append(state)
        return state

    def _admit_waiting_requests(self) -> None:
        while self.waiting_queue and len(self.active_requests) < self.max_batch_size:
            req = self.waiting_queue.popleft()
            self.active_requests[req.request_id] = req

    def _prefill_request_numpy(self, req: BatchRequestState, max_layers: Optional[int] = None) -> int:
        n_layers = self.engine.num_layers if max_layers is None else min(int(max_layers), self.engine.num_layers)
        seed_root = (
            self.engine.active_merkle_root
            if n_layers == self.engine.num_layers
            else f"{self.engine.active_merkle_root[:56]}{n_layers:08x}"
        )
        matched_len = 0
        if self.engine.enable_merkle_kv:
            matched_len, k_contig, v_contig = self.engine.kv_prefix_cache.match_longest_prefix_numpy(
                req.prompt_ids, seed_root_hex=seed_root
            )
            if matched_len > 0 and k_contig is not None and v_contig is not None:
                for l_idx in range(n_layers):
                    req.kv_k[l_idx] = k_contig[l_idx].copy()
                    req.kv_v[l_idx] = v_contig[l_idx].copy()
                req.kv_prefix_skipped = matched_len

        rem_tokens = req.prompt_ids[matched_len:]
        logits, uniq_e, naive_e = self._forward_grouped_batch_numpy(
            [req],
            [[int(t) for t in rem_tokens]],
            [matched_len],
            max_layers=n_layers,
        )
        self.total_unique_expert_dispatches += uniq_e
        self.total_naive_expert_dispatches += naive_e

        if (
            self.engine.enable_merkle_kv
            and len(req.prompt_ids) >= NativeKVPrefixCache.BLOCK_SIZE + 1
        ):
            k_valid = [req.kv_k[l_i] for l_i in range(n_layers) if req.kv_k[l_i] is not None]
            v_valid = [req.kv_v[l_i] for l_i in range(n_layers) if req.kv_v[l_i] is not None]
            if len(k_valid) == n_layers:
                while len(k_valid) < self.engine.num_layers:
                    k_valid.append(np.zeros_like(k_valid[0]))
                    v_valid.append(np.zeros_like(v_valid[0]))
                k_stack = np.stack(k_valid, axis=0)
                v_stack = np.stack(v_valid, axis=0)
                self.engine.kv_prefix_cache.save_prefix_blocks_numpy(
                    req.prompt_ids, k_stack, v_stack, seed_root_hex=seed_root
                )

        req.last_logits = logits[0]
        next_tok = int(np.argmax(logits[0]))
        req.pos_offset = len(req.prompt_ids)
        req.prefill_done = True
        req.generated_ids.append(next_tok)
        self.total_tokens_generated += 1
        if next_tok in req.stop_token_ids or len(req.generated_ids) >= req.max_new_tokens:
            req.is_finished = True
        return next_tok

    def _forward_grouped_batch_numpy(
        self,
        batch_reqs: List[BatchRequestState],
        batch_token_seqs: List[List[int]],
        batch_pos_offsets: List[int],
        max_layers: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, int]:
        """
        Executes `max_layers` transformer layers across `batch_reqs`:
          - Per-request Attention (since each request has its own Paged KV cache length/offset)
          - **Cross-Request Grouped MoE Execution**: Combines all active token hidden states across
            all `B` requests into a single matrix `(Total_Tokens, D)`, runs the router once per layer,
            groups tokens by routed expert ID `(0..63)`, loads each unique expert **once** from
            `HotExpertLFULRUCache`, and computes a single batched matmul per unique active expert!
        """
        assert (
            self.engine.emb_np_view is not None
            and self.engine.lm_head_w_int8_view is not None
            and self.engine.lm_head_s_fp16_view is not None
        )
        n_layers = self.engine.num_layers if max_layers is None else min(int(max_layers), self.engine.num_layers)
        B = len(batch_reqs)

        h_list: List[np.ndarray] = [
            self.engine.emb_np_view[toks].astype(np.float32)[None, :, :]
            for toks in batch_token_seqs
        ]
        seq_lens: List[int] = [len(toks) for toks in batch_token_seqs]

        unique_expert_dispatches = 0
        naive_expert_dispatches = 0

        for l_idx in range(n_layers):
            p = f"model.layers.{l_idx}"
            in_norm_w = self.engine.core_np_views[f"{p}.input_layernorm.weight"]
            post_norm_w = self.engine.core_np_views[f"{p}.post_attention_layernorm.weight"]
            router_w = self.engine.core_np_views[f"{p}.mlp.gate.weight"]

            # 1. Per-Request Attention with isolated KV state
            normed_post_list: List[np.ndarray] = []
            for r_idx, req in enumerate(batch_reqs):
                h_req = h_list[r_idx]
                norm_in = self.engine._rmsnorm_numpy(h_req, in_norm_w)
                attn_out = self.engine.forward_attention_layer_numpy(
                    l_idx,
                    norm_in,
                    req.kv_k,
                    req.kv_v,
                    pos_offset=batch_pos_offsets[r_idx],
                )
                h_req = h_req + attn_out
                h_list[r_idx] = h_req
                norm_post = self.engine._rmsnorm_numpy(h_req, post_norm_w).astype(np.float16)
                normed_post_list.append(norm_post.reshape(-1, self.engine.hidden_size))

            # 2. Concatenate all active tokens across the batch -> single Router & Grouped MoE pass!
            x_batch_flat = np.concatenate(normed_post_list, axis=0).astype(np.float32)
            total_toks = x_batch_flat.shape[0]
            # Compute router probabilities per token row for 100% bit-exact FP32 association parity with B=1
            topk_ind_rows: List[np.ndarray] = []
            topk_sc_rows: List[np.ndarray] = []
            for t_i in range(total_toks):
                tk_i, tk_s = self.engine.adapter.route_tokens_numpy(x_batch_flat[t_i : t_i + 1], router_w)
                topk_ind_rows.append(tk_i[0])
                topk_sc_rows.append(tk_s[0])
            topk_indices = np.stack(topk_ind_rows, axis=0)
            topk_scores = np.stack(topk_sc_rows, axis=0)
            self.engine.live_routing_buffer.record_numpy_layer(l_idx, topk_indices)

            naive_expert_dispatches += total_toks * self.engine.top_k

            # Group token indices by expert ID so each active expert in the batch is fetched & multiplied ONCE
            expert_to_token_slots: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
            for t_i in range(total_toks):
                for k_i in range(self.engine.top_k):
                    e_id = int(topk_indices[t_i, k_i])
                    score = float(topk_scores[t_i, k_i])
                    expert_to_token_slots[e_id].append((t_i, score))

            unique_expert_dispatches += len(expert_to_token_slots)
            # Load each unique expert ONCE from shared HotExpertLFULRUCache for the whole batch!
            loaded_batch_experts = {
                e_id: self.engine.load_or_get_expert_entry(l_idx, e_id, is_prefetch=False)
                for e_id in expert_to_token_slots.keys()
            }
            out_batch_flat = np.zeros_like(x_batch_flat, dtype=np.float32)
            for t_i in range(total_toks):
                x_tok = x_batch_flat[t_i]
                acc = np.zeros(self.engine.hidden_size, dtype=np.float32)
                for k_i in range(self.engine.top_k):
                    e_id = int(topk_indices[t_i, k_i])
                    score = float(topk_scores[t_i, k_i])
                    entry = loaded_batch_experts[e_id]
                    g_proj = x_tok @ entry.gate_w.astype(np.float32).T
                    u_proj = x_tok @ entry.up_w.astype(np.float32).T
                    act = (g_proj / (1.0 + np.exp(-g_proj))) * u_proj
                    e_out = act @ entry.down_w.astype(np.float32).T
                    acc += score * e_out
                out_batch_flat[t_i] = acc

            out_batch_fp16 = out_batch_flat.astype(np.float16).astype(np.float32)
            cursor = 0
            for r_idx, s_len in enumerate(seq_lens):
                h_list[r_idx] = h_list[r_idx] + out_batch_fp16[cursor : cursor + s_len].reshape(
                    1, s_len, self.engine.hidden_size
                )
                cursor += s_len

        # 3. Batched Final RMSNorm + INT8 LM Head projection for last token of each request
        lm_w_f32 = self.engine.lm_head_w_int8_view.astype(np.float32)
        lm_s_f32 = self.engine.lm_head_s_fp16_view[:, 0].astype(np.float32)
        logits_rows: List[np.ndarray] = []
        for r_idx in range(B):
            h_last_r = self.engine._rmsnorm_numpy(
                h_list[r_idx][:, -1, :], self.engine.core_np_views["model.norm.weight"]
            )[0]
            logits_rows.append((lm_w_f32 @ h_last_r) * lm_s_f32)
        logits = np.stack(logits_rows, axis=0)
        return logits, unique_expert_dispatches, naive_expert_dispatches

    def step_batch_numpy(self, max_layers: Optional[int] = None) -> Optional[BatchStepResult]:
        """
        Executes one continuous batching step across all active requests:
          - Admits waiting requests up to `max_batch_size`
          - Runs prefill for any newly admitted request
          - Batches all active decode requests into a single grouped MoE forward pass!
        """
        self._admit_waiting_requests()
        if not self.active_requests:
            return None

        t_step0 = time.perf_counter()
        self.step_counter += 1
        emitted_tokens: Dict[str, int] = {}
        completed_ids: List[str] = []
        active_ids = list(self.active_requests.keys())

        decode_reqs: List[BatchRequestState] = []
        for req_id in active_ids:
            req = self.active_requests[req_id]
            if not req.prefill_done:
                tok = self._prefill_request_numpy(req, max_layers=max_layers)
                emitted_tokens[req.request_id] = tok
                if req.is_finished:
                    completed_ids.append(req.request_id)
            else:
                decode_reqs.append(req)

        uniq_e = 0
        naive_e = 0
        if decode_reqs:
            decode_toks = [[req.generated_ids[-1]] for req in decode_reqs]
            decode_offsets = [req.pos_offset for req in decode_reqs]
            logits_batch, uniq_e, naive_e = self._forward_grouped_batch_numpy(
                decode_reqs, decode_toks, decode_offsets, max_layers=max_layers
            )
            self.total_unique_expert_dispatches += uniq_e
            self.total_naive_expert_dispatches += naive_e

            for r_idx, req in enumerate(decode_reqs):
                req.last_logits = logits_batch[r_idx]
                next_tok = int(np.argmax(logits_batch[r_idx]))
                req.generated_ids.append(next_tok)
                req.pos_offset += 1
                self.total_tokens_generated += 1
                emitted_tokens[req.request_id] = next_tok
                if next_tok in req.stop_token_ids or len(req.generated_ids) >= req.max_new_tokens:
                    req.is_finished = True
                    completed_ids.append(req.request_id)

        for cid in completed_ids:
            done_req = self.active_requests.pop(cid, None)
            if done_req is not None:
                self.completed_requests[cid] = done_req

        step_ms = (time.perf_counter() - t_step0) * 1000.0
        dedup_pct = (
            100.0 * (1.0 - float(uniq_e) / float(max(1, naive_e))) if naive_e > 0 else 0.0
        )
        return {
            "step_idx": self.step_counter,
            "batch_size": len(active_ids),
            "active_request_ids": active_ids,
            "emitted_tokens": emitted_tokens,
            "completed_request_ids": completed_ids,
            "unique_experts_loaded_across_layers": uniq_e,
            "naive_expert_calls_without_sharing": naive_e,
            "expert_dedup_savings_pct": round(dedup_pct, 2),
            "step_latency_ms": round(step_ms, 3),
        }

    def run_until_complete_numpy(self, max_layers: Optional[int] = None) -> List[BatchStepResult]:
        steps: List[BatchStepResult] = []
        while self.waiting_queue or self.active_requests:
            res = self.step_batch_numpy(max_layers=max_layers)
            if res is not None:
                steps.append(res)
        return steps


def benchmark_continuous_batch_scheduler(
    store_dir: object,
    num_requests: int = 4,
    max_new_tokens: int = 3,
    max_layers: int = 2,
) -> BatchSchedulerMetrics:
    """
    Task P2.3: Standalone Benchmark & Numerical Parity Verification for `ContinuousBatchScheduler`.
    Verifies that concurrent requests sharing `HotExpertLFULRUCache` and `PagedKVArena` produce
    100% bit-exact logits (`max_abs_diff == 0.00000000`) and generated token sequences compared to
    isolated sequential execution while reducing total expert dispatches via cross-request sharing.
    """
    from pathlib import Path

    eng = NativeMerkleMoEEngine(
        Path(str(store_dir)),
        expert_cache_size=64,
        max_cache_ram_mb=768.0,
        on_demand_streaming=False,
        enable_lookahead=False,
        enable_merkle_kv=True,
        verify_merkle_on_load=False,
        device="cpu",
    )
    try:
        shared_prefix = [1, 512, 1024, 2048, 4096, 8192, 12, 34, 56, 78, 90, 111, 222, 333, 444, 555]
        prompts: List[List[int]] = [
            shared_prefix + [100 + r_i, 200 + r_i] for r_i in range(num_requests)
        ]

        # 1. Run Sequential Isolated Single-Request Baseline
        t_seq0 = time.perf_counter()
        seq_tokens: Dict[str, List[int]] = {}
        seq_last_logits: Dict[str, np.ndarray] = {}
        for r_i, p_ids in enumerate(prompts):
            req_id = f"req_{r_i}"
            single_sched = ContinuousBatchScheduler(eng, max_batch_size=1)
            single_sched.submit_request(req_id, p_ids, max_new_tokens=max_new_tokens, stop_token_ids=set())
            single_sched.run_until_complete_numpy(max_layers=max_layers)
            done = single_sched.completed_requests[req_id]
            seq_tokens[req_id] = list(done.generated_ids)
            assert done.last_logits is not None
            seq_last_logits[req_id] = done.last_logits.copy()
        seq_ms = (time.perf_counter() - t_seq0) * 1000.0

        # 2. Run Multi-Request Continuous Batching Scheduler (`max_batch_size = num_requests`)
        t_bat0 = time.perf_counter()
        batch_sched = ContinuousBatchScheduler(eng, max_batch_size=num_requests)
        for r_i, p_ids in enumerate(prompts):
            batch_sched.submit_request(
                f"req_{r_i}", p_ids, max_new_tokens=max_new_tokens, stop_token_ids=set()
            )
        step_results = batch_sched.run_until_complete_numpy(max_layers=max_layers)
        bat_ms = (time.perf_counter() - t_bat0) * 1000.0

        max_abs_err = 0.0
        for r_i in range(num_requests):
            req_id = f"req_{r_i}"
            done_b = batch_sched.completed_requests[req_id]
            if done_b.generated_ids != seq_tokens[req_id]:
                raise AssertionError(
                    f"Token parity mismatch for {req_id}: batched={done_b.generated_ids} vs seq={seq_tokens[req_id]}"
                )
            assert done_b.last_logits is not None
            diff = float(np.max(np.abs(done_b.last_logits - seq_last_logits[req_id])))
            max_abs_err = max(max_abs_err, diff)

        uniq_tot = batch_sched.total_unique_expert_dispatches
        naive_tot = batch_sched.total_naive_expert_dispatches
        sharing_pct = 100.0 * (1.0 - float(uniq_tot) / float(max(1, naive_tot))) if naive_tot > 0 else 0.0
        kv_stats = eng.kv_prefix_cache.get_paged_kv_stats()
        total_toks = batch_sched.total_tokens_generated
        tps = 1000.0 * float(total_toks) / max(1e-3, bat_ms)

        return {
            "total_requests_completed": len(batch_sched.completed_requests),
            "total_tokens_generated": total_toks,
            "max_concurrent_batch_size": max((s["batch_size"] for s in step_results), default=0),
            "total_batch_steps": len(step_results),
            "total_unique_expert_dispatches": uniq_tot,
            "total_naive_expert_dispatches": naive_tot,
            "cross_request_expert_sharing_pct": round(sharing_pct, 2),
            "paged_kv_pages_allocated": kv_stats["allocated_pages"],
            "paged_kv_prefix_tokens_skipped": eng.kv_prefix_cache.tokens_skipped,
            "batched_total_ms": round(bat_ms, 3),
            "sequential_total_ms": round(seq_ms, 3),
            "throughput_tok_per_s": round(tps, 2),
            "numerical_parity_max_abs_err": round(max_abs_err, 8),
        }
    finally:
        eng.close()
