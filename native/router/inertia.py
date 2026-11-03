#!/usr/bin/env python3
"""
SynapCore Thalamic Router v2 — Component 1: Working-Memory Neural Inertia (`native/router/inertia.py`)
======================================================================================================
WHAT THIS FILE DOES:
  Implements biological **Working-Memory Neural Inertia (Attractor State `s_t`)** per layer.
  In the human brain, once a cortical circuit (e.g., coding or mathematical derivation) is
  activated, recurrent thalamocortical loops maintain an attractor state across consecutive
  words so the brain does not randomly switch expert circuits mid-sentence.

MATHEMATICAL FORMULATION:
  1. At token step `t` and layer `l`, if `inertia_steps[l] > 0`, the router adds a log-prior
     continuity bias scaled by the layer's base logit standard deviation `sigma_z`:
       `z_inertia^{(l)} = gamma_inertia * sigma_z * NormalizeZeroMean(log(max(s_{t-1}^{(l)}, eps)))`
     Wait—to ensure direct, predictable attraction toward recently active experts without
     penalizing cold experts by `-3.8` logits, we use the centered log-probability bonus
     relative to the uniform distribution `u = 1 / N`:
       `log_ratio = log(max(s_{t-1}^{(l)}, eps) / u)`
       `z_inertia^{(l)} = gamma_inertia * clip(log_ratio, -2.0, +4.0)`
  2. After final routing probabilities `p_t^{(l)}` are computed, the working-memory state
     updates via Exponential Moving Average (EMA):
       `s_t^{(l)} = alpha_inertia * s_{t-1}^{(l)} + (1 - alpha_inertia) * p_t^{(l)}`

AUDITOR FIXES INCLUDED HERE:
  - Bug 1.4 : `mlx_inertia_steps[layer_idx]` is tracked per layer directly inside
    `apply_and_update_mlx`, completely decoupled from `update_topdown_context_mlx`.
  - Bug 1.5 : Both NumPy and MLX compute `log(max(s, 1e-6))` in `float32` using the exact
    same `inertia_eps = 1e-6` constant and the exact same token-sequential EMA rule during
    multi-token prefill (`T >= 1`), achieving `< 1e-4` numerical parity between CPU and GPU.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np

from native.native_engine import HAS_MLX_METAL
from native.router.config import ThalamicRouterV2Config

if HAS_MLX_METAL:
    import mlx.core as mx


class NeuralInertiaWorkingMemory:
    """
    Per-layer working-memory EMA state `s_t` (`16 x 256`) with identical NumPy and MLX semantics.
    """

    def __init__(self, cfg: ThalamicRouterV2Config):
        self.cfg: ThalamicRouterV2Config = cfg
        self.uniform_prob: float = 1.0 / float(self.cfg.total_experts)
        self.log_uniform: float = math.log(self.uniform_prob)

        # CPU / NumPy session state
        self.s_inertia_np: np.ndarray = np.full(
            (self.cfg.num_layers, self.cfg.total_experts),
            self.uniform_prob,
            dtype=np.float32,
        )
        self.inertia_steps_np: np.ndarray = np.zeros((self.cfg.num_layers,), dtype=np.int32)
        self.prev_top8_per_layer: Dict[int, Set[int]] = {}
        self.jaccard_overlaps: List[float] = []

        # GPU / MLX session state
        self.mlx_inertia_steps: List[int] = [0] * self.cfg.num_layers
        if HAS_MLX_METAL:
            self.mlx_s_inertia: List["mx.array"] = [
                mx.full((1, self.cfg.total_experts), self.uniform_prob, dtype=mx.float32)
                for _ in range(self.cfg.num_layers)
            ]

    def reset(self) -> None:
        """Resets working-memory inertia state and Jaccard overlap history for a new session."""
        self.s_inertia_np.fill(self.uniform_prob)
        self.inertia_steps_np.fill(0)
        self.prev_top8_per_layer.clear()
        self.jaccard_overlaps.clear()
        self.mlx_inertia_steps = [0] * self.cfg.num_layers
        if HAS_MLX_METAL:
            self.mlx_s_inertia = [
                mx.full((1, self.cfg.total_experts), self.uniform_prob, dtype=mx.float32)
                for _ in range(self.cfg.num_layers)
            ]

    def compute_inertia_bias_numpy(self, layer_idx: int) -> np.ndarray:
        """
        Returns `(1, 256)` float32 inertia logit bias for `layer_idx`.
        If `inertia_steps_np[layer_idx] == 0`, returns all zeros.
        """
        if self.inertia_steps_np[layer_idx] <= 0:
            return np.zeros((1, self.cfg.total_experts), dtype=np.float32)
        s_prev = np.maximum(self.s_inertia_np[layer_idx], self.cfg.inertia_eps)
        log_ratio = np.log(s_prev) - self.log_uniform
        bias_1d = self.cfg.gamma_inertia * np.clip(log_ratio, -2.0, 4.0)
        return bias_1d[None, :].astype(np.float32)

    def update_state_numpy(
        self,
        layer_idx: int,
        probs_2d: np.ndarray,
        topk_indices_2d: np.ndarray,
    ) -> None:
        """
        Updates `s_inertia_np[layer_idx]` across `T` tokens in `probs_2d` (`(T, 256)`)
        and records exact Top-8 Jaccard overlap across consecutive tokens.
        """
        alpha = self.cfg.alpha_inertia
        num_tokens = probs_2d.shape[0]
        for t_i in range(num_tokens):
            p_tok = probs_2d[t_i].astype(np.float32)
            if self.inertia_steps_np[layer_idx] == 0:
                self.s_inertia_np[layer_idx] = p_tok
            else:
                self.s_inertia_np[layer_idx] = (
                    alpha * self.s_inertia_np[layer_idx] + (1.0 - alpha) * p_tok
                )
            self.inertia_steps_np[layer_idx] += 1

            curr_set = {int(x) for x in topk_indices_2d[t_i]}
            if layer_idx in self.prev_top8_per_layer:
                prev_set = self.prev_top8_per_layer[layer_idx]
                u_sz = len(prev_set | curr_set)
                if u_sz > 0:
                    self.jaccard_overlaps.append(len(prev_set & curr_set) / float(u_sz))
            self.prev_top8_per_layer[layer_idx] = curr_set

    def compute_inertia_bias_mlx(self, layer_idx: int, dtype: "mx.Dtype") -> "mx.array":
        """
        Zero-sync on-GPU inertia logit bias computation in `mx.float32` (preventing FP16 log underflow),
        then cast to `dtype`.
        """
        if self.mlx_inertia_steps[layer_idx] <= 0:
            return mx.zeros((1, self.cfg.total_experts), dtype=dtype)
        s_prev_f32 = mx.maximum(self.mlx_s_inertia[layer_idx].astype(mx.float32), self.cfg.inertia_eps)
        log_ratio = mx.log(s_prev_f32) - self.log_uniform
        bias_f32 = self.cfg.gamma_inertia * mx.clip(log_ratio, -2.0, 4.0)
        return bias_f32.astype(dtype)

    def update_state_mlx(self, layer_idx: int, routing_weights: "mx.array") -> None:
        """
        Updates `self.mlx_s_inertia[layer_idx]` in `mx.float32` matching `update_state_numpy`
        across both single-token decode (`T == 1`) and multi-token prefill (`T > 1`).
        """
        alpha = self.cfg.alpha_inertia
        probs_f32 = mx.stop_gradient(routing_weights.astype(mx.float32))
        num_tokens = int(probs_f32.shape[0])

        s_curr = self.mlx_s_inertia[layer_idx]
        steps_curr = self.mlx_inertia_steps[layer_idx]
        for t_i in range(num_tokens):
            p_tok = probs_f32[t_i : t_i + 1, :]
            if steps_curr == 0:
                s_curr = p_tok
            else:
                s_curr = alpha * s_curr + (1.0 - alpha) * p_tok
            steps_curr += 1

        self.mlx_s_inertia[layer_idx] = s_curr
        self.mlx_inertia_steps[layer_idx] = steps_curr

    def record_post_generation_jaccard(self, all_routed_np: np.ndarray) -> None:
        """Computes exact Top-8 Jaccard stickiness from flushed MLX routing trace `(steps, 16, 8)`."""
        if all_routed_np.size == 0:
            return
        num_steps, num_layers, _ = all_routed_np.shape
        for l_idx in range(num_layers):
            prev_set: Set[int] = set()
            for s_idx in range(num_steps):
                curr_set = {int(x) for x in all_routed_np[s_idx, l_idx]}
                if s_idx > 0:
                    u_sz = len(prev_set | curr_set)
                    if u_sz > 0:
                        self.jaccard_overlaps.append(len(prev_set & curr_set) / float(u_sz))
                prev_set = curr_set
