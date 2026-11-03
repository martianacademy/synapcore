#!/usr/bin/env python3
"""
SynapCore Thalamic Router v2 — Component 2: Top-Down Thalamocortical Loop (`native/router/topdown.py`)
======================================================================================================
WHAT THIS FILE DOES:
  Implements the **Corticothalamic Feedback Loop (`Layer 15 -> Layers 0..15` across ALL `0..255` Experts)**.
  In a standard Transformer/MoE, Layer 0 routes blindly without knowing what the upper
  cortical layers (`Layer 15`) concluded on previous tokens. In the human brain,
  Layer-6 corticothalamic pyramidal neurons maintain a slow-wave cortical context (`rho_td = 0.75`)
  and project top-down intention signals back to the Thalamus so both core sensory/syntactic
  assemblies (`Base Experts 0..63`) and high-level association assemblies (`Domain Experts 64..255`)
  route in alignment with the multi-token plan.

MATHEMATICAL FORMULATION:
  1. At the end of token `t - 1`, the final Layer-15 hidden state `h^{(15)}_{t-1} in R^{2048}`
     is normalized in `float32` and integrated into a slow-wave cortical EMA `h_bar_{t-1}`:
       `h_unit = h^{(15)}_{t-1} / ||h^{(15)}_{t-1}||_2`
       `h_bar_raw = rho_td * h_bar_{t-2} + (1 - rho_td) * h_unit`
       `h_bar = h_bar_raw / ||h_bar_raw||_2`
  2. Projected into a rank-32 cortical intention vector `c_{t-1} in R^{32}` with unit RMS
     scaling (so low-rank projection from 2048 -> 32 does not shrink the signal by `sqrt(32/2048)`):
       `c_raw = tanh(h_bar @ U_td.T)`
       `c_unit = c_raw / max(rms(c_raw), 1e-6)`
  3. At token `t`, each layer `l in 0..15` receives a top-down routing bias across all `256` experts
     (`0..63` Base + `64..255` Domain) normalized by `1 / sqrt(rank)` and scaled by `beta_topdown * sigma_z`:
       `z_topdown^{(l)} = beta_topdown * sigma_z * ((c_unit @ W_td^{(l)}.T) / sqrt(r))`

AUDITOR FIXES & V2.2 UPGRADES INCLUDED HERE:
  - Bug 1.1 (`fp16` Squaring Overflow): L2 norm of `layer15_h_mlx` is computed strictly in
    `mx.float32` (`sum(h_f32 * h_f32)`) before casting the unit vector to `mx.float16`.
  - Mismatch 2.4 (`u_topdown` & Full `0..255` `w_topdown` Gradient Training): Implements full
    chain-rule backpropagation through `c = tanh(h @ U_td.T)` and updates both Base (`0..63`)
    and Domain (`64..255`) rows of `W_td` alongside `U_td`.
  - Full `0..255` Cortical Coupling + Slow-Wave EMA (`rho_td = 0.75`): Ensures top-down
    feedback actively stabilizes both the 6 Base Experts and 2 Domain Experts selected in Top-8.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from native.native_engine import HAS_MLX_METAL
from native.router.config import ThalamicRouterV2Config

if HAS_MLX_METAL:
    import mlx.core as mx


class TopDownThalamocorticalLoop:
    """
    Manages the Layer-15 -> Layers 0..15 top-down feedback matrices `u_topdown` `(32, 2048)`
    and `w_topdown` `(16, 256, 32)` plus the slow-wave cortical context `last_layer15_h`.
    """

    def __init__(self, cfg: ThalamicRouterV2Config):
        self.cfg: ThalamicRouterV2Config = cfg
        self.inv_sqrt_rank: float = 1.0 / math.sqrt(float(max(1, self.cfg.topdown_rank)))
        self.u_topdown: np.ndarray = np.zeros(
            (self.cfg.topdown_rank, self.cfg.hidden_size), dtype=np.float32
        )
        self.w_topdown: np.ndarray = np.zeros(
            (self.cfg.num_layers, self.cfg.total_experts, self.cfg.topdown_rank), dtype=np.float32
        )

        # Session state (CPU & MLX)
        self.last_layer15_h_np: Optional[np.ndarray] = None  # (1, 2048) float32 unit-norm
        self.mlx_last_layer15_h: Optional["mx.array"] = None  # (1, 2048) float16 unit-norm
        self.topdown_steps_count: int = 0

        if HAS_MLX_METAL:
            self.mlx_u_topdown: "mx.array" = mx.zeros(
                (self.cfg.topdown_rank, self.cfg.hidden_size), dtype=mx.float16
            )
            self.mlx_w_topdown: List["mx.array"] = [
                mx.zeros((self.cfg.total_experts, self.cfg.topdown_rank), dtype=mx.float16)
                for _ in range(self.cfg.num_layers)
            ]

    def sync_mlx_weights(self) -> None:
        """
        Syncs persistent `u_topdown` and `w_topdown` to MLX GPU buffers WITHOUT resetting
        `self.mlx_last_layer15_h` session state (Fixing Auditor Bug 1.4).
        """
        if not HAS_MLX_METAL:
            return
        self.mlx_u_topdown = mx.array(self.u_topdown.astype(np.float16))
        self.mlx_w_topdown = [
            mx.array(self.w_topdown[l_idx].astype(np.float16))
            for l_idx in range(self.cfg.num_layers)
        ]
        mx.eval(self.mlx_u_topdown, self.mlx_w_topdown)

    def reset_session(self) -> None:
        """Clears only the per-turn cortical context vector `last_layer15_h`."""
        self.last_layer15_h_np = None
        self.mlx_last_layer15_h = None
        self.topdown_steps_count = 0

    def update_context_numpy(self, layer15_h: np.ndarray) -> None:
        """
        Normalizes `layer15_h` in `float32` (preventing overflow even when `||h||_2 > 1000`)
        and updates the slow-wave cortical context EMA (`rho = topdown_ema_decay`) with
        exact unit L2 re-normalization.
        """
        arr_f32 = np.asarray(layer15_h, dtype=np.float32).reshape(-1, self.cfg.hidden_size)
        last_vec = arr_f32[-1:]
        norm_f32 = np.sqrt(np.maximum(np.sum(last_vec * last_vec, axis=-1, keepdims=True), 1e-12))
        unit_vec = (last_vec / norm_f32).astype(np.float32)

        if self.last_layer15_h_np is None:
            self.last_layer15_h_np = unit_vec
        else:
            rho = float(self.cfg.topdown_ema_decay)
            blended = rho * self.last_layer15_h_np + (1.0 - rho) * unit_vec
            b_norm = np.sqrt(np.maximum(np.sum(blended * blended, axis=-1, keepdims=True), 1e-12))
            self.last_layer15_h_np = (blended / b_norm).astype(np.float32)

    def update_context_mlx(self, layer15_h_mlx: "mx.array") -> None:
        """
        Auditor Bug 1.1 Fix + Slow-Wave Cortical EMA:
        Computes L2 norm strictly in `mx.float32` BEFORE casting to `mx.float16`.
        Prevents `fp16` squaring overflow when `||h||_2 > 255.9` and maintains the unit-norm
        slow-wave cortical context EMA on GPU.
        """
        if not HAS_MLX_METAL:
            return
        last_tok_f32 = layer15_h_mlx[:, -1:, :].reshape(1, self.cfg.hidden_size).astype(mx.float32)
        norm_f32 = mx.sqrt(
            mx.maximum(mx.sum(last_tok_f32 * last_tok_f32, axis=-1, keepdims=True), 1e-12)
        )
        unit_f32 = last_tok_f32 / norm_f32

        if self.mlx_last_layer15_h is None:
            self.mlx_last_layer15_h = unit_f32.astype(mx.float16)
        else:
            rho = float(self.cfg.topdown_ema_decay)
            blended_f32 = (
                rho * self.mlx_last_layer15_h.astype(mx.float32) + (1.0 - rho) * unit_f32
            )
            b_norm_f32 = mx.sqrt(
                mx.maximum(mx.sum(blended_f32 * blended_f32, axis=-1, keepdims=True), 1e-12)
            )
            self.mlx_last_layer15_h = (blended_f32 / b_norm_f32).astype(mx.float16)
        self.topdown_steps_count += 1

    def compute_topdown_bias_numpy(self, layer_idx: int, sigma_z: float = 1.0) -> np.ndarray:
        """
        Computes `(1, 256)` top-down logit bias across all `0..255` experts for `layer_idx`,
        normalized by `1 / sqrt(topdown_rank)` so dot-product variance is invariant to rank `r`.
        """
        if self.last_layer15_h_np is None:
            return np.zeros((1, self.cfg.total_experts), dtype=np.float32)
        c_raw = np.tanh(self.last_layer15_h_np @ self.u_topdown.T)  # (1, 32)
        c_rms = np.sqrt(np.maximum(np.mean(c_raw * c_raw, axis=-1, keepdims=True), 1e-8))
        c_normed = c_raw / c_rms
        raw_proj = (c_normed @ self.w_topdown[layer_idx].T) * self.inv_sqrt_rank  # (1, 256)
        if layer_idx == 0:
            self.topdown_steps_count += 1
        return (self.cfg.beta_topdown * float(sigma_z) * raw_proj).astype(np.float32)

    def compute_topdown_bias_mlx(
        self, layer_idx: int, sigma_z: "mx.array", dtype: "mx.Dtype"
    ) -> "mx.array":
        """
        Zero-sync on-GPU top-down logit bias computation matching `compute_topdown_bias_numpy`.
        """
        if self.mlx_last_layer15_h is None:
            return mx.zeros((1, self.cfg.total_experts), dtype=dtype)
        c_raw = mx.tanh(
            self.mlx_last_layer15_h.astype(mx.float32) @ self.mlx_u_topdown.astype(mx.float32).T
        )
        c_rms = mx.sqrt(mx.maximum(mx.mean(c_raw * c_raw, axis=-1, keepdims=True), 1e-8))
        c_normed = c_raw / c_rms
        raw_proj = (
            c_normed @ self.mlx_w_topdown[layer_idx].astype(mx.float32).T
        ) * self.inv_sqrt_rank
        td_f32 = self.cfg.beta_topdown * sigma_z * raw_proj
        return td_f32.astype(dtype)

    def backward_and_update_numpy(
        self,
        layer_idx: int,
        h15_unit_batch: np.ndarray,
        grad_z_topdown: np.ndarray,
        learning_rate: float,
    ) -> Tuple[float, float]:
        """
        Auditor Mismatch 2.4 Fix + Full `0..255` Base & Domain Update:
        Backpropagates `grad_z_topdown` (`(N, 256)`) through both `w_topdown[layer_idx]`
        (updating Base Experts `0..63` with `0.15 * base_topdown_scale * lr` and Domain Experts `64..255`
        with `lr`) AND `u_topdown` via chain rule through `c = tanh(h15 @ u_topdown.T)`.
        Returns `(||grad_u_td||_F, ||grad_w_td||_F)`.
        """
        n_samples = float(max(1, h15_unit_batch.shape[0]))
        c_raw = np.tanh(h15_unit_batch @ self.u_topdown.T)  # (N, 32)
        c_rms = np.sqrt(np.maximum(np.mean(c_raw * c_raw, axis=-1, keepdims=True), 1e-8))
        c_normed = c_raw / c_rms

        eff_beta = self.cfg.beta_topdown * self.inv_sqrt_rank
        # 1. Gradient wrt w_topdown[layer_idx] (256, 32)
        grad_w_td = eff_beta * (grad_z_topdown.T @ c_normed) / n_samples

        # 2. Chain-rule gradient wrt u_topdown (32, 2048) through tanh
        grad_c_normed = eff_beta * (grad_z_topdown @ self.w_topdown[layer_idx])  # (N, 32)
        dtanh = (1.0 - c_raw * c_raw) / c_rms
        grad_u_td = ((grad_c_normed * dtanh).T @ h15_unit_batch) / n_samples  # (32, 2048)

        # Update both Base Experts (0..63) and Expanded Domain Experts (64..255)
        base_e = self.cfg.base_experts
        self.w_topdown[layer_idx, :base_e] -= (
            learning_rate * (0.15 * self.cfg.base_topdown_scale) * grad_w_td[:base_e]
        )
        self.w_topdown[layer_idx, base_e:] -= learning_rate * grad_w_td[base_e:]

        # Update u_topdown and row-normalize to keep unit row norms
        self.u_topdown -= (learning_rate * 0.25) * grad_u_td
        u_norms = np.maximum(np.linalg.norm(self.u_topdown, axis=-1, keepdims=True), 1e-6)
        self.u_topdown = self.u_topdown / u_norms

        return float(np.linalg.norm(grad_u_td)), float(np.linalg.norm(grad_w_td))
