#!/usr/bin/env python3
"""
SynapCore Thalamic Router v2 — Component 4: Symmetric Oja-Hebbian Synaptic Plasticity & Fatigue
(`native/router/hebbian.py`)
===============================================================================================
WHAT THIS FILE DOES:
  Implements biological synaptic plasticity & lateral excitation across ALL `0..255` experts
  (`Base Experts 0..63` + `Specialized Domain Experts 64..255`) in each layer:
    1. **Full `256 x 256` Symmetric Top-8 Normalized Oja-Hebbian Co-Activation Matrix (`H_coact`)**:
       *"Neurons that fire together, wire together."*
       Connects Base <-> Base (`0..63 x 0..63`), Base <-> Domain (`0..63 x 64..255`), and
       Domain <-> Domain (`64..255 x 64..255`) expert assemblies.
       During forward routing, pre-synaptic activation `p_pre` blends the active working-memory
       trace `s_{t-1}` with current token resonance `p_init` (`0.70*s_{t-1} + 0.30*p_init`),
       projects through `H_coact[l]`, and scales by peak synaptic drive so co-wired cliques
       mutually stabilize each other across tokens without distorting Base/Domain bank shares.
    2. **Symmetric Top-8 Bilinear Oja Learning Rule**:
         `p_tilde_i = p_i / sum_{k in Top8} p_k`
         `dH_ij = lr * reward * (p_tilde_i * p_tilde_j - H_ij * (p_tilde_i^2 + p_tilde_j^2) / 2)`
       Strictly symmetric (`H_ij == H_ji`) with zero diagonal, learning both intra-bank and
       cross-bank (`Base <-> Domain`) co-activations.
    3. **Homeostatic Firing-Rate Fatigue**:
       Tracks an exponential moving firing rate `firing_ema[l, e]` across all `256` experts.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from native.native_engine import HAS_MLX_METAL
from native.router.config import ThalamicRouterV2Config

if HAS_MLX_METAL:
    import mlx.core as mx


class HebbianSynapticPlasticity:
    """
    Manages the `16 x 256 x 256` symmetric synaptic co-activation matrix `h_coact`
    (spanning both Base `0..63` and Domain `64..255` experts) and homeostatic `firing_ema`.
    """

    def __init__(self, cfg: ThalamicRouterV2Config):
        self.cfg: ThalamicRouterV2Config = cfg
        self.h_coact: np.ndarray = np.zeros(
            (self.cfg.num_layers, self.cfg.total_experts, self.cfg.total_experts),
            dtype=np.float32,
        )
        self.target_exp_rate: float = 1.0 / float(
            max(1, self.cfg.total_experts - self.cfg.base_experts)
        )
        self.firing_ema: np.ndarray = np.full(
            (self.cfg.num_layers, self.cfg.total_experts),
            self.target_exp_rate,
            dtype=np.float32,
        )
        self.hebbian_updates_applied: int = 0

        if HAS_MLX_METAL:
            self.mlx_h_coact: List["mx.array"] = [
                mx.zeros((self.cfg.total_experts, self.cfg.total_experts), dtype=mx.float16)
                for _ in range(self.cfg.num_layers)
            ]

    def sync_mlx_weights(self, layer_idx: int = -1) -> None:
        """
        Syncs `h_coact` to MLX GPU buffers without touching session state (Auditor Bug 1.4 fix).
        If `layer_idx >= 0`, syncs only that layer for fast online Oja updates.
        """
        if not HAS_MLX_METAL:
            return
        if 0 <= layer_idx < self.cfg.num_layers:
            self.mlx_h_coact[layer_idx] = mx.array(self.h_coact[layer_idx].astype(np.float16))
            mx.eval(self.mlx_h_coact[layer_idx])
        else:
            self.mlx_h_coact = [
                mx.array(self.h_coact[l_i].astype(np.float16))
                for l_i in range(self.cfg.num_layers)
            ]
            mx.eval(self.mlx_h_coact)

    def reset_fatigue(self) -> None:
        """Resets short-term homeostatic firing-rate fatigue for a new session."""
        self.firing_ema.fill(self.target_exp_rate)

    def compute_hebbian_bias_numpy(
        self,
        layer_idx: int,
        z_mod_f32: np.ndarray,
        sigma_z: float = 1.0,
        apply_fatigue: bool = False,
        s_inertia_np: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes `(T, 256)` lateral co-activation logit bias across all `0..255` experts:
          1. Blends active working-memory trace `s_inertia_np` (`0.70`) with current token
             resonance `p_init` (`0.30`), so the active Top-8 clique from step `t-1` laterally
             excites its synaptically co-wired partners at step `t`.
          2. Normalizes positive lateral drive by its peak magnitude (`max(drive, 0.015)`)
             without subtracting a cross-bank mean, so lateral facilitation never cannibalizes
             Domain Expert routing share.
        """
        z_shift = z_mod_f32 - np.max(z_mod_f32, axis=-1, keepdims=True)
        p_init = np.exp(z_shift)
        p_init = p_init / np.sum(p_init, axis=-1, keepdims=True)

        if s_inertia_np is not None:
            p_pre = 0.70 * s_inertia_np.reshape(1, self.cfg.total_experts) + 0.30 * p_init
        else:
            p_pre = p_init

        raw_hebb = p_pre @ self.h_coact[layer_idx]  # (T, 256)
        peak_drive = np.maximum(np.max(np.abs(raw_hebb), axis=-1, keepdims=True), 0.015)
        drive_scaled = raw_hebb / peak_drive
        scaled_hebb = self.cfg.lambda_hebbian * float(sigma_z) * drive_scaled

        if not apply_fatigue:
            return scaled_hebb.astype(np.float32)

        base_e = self.cfg.base_experts
        homeo_bias = np.zeros((1, self.cfg.total_experts), dtype=np.float32)
        exp_rates = self.firing_ema[layer_idx, base_e:]
        mean_rate = float(np.mean(exp_rates))
        homeo_bias[0, base_e:] = self.cfg.homeostatic_fatigue_gain * (mean_rate - exp_rates) / max(
            mean_rate, 1e-4
        )
        return (scaled_hebb + homeo_bias).astype(np.float32)

    def compute_hebbian_bias_mlx(
        self,
        layer_idx: int,
        z_mod: "mx.array",
        sigma_z: "mx.array",
        dtype: "mx.Dtype",
        s_inertia_mlx: Optional["mx.array"] = None,
    ) -> "mx.array":
        """Zero-sync on-GPU Hebbian lateral facilitation matching `compute_hebbian_bias_numpy`."""
        p_init = mx.softmax(z_mod.astype(mx.float32), axis=-1, precise=True)
        if s_inertia_mlx is not None:
            p_pre = 0.70 * s_inertia_mlx.astype(mx.float32) + 0.30 * p_init
        else:
            p_pre = p_init

        raw_hebb = p_pre @ self.mlx_h_coact[layer_idx].astype(mx.float32)
        peak_drive = mx.maximum(mx.max(mx.abs(raw_hebb), axis=-1, keepdims=True), 0.015)
        drive_scaled = raw_hebb / peak_drive
        scaled_hebb = self.cfg.lambda_hebbian * sigma_z * drive_scaled
        return scaled_hebb.astype(dtype)

    def update_firing_fatigue_numpy(self, layer_idx: int, probs_2d: np.ndarray) -> None:
        """Updates the homeostatic firing-rate EMA for all experts."""
        base_e = self.cfg.base_experts
        decay = self.cfg.fatigue_ema_decay
        exp_probs = probs_2d[:, base_e:].astype(np.float32)
        exp_sum = np.maximum(np.sum(exp_probs, axis=-1, keepdims=True), 1e-9)
        norm_exp_probs = np.mean(exp_probs / exp_sum, axis=0)
        self.firing_ema[layer_idx, base_e:] = (
            decay * self.firing_ema[layer_idx, base_e:] + (1.0 - decay) * norm_exp_probs
        )

    def apply_symmetric_oja_step_numpy(
        self,
        layer_idx: int,
        probs_1d: np.ndarray,
        topk_indices_1d: np.ndarray,
        reward_signal: float = 1.0,
        sync_mlx: bool = True,
    ) -> float:
        """
        Auditor Design #4 & Bug 1.4 Fix:
        Applies the **Symmetric Top-8 Normalized Oja Rule** across all co-activated Top-8
        experts (Base <-> Base, Base <-> Domain, and Domain <-> Domain):
          `p_tilde_i = p_i / sum_{k in Top8} p_k`
          `dH_ij = lr * reward * (p_tilde_i * p_tilde_j - H_ij * (p_tilde_i^2 + p_tilde_j^2) / 2)`
        Guarantees `H_ij == H_ji` (exact symmetry) and fixed point `2*p_i*p_j / (p_i^2 + p_j^2) in [0, 1]`,
        and immediately syncs `mlx_h_coact[layer_idx]` when `sync_mlx=True`.
        Returns Frobenius norm of the synaptic update `||dH||_F`.
        """
        lr = self.cfg.oja_learning_rate * float(reward_signal)
        active_ids = [int(x) for x in topk_indices_1d]
        raw_p = np.array([max(float(probs_1d[i]), 1e-7) for i in active_ids], dtype=np.float32)
        p_tilde = raw_p / float(np.sum(raw_p))

        h_mat = self.h_coact[layer_idx]
        delta_norm_sq = 0.0
        n_act = len(active_ids)

        for a_idx in range(n_act):
            i = active_ids[a_idx]
            pi = float(p_tilde[a_idx])
            for b_idx in range(a_idx + 1, n_act):
                j = active_ids[b_idx]
                pj = float(p_tilde[b_idx])
                old_val = 0.5 * (float(h_mat[i, j]) + float(h_mat[j, i]))
                sym_decay = 0.5 * (pi * pi + pj * pj)
                d_val = lr * (pi * pj - old_val * sym_decay)
                new_val = float(np.clip(old_val + d_val, -0.35, 0.65))
                h_mat[i, j] = new_val
                h_mat[j, i] = new_val
                delta_norm_sq += 2.0 * ((new_val - old_val) ** 2)

        self.hebbian_updates_applied += 1
        if sync_mlx and HAS_MLX_METAL:
            self.sync_mlx_weights(layer_idx=layer_idx)
        return float(math.sqrt(delta_norm_sq))

    def update_from_post_generation_trace(self, all_routed_np: np.ndarray) -> float:
        """
        Auditor Mismatch 2.3 Fix:
        Applies unsupervised Symmetric Oja updates across the routed Top-8 expert pairs
        (both Base `0..63` and Domain `64..255`) from the completed generation trace.
        """
        if all_routed_np.size == 0:
            return 0.0
        num_steps, num_layers, _ = all_routed_np.shape
        total_delta_sq = 0.0
        for l_idx in range(num_layers):
            counts = np.zeros((self.cfg.total_experts,), dtype=np.float32)
            for s_idx in range(num_steps):
                for e_id in all_routed_np[s_idx, l_idx]:
                    counts[int(e_id)] += 1.0
            probs_emp = counts / max(1.0, float(np.sum(counts)))
            top8_emp = np.argpartition(-probs_emp, kth=7)[:8].astype(np.int32)
            d_norm = self.apply_symmetric_oja_step_numpy(
                l_idx, probs_emp, top8_emp, reward_signal=0.5, sync_mlx=False
            )
            total_delta_sq += d_norm * d_norm
        if HAS_MLX_METAL:
            self.sync_mlx_weights(layer_idx=-1)
        return float(math.sqrt(total_delta_sq))
