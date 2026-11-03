#!/usr/bin/env python3
"""
SynapCore Thalamic Router v2 — Component 3: Homeostatic Entropy Gate & Lobe Mass Conservation
(`native/router/neuromodulator.py`)
=============================================================================================
WHAT THIS FILE DOES:
  Implements two biological cortical-gating functions:
    1. **Locus Coeruleus Noradrenaline Uncertainty Gate (Homeostatic Z-Score Entropy)**:
       Measures the normalized Shannon entropy `H(p_base) in [0, 1]` of the 64 Frozen Base
       Experts on every token and layer. Instead of a static threshold (`0.64`) that saturates
       at 99% System-2 activation because unscaled 64-way softmax naturally sits around `H ~ 0.88`,
       this module tracks a per-layer homeostatic running mean `mu_H^{(l)}` and standard
       deviation `sigma_H^{(l)}`. System-2 (`eta_t >= 0.5`) triggers ONLY when the current token's
       entropy exceeds the layer's running baseline (`z_H = (H - mu_H) / sigma_H >= z_threshold`).
    2. **Base-64 Top-8 Probability Mass Conservation (`preserve_base_topk_mass`)**:
       In `OLMoE-1B-7B` (`norm_topk_prob = False`), expanding the softmax denominator from
       `64` experts to `256` experts leaks probability mass into the `192` non-selected tail
       experts, shrinking the sum of the Top-8 weights (`sum_{k=1}^8 s_k`) and attenuating the
       FFN output norm. This module computes the exact 64-expert Top-8 reference mass
       `M_base8 = sum_{k=1}^8 p_{base,(k)}` and rescales the selected 256-expert Top-8 weights
       to conserve `M_base8` (`0.00%` FFN norm shrinkage!).

AUDITOR FIXES INCLUDED HERE:
  - Mismatch 2.1 : Both routing and telemetry use the exact same additive logit formula
    `domain_logit_boost = kappa_neuromod * sigma_z * (eta - 0.35)` and report both
    `mean_eta_gate` and `mean_domain_logit_boost`.
  - Mismatch 2.2 : Lazy GPU telemetry buffer (`_pending_mlx_telemetry`) records `(H, z_H, eta, boost)`
    handles during MLX decode without `mx.eval` stalls and flushes them at the end of generation.
  - Design #2 : Conserves the 64-expert Top-8 weight sum when `norm_topk_prob = False`.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from native.native_engine import HAS_MLX_METAL
from native.router.config import ThalamicRouterV2Config

if HAS_MLX_METAL:
    import mlx.core as mx


class HomeostaticNeuromodulator:
    """
    Computes Homeostatic Z-Score Shannon Entropy Neuromodulation (`System 1` vs `System 2`)
    and conserves Base-64 Top-8 probability mass across both NumPy and MLX paths.
    """

    def __init__(self, cfg: ThalamicRouterV2Config):
        self.cfg: ThalamicRouterV2Config = cfg
        self.inv_log_base: float = 1.0 / float(math.log(self.cfg.base_experts))

        # Per-layer running homeostatic entropy statistics (`mu_H` and `sigma_H`)
        self.mu_h_np: np.ndarray = np.full(
            (self.cfg.num_layers,), self.cfg.init_entropy_mean, dtype=np.float32
        )
        self.sigma_h_np: np.ndarray = np.full(
            (self.cfg.num_layers,), self.cfg.init_entropy_std, dtype=np.float32
        )
        self.stats_steps_np: np.ndarray = np.zeros((self.cfg.num_layers,), dtype=np.int32)

        # Telemetry accumulators
        self.shannon_entropies: List[float] = []
        self.z_score_entropies: List[float] = []
        self.eta_gates: List[float] = []
        self.domain_logit_boosts: List[float] = []
        self.system1_count: int = 0
        self.system2_count: int = 0

        # Lazy MLX telemetry queue: list of `(entropies_1d, z_h_1d, eta_1d, boost_1d)` mx.arrays
        self._pending_mlx_telemetry: List[Tuple["mx.array", "mx.array", "mx.array", "mx.array"]] = []

    def reset_session(self) -> None:
        """Resets per-turn telemetry accumulators while preserving learned homeostatic baselines."""
        self.shannon_entropies.clear()
        self.z_score_entropies.clear()
        self.eta_gates.clear()
        self.domain_logit_boosts.clear()
        self.system1_count = 0
        self.system2_count = 0
        self._pending_mlx_telemetry.clear()

    def compute_neuromodulation_numpy(
        self,
        layer_idx: int,
        z_raw_f32: np.ndarray,
        update_homeostasis: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """
        Given `z_raw_f32` of shape `(T, 256)`:
          1. Computes base-64 probabilities `p_base` and reference Top-8 mass `base_top8_mass` `(T, 1)`.
          2. Computes normalized Shannon entropy `H(p_base) in [0, 1]`.
          3. Computes homeostatic Z-score `z_H = (H - mu_H[l]) / max(sigma_H[l], 0.015)`
             and sigmoid gate `eta = sigmoid((z_H - z_threshold) / z_temp)`.
          4. Computes `domain_logit_boost = kappa_neuromod * sigma_z * (eta - 0.35)` `(T, 1)`.
        Returns `(domain_logit_boost_2d, base_top8_mass_2d, sigma_z_scalar, mean_H, mean_zH, mean_eta)`.
        """
        base_e = self.cfg.base_experts
        z_base = z_raw_f32[:, :base_e]  # (T, 64)
        raw_std = float(np.std(z_base))
        sigma_z = float(max(raw_std, 0.25))

        # Reference 64-expert Top-8 probability mass on raw logits (Auditor Design #2 conservation)
        z_base_shift = z_base - np.max(z_base, axis=-1, keepdims=True)
        p_base_raw = np.exp(z_base_shift)
        p_base_raw = p_base_raw / np.sum(p_base_raw, axis=-1, keepdims=True)
        top8_base_part = np.partition(-p_base_raw, kth=self.cfg.top_k - 1, axis=-1)[:, : self.cfg.top_k]
        base_top8_mass = (-np.sum(top8_base_part, axis=-1, keepdims=True)).astype(np.float32)  # (T, 1)

        # Scale-invariant standardized logits for Shannon entropy gauge (works identically on raw embeddings & RMSNorm states)
        z_std_per_tok = np.maximum(np.std(z_base, axis=-1, keepdims=True), 1e-6)
        z_normed = (z_base - np.mean(z_base, axis=-1, keepdims=True)) / z_std_per_tok
        z_norm_shift = z_normed - np.max(z_normed, axis=-1, keepdims=True)
        p_ent = np.exp(z_norm_shift)
        p_ent = p_ent / np.sum(p_ent, axis=-1, keepdims=True)

        # Exact normalized Shannon entropy in [0, 1]
        entropies = (
            -np.sum(p_ent * np.log(np.clip(p_ent, 1e-9, 1.0)), axis=-1) * self.inv_log_base
        ).astype(np.float32)

        if self.stats_steps_np[layer_idx] == 0:
            self.mu_h_np[layer_idx] = float(np.mean(entropies))
            self.sigma_h_np[layer_idx] = max(float(np.std(entropies)), 0.008)

        mu_l = float(self.mu_h_np[layer_idx])
        sig_l = max(float(self.sigma_h_np[layer_idx]), 0.005)
        z_h = (entropies - mu_l) / sig_l

        # Update homeostatic running mean/std AFTER computing z_h so spikes are detected fairly
        if update_homeostasis:
            decay = self.cfg.entropy_ema_decay
            batch_mean = float(np.mean(entropies))
            batch_abs_dev = float(np.mean(np.abs(entropies - mu_l)))
            self.mu_h_np[layer_idx] = decay * mu_l + (1.0 - decay) * batch_mean
            self.sigma_h_np[layer_idx] = max(
                decay * sig_l + (1.0 - decay) * max(batch_abs_dev, 0.005), 0.005
            )
            self.stats_steps_np[layer_idx] += 1

        eta = 1.0 / (
            1.0 + np.exp(-(z_h - self.cfg.z_entropy_threshold) / self.cfg.z_entropy_temp)
        )
        domain_boost_1d = (self.cfg.kappa_neuromod * sigma_z * (eta - 0.35)).astype(np.float32)

        for t_i in range(len(entropies)):
            h_val = float(entropies[t_i])
            zh_val = float(z_h[t_i])
            eta_val = float(eta[t_i])
            b_val = float(domain_boost_1d[t_i])
            self.shannon_entropies.append(h_val)
            self.z_score_entropies.append(zh_val)
            self.eta_gates.append(eta_val)
            self.domain_logit_boosts.append(b_val)
            if zh_val >= self.cfg.z_entropy_threshold:
                self.system2_count += 1
            else:
                self.system1_count += 1

        return (
            domain_boost_1d[:, None],
            base_top8_mass,
            sigma_z,
            float(np.mean(entropies)),
            float(np.mean(z_h)),
            float(np.mean(eta)),
        )

    def compute_neuromodulation_mlx(
        self,
        layer_idx: int,
        z_raw: "mx.array",
    ) -> Tuple["mx.array", "mx.array", "mx.array"]:
        """
        Zero-Sync On-GPU Neuromodulation for Apple MLX Metal (`0` `mx.eval` calls during decode!).
        Queues `(entropies, z_h, eta, domain_boost)` handles in `_pending_mlx_telemetry` for
        post-generation flush (Fixing Auditor Mismatch 2.2).
        Returns `(domain_boost_2d, base_top8_mass_2d, sigma_z_scalar_mlx)`.
        """
        base_e = self.cfg.base_experts
        z_base_f32 = z_raw[:, :base_e].astype(mx.float32)
        mean_z = mx.mean(z_base_f32, axis=-1, keepdims=True)
        var_z = mx.mean((z_base_f32 - mean_z) * (z_base_f32 - mean_z), axis=-1, keepdims=True)
        sigma_z = mx.maximum(mx.sqrt(mx.mean(var_z)), 0.25)

        p_base = mx.softmax(z_base_f32, axis=-1, precise=True)
        top8_base_idx = mx.stop_gradient(mx.argpartition(-p_base, kth=7, axis=-1)[..., :8])
        top8_base_scores = mx.take_along_axis(p_base, top8_base_idx, axis=-1)
        base_top8_mass = mx.sum(top8_base_scores, axis=-1, keepdims=True)  # (T, 1)

        entropies = (
            -mx.sum(p_base * mx.log(mx.maximum(p_base, 1e-9)), axis=-1, keepdims=True)
            * self.inv_log_base
        )
        mu_l = float(self.mu_h_np[layer_idx])
        sig_l = max(float(self.sigma_h_np[layer_idx]), 0.012)
        z_h = (entropies - mu_l) / sig_l
        eta = mx.sigmoid((z_h - self.cfg.z_entropy_threshold) / self.cfg.z_entropy_temp)
        domain_boost = (self.cfg.kappa_neuromod * sigma_z * (eta - 0.35)).astype(z_raw.dtype)

        # Queue 1D slices lazily for single-shot evaluation after generation completes
        self._pending_mlx_telemetry.append(
            (
                entropies.reshape(-1),
                z_h.reshape(-1),
                eta.reshape(-1),
                domain_boost.astype(mx.float32).reshape(-1),
            )
        )
        return domain_boost, base_top8_mass, sigma_z

    def flush_mlx_telemetry(self) -> None:
        """
        Auditor Mismatch 2.2 Fix: Evaluates all queued MLX entropy & System-1/System-2 tensors
        in a single batched `mx.eval()` call after generation finishes!
        """
        if not HAS_MLX_METAL or not self._pending_mlx_telemetry:
            return
        ent_list = [item[0] for item in self._pending_mlx_telemetry]
        zh_list = [item[1] for item in self._pending_mlx_telemetry]
        eta_list = [item[2] for item in self._pending_mlx_telemetry]
        bst_list = [item[3] for item in self._pending_mlx_telemetry]
        self._pending_mlx_telemetry.clear()

        ent_cat = mx.concatenate(ent_list, axis=0)
        zh_cat = mx.concatenate(zh_list, axis=0)
        eta_cat = mx.concatenate(eta_list, axis=0)
        bst_cat = mx.concatenate(bst_list, axis=0)
        mx.eval(ent_cat, zh_cat, eta_cat, bst_cat)

        ent_np = np.array(ent_cat, dtype=np.float32)
        zh_np = np.array(zh_cat, dtype=np.float32)
        eta_np = np.array(eta_cat, dtype=np.float32)
        bst_np = np.array(bst_cat, dtype=np.float32)

        self.shannon_entropies.extend(float(x) for x in ent_np)
        self.z_score_entropies.extend(float(x) for x in zh_np)
        self.eta_gates.extend(float(x) for x in eta_np)
        self.domain_logit_boosts.extend(float(x) for x in bst_np)

        sys2 = int(np.sum(zh_np >= self.cfg.z_entropy_threshold))
        sys1 = int(len(zh_np) - sys2)
        self.system2_count += sys2
        self.system1_count += sys1

    def conserve_topk_mass_numpy(
        self,
        topk_scores: np.ndarray,
        base_top8_mass: np.ndarray,
        norm_topk_prob: bool,
    ) -> np.ndarray:
        """
        Auditor Design #2 Fix:
          - If `norm_topk_prob=True`, normalizes `topk_scores` to sum to `1.0`.
          - If `norm_topk_prob=False` and `preserve_base_topk_mass=True`, rescales `topk_scores`
            so `sum(topk_scores) == base_top8_mass`, preventing 256-way softmax dilution!
        """
        if norm_topk_prob:
            return topk_scores / np.maximum(np.sum(topk_scores, axis=-1, keepdims=True), 1e-9)
        if self.cfg.preserve_base_topk_mass:
            curr_sum = np.maximum(np.sum(topk_scores, axis=-1, keepdims=True), 1e-9)
            return topk_scores * (base_top8_mass / curr_sum)
        return topk_scores

    def conserve_topk_mass_mlx(
        self,
        topk_scores: "mx.array",
        base_top8_mass: "mx.array",
        norm_topk_prob: bool,
    ) -> "mx.array":
        """Zero-sync on-GPU Top-8 probability mass conservation matching `conserve_topk_mass_numpy`."""
        if norm_topk_prob:
            return topk_scores / mx.maximum(mx.sum(topk_scores, axis=-1, keepdims=True), 1e-9)
        if self.cfg.preserve_base_topk_mass:
            curr_sum = mx.maximum(mx.sum(topk_scores, axis=-1, keepdims=True), 1e-9)
            return (topk_scores.astype(mx.float32) * (base_top8_mass / curr_sum)).astype(
                topk_scores.dtype
            )
        return topk_scores
