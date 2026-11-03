# SynapCore Thalamic Router v2 — Comprehensive Audit Verification & Engineering Fix Plan

**Target Module:** [`native/thalamic_router_v2.py`](file:///Users/suru.martian/Documents/merkle-moe/native/thalamic_router_v2.py)  
**Target Store:** [`olmoe_merkle_store/`](file:///Users/suru.martian/Documents/merkle-moe/olmoe_merkle_store)  
**Audit Verdict:** **100% Validated (All 16 Auditor Findings Confirmed + Merged with 6 Architectural Upgrades)**

---

## Executive Verdict on the Auditor's Review

The auditor's review of `native/thalamic_router_v2.py` is **100% accurate, mathematically sound, and grounded in the actual code**:
- **Part I — 6 Real Bugs (Auditor):** Every single bug identified (FP16 norm overflow on Metal GPU, non-atomic state overwrite, `sub_clusters[0]` batchbroadcast bug, `mlx_has_inertia` coupling, NumPy vs. MLX numerical drift, and missing typing/lint imports) is a genuine defect in `native/thalamic_router_v2.py`.
- **Part II — 6 Docstring vs. Code Mismatches (Auditor):** Every discrepancy between docstrings/telemetry fields and the executed code path is confirmed.
- **Part III — 4 Mathematical Design Concerns (Auditor) + 6 Cortical-Thalamic Upgrades (Our Roadmap):** The auditor's 4 design observations directly explain the empirical numbers we observed (`H(base) = 0.895`, `Sys1/Sys2 = 1%/99%`, `loss: 1.7260 -> 1.7256`) and unify naturally with our 6 next-stage router upgrades.

Below is the complete, itemized engineering specification detailing **WHY** each issue happens and **HOW** we will fix it in code and mathematics.

---

## Part I: Auditor's 6 Real Bugs (`WHY` & `HOW`)

### Bug 1.1: `fp16` Squaring Overflow in `update_topdown_context_mlx` (Lines 341–343)
- **Auditor Finding:** `layer15_h_mlx` is cast to `mx.float16` *before* computing `mx.sum(last_tok_h * last_tok_h)`. When $\|\mathbf{h}^{(15)}\|_2 > \sqrt{65504} \approx 255.9$ (in practice post-RMSNorm/residual hidden states often have $\|\mathbf{h}\|_2 \sim 300\text{–}900$), `last_tok_h * last_tok_h` or its sum overflows IEEE-754 FP16 (`max = 65504`) to `inf`, silently zeroing out `last_tok_h / inf -> 0` (or producing `NaN` if any single coordinate exceeds `256`).
- **Why It Happens:** Premature `.astype(mx.float16)` prior to the quadratic L2 reduction $\sum_{d=1}^{2048} h_d^2$.
- **How to Fix:**
  1. Cast `layer15_h_mlx[:, -1:, :]` to `mx.float32` first.
  2. Compute the L2 norm in `float32`: `norm_f32 = mx.sqrt(mx.maximum(mx.sum(h_f32 * h_f32, axis=-1, keepdims=True), 1e-12))`.
  3. Normalize in `float32` and cast the unit-norm vector (`||h||_2 == 1.0`, every element $\in [-1, 1]$) to `mx.float16`:
     ```python
     last_tok_f32 = layer15_h_mlx[:, -1:, :].reshape(1, self.cfg.hidden_size).astype(mx.float32)
     norm_f32 = mx.sqrt(mx.maximum(mx.sum(last_tok_f32 * last_tok_f32, axis=-1, keepdims=True), 1e-12))
     self.mlx_last_layer15_h = (last_tok_f32 / norm_f32).astype(mx.float16)
     ```

### Bug 1.2: Silent Overwrite of Trained State & Non-Atomic `np.savez` (Lines 188–210, 250–255)
- **Auditor Finding:** In `_load_or_initialize_persistent_weights()`, any load error or shape mismatch hits `except Exception: pass` and falls through to `save_persistent_state()`, silently overwriting a previously trained `thalamic_router_v2_state.npz`. Furthermore, `np.savez(self.state_npz_path, ...)` writes directly in-place rather than atomically.
- **Why It Happens:** Broad exception swallowing coupled with unconditional `save_persistent_state()` at the end of initialization, plus `np.savez`'s non-atomic file truncation.
- **How to Fix:**
  1. If `self.state_npz_path.exists()` fails to load or has a shape mismatch, **log a clear warning, create a timestamped backup copy** (`thalamic_router_v2_state.corrupt.<ts>.npz.bak`), and raise a `RuntimeError` unless `allow_reinit_on_mismatch=True` is explicitly passed.
  2. In `save_persistent_state()`, write to a temporary file `tmp_npz = self.state_npz_path.with_name("thalamic_router_v2_state.tmp.npz")` (since `np.savez` appends `.npz` if not present) and atomically replace via `os.replace(tmp_npz, self.state_npz_path)`. Do the same for `thalamic_router_v2_manifest.json` via `.json.tmp` + `os.replace`.

### Bug 1.3: `sub_clusters[0]` Broadcast Across Entire Batch in `train_v2_on_sft_batch` (Line 638)
- **Auditor Finding:** `target_cluster = int(sub_clusters[0])` uses only the first sample's sub-cluster hint for all `N` samples in the batch, training every sample in the batch toward sample `0`'s sub-cluster.
- **Why It Happens:** Vectorized `target_p[:, c_start:c_end]` assignment used a scalar `sub_clusters[0]` instead of per-row indexing over `sub_clusters[i]`.
- **How to Fix:**
  Construct `target_p` row-by-row for each sample $i \in \{0, \dots, N-1\}$ using its own `sub_clusters[i]`:
  ```python
  target_p = np.zeros((len(X_p), self.cfg.total_experts), dtype=np.float32)
  for i, sc_hint in enumerate(sub_clusters):
      t_cluster = int(sc_hint) % max(1, min(16, epd))
      c_start = start_e + (t_cluster * cluster_span) % epd
      c_end = min(end_e, c_start + cluster_span)
      target_p[i, c_start:c_end] = 0.65 / max(1, c_end - c_start)
      target_p[i, start_e:end_e] += 0.35 / float(epd)
  target_p /= np.maximum(np.sum(target_p, axis=-1, keepdims=True), 1e-6)
  ```

### Bug 1.4: `mlx_has_inertia` Coupled to `update_topdown_context_mlx` & Session Reset on Save
- **Auditor Finding:**
  1. `self.mlx_has_inertia` only becomes `True` inside `update_topdown_context_mlx()`. If `route_layer_mlx` is called standalone (or before `update_topdown_context_mlx`), inertia never activates.
  2. Calling `save_persistent_state()` invokes `_sync_mlx_buffers()`, which overwrites `self.mlx_s_inertia` and `self.mlx_last_layer15_h`, wiping active GPU session state mid-conversation.
  3. Online NumPy Oja updates to `self.h_coact[layer_idx]` do not sync to `self.mlx_h_coact[layer_idx]` until `save_persistent_state()` is called.
- **Why It Happens:** Single boolean flag `mlx_has_inertia` shared with top-down context update, and `_sync_mlx_buffers()` mixing persistent parameter sync with session state initialization.
- **How to Fix:**
  1. Replace `self.mlx_has_inertia: bool` with a per-layer step counter `self.mlx_inertia_steps: List[int] = [0] * self.cfg.num_layers` that increments inside `route_layer_mlx(layer_idx, ...)` itself right after updating `self.mlx_s_inertia[layer_idx]`.
  2. Split `_sync_mlx_buffers()` into `_sync_mlx_persistent_weights()` (updates only `mlx_u_topdown`, `mlx_w_topdown`, `mlx_h_coact` without touching `mlx_s_inertia` or `mlx_last_layer15_h`) and `reset_session_state()`.
  3. In `apply_oja_hebbian_step_numpy(layer_idx, ...)`, if `HAS_MLX_METAL and self._mlx_ready`, update `self.mlx_h_coact[layer_idx] = mx.array(self.h_coact[layer_idx].astype(np.float16))`.

### Bug 1.5: Numerical Parity Mismatch Between NumPy and MLX Paths
- **Auditor Finding:**
  1. Inertia probability floor is `1e-6` in NumPy (`np.clip(..., 1e-6, 1.0)`) vs. `1e-5` in MLX (`mx.maximum(s_prev, 1e-5)`), causing a $\gamma \ln(10) = 0.28 \times 2.3026 \approx 0.6447$ logit discrepancy on cold experts.
  2. During multi-token prefill (`T > 1`), NumPy updates `s_inertia_np` token-by-token inside a `for t_i in range(num_tokens)` loop, whereas MLX updates `mlx_s_inertia` using only `routing_weights[-1:, :]` and applies the same `s_prev` to all `T` tokens in the prefill batch.
- **Why It Happens:** Different epsilon constants (`1e-6` vs `1e-5`) and looped vs. batched prefill handling.
- **How to Fix:**
  1. Unify the inertia clamp constant to `INERTIA_EPS = 1e-5` (safe in both FP16 and FP32, since FP16 min positive normal is $6.1 \times 10^{-5}$; even better, compute `mx.log(mx.maximum(s_prev.astype(mx.float32), 1e-6))` in `float32` on both NumPy and MLX so `1e-6` never underflows FP16!).
  2. For multi-token prefill (`T > 1`), unify the exact EMA update rule between NumPy and MLX: during prefill of length $T$, compute the exponentially weighted moving average across the $T$ prompt tokens with decay weights $\alpha^{T-1-t}(1-\alpha)$ on both paths (or step-by-step on single-token decode where $T=1$), and add a numerical parity test in `tests/test_thalamic_router_v2.py` asserting `max_abs_err < 1e-3` between `route_layer_numpy` and `route_layer_mlx`.

### Bug 1.6: Missing `Set` Import & Unused Symbols (Lint / `mypy` / `pyflakes`)
- **Auditor Finding:** `Set` is used in annotations on L169 and L573 without `from typing import Set`. `Union`, `BASE_EXPERTS_PER_LAYER`, and `epd` (L215) are unused.
- **Why It Happens:** `from __future__ import annotations` defers annotation evaluation at runtime, masking the missing `Set` import.
- **How to Fix:**
  Import `Set` in `from typing import Dict, List, Optional, Set, Tuple, TypedDict` and remove unused imports/variables (`Union`, `BASE_EXPERTS_PER_LAYER`, unused `epd`), then verify with `flake8` / `mypy` / `pyflakes`.

---

## Part II: Auditor's 6 Docstring vs. Code Mismatches (`WHY` & `HOW`)

| # | Auditor Finding | Why It Happened | Exact Fix (`HOW`) |
| :--- | :--- | :--- | :--- |
| **2.1** | `mean_neuromodulator_gain` reports `1 + kappa * eta`, but routing actually adds `kappa * (eta - 0.35)` to logits. | Telemetry tracked a multiplicative gain formula while logit modulation used a centered additive offset. | Unify both code and telemetry around a single transparent metric: `domain_logit_boost = kappa * (eta - eta_baseline)` (and report `mean_domain_logit_boost` in logits + `mean_eta_gate` $\in [0, 1]$). |
| **2.2** | MLX GPU path (`route_layer_mlx`) never records Shannon entropy or `system1_count`/`system2_count`, so GPU telemetry displays `0.000` and `0%/0%`. | To avoid `mx.eval()` inside `route_layer_mlx`, entropy tensors were discarded instead of buffered lazily like `LiveRouteRingBuffer`. | Append `(entropies[-1], eta[-1])` `mx.array` handles to a lightweight `self._pending_mlx_telemetry` list during decode and evaluate them **once** inside `record_post_generation_trace()` after `mx.synchronize()`, populating `shannon_entropies`, `system1_count`, and `system2_count` on GPU with **zero** mid-decode stalls! |
| **2.3** | `record_post_generation_trace` docstring says it applies unsupervised Hebbian updates, but code only computes Jaccard overlap. | Hebbian update call was left out of the post-generation trace loop. | In `record_post_generation_trace(all_routed_np)`, compute the empirical co-occurrence frequency of each routed Top-8 pair across the generated tokens and call `apply_oja_hebbian_step_numpy()` once per layer at the end of generation. |
| **2.4** | `train_v2_on_sft_batch` docstring claims `u_topdown` is trained, but only `w_topdown` is updated; header docstring calls the function `train_v2_router_on_samples`. | Backprop step stopped at `grad_w_td` without backpropagating through `c_td = tanh(Y_r_unit @ u_topdown.T)`, and header docstring used an older method name. | Compute the chain-rule gradient wrt `u_topdown`: `grad_c = (diff @ self.w_topdown[l_idx]) * (1.0 - c_td**2)` and `grad_u_td = (grad_c.T @ Y_r_unit) / len(X_p)`, updating `self.u_topdown` with orthonormal row projection, and align the header docstring name to `train_v2_on_sft_batch`. |
| **2.5** | `loss_after` is measured *before* the last epoch's gradient step, and the reported loss (domain-mass NLL) differs from the optimized objective (KL/CE against `target_p`). | Loss was logged inside the loop at `ep == epochs - 1` prior to the parameter update, and measured only total domain mass instead of full KL divergence $D_{\text{KL}}(\text{target\_p} \parallel \text{probs\_tot})$. | Perform a post-update forward pass after epoch `epochs - 1` to measure the true post-update loss, and report **both** `ce_loss_before -> ce_loss_after` (the exact optimized objective $-\sum \text{target\_p} \log \text{probs\_tot}$) and `domain_mass_before -> domain_mass_after`. |
| **2.6** | `tokens_routed` counts `self.layer_calls // 16` instead of actual tokens routed (`num_tokens` during prefill or single-layer calls). | `self.layer_calls` incremented by `1` per call regardless of `x_flat.shape[0]`. | Track `self.tokens_routed_total += int(x_flat.shape[0])` specifically when `layer_idx == 0` (or track per-layer token counts `self.layer_token_counts[layer_idx] += num_tokens` and report `max(self.layer_token_counts)`). |

---

## Part III: Auditor's 4 Design Concerns + Our 6 Cortical-Thalamic Upgrades (`WHY` & `HOW`)

The auditor's 4 design concerns and our 6 proposed improvements address the exact same underlying mathematical dynamics. Here is how they combine into a unified **Thalamic Router v2.1 Architecture**:

### 3.1 Per-Stage Bias Magnitude Telemetry & Signal Scaling (Auditor Design #1 + Our Upgrade #5)
- **Why (The Problem):**
  - As the auditor noted, because $\mathbf{h}^{(15)}_{t-1}$ is unit-norm and $\mathbf{U}_{\text{td}}$ is orthonormal, $\|\mathbf{c}_{t-1}\|_2 \le 1$, so the top-down bias is bounded by $|\mathbf{z}_{\text{td}}| \le \beta_{\text{td}} = 0.22$ logits.
  - Similarly, the Hebbian term $\lambda_{\text{hebb}} (\mathbf{p}_{\text{init}} \mathbf{H}_{\text{coact}})$ is $\le 0.02\text{–}0.09$ logits at initialization, whereas raw router logit gaps (`max - 8th largest`) in `OLMoE-1B-7B` are typically $O(1.0\text{–}2.5)$ logits!
  - Consequently, in our test run, almost all of the Jaccard stickiness gain (`0.1895 -> 0.2358`) came from **Neural Inertia (`s_t`)**, while Top-Down and Hebbian had minimal effect.
- **How to Fix:**
  1. **Scale Alignment to Local Router Logit Std-Dev ($\sigma_z^{(l)}$):**
     Measure the per-token standard deviation of the base router logits $\sigma_z = \text{std}(\mathbf{z}_{\text{raw}}[:, :64])$ (typically $\sim 1.2\text{–}1.8$). Scale the Top-Down, Hebbian, and Inertia terms relative to $\sigma_z$ (and normalize $\mathbf{c}_{t-1}$ to unit L2 norm $\hat{\mathbf{c}}_{t-1} = \mathbf{c}_{t-1} / \|\mathbf{c}_{t-1}\|_2$ so it does not shrink by $1/\sqrt{2048}$ when projected onto 32 dimensions!).
  2. **Vertical Cortical Column Transition ($\mathbf{p}^{(l-1)}_t \to \text{Layer } l$):**
     Feed Layer $l-1$'s routing probability $\mathbf{p}^{(l-1)}_t$ into Layer $l$ via a learned cross-layer transition matrix $\mathbf{T}^{(l-1 \to l)}$ so within-token vertical pathways reinforce Top-Down context.
  3. **Built-In Ablation & Per-Stage Mean $|\text{Bias}|$ Telemetry:**
     Record `mean_abs_logit_raw`, `mean_abs_logit_inertia`, `mean_abs_logit_topdown`, `mean_abs_logit_hebbian`, and `mean_abs_logit_neuromod` in `ThalamicV2Telemetry`, and include an automated 4-way ablation benchmark (`raw` vs `+inertia` vs `+topdown` vs `+hebbian`) in `tests/test_thalamic_router_v2.py`.

### 3.2 Two-Level Hierarchical Lobe Routing & Top-8 Mass Preservation (Auditor Design #2 + Our Upgrades #2 & #3)
- **Why (The Problem):**
  - `OLMoE-1B-7B` has `norm_topk_prob = False` in `config.json`—meaning the FFN output is multiplied directly by the raw softmax probabilities $\sum_{k=1}^8 s_k \text{Expert}_k(x)$ without renormalizing $\sum_{k=1}^8 s_k = 1$.
  - When we expand a flat softmax from `64` experts to `256` experts, probability mass leaks into the `192` non-selected expanded tail experts, **shrinking the sum of the Top-8 weights $\sum_{k=1}^8 s_k$** and reducing the FFN residual norm compared to the original 64-expert model!
- **How to Fix:**
  1. **Two-Level Hierarchical Lobe Gate ($4\text{-way Macro Lobe} \times 64\text{-way Intra-Lobe}$):**
     - Compute the 64-way softmax over Base Experts (`0..63`) **independently**: $\mathbf{p}_{\text{base}} = \text{softmax}(\mathbf{z}_{0..63})$, and 64-way softmaxes inside each specialized lobe (`thinking 64..127`, `tool 128..191`, `reasoning 192..255`).
     - A 4-way **Macro Lobe Gate** $\mathbf{g}_{\text{lobe}} = [g_{\text{base}}, g_{\text{think}}, g_{\text{tool}}, g_{\text{reason}}]$ ($\sum g = 1$, with $g_{\text{base}} \in [0.78, 0.92]$ modulated by the Neuromodulator Gate $\eta_t$) scales each lobe's internal distribution.
  2. **Exact Base Top-8 Mass Conservation (`preserve_base_topk_mass=True`):**
     - Compute the reference 64-expert Top-8 probability sum $M_{\text{base8}} = \sum_{k=1}^8 p_{\text{base},(k)}$ and rescale the selected 256-expert Top-8 scores so $\sum_{k=1}^8 s^{\text{v2}}_k \equiv M_{\text{base8}}$ even when `norm_topk_prob = False`! This guarantees **zero FFN norm shrinkage** while also supporting **Nucleus (`Top-p`) Dynamic Active Masking** on ultra-confident System-1 tokens.

### 3.3 Homeostatic Z-Score Entropy Gate (Fixing `Sys1/Sys2: 1%/99%` — Our Upgrade #1)
- **Why (The Problem):**
  - In our live test, `H(base)` averaged `0.895` (`Sys1/Sys2: 1%/99%`) because a 64-way softmax over unscaled inner products naturally has normalized entropy in `[0.84, 0.94]`, well above the static threshold `tau_entropy = 0.64`.
- **How to Fix:**
  - Maintain a per-layer running EMA of entropy mean $\mu_H^{(l)}$ (initialized to `0.88`) and standard deviation $\sigma_H^{(l)}$ (initialized to `0.04`):
    $$z_H^{(l)} = \frac{H(\mathbf{p}_{\text{base}}) - \mu_H^{(l)}}{\max(\sigma_H^{(l)}, 0.01)}, \quad \eta_t = \sigma\left(\frac{z_H^{(l)} - 0.50}{0.50}\right)$$
  - Now **System 1 ($z_H < 0.50$, ~70–75% of tokens)** activates on normal/predictable tokens, and **System 2 ($z_H \ge 0.50$, ~25–30% of tokens)** activates only on genuine above-average uncertainty spikes!

### 3.4 True Layer-Hidden-State Training & Counterfactual Token-Loss Credit Assignment (Auditor Design #3 + Our Upgrade #4)
- **Why (The Problem):**
  - As the auditor pointed out, `train_v2_on_sft_batch` previously fed mean token embeddings `np.mean(p_emb, axis=0)` to all 16 layers and used `np.mean(r_emb, axis=0)` as a proxy for Layer-15's hidden state $\mathbf{h}^{(15)}_{t-1}$, while omitting inertia and entropy terms during training. Because intermediate layer hidden states $\mathbf{h}^{(l)}_t$ rotate across layers (`L0..L15`), weights trained on raw token embeddings do not match inference hidden states!
- **How to Fix:**
  1. **True Hidden-State Trajectory Extraction:**
     In `train_v2_on_sft_batch`, run a real forward pass through `eng` (`forward_full_model_numpy` with a lightweight hook or layer-by-layer pass on a short token window of the SFT sample) to capture the **actual per-layer post-attention RMSNorm hidden states $\mathbf{h}^{(l)}_t$ (`l = 0..15`) and actual final Layer-15 state $\mathbf{h}^{(15)}_{t-1}$**!
  2. **Include Full 4-Stage Forward Equations in Training:**
     Run the exact same 4 stages (Raw + Entropy Gate + Inertia + Top-Down + Hebbian) during the training forward pass so trained `u_topdown`, `w_topdown`, and `h_coact` transfer 1:1 to inference.
  3. **Counterfactual Next-Token Cross-Entropy Reward ($\Delta \mathcal{L}_{\text{CE}}$):**
     Weight the gradient update by the actual reduction in next-token prediction loss on the ground-truth response tokens $y_{t+1}$.

### 3.5 Symmetric Covariance Oja Rule + Homeostatic Fatigue Load-Balancing (Auditor Design #4 + Our Upgrade #6)
- **Why (The Problem):**
  - As the auditor noted, standard single-neuron Oja's rule $\Delta H_{ij} = \eta (p_i p_j - H_{ij} p_i^2)$ has fixed point $H_{ij}^* = p_j / p_i$, which is **asymmetric** ($H_{ij}^* \ne H_{ji}^*$) and blows up to the `0.50` clip limit whenever a rare expert $i$ ($p_i \ll p_j$) co-fires with a frequent expert $j$. Moreover, raw $p_i p_j \sim (0.05)^2 = 2.5 \times 10^{-3}$ makes $\eta p_i p_j \sim 3.7 \times 10^{-5}$ tiny per step.
- **How to Fix:**
  1. **Symmetric Top-K Normalized Covariance Oja Rule:**
     Normalize the active Top-8 scores to $\tilde{p}_i = s_i / \sum_{k \in \text{Top8}} s_k \in [0, 1]$ (so $\tilde{p}_i \sim 0.125$, making $\tilde{p}_i \tilde{p}_j \sim 0.0156$ — **6x stronger signal**), and use the **Symmetric Oja Decay** $( \tilde{p}_i^2 + \tilde{p}_j^2 ) / 2$:
     $$\Delta H^{(l)}_{ij} = \eta_{\text{oja}} \left( \tilde{p}_i \tilde{p}_j - H^{(l)}_{ij} \frac{\tilde{p}_i^2 + \tilde{p}_j^2}{2} \right), \quad i \ne j \in \text{Top-8}$$
     This guarantees exact symmetry ($H_{ij} \equiv H_{ji}$) and a bounded, well-conditioned fixed point $H_{ij}^* = \frac{2 \tilde{p}_i \tilde{p}_j}{\tilde{p}_i^2 + \tilde{p}_j^2} \in [0, 1]$ (the harmonic-quadratic similarity between co-activated experts!).
  2. **Homeostatic Firing-Rate Fatigue (Preventing Expert Starvation across `3,072` Leaves):**
     Track an exponential moving firing rate $f^{(l)}_i$ per expanded expert (`64..255`). Apply a small homeostatic excitability offset $b_{\text{homeo}, i}^{(l)} = \gamma_{\text{homeo}} (\bar{f} - f^{(l)}_i)$ so over-used experts experience slight refractory fatigue and under-used experts get explored during training.

---

## Summary Implementation Checklist (Ready to Execute on Approval)

1. **`native/thalamic_router_v2.py`**:
   - [ ] Fix Bug 1.1: `float32` L2 norm before `float16` cast in `update_topdown_context_mlx`.
   - [ ] Fix Bug 1.2: Atomic `.tmp.npz` + `os.replace()` saves and backup-on-mismatch protection.
   - [ ] Fix Bug 1.3: Per-sample `sub_clusters[i]` target matrix in `train_v2_on_sft_batch`.
   - [ ] Fix Bug 1.4: Per-layer `mlx_inertia_steps[l]`, split `_sync_mlx_persistent_weights()` from `reset_session_state()`, and sync online Oja updates to `mlx_h_coact`.
   - [ ] Fix Bug 1.5: Unified `float32` log-inertia clamp (`1e-6`) and prefill EMA parity between NumPy and MLX.
   - [ ] Fix Bug 1.6: Import `Set`, remove unused imports/variables.
   - [ ] Fix Mismatches 2.1–2.6: Lazy GPU entropy telemetry flush, post-update loss measurement, `u_topdown` chain-rule gradient update, true token counter, and post-generation Hebbian update.
   - [ ] Implement Design Upgrades 3.1–3.5: Per-stage $|\text{bias}|$ telemetry + $\sigma_z$ scaling, Top-8 mass conservation (`preserve_base_topk_mass`), Homeostatic Z-Score Entropy Gate ($\mu_H, \sigma_H$), True Layer-Hidden-State SFT training, and Symmetric Oja rule $\frac{2\tilde{p}_i\tilde{p}_j}{\tilde{p}_i^2 + \tilde{p}_j^2}$ + homeostatic load balancing.
2. **`tests/test_thalamic_router_v2.py`**:
   - [ ] Add FP16 high-norm (`||h||_2 = 1000`) overflow regression test.
   - [ ] Add NumPy vs. MLX numerical parity test (`max_abs_err < 1e-3`).
   - [ ] Add 4-stage ablation test measuring each stage's exact $|\text{bias}|$ and Jaccard contribution.
   - [ ] Add Top-8 probability mass conservation test (`norm_topk_prob=False` vs. 64-expert baseline).
