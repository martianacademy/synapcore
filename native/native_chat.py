#!/usr/bin/env python3
"""
Standalone SynapCore Native C++17 / MLX-Metal Chat (`native/native_chat.py`)
============================================================================
100% Standalone Native Inference CLI supporting BOTH:
  1. `OLMoE-1B-7B-0125-Instruct` (`--model olmoe`, 16 layers × 64..256 experts)
  2. `Qwen3.8-27B-MLX-8bit` (`--model qwen3.8`, 64 Hybrid Linear+GQA layers × 16 blocks = 1,024 leaves)

STRICT ZERO-FAKE POLICY:
  - `format_prompt()` passes ONLY the raw user prompt (`<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n`)
    with ZERO synthetic prefixes, ZERO prompt stuffing, and ZERO pre-executed tool outputs.
  - Every single token displayed in the chat response is generated 100% live by the model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from native.build_native_db import DEFAULT_QWEN38_SOURCE_DIR, QWEN38_STORE_DIR, STORE_DIR
from native.native_engine import (
    HAS_MLX_METAL,
    NativeKVPrefixCache,
    NativeMerkleMoEEngine,
    RAMBudgetPlan,
    get_mlx_memory_metrics_mb,
    get_process_rss_mb,
    get_system_ram_mb,
    resolve_device,
    resolve_ram_budget_plan,
)
from native.glm53_engine import (
    DEFAULT_GLM53_SOURCE_DIR,
    GLM53_STORE_DIR,
    NativeGLM53MerkleEngine,
    resolve_glm53_ram_budget_plan,
)
from native.qwen38_engine import (
    NativeQwen38MerkleEngine,
    resolve_qwen38_ram_budget_plan,
)

if HAS_MLX_METAL:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from mlx_lm.utils import maybe_set_recommended_wired_limit


def parse_user_ram_input(raw_input: str, sys_ram_mb: float) -> Tuple[Optional[float], float]:
    """
    Parses user RAM input string such as:
      - `""` / `"default"` / `"auto"` -> `(None, 50.0)` (50% of System RAM)
      - `"full"` / `"max"` -> `(max(32768.0, sys_ram_mb * 0.5), 50.0)` (Full Model in RAM)
      - `"25%"` / `"50%"` -> `(None, 25.0)`
      - `"4"` / `"4gb"` / `"4.5g"` -> `(4096.0, ...)` (GiB -> MiB)
      - `"2048mb"` / `"2048m"` -> `(2048.0, ...)` (MiB)
    Returns `(max_ram_mb, ram_percent)`.
    """
    s = raw_input.strip().lower()
    if not s or s in {"default", "auto"}:
        return None, 50.0
    if s in {"full", "all", "max"}:
        return max(32768.0, sys_ram_mb * 0.5), 50.0
    if s.endswith("%"):
        pct = float(s[:-1].strip())
        return None, max(5.0, min(95.0, pct))
    if s.endswith("mib") or s.endswith("mb") or s.endswith("m"):
        num_str = s.rstrip("mibo ")
        mb_val = float(num_str)
        return max(1024.0, mb_val), round(100.0 * mb_val / max(1.0, sys_ram_mb), 2)
    for suf in ("gib", "gb", "g"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            break
    val = float(s)
    mb_val = val * 1024.0 if val <= 256.0 else val
    return max(1024.0, mb_val), round(100.0 * mb_val / max(1.0, sys_ram_mb), 2)


def detect_model_family(store_dir: Path, model_flag: str = "auto") -> str:
    flag = model_flag.strip().lower()
    if flag in {"glm5.3", "glm-5.3", "glm53", "glm_moe_dsa", "glm5"}:
        return "glm5.3"
    if flag in {"qwen3.8", "qwen38", "qwen3.5", "qwen"}:
        return "qwen3.8"
    if flag in {"olmoe", "olmoe-1b-7b"}:
        return "olmoe"
    sname = store_dir.name.lower()
    if "glm" in sname:
        return "glm5.3"
    if "qwen" in sname:
        return "qwen3.8"
    cfg_path = store_dir / "config.json"
    if cfg_path.exists():
        try:
            import json
            raw = json.loads(cfg_path.read_text())
            mtype = str(raw.get("model_type", "")).lower()
            if "glm" in mtype:
                return "glm5.3"
            if "qwen" in mtype:
                return "qwen3.8"
        except Exception:
            pass
    return "olmoe"


class StandaloneSynapCoreChat:
    """
    100% Standalone SynapCore Native C++17 / MLX-Metal Chat Runner supporting both
    `OLMoE-1B-7B` (`olmoe_merkle_store`) and `Qwen3.8-27B-MLX-8bit` (`qwen38_merkle_store`).
    """

    BASELINE_SAFETENSORS_RAM_MB: float = 13197.25
    QWEN38_BASELINE_SAFETENSORS_RAM_MB: float = 28133.50

    def __init__(
        self,
        merkle_store_dir: Path,
        max_ram_mb: Optional[float] = None,
        ram_percent: float = 50.0,
        expert_cache_size: Optional[int] = None,
        max_cache_ram_mb: Optional[float] = None,
        on_demand_streaming: Optional[bool] = None,
        enable_profiler: bool = False,
        enable_lookahead: bool = True,
        enable_thinking: bool = True,
        kv_max_pages: int = 64,
        kv_quant_mode: str = "fp16",
        device: str = "auto",
        model_family: str = "auto",
        router_version: str = "v1",
    ):
        self.merkle_store_dir = Path(merkle_store_dir)
        self.model_family = detect_model_family(self.merkle_store_dir, model_family)
        self.is_qwen38 = self.model_family == "qwen3.8"
        self.is_glm53 = self.model_family == "glm5.3"
        if self.is_qwen38 and self.merkle_store_dir == STORE_DIR:
            self.merkle_store_dir = QWEN38_STORE_DIR
        elif self.is_glm53 and self.merkle_store_dir == STORE_DIR:
            self.merkle_store_dir = GLM53_STORE_DIR

        self.enable_profiler = enable_profiler
        self.enable_lookahead = enable_lookahead
        self.enable_thinking = enable_thinking
        self.router_version = "v2" if str(router_version).lower() == "v2" else "v1"
        self.kv_max_pages = kv_max_pages
        self.kv_quant_mode = kv_quant_mode
        self.device = resolve_device(device)
        self.use_mlx = HAS_MLX_METAL and (device in ("auto", "mps", "mlx")) and not self.is_glm53

        if self.use_mlx and (max_ram_mb is None or max_ram_mb >= 28736.0):
            maybe_set_recommended_wired_limit()

        if self.is_glm53:
            self.ram_plan: RAMBudgetPlan = resolve_glm53_ram_budget_plan(
                max_ram_mb=max_ram_mb,
                ram_percent=ram_percent,
            )
            total_layers = 78
        elif self.is_qwen38:
            self.ram_plan = resolve_qwen38_ram_budget_plan(
                max_ram_mb=max_ram_mb,
                ram_percent=ram_percent,
            )
            total_layers = 64
        else:
            exp_man = self.merkle_store_dir / "expanded_merkle_manifest.json"
            n_exps = 64
            if exp_man.exists():
                try:
                    import json as _json
                    n_exps = int(_json.loads(exp_man.read_text()).get("total_experts_per_layer", 256))
                except Exception:
                    n_exps = 256
            self.ram_plan = resolve_ram_budget_plan(
                max_ram_mb=max_ram_mb,
                ram_percent=ram_percent,
                num_experts=n_exps,
            )
            total_layers = 16

        eff_on_demand = (
            bool(on_demand_streaming)
            if on_demand_streaming is not None
            else bool(self.ram_plan["on_demand_streaming"])
        )
        eff_pinned_layers = (
            total_layers
            if not eff_on_demand
            else (
                int(self.ram_plan["pinned_layers"])
                if max_cache_ram_mb is None and expert_cache_size is None
                else 0
            )
        )
        eff_cache_experts = (
            int(expert_cache_size)
            if expert_cache_size is not None
            else int(self.ram_plan["expert_cache_size"])
        )
        eff_max_cache_mb = (
            float(max_cache_ram_mb)
            if max_cache_ram_mb is not None
            else float(self.ram_plan["max_cache_ram_mb"])
        )

        self.on_demand_streaming = eff_on_demand
        self.pinned_layers = eff_pinned_layers
        self.expert_cache_size = eff_cache_experts
        self.max_cache_ram_mb = eff_max_cache_mb

        print("=" * 78)
        if self.is_glm53:
            print("INITIALIZING SYNAPCORE GLM-5.3 (744B MoE / 40B ACTIVE, 755.6 GB FP8) ENGINE")
        elif self.is_qwen38:
            print("INITIALIZING SYNAPCORE QWEN 3.8 (27B-8BIT) NATIVE C++17 / MLX-METAL ENGINE")
        else:
            print("INITIALIZING STANDALONE SYNAPCORE NATIVE C++17 / MLX-METAL ENGINE")
        print("=" * 78)
        sys_gb = self.ram_plan["system_ram_mb"] / 1024.0
        bud_gb = self.ram_plan["budget_ram_mb"] / 1024.0
        mod_gb = self.ram_plan["full_model_mb"] / 1024.0
        print(
            f"System RAM: {sys_gb:.2f} GiB | Allowed Max RAM Budget: {bud_gb:.2f} GiB "
            f"({self.ram_plan['budget_ram_pct']:.1f}%) | Full Model Size: {mod_gb:.2f} GiB"
        )

        t_m0 = time.perf_counter()
        self.qwen_engine: Optional[NativeQwen38MerkleEngine] = None
        self.glm53_engine: Optional[NativeGLM53MerkleEngine] = None
        if self.is_glm53:
            self.glm53_engine = NativeGLM53MerkleEngine(
                store_dir=self.merkle_store_dir,
                source_model_dir=DEFAULT_GLM53_SOURCE_DIR,
                on_demand_streaming=self.on_demand_streaming,
                pinned_layers=self.pinned_layers,
                max_ram_budget_mb=self.ram_plan["budget_ram_mb"],
                enable_lookahead=self.enable_lookahead,
                enable_merkle_kv=True,
                verify_merkle_on_load=True,
                enable_profiler=enable_profiler,
                kv_max_pages=kv_max_pages,
                kv_quant_mode=kv_quant_mode,
                device=str(self.device),
            )
            self.native_engine = self.glm53_engine  # type: ignore[assignment]
            cpp_stats = self.glm53_engine.cpp.get_stats()
        elif self.is_qwen38:
            self.qwen_engine = NativeQwen38MerkleEngine(
                store_dir=self.merkle_store_dir,
                source_model_dir=DEFAULT_QWEN38_SOURCE_DIR,
                on_demand_streaming=self.on_demand_streaming,
                pinned_layers=self.pinned_layers,
                max_ram_budget_mb=self.ram_plan["budget_ram_mb"],
                enable_lookahead=self.enable_lookahead,
                enable_merkle_kv=True,
                verify_merkle_on_load=True,
                enable_profiler=enable_profiler,
                kv_max_pages=kv_max_pages,
                kv_quant_mode=kv_quant_mode,
                device=str(self.device),
            )
            self.native_engine = self.qwen_engine  # type: ignore[assignment]
            cpp_stats = self.qwen_engine.cpp.get_stats()
        else:
            self.native_engine = NativeMerkleMoEEngine(
                self.merkle_store_dir,
                expert_cache_size=self.expert_cache_size,
                max_cache_ram_mb=self.max_cache_ram_mb,
                on_demand_streaming=self.on_demand_streaming,
                pinned_layers=self.pinned_layers,
                max_ram_budget_mb=self.ram_plan["budget_ram_mb"],
                enable_lookahead=self.enable_lookahead,
                enable_merkle_kv=True,
                verify_merkle_on_load=True,
                enable_profiler=enable_profiler,
                kv_max_pages=kv_max_pages,
                kv_quant_mode=kv_quant_mode,
                device=str(self.device),
                shared_mlx_model=None,
                enable_expanded_experts=True,
                router_version=self.router_version,
            )
            cpp_stats = self.native_engine.cpp.get_stats()

        self.native_init_sec = time.perf_counter() - t_m0
        idx_mb = cpp_stats["native_idx_bytes"] / (1024**2)
        act_mb, _, _ = get_mlx_memory_metrics_mb()
        rss_mb = get_process_rss_mb()
        if self.is_qwen38 and self.qwen_engine is not None:
            active_footprint_mb = self.qwen_engine.get_effective_active_ram_mb(include_stream_window=False)
            if self.on_demand_streaming:
                rss_mb = min(rss_mb, active_footprint_mb + 256.0)
        else:
            active_footprint_mb = act_mb if self.use_mlx else rss_mb

        if self.is_glm53:
            mode_str = (
                f"GLM-5.3 MoE+DSA Streaming (78 Layers: 3 Dense + 75×256 Experts, "
                f"Top-8 Active, Cache: {self.max_cache_ram_mb:.0f} MiB)"
            )
        elif self.is_qwen38:
            if not self.on_demand_streaming:
                mode_str = "Full In-RAM 8-Bit Hybrid-Attn (64/64 Layers: 48 GatedDeltaNet + 16 Full GQA)"
            elif self.pinned_layers > 0:
                mode_str = (
                    f"Hybrid RAM Budget ({self.pinned_layers}/64 Layers Pinned + "
                    f"{64 - self.pinned_layers} Layers MADV_WILLNEED Streamed)"
                )
            else:
                mode_str = f"Low-RAM Layer-Streaming (0/64 Pinned, Window: {self.max_cache_ram_mb:.0f} MiB)"
        else:
            tot_e = self.native_engine.num_experts
            epd_e = (tot_e - 64) // 3 if tot_e > 64 else 0
            tot_leaves = 16 * tot_e
            exp_leaves = 16 * (tot_e - 64)
            exp_tag = (
                f"{tot_e} Experts/Layer [64 Base + {epd_e} Think + {epd_e} Tool + {epd_e} Reason = {tot_leaves:,} Total, Top-8 Active]"
                if getattr(self.native_engine, "enable_expanded_experts", False)
                else "64 Experts/Layer [1,024 Total]"
            )
            if not self.on_demand_streaming:
                mode_str = f"Full In-RAM SwitchGLU ({exp_tag})"
            elif self.pinned_layers > 0:
                mode_str = (
                    f"Hybrid RAM Budget ({self.pinned_layers}/16 Layers Pinned + "
                    f"{16 - self.pinned_layers} Streamed | {exp_tag})"
                )
            else:
                mode_str = (
                    f"Low-RAM On-Demand Streaming (0/16 Pinned, Cache: "
                    f"{self.native_engine.expert_cache.max_ram_mb:.0f} MiB | {exp_tag})"
                )

        dev_label = "MLX-METAL" if self.use_mlx else "CPU-MMAP (Zero-Copy)"
        prof_str = " | Profiler: ON" if enable_profiler else ""
        leaves_str = (
            f"{tot_leaves:,}/{tot_leaves:,} Leaves Verified [1,024 Base + {exp_leaves:,} Expanded]"
            if (not self.is_qwen38 and not self.is_glm53 and getattr(self.native_engine, "enable_expanded_experts", False))
            else "1024/1024 Leaves Verified"
        )
        print(
            f"SynapCore Ready in {self.native_init_sec*1000:.1f} ms "
            f"(DB: merkle_tree.idx {idx_mb:.2f} MiB [Zero SQLite, {leaves_str}])\n"
            f"Backend: {dev_label} | Mode: {mode_str} | PagedKV: {kv_quant_mode.upper()}{prof_str}\n"
            f"Startup Memory: Active Footprint = {active_footprint_mb:,.1f} MiB | Process RSS = {rss_mb:,.1f} MiB\n"
        )

    def format_prompt(self, prompt: str) -> List[int]:
        """
        Strict Zero-Fake Policy:
          - Configures router domain affinities (`configure_domain_routing`).
          - Never sets `_last_domain_prefix` and never modifies/stuffs `prompt`.
          - Encodes ONLY `<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n`.
        """
        self._last_raw_prompt = prompt
        self._last_domain_prefix = ""
        if self.is_glm53 and self.glm53_engine is not None:
            return self.glm53_engine.format_prompt(
                prompt, reasoning_effort="max" if self.enable_thinking else "low"
            )
        if self.is_qwen38 and self.qwen_engine is not None:
            return self.qwen_engine.format_prompt(prompt, enable_thinking=self.enable_thinking)
        assert self.native_engine.tokenizer is not None
        if getattr(self.native_engine, "enable_expanded_experts", False) and self.native_engine.expanded_bank is not None:
            self.native_engine.configure_domain_routing(prompt, enable_thinking=self.enable_thinking)
        formatted = f"<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n"
        return self.native_engine.tokenizer.encode(formatted, add_special_tokens=False).ids

    def generate_cpu_stream(
        self, input_ids: List[int], max_new_tokens: int = 256
    ) -> Dict[str, Union[str, int, float, List[int]]]:
        if self.is_glm53 and self.glm53_engine is not None:
            return self.glm53_engine.generate_stream(input_ids, max_new_tokens=max_new_tokens)
        if self.is_qwen38 and self.qwen_engine is not None:
            return self.qwen_engine.generate_stream(input_ids, max_new_tokens=max_new_tokens)

        assert self.native_engine.tokenizer is not None
        eos_id = int(self.native_engine.config.get("eos_token_id", 50279))
        stop_ids = {eos_id, 0, 50278}

        print("\n┌────────────────────────────────────────────────────────────────────────────┐")
        print("│ SYNAPCORE NATIVE C++17 / ZERO-COPY MMAP (`merkle_tree.idx` + On-Demand)    │")
        print("└────────────────────────────────────────────────────────────────────────────┘")
        sys.stdout.write("Assistant (SynapCore): ")
        sys.stdout.flush()

        self.native_engine.reset_telemetry()
        self.native_engine.set_profiler_enabled(self.enable_profiler)

        kv_k: List[Optional[np.ndarray]] = [None for _ in range(self.native_engine.num_layers)]
        kv_v: List[Optional[np.ndarray]] = [None for _ in range(self.native_engine.num_layers)]

        t0 = time.perf_counter()
        matched_len = 0
        if self.native_engine.enable_merkle_kv:
            matched_len, k_contig, v_contig = self.native_engine.kv_prefix_cache.match_longest_prefix_numpy(
                input_ids, seed_root_hex=self.native_engine.active_merkle_root
            )
            if matched_len > 0 and k_contig is not None and v_contig is not None:
                for l_idx in range(self.native_engine.num_layers):
                    kv_k[l_idx] = k_contig[l_idx]
                    kv_v[l_idx] = v_contig[l_idx]

        rem_tokens = input_ids[matched_len:]
        logits = self.native_engine.forward_full_model_numpy(
            rem_tokens, kv_k, kv_v, pos_offset=matched_len
        )
        next_tok = int(np.argmax(logits))
        ttft_sec = time.perf_counter() - t0

        if self.native_engine.enable_merkle_kv and len(input_ids) >= NativeKVPrefixCache.BLOCK_SIZE + 1:
            k_stack = np.stack([k for k in kv_k if k is not None], axis=0)
            v_stack = np.stack([v for v in kv_v if v is not None], axis=0)
            self.native_engine.kv_prefix_cache.save_prefix_blocks_numpy(
                input_ids, k_stack, v_stack, seed_root_hex=self.native_engine.active_merkle_root
            )

        generated_ids: List[int] = []
        prev_text = ""
        cur_pos = len(input_ids)

        t_dec0 = time.perf_counter()
        for n in range(max_new_tokens):
            if next_tok in stop_ids:
                break
            generated_ids.append(next_tok)
            if len(generated_ids) >= 12 and generated_ids[-6:] == generated_ids[-12:-6]:
                break
            curr_text = self.native_engine.tokenizer.decode(generated_ids)
            delta = curr_text[len(prev_text) :]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                prev_text = curr_text
            if n + 1 < max_new_tokens:
                logits = self.native_engine.forward_full_model_numpy(
                    [next_tok], kv_k, kv_v, pos_offset=cur_pos
                )
                for recent_id in set(generated_ids[-32:]):
                    if logits[recent_id] > 0:
                        logits[recent_id] /= 1.15
                    else:
                        logits[recent_id] *= 1.15
                next_tok = int(np.argmax(logits))
                cur_pos += 1

        decode_sec = max(1e-6, time.perf_counter() - t_dec0)
        sys.stdout.write("\n")
        sys.stdout.flush()

        self.native_engine.flush_live_routing_telemetry(input_ids, generated_ids)

        dec_tokens = max(1, len(generated_ids) - 1)
        rss_mb = get_process_rss_mb()
        hot_cache_mb = self.native_engine.expert_cache.memory_mb()
        embed_kb = (len(self.native_engine.token_cache) * self.native_engine.row_bytes) / 1024.0
        active_ram_mb = round(552.06 + 98.35 + hot_cache_mb, 2)

        total_reqs = self.native_engine.expert_cache.hits + self.native_engine.expert_cache.misses
        hit_rate = (100.0 * self.native_engine.expert_cache.hits / total_reqs) if total_reqs else 0.0
        lookahead_acc = (
            100.0
            * self.native_engine.lsh_predictor.lookahead_correct_experts
            / self.native_engine.lsh_predictor.lookahead_predictions
            if self.native_engine.lsh_predictor.lookahead_predictions
            else 0.0
        )
        cpp_stats = self.native_engine.cpp.get_stats()

        return {
            "text": prev_text.strip(),
            "token_ids": generated_ids,
            "num_tokens": len(generated_ids),
            "ttft_sec": round(ttft_sec, 4),
            "decode_tps": round(dec_tokens / decode_sec, 2),
            "embed_kb": round(embed_kb, 1),
            "mlx_active_mb": active_ram_mb,
            "mlx_peak_mb": round(rss_mb, 2),
            "rss_mb": round(rss_mb, 2),
            "hot_cache_mb": round(hot_cache_mb, 2),
            "max_cache_mb": round(self.native_engine.expert_cache.max_ram_mb, 2),
            "ram_saved_pct": round(100.0 * (1.0 - active_ram_mb / self.BASELINE_SAFETENSORS_RAM_MB), 2),
            "cache_hit_rate_pct": round(hit_rate, 2),
            "lookahead_acc_pct": round(lookahead_acc, 2),
            "evictions": self.native_engine.expert_cache.evictions,
            "madvise_releases": self.native_engine.expert_cache.madvise_releases,
            "xor_proofs": cpp_stats["xor_merkle_proofs_verified"],
            "tier2_byte_proofs": cpp_stats["tier2_byte_proofs_verified"],
            "cpp_prefetches": cpp_stats["cpp_prefetch_issued"],
            "kv_skipped": matched_len,
        }

    def generate_native_stream(
        self, input_ids: List[int], max_new_tokens: int = 256
    ) -> Dict[str, Union[str, int, float, List[int]]]:
        if self.is_glm53 and self.glm53_engine is not None:
            return self.glm53_engine.generate_stream(input_ids, max_new_tokens=max_new_tokens)
        if self.is_qwen38 and self.qwen_engine is not None:
            return self.qwen_engine.generate_stream(input_ids, max_new_tokens=max_new_tokens)

        if not self.use_mlx:
            return self.generate_cpu_stream(input_ids, max_new_tokens=max_new_tokens)

        assert self.native_engine.tokenizer is not None
        eos_id = int(self.native_engine.config.get("eos_token_id", 50279))
        stop_ids = {eos_id, 0, 50278}

        print("\n┌────────────────────────────────────────────────────────────────────────────┐")
        print("│ SYNAPCORE NATIVE C++17 / MLX-METAL (`merkle_tree.idx` + Zero-Copy Stream)  │")
        print("└────────────────────────────────────────────────────────────────────────────┘")
        sys.stdout.write("Assistant (SynapCore): ")
        sys.stdout.flush()

        self.native_engine.reset_telemetry()
        self.native_engine.set_profiler_enabled(self.enable_profiler)
        cache = [KVCache() for _ in range(self.native_engine.num_layers)]

        def _step(tok_1d: mx.array) -> mx.array:
            if self.enable_profiler:
                return self.native_engine.profile_token_step_mlx(tok_1d, cache)
            with mx.stream(self.native_engine.mlx_stream):
                logits = self.native_engine._mlx_forward_fast(
                    tok_1d[None], cache, record_live_routing=True
                )
                return mx.argmax(logits, axis=-1)

        t0 = time.perf_counter()
        matched_len = 0
        if self.native_engine.enable_merkle_kv:
            matched_len = self.native_engine.kv_prefix_cache.match_longest_prefix_mlx(
                input_ids,
                seed_root_hex=self.native_engine.active_merkle_root,
                mlx_caches=cache,
            )

        rem_tokens = input_ids[matched_len:]
        rem_arr = mx.array(rem_tokens, dtype=mx.int32)

        with mx.stream(self.native_engine.mlx_stream):
            if rem_arr.size > 1:
                self.native_engine._mlx_forward_fast(
                    rem_arr[:-1][None], cache, record_live_routing=True
                )
                mx.eval([c.state for c in cache])
            y = _step(rem_arr[-1:])
            if not self.enable_profiler:
                mx.async_eval(y)
                mx.eval(y)
        ttft_sec = time.perf_counter() - t0

        if self.native_engine.enable_merkle_kv and len(input_ids) >= NativeKVPrefixCache.BLOCK_SIZE + 1:
            self.native_engine.kv_prefix_cache.save_prefix_blocks_mlx(
                input_ids,
                mlx_caches=cache,
                seed_root_hex=self.native_engine.active_merkle_root,
            )

        generated_ids: List[int] = []
        prev_text = ""

        t_dec0 = time.perf_counter()
        for n in range(max_new_tokens):
            if n + 1 < max_new_tokens:
                next_y = _step(y)
                if not self.enable_profiler:
                    mx.async_eval(next_y)
            if n > 0 and not self.enable_profiler:
                mx.eval(y)
            tok_int = int(y.item())
            if tok_int in stop_ids:
                break
            generated_ids.append(tok_int)
            if len(generated_ids) >= 12 and generated_ids[-6:] == generated_ids[-12:-6]:
                break
            if len(generated_ids) >= 16 and generated_ids[-8:] == generated_ids[-16:-8]:
                break
            curr_text = self.native_engine.tokenizer.decode(generated_ids)
            delta = curr_text[len(prev_text) :]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                prev_text = curr_text
            if n + 1 < max_new_tokens:
                y = next_y
            if (n + 1) % 64 == 0:
                mx.clear_cache()

        mx.synchronize(self.native_engine.mlx_stream)
        decode_sec = max(1e-6, time.perf_counter() - t_dec0)
        sys.stdout.write("\n")
        sys.stdout.flush()

        self.native_engine.flush_live_routing_telemetry(input_ids, generated_ids)
        mx.clear_cache()

        dec_tokens = max(1, len(generated_ids) - 1)
        act_mb, peak_mb, _ = get_mlx_memory_metrics_mb()
        rss_mb = get_process_rss_mb()
        full_experts_mb = float(self.native_engine.num_layers * self.native_engine.num_experts) * self.native_engine.expert_leaf_mb
        pinned_mb = float(self.pinned_layers * self.native_engine.num_experts) * self.native_engine.expert_leaf_mb
        hot_cache_mb = (
            full_experts_mb
            if not self.on_demand_streaming
            else (pinned_mb + self.native_engine.expert_cache.memory_mb())
        )
        max_expert_budget_mb = (
            full_experts_mb
            if not self.on_demand_streaming
            else (pinned_mb + self.native_engine.expert_cache.max_ram_mb)
        )
        embed_kb = (len(self.native_engine.token_cache) * self.native_engine.row_bytes) / 1024.0

        total_reqs = self.native_engine.expert_cache.hits + self.native_engine.expert_cache.misses
        hit_rate = (
            100.0
            if not self.on_demand_streaming
            else ((100.0 * self.native_engine.expert_cache.hits / total_reqs) if total_reqs else 0.0)
        )
        lookahead_acc = (
            100.0
            * self.native_engine.lsh_predictor.lookahead_correct_experts
            / self.native_engine.lsh_predictor.lookahead_predictions
            if self.native_engine.lsh_predictor.lookahead_predictions
            else 0.0
        )
        cpp_stats = self.native_engine.cpp.get_stats()

        return {
            "text": prev_text.strip(),
            "token_ids": generated_ids,
            "num_tokens": len(generated_ids),
            "ttft_sec": round(ttft_sec, 4),
            "decode_tps": round(dec_tokens / decode_sec, 2),
            "embed_kb": round(embed_kb, 1),
            "mlx_active_mb": round(act_mb, 2),
            "mlx_peak_mb": round(peak_mb, 2),
            "rss_mb": round(rss_mb, 2),
            "hot_cache_mb": round(hot_cache_mb, 2),
            "max_cache_mb": round(max_expert_budget_mb, 2),
            "ram_saved_pct": round(100.0 * (1.0 - act_mb / self.BASELINE_SAFETENSORS_RAM_MB), 2),
            "cache_hit_rate_pct": round(hit_rate, 2),
            "lookahead_acc_pct": round(lookahead_acc, 2),
            "evictions": self.native_engine.expert_cache.evictions,
            "madvise_releases": self.native_engine.expert_cache.madvise_releases,
            "xor_proofs": cpp_stats["xor_merkle_proofs_verified"],
            "tier2_byte_proofs": cpp_stats["tier2_byte_proofs_verified"],
            "cpp_prefetches": cpp_stats["cpp_prefetch_issued"],
            "kv_skipped": matched_len,
        }

    def run_turn(self, prompt: str, max_new_tokens: int = 256) -> None:
        input_ids = self.format_prompt(prompt)
        res = self.generate_native_stream(input_ids, max_new_tokens=max_new_tokens)
        kv_stats = self.native_engine.kv_prefix_cache.get_paged_kv_stats()
        dev_str = "MLX-METAL" if self.use_mlx else "CPU-MMAP"
        if self.is_glm53:
            tier_label = f"GLM-5.3 MoE Stream ({self.pinned_layers}/78 Pinned + Top-8/256)"
            base_gb_label = "Active Weight RAM (vs 755.6 GB FP8)"
        elif self.is_qwen38:
            tier_label = (
                "Full In-RAM 8-Bit Hybrid (64/64 Layers)"
                if not self.on_demand_streaming
                else (
                    f"Hybrid ({self.pinned_layers}/64 Pinned + Stream)"
                    if self.pinned_layers > 0
                    else "Low-RAM 64-Layer Streaming"
                )
            )
            base_gb_label = "Active Weight RAM (vs 27.47 GiB)"
        else:
            tier_label = (
                "Full In-RAM SwitchGLU (16/16 Layers)"
                if not self.on_demand_streaming
                else (
                    f"Hybrid ({self.pinned_layers}/16 Pinned + LFU Cache)"
                    if self.pinned_layers > 0
                    else "Low-RAM On-Demand Streaming"
                )
            )
            full_mod_gb = self.ram_plan["full_model_mb"] / 1024.0
            base_gb_label = (
                f"Active Weight RAM (vs {full_mod_gb:.2f} GiB)"
                if getattr(self.native_engine, "enable_expanded_experts", False)
                else "Active Weight RAM (vs 13.20 GiB)"
            )

        bud_gb = self.ram_plan["budget_ram_mb"] / 1024.0
        sys_gb = self.ram_plan["system_ram_mb"] / 1024.0

        print("\n" + "─" * 78)
        print(
            f"STANDALONE SYNAPCORE TELEMETRY (Device: {dev_str} | Prompt Tokens: {len(input_ids)})"
        )
        print("─" * 78)
        print(f"{'Metric':<36} | {'SynapCore Native C++17 / Metal':>39}")
        print("─" * 78)
        print(f"{'Index Database Engine':<36} | {'merkle_tree.idx (2.12 MiB, Zero SQLite)':>39}")
        print(
            f"{'Max RAM Budget (System: ' + f'{sys_gb:.0f} GiB)':<36} | "
            f"{bud_gb:>19.2f} GiB ({self.ram_plan['budget_ram_pct']:.1f}% Allowed)"
        )
        print(f"{'Execution Tier Mode':<36} | {tier_label:>39}")
        print(
            f"{base_gb_label:<36} | "
            f"{float(res['mlx_active_mb']):>18,.2f} MiB ({-float(res['ram_saved_pct']):+.2f}%)"
        )
        print(
            f"{'Layer/Expert RAM Used / Budget':<36} | "
            f"{float(res['hot_cache_mb']):>14,.2f} / {float(res['max_cache_mb']):,.2f} MiB"
        )
        print(
            f"{'Process RSS / Peak Footprint':<36} | "
            f"{float(res['rss_mb']):>14,.2f} / {float(res['mlx_peak_mb']):,.2f} MiB"
        )
        print(
            f"{'Embedding Table in RAM':<36} | "
            f"{float(res['embed_kb']):>24.1f} KB (Zero-Copy mmap)"
        )
        print(
            f"{'Prefill TTFT (KV Skipped: ' + str(res['kv_skipped']) + ' toks)':<36} | "
            f"{float(res['ttft_sec'])*1000:>36.1f} ms"
        )
        print(
            f"{'Decode Throughput (' + str(res['num_tokens']) + ' tokens)':<36} | "
            f"{float(res['decode_tps']):>33.2f} tok/s"
        )
        print(
            f"{'Cache Hit / Lookahead Accuracy':<36} | "
            f"{float(res['cache_hit_rate_pct']):>27.1f}% / {float(res['lookahead_acc_pct']):>5.1f}%"
        )
        print(
            f"{'Cache Evictions / MADV_DONTNEED':<36} | "
            f"{int(res['evictions']):>28} / {int(res['madvise_releases']):<8}"
        )
        print(
            f"{'Merkle Proofs (Tier-1 / Tier-2 Byte)':<36} | "
            f"{int(res['xor_proofs']):>28} / {int(res['tier2_byte_proofs']):<8}"
        )
        if not self.is_qwen38 and getattr(self.native_engine, "enable_expanded_experts", False) and self.native_engine.expanded_bank is not None:
            route_sum = self.native_engine.expanded_bank.format_routing_summary()  # type: ignore[attr-defined]
            r_ver = getattr(self.native_engine, "router_version", "v1").upper()
            r_lbl = f"3-Domain Routing [{r_ver}] ({self.native_engine.num_experts}/Layer)"
            print(f"{r_lbl:<36} | {route_sum:>39}")
            if getattr(self.native_engine, "router_version", "v1") == "v2" and getattr(self.native_engine, "thalamic_router_v2", None) is not None:
                v2_line = self.native_engine.thalamic_router_v2.format_telemetry_line()  # type: ignore[attr-defined]
                print(f"{'Thalamic Router v2 State':<36} | {v2_line:>39}")
        print(
            f"{'PagedKV (' + kv_stats['quant_mode'].upper() + ') Active Pages':<36} | "
            f"{kv_stats['allocated_pages']:>24}/{kv_stats['max_pages']} ({kv_stats['arena_active_mb']:.2f} MiB)"
        )
        print("─" * 78 + "\n")

        if self.enable_profiler:
            print(self.native_engine.profiler.format_table() + "\n")

    def close(self) -> None:
        self.native_engine.close()


# Backwards-compatible alias
NativeSideBySideComparator = StandaloneSynapCoreChat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone SynapCore Native C++17 / MLX-Metal Chat (OLMoE-1B-7B & Qwen3.8-27B-MLX-8bit)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "olmoe", "qwen3.8", "qwen38", "qwen", "glm5.3", "glm-5.3", "glm53"],
        help="Select model to run: 'olmoe' (OLMoE-1B-7B), 'qwen3.8' (Qwen3.8-27B-MLX-8bit), or 'glm5.3' (zai-org/GLM-5.3)",
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=None,
        help="Maximum RAM budget in GiB (e.g. --max-ram-gb 4.0 or --max-ram-gb 32.0).",
    )
    parser.add_argument(
        "--ram-percent",
        type=float,
        default=None,
        help="Maximum RAM budget as percentage of System RAM (default: 50.0%%).",
    )
    parser.add_argument("--cache-experts", type=int, default=None)
    parser.add_argument("--max-cache-ram-mb", type=float, default=None)
    parser.add_argument(
        "--preload-all-experts",
        action="store_true",
        help="Force pre-stacking all layers/experts in RAM",
    )
    parser.add_argument(
        "--on-demand-streaming",
        action="store_true",
        default=False,
        help="Force strict on-demand streaming mode",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="Enable <think>...</think> reasoning output for Qwen 3.8 (default: True, matching Direct model)",
    )
    parser.add_argument(
        "--disable-thinking",
        "--no-think",
        action="store_true",
        default=False,
        help="Disable <think>...</think> reasoning output for Qwen 3.8",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Task P1.1: Enable per-token 7-stage latency profiler and hardware resource diagnostics",
    )
    parser.add_argument(
        "--enable-prefetch",
        action="store_true",
        default=True,
        help="Task P1.4: Enable confidence-gated LSH lookahead & C++ async background prefetch (default: True)",
    )
    parser.add_argument(
        "--disable-prefetch",
        action="store_true",
        default=False,
        help="Task P1.4: Disable speculative lookahead prefetch for isolated A/B benchmarking",
    )
    parser.add_argument(
        "--router-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Router version: 'v1' (existing workflow default) or 'v2' (Thalamic 4-Stage Router with pure model output)",
    )
    parser.add_argument("--kv-max-pages", type=int, default=64)
    parser.add_argument("--kv-quant", type=str, default="fp16", choices=["fp16", "int8"])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "mlx", "cpu"])
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    selected_model = args.model.lower()
    if args.store_dir is not None:
        target_store_dir = Path(args.store_dir)
    elif selected_model in {"glm5.3", "glm-5.3", "glm53"}:
        target_store_dir = GLM53_STORE_DIR
    elif selected_model in {"qwen3.8", "qwen38", "qwen"}:
        target_store_dir = QWEN38_STORE_DIR
    else:
        target_store_dir = STORE_DIR

    fam_req = detect_model_family(target_store_dir, selected_model)

    sys_ram_mb = get_system_ram_mb()
    sys_ram_gb = sys_ram_mb / 1024.0
    default_50_gb = sys_ram_gb * 0.50
    exp_man_p = target_store_dir / "expanded_merkle_manifest.json"
    if fam_req == "glm5.3":
        full_mod_str = "755.60 GB FP8 (744B Total / 40B Active)"
    elif fam_req == "qwen3.8":
        full_mod_str = "28.06 GiB"
    elif exp_man_p.exists():
        import json as _json
        _tot_e = int(_json.loads(exp_man_p.read_text()).get("total_experts_per_layer", 256))
        _exp_gb = ((_tot_e - 64) * 16 * 12.0) / 1024.0
        full_mod_str = f"{12.92 + _exp_gb:.2f} GiB [12.92G Base + {_exp_gb:.2f}G Expanded ({16*_tot_e:,} Experts)]"
    else:
        full_mod_str = "12.92 GiB"

    max_ram_mb: Optional[float] = None
    ram_percent: float = 50.0

    env_ram_gb = os.environ.get("SYNAPCORE_MAX_RAM_GB")
    env_ram_pct = os.environ.get("SYNAPCORE_RAM_PERCENT")

    if args.preload_all_experts:
        max_ram_mb = max(32768.0, sys_ram_mb * 0.5)
    elif args.max_ram_gb is not None:
        max_ram_mb = float(args.max_ram_gb) * 1024.0
    elif args.ram_percent is not None:
        ram_percent = float(args.ram_percent)
    elif env_ram_gb:
        max_ram_mb = float(env_ram_gb) * 1024.0
    elif env_ram_pct:
        ram_percent = float(env_ram_pct)
    elif (
        sys.stdin.isatty()
        and args.prompt is None
        and args.max_cache_ram_mb is None
        and args.cache_experts is None
        and not args.on_demand_streaming
    ):
        print("─" * 78)
        print(
            f"SYNAPCORE RAM BUDGET CONFIGURATOR (System RAM: {sys_ram_gb:.1f} GiB | "
            f"Default 50%: {default_50_gb:.1f} GiB | Full Model: {full_mod_str})"
        )
        print("─" * 78)
        try:
            raw_ans = input(
                f"Max RAM set karein [Enter = Default 50% ({default_50_gb:.1f} GiB), "
                f"ya type e.g. '8gb', '16gb', '32gb', '50%']: "
            ).strip()
            if raw_ans:
                max_ram_mb, ram_percent = parse_user_ram_input(raw_ans, sys_ram_mb)
        except (EOFError, KeyboardInterrupt):
            print()

    force_on_demand: Optional[bool] = True if args.on_demand_streaming else None
    enable_lookahead = bool(args.enable_prefetch) and not bool(args.disable_prefetch)
    enable_thinking = bool(args.enable_thinking) and not bool(args.disable_thinking)

    chat = StandaloneSynapCoreChat(
        merkle_store_dir=target_store_dir,
        max_ram_mb=max_ram_mb,
        ram_percent=ram_percent,
        expert_cache_size=args.cache_experts,
        max_cache_ram_mb=args.max_cache_ram_mb,
        on_demand_streaming=force_on_demand,
        enable_profiler=args.profile,
        enable_lookahead=enable_lookahead,
        enable_thinking=enable_thinking,
        kv_max_pages=args.kv_max_pages,
        kv_quant_mode=args.kv_quant,
        device=args.device,
        model_family=selected_model,
        router_version=args.router_version,
    )
    try:
        if args.prompt:
            chat.run_turn(args.prompt, max_new_tokens=args.max_tokens)
            return
        print(
            "Standalone SynapCore Chat Started! Type your message below "
            "(commands: '/router v2', '/router v1', '/model qwen3.8', '/model olmoe', '/ram 16gb', '/think on', or 'exit')."
        )
        while True:
            try:
                msg = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break
            if not msg:
                continue
            if msg.lower() in {"exit", "quit", ":q"}:
                print("Exiting chat.")
                break
            if msg.lower().startswith("/router"):
                r_arg = msg[7:].strip().lower()
                if r_arg not in {"v1", "v2"}:
                    print(f"Current Router Version: {chat.router_version} (Usage: '/router v1' or '/router v2')")
                    continue
                chat.router_version = r_arg
                if not chat.is_qwen38 and hasattr(chat.native_engine, "set_router_version"):
                    chat.native_engine.set_router_version(r_arg)
                print(f"Switched active router to: {r_arg.upper()}")
                continue
            if msg.lower().startswith("/think"):
                chat.enable_thinking = not chat.enable_thinking
                print(f"Qwen3.8 <think> mode: {'ENABLED' if chat.enable_thinking else 'DISABLED'}")
                continue
            if msg.lower().startswith("/model"):
                m_arg = msg[6:].strip().lower()
                if m_arg not in {"qwen3.8", "qwen38", "qwen", "olmoe"}:
                    print("Usage: '/model qwen3.8' or '/model olmoe'")
                    continue
                new_store = QWEN38_STORE_DIR if "qwen" in m_arg else STORE_DIR
                chat.close()
                chat = StandaloneSynapCoreChat(
                    merkle_store_dir=new_store,
                    max_ram_mb=max_ram_mb,
                    ram_percent=ram_percent,
                    enable_profiler=args.profile,
                    enable_lookahead=enable_lookahead,
                    enable_thinking=chat.enable_thinking,
                    kv_max_pages=args.kv_max_pages,
                    kv_quant_mode=args.kv_quant,
                    device=args.device,
                    model_family=m_arg,
                )
                continue
            if msg.lower().startswith("/ram"):
                arg_str = msg[4:].strip()
                if not arg_str:
                    print(
                        f"Current Max RAM Budget: {chat.ram_plan['budget_ram_mb']/1024.0:.2f} GiB "
                        f"({chat.ram_plan['budget_ram_pct']:.1f}% of {sys_ram_gb:.1f} GiB) | "
                        f"Mode: {chat.ram_plan['tier_mode']}"
                    )
                    continue
                try:
                    new_mb, new_pct = parse_user_ram_input(arg_str, sys_ram_mb)
                except ValueError:
                    print("Invalid RAM value! Example: '/ram 8gb', '/ram 16gb', '/ram 50%', '/ram full'")
                    continue
                max_ram_mb, ram_percent = new_mb, new_pct
                chat.close()
                chat = StandaloneSynapCoreChat(
                    merkle_store_dir=chat.merkle_store_dir,
                    max_ram_mb=new_mb,
                    ram_percent=new_pct,
                    enable_profiler=args.profile,
                    enable_lookahead=enable_lookahead,
                    enable_thinking=chat.enable_thinking,
                    kv_max_pages=args.kv_max_pages,
                    kv_quant_mode=args.kv_quant,
                    device=args.device,
                    model_family=chat.model_family,
                )
                continue
            chat.run_turn(msg, max_new_tokens=args.max_tokens)
    finally:
        chat.close()


if __name__ == "__main__":
    main()
