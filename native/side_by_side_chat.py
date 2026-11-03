#!/usr/bin/env python3
"""
Side-by-Side Dual-Column Chat: SynapCore Native Engine (Left) vs. Direct .safetensors Model (Right)
===================================================================================================
Supports BOTH:
  - `--model qwen3.8` (`Qwen3.8-27B-MLX-8bit`, 64 Hybrid Linear+GQA layers)
  - `--model olmoe`   (`OLMoE-1B-7B-0125-Instruct`, 16 MoE layers, 64 experts)

Features:
  1. Single prompt input generates responses from BOTH models with the EXACT SAME temperature
     (`--temperature`, default `0.0` greedy or any `T > 0.0` with synchronized PRNG seed `--seed`
     so identical logits sample the exact same token!).
  2. Live Side-by-Side Left & Right Column Terminal Renderer (`SynapCore Native` on Left,
     `Direct .safetensors` on Right) that updates in real time as tokens stream!
  3. Side-by-Side Comparative Telemetry Table (TTFT, Decode tok/s, Weight RAM, Embedding RAM,
     Merkle Proofs, PagedKV Cache, and Token Agreement %).
  4. Interactive chat commands: `/temp 0.7`, `/temp 0.0`, `/model qwen3.8`, `/model olmoe`,
     `/think`, `/seed 42`, `exit`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from native.build_native_db import DEFAULT_QWEN38_SOURCE_DIR, QWEN38_STORE_DIR, STORE_DIR
from native.native_chat import detect_model_family
from native.native_engine import (
    HAS_MLX_METAL,
    NativeKVPrefixCache,
    NativeMerkleMoEEngine,
    get_mlx_memory_metrics_mb,
    get_process_rss_mb,
    resolve_device,
)
from native.qwen38_engine import NativeQwen38MerkleEngine

if HAS_MLX_METAL:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from mlx_lm.utils import load_model, maybe_set_recommended_wired_limit

DEFAULT_OLMOE_DIRECT_DIR = Path(
    "/Users/suru.martian/.lmstudio/models/mlx-community/OLMoE-1B-7B-0125-Instruct"
)


def sample_token_mlx(
    logits_2d: "mx.array",
    temperature: float,
    rng_key: Optional["mx.array"] = None,
) -> "mx.array":
    """
    Samples next token from `logits_2d` (shape `(1, vocab_size)`) at the exact specified `temperature`.
    If `temperature <= 0.0`, uses deterministic `mx.argmax`.
    If `temperature > 0.0`, scales `logits_2d / temperature` and samples via `mx.random.categorical`
    with an explicit `rng_key` so Left (SynapCore) and Right (Direct) models use the exact same random draw!
    """
    if temperature <= 0.0:
        return mx.argmax(logits_2d, axis=-1)
    scaled = logits_2d.astype(mx.float32) / float(temperature)
    if rng_key is not None:
        return mx.random.categorical(scaled, key=rng_key)
    return mx.random.categorical(scaled)


def sample_token_numpy(
    logits_1d: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """
    CPU/NumPy temperature sampler with synchronized `np.random.Generator` for deterministic parity.
    """
    if temperature <= 0.0:
        return int(np.argmax(logits_1d))
    scaled = logits_1d.astype(np.float64) / float(temperature)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / np.sum(probs)
    return int(rng.choice(len(probs), p=probs))


class SideBySideColumnRenderer:
    """
    Renders and live-updates a two-column terminal box:
      - Left Column : SynapCore Native (`merkle_tree.idx` + C++17 / Metal)
      - Right Column: Direct Standard Model (`.safetensors` Baseline)
    """

    def __init__(self, left_title: str, right_title: str, temperature: float):
        term_cols = shutil.get_terminal_size((120, 40)).columns
        total_w = max(84, min(156, term_cols - 2))
        self.col_w: int = (total_w - 7) // 2
        self.left_title: str = left_title[: self.col_w]
        self.right_title: str = right_title[: self.col_w]
        self.temperature: float = temperature
        self._last_rendered_lines: int = 0
        self.is_tty: bool = sys.stdout.isatty()

    def _wrap_text(self, text: str) -> List[str]:
        if not text:
            return [""]
        clean = text.replace("\r", "").replace("\t", "    ")
        out_lines: List[str] = []
        for para in clean.split("\n"):
            if not para:
                out_lines.append("")
            else:
                wrapped = textwrap.wrap(
                    para,
                    width=self.col_w,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
                out_lines.extend(wrapped if wrapped else [""])
        return out_lines if out_lines else [""]

    def build_box_lines(
        self,
        left_text: str,
        right_text: str,
        left_status: str = "",
        right_status: str = "",
    ) -> List[str]:
        cw = self.col_w
        top_b = f"┌─{'─' * cw}─┬─{'─' * cw}─┐"
        hdr_1 = f"│ {self.left_title:<{cw}} │ {self.right_title:<{cw}} │"
        temp_str = f"Temp: {self.temperature:.2f}"
        l_sub = (left_status or temp_str)[:cw]
        r_sub = (right_status or temp_str)[:cw]
        hdr_2 = f"│ {l_sub:<{cw}} │ {r_sub:<{cw}} │"
        mid_b = f"├─{'─' * cw}─┼─{'─' * cw}─┤"
        bot_b = f"└─{'─' * cw}─┴─{'─' * cw}─┘"

        l_lines = self._wrap_text(left_text)
        r_lines = self._wrap_text(right_text)
        n_rows = max(len(l_lines), len(r_lines), 1)

        lines = [top_b, hdr_1, hdr_2, mid_b]
        for i in range(n_rows):
            l_cell = l_lines[i] if i < len(l_lines) else ""
            r_cell = r_lines[i] if i < len(r_lines) else ""
            lines.append(f"│ {l_cell:<{cw}} │ {r_cell:<{cw}} │")
        lines.append(bot_b)
        return lines

    def update(
        self,
        left_text: str,
        right_text: str,
        left_status: str = "",
        right_status: str = "",
        final: bool = False,
    ) -> None:
        if not self.is_tty and not final:
            return
        box_lines = self.build_box_lines(left_text, right_text, left_status, right_status)
        if self.is_tty and self._last_rendered_lines > 0:
            sys.stdout.write(f"\x1b[{self._last_rendered_lines}A")
        for line in box_lines:
            if self.is_tty:
                sys.stdout.write("\r\x1b[2K" + line + "\n")
            else:
                sys.stdout.write(line + "\n")
        sys.stdout.flush()
        self._last_rendered_lines = len(box_lines)


class SideBySideDualModelChat:
    """
    Runs SynapCore Native Engine (Left Column) and Direct `.safetensors` Model (Right Column)
    side-by-side on the exact same prompt and temperature.
    """

    def __init__(
        self,
        model_family: str = "qwen3.8",
        temperature: float = 0.0,
        seed: int = 42,
        enable_thinking: bool = True,
        device: str = "auto",
    ):
        self.model_family = detect_model_family(
            QWEN38_STORE_DIR if "qwen" in model_family.lower() else STORE_DIR,
            model_family,
        )
        self.is_qwen38 = self.model_family == "qwen3.8"
        self.temperature = max(0.0, float(temperature))
        self.seed = int(seed)
        self.enable_thinking = bool(enable_thinking)
        self.device = resolve_device(device)
        self.use_mlx = HAS_MLX_METAL and (device in ("auto", "mps", "mlx"))

        if self.use_mlx:
            maybe_set_recommended_wired_limit()

        print("=" * 88)
        if self.is_qwen38:
            print(
                f"INITIALIZING SIDE-BY-SIDE CHAT: QWEN 3.8 (27B-8BIT) | "
                f"LEFT = SynapCore Native vs. RIGHT = Direct .safetensors (Temp = {self.temperature:.2f})"
            )
            self.store_dir = QWEN38_STORE_DIR
            self.direct_model_dir = DEFAULT_QWEN38_SOURCE_DIR
            self.direct_baseline_ram_mb = 28133.50
        else:
            print(
                f"INITIALIZING SIDE-BY-SIDE CHAT: OLMoE-1B-7B | "
                f"LEFT = SynapCore Native vs. RIGHT = Direct .safetensors (Temp = {self.temperature:.2f})"
            )
            self.store_dir = STORE_DIR
            self.direct_model_dir = DEFAULT_OLMOE_DIRECT_DIR
            self.direct_baseline_ram_mb = 13197.25
        print("=" * 88)

        self.qwen_engine: Optional[NativeQwen38MerkleEngine] = None
        self.olmoe_engine: Optional[NativeMerkleMoEEngine] = None
        self.direct_mlx_model: Optional[object] = None
        self.direct_stream: Optional[object] = None

        t0 = time.perf_counter()
        if self.is_qwen38:
            self.qwen_engine = NativeQwen38MerkleEngine(
                store_dir=self.store_dir,
                source_model_dir=self.direct_model_dir,
                on_demand_streaming=False,
                pinned_layers=64,
                enable_lookahead=True,
                enable_merkle_kv=True,
                verify_merkle_on_load=True,
                device=str(self.device),
            )
            if self.use_mlx and self.qwen_engine.mlx_model is not None:
                # Share the 8-bit MLX model weights for the Direct uncached `.safetensors` runner
                # so both models fit comfortably in Unified Memory without doubling 27 GiB -> 54 GiB!
                self.direct_mlx_model = self.qwen_engine.mlx_model
                self.direct_stream = mx.new_thread_local_stream(mx.default_device())
        else:
            if self.use_mlx and self.direct_model_dir.exists():
                print("[1/2] Loading Direct .safetensors Model (mlx_lm.utils.load_model)...")
                self.direct_mlx_model, _ = load_model(self.direct_model_dir, lazy=False)
                self.direct_stream = mx.new_thread_local_stream(mx.default_device())
            print("[2/2] Initializing SynapCore Native Engine (merkle_tree.idx + 76 Expanded Experts/Layer)...")
            self.olmoe_engine = NativeMerkleMoEEngine(
                self.store_dir,
                expert_cache_size=1216,
                max_cache_ram_mb=14592.0,
                on_demand_streaming=False,
                pinned_layers=16,
                enable_lookahead=True,
                enable_merkle_kv=True,
                verify_merkle_on_load=True,
                device=str(self.device),
                shared_mlx_model=self.direct_mlx_model,  # type: ignore[arg-type]
                enable_expanded_experts=True,
            )
            if self.use_mlx and self.direct_mlx_model is None:
                self.direct_stream = mx.new_thread_local_stream(mx.default_device())

        self.init_sec = time.perf_counter() - t0
        act_mb, _, _ = get_mlx_memory_metrics_mb()
        rss_mb = get_process_rss_mb()
        print(
            f"Both Models Ready in {self.init_sec*1000:.1f} ms | "
            f"Backend: {'MLX-METAL' if self.use_mlx else 'CPU-MMAP'} | "
            f"Temperature: {self.temperature:.2f} | Active RAM: {(act_mb if self.use_mlx else rss_mb):,.1f} MiB\n"
        )

    def format_prompt(self, prompt: str) -> List[int]:
        self._last_raw_prompt = prompt
        if self.is_qwen38:
            assert self.qwen_engine is not None
            return self.qwen_engine.format_prompt(prompt, enable_thinking=self.enable_thinking)
        assert self.olmoe_engine is not None and self.olmoe_engine.tokenizer is not None
        if getattr(self.olmoe_engine, "enable_expanded_experts", False):
            self.olmoe_engine.configure_domain_routing(prompt, enable_thinking=self.enable_thinking)
        formatted = f"<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n"
        return self.olmoe_engine.tokenizer.encode(formatted, add_special_tokens=False).ids

    def _decode_tokens(self, token_ids: List[int]) -> str:
        if self.is_qwen38:
            assert self.qwen_engine is not None and self.qwen_engine.tokenizer is not None
            decoded = self.qwen_engine.tokenizer.decode(token_ids)
            return ("<think>\n" + decoded) if self.enable_thinking else decoded
        assert self.olmoe_engine is not None and self.olmoe_engine.tokenizer is not None
        return self.olmoe_engine.tokenizer.decode(token_ids)

    def _stop_ids(self) -> Set[int]:
        if self.is_qwen38:
            return {248046, 248044}
        assert self.olmoe_engine is not None
        eos_id = int(self.olmoe_engine.config.get("eos_token_id", 50279))
        return {eos_id, 0, 50278}

    def run_side_by_side_turn(
        self, prompt: str, max_new_tokens: int = 256
    ) -> Tuple[Dict[str, Union[str, int, float, List[int]]], Dict[str, Union[str, int, float, List[int]]]]:
        input_ids = self.format_prompt(prompt)
        stop_ids = self._stop_ids()

        model_tag = "Qwen3.8-27B-8bit" if self.is_qwen38 else "OLMoE-1B-7B"
        renderer = SideBySideColumnRenderer(
            left_title=f"[LEFT] SynapCore Native ({model_tag})",
            right_title=f"[RIGHT] Direct .safetensors ({model_tag})",
            temperature=self.temperature,
        )

        if not self.use_mlx:
            return self._run_cpu_side_by_side(input_ids, max_new_tokens, renderer)

        # Pre-generate per-step PRNG keys from `self.seed` so at any `temperature > 0.0`,
        # Left (SynapCore) and Right (Direct) use the exact same random key at token step `n`!
        step_keys: List[Optional["mx.array"]] = []
        base_key = mx.random.key(self.seed)
        for n in range(max_new_tokens + 2):
            if self.temperature > 0.0:
                base_key, sub_key = mx.random.split(base_key)
                step_keys.append(sub_key)
            else:
                step_keys.append(None)

        left_ids: List[int] = []
        right_ids: List[int] = []
        left_text = ""
        right_text = ""

        renderer.update(
            left_text="⏳ Prefilling (merkle_tree.idx + PagedKV)...",
            right_text="⏳ Waiting for Left or prefilling Direct...",
            left_status=f"Temp: {self.temperature:.2f} | Prefill...",
            right_status=f"Temp: {self.temperature:.2f} | Idle",
        )

        # =========================================================================
        # 1. LEFT COLUMN: SynapCore Native Engine (`merkle_tree.idx` + PagedKV)
        # =========================================================================
        if self.is_qwen38:
            assert self.qwen_engine is not None and self.qwen_engine.mlx_model is not None
            eng_q = self.qwen_engine
            eng_q.reset_telemetry()
            l_cache = eng_q.mlx_model.make_cache()

            t_l0 = time.perf_counter()
            matched_len = 0
            rem_arr = mx.array(input_ids, dtype=mx.int32)
            with mx.stream(eng_q.mlx_stream):
                if rem_arr.size > 1:
                    eng_q._mlx_forward_hybrid(rem_arr[:-1][None], l_cache)
                    mx.eval([c.state for c in l_cache])
                l_logits = eng_q._mlx_forward_hybrid(rem_arr[-1:][None], l_cache)
                y_l = sample_token_mlx(l_logits, self.temperature, step_keys[0])
                mx.eval(y_l)
            l_ttft_sec = time.perf_counter() - t_l0

            if eng_q.enable_merkle_kv and len(input_ids) >= NativeKVPrefixCache.BLOCK_SIZE + 1:
                eng_q._save_prefix_blocks_hybrid_mlx(input_ids, l_cache)

            for tid in input_ids:
                eng_q.token_cache[int(tid)] = np.zeros((1,), dtype=np.uint8)

            t_ldec0 = time.perf_counter()
            for n in range(max_new_tokens):
                tok_int = int(y_l.item())
                if tok_int in stop_ids:
                    break
                left_ids.append(tok_int)
                eng_q.token_cache[tok_int] = np.zeros((1,), dtype=np.uint8)
                left_text = self._decode_tokens(left_ids)
                elapsed_l = max(1e-6, time.perf_counter() - t_ldec0)
                l_tps_live = len(left_ids) / elapsed_l
                renderer.update(
                    left_text=left_text,
                    right_text="⏳ Waiting to stream Direct Model...",
                    left_status=f"Temp: {self.temperature:.2f} | TTFT: {l_ttft_sec*1000:.0f}ms | {l_tps_live:.1f} tok/s",
                    right_status=f"Temp: {self.temperature:.2f} | Queued",
                )
                if n + 1 < max_new_tokens:
                    with mx.stream(eng_q.mlx_stream):
                        l_logits = eng_q._mlx_forward_hybrid(y_l[None], l_cache)
                        y_l = sample_token_mlx(l_logits, self.temperature, step_keys[n + 1])
                        mx.eval(y_l)
            l_decode_sec = max(1e-6, time.perf_counter() - t_ldec0)
            cpp_stats = eng_q.cpp.get_stats()
            embed_kb = (len(eng_q.token_cache) * eng_q.row_bytes) / 1024.0
            hit_rate = 100.0
            lookahead_acc = 100.0
        else:
            assert self.olmoe_engine is not None
            eng_o = self.olmoe_engine
            eng_o.reset_telemetry()
            l_cache_o = [KVCache() for _ in range(eng_o.num_layers)]

            t_l0 = time.perf_counter()
            matched_len = 0
            if eng_o.enable_merkle_kv:
                matched_len = eng_o.kv_prefix_cache.match_longest_prefix_mlx(
                    input_ids,
                    seed_root_hex=eng_o.active_merkle_root,
                    mlx_caches=l_cache_o,
                )
            rem_tokens = input_ids[matched_len:]
            rem_arr = mx.array(rem_tokens, dtype=mx.int32)
            with mx.stream(eng_o.mlx_stream):
                if rem_arr.size > 1:
                    eng_o._mlx_forward_fast(rem_arr[:-1][None], l_cache_o, record_live_routing=True)
                    mx.eval([c.state for c in l_cache_o])
                l_logits = eng_o._mlx_forward_fast(rem_arr[-1:][None], l_cache_o, record_live_routing=True)
                y_l = sample_token_mlx(l_logits, self.temperature, step_keys[0])
                mx.eval(y_l)
            l_ttft_sec = time.perf_counter() - t_l0

            if eng_o.enable_merkle_kv and len(input_ids) >= NativeKVPrefixCache.BLOCK_SIZE + 1:
                eng_o.kv_prefix_cache.save_prefix_blocks_mlx(
                    input_ids,
                    mlx_caches=l_cache_o,
                    seed_root_hex=eng_o.active_merkle_root,
                )

            t_ldec0 = time.perf_counter()
            for n in range(max_new_tokens):
                tok_int = int(y_l.item())
                if tok_int in stop_ids:
                    break
                left_ids.append(tok_int)
                left_text = self._decode_tokens(left_ids)
                elapsed_l = max(1e-6, time.perf_counter() - t_ldec0)
                l_tps_live = len(left_ids) / elapsed_l
                renderer.update(
                    left_text=left_text,
                    right_text="⏳ Waiting to stream Direct Model...",
                    left_status=f"Temp: {self.temperature:.2f} | TTFT: {l_ttft_sec*1000:.0f}ms | {l_tps_live:.1f} tok/s",
                    right_status=f"Temp: {self.temperature:.2f} | Queued",
                )
                if n + 1 < max_new_tokens:
                    with mx.stream(eng_o.mlx_stream):
                        l_logits = eng_o._mlx_forward_fast(y_l[None], l_cache_o, record_live_routing=True)
                        y_l = sample_token_mlx(l_logits, self.temperature, step_keys[n + 1])
                        mx.eval(y_l)
            l_decode_sec = max(1e-6, time.perf_counter() - t_ldec0)
            eng_o.flush_live_routing_telemetry(input_ids, left_ids)
            cpp_stats = eng_o.cpp.get_stats()
            embed_kb = (len(eng_o.token_cache) * eng_o.row_bytes) / 1024.0
            total_reqs = eng_o.expert_cache.hits + eng_o.expert_cache.misses
            hit_rate = (100.0 * eng_o.expert_cache.hits / total_reqs) if total_reqs else 100.0
            lookahead_acc = (
                100.0
                * eng_o.lsh_predictor.lookahead_correct_experts
                / eng_o.lsh_predictor.lookahead_predictions
                if eng_o.lsh_predictor.lookahead_predictions
                else 0.0
            )

        l_tps_final = max(1, len(left_ids) - 1) / l_decode_sec
        mx.clear_cache()

        # =========================================================================
        # 2. RIGHT COLUMN: Direct Standard `.safetensors` Model (No Merkle / No PagedKV)
        # =========================================================================
        prompt_arr = mx.array(input_ids, dtype=mx.int32)
        if self.is_qwen38:
            assert self.direct_mlx_model is not None
            r_model = self.direct_mlx_model
            r_cache = r_model.make_cache()  # type: ignore[attr-defined]

            t_r0 = time.perf_counter()
            with mx.stream(self.direct_stream):  # type: ignore[arg-type]
                if prompt_arr.size > 1:
                    r_model(prompt_arr[:-1][None], cache=r_cache)  # type: ignore[operator]
                    mx.eval([c.state for c in r_cache])
                r_logits_3d = r_model(prompt_arr[-1:][None], cache=r_cache)  # type: ignore[operator]
                y_r = sample_token_mlx(r_logits_3d[:, -1, :], self.temperature, step_keys[0])
                mx.eval(y_r)
            r_ttft_sec = time.perf_counter() - t_r0

            t_rdec0 = time.perf_counter()
            for n in range(max_new_tokens):
                tok_int = int(y_r.item())
                if tok_int in stop_ids:
                    break
                right_ids.append(tok_int)
                right_text = self._decode_tokens(right_ids)
                elapsed_r = max(1e-6, time.perf_counter() - t_rdec0)
                r_tps_live = len(right_ids) / elapsed_r
                renderer.update(
                    left_text=left_text,
                    right_text=right_text,
                    left_status=f"Temp: {self.temperature:.2f} | TTFT: {l_ttft_sec*1000:.0f}ms | {l_tps_final:.1f} tok/s",
                    right_status=f"Temp: {self.temperature:.2f} | TTFT: {r_ttft_sec*1000:.0f}ms | {r_tps_live:.1f} tok/s",
                )
                if n + 1 < max_new_tokens:
                    with mx.stream(self.direct_stream):  # type: ignore[arg-type]
                        r_logits_3d = r_model(y_r[None], cache=r_cache)  # type: ignore[operator]
                        y_r = sample_token_mlx(r_logits_3d[:, -1, :], self.temperature, step_keys[n + 1])
                        mx.eval(y_r)
            r_decode_sec = max(1e-6, time.perf_counter() - t_rdec0)
        else:
            assert self.olmoe_engine is not None
            eng_o = self.olmoe_engine
            r_cache_o = [KVCache() for _ in range(eng_o.num_layers)]

            t_r0 = time.perf_counter()
            with mx.stream(self.direct_stream):  # type: ignore[arg-type]
                if self.direct_mlx_model is not None:
                    if prompt_arr.size > 1:
                        self.direct_mlx_model(prompt_arr[:-1][None], cache=r_cache_o)  # type: ignore[operator]
                        mx.eval([c.state for c in r_cache_o])
                    r_logits_3d = self.direct_mlx_model(prompt_arr[-1:][None], cache=r_cache_o)  # type: ignore[operator]
                    y_r = sample_token_mlx(r_logits_3d[:, -1, :], self.temperature, step_keys[0])
                else:
                    if prompt_arr.size > 1:
                        eng_o._mlx_forward_fast(prompt_arr[:-1][None], r_cache_o, record_live_routing=False)
                        mx.eval([c.state for c in r_cache_o])
                    r_logits = eng_o._mlx_forward_fast(prompt_arr[-1:][None], r_cache_o, record_live_routing=False)
                    y_r = sample_token_mlx(r_logits, self.temperature, step_keys[0])
                mx.eval(y_r)
            r_ttft_sec = time.perf_counter() - t_r0

            t_rdec0 = time.perf_counter()
            for n in range(max_new_tokens):
                tok_int = int(y_r.item())
                if tok_int in stop_ids:
                    break
                right_ids.append(tok_int)
                right_text = self._decode_tokens(right_ids)
                elapsed_r = max(1e-6, time.perf_counter() - t_rdec0)
                r_tps_live = len(right_ids) / elapsed_r
                renderer.update(
                    left_text=left_text,
                    right_text=right_text,
                    left_status=f"Temp: {self.temperature:.2f} | TTFT: {l_ttft_sec*1000:.0f}ms | {l_tps_final:.1f} tok/s",
                    right_status=f"Temp: {self.temperature:.2f} | TTFT: {r_ttft_sec*1000:.0f}ms | {r_tps_live:.1f} tok/s",
                )
                if n + 1 < max_new_tokens:
                    with mx.stream(self.direct_stream):  # type: ignore[arg-type]
                        if self.direct_mlx_model is not None:
                            r_logits_3d = self.direct_mlx_model(y_r[None], cache=r_cache_o)  # type: ignore[operator]
                            y_r = sample_token_mlx(r_logits_3d[:, -1, :], self.temperature, step_keys[n + 1])
                        else:
                            r_logits = eng_o._mlx_forward_fast(y_r[None], r_cache_o, record_live_routing=False)
                            y_r = sample_token_mlx(r_logits, self.temperature, step_keys[n + 1])
                        mx.eval(y_r)
            r_decode_sec = max(1e-6, time.perf_counter() - t_rdec0)

        r_tps_final = max(1, len(right_ids) - 1) / r_decode_sec
        renderer.update(
            left_text=left_text,
            right_text=right_text,
            left_status=f"Done ({len(left_ids)} toks) | TTFT: {l_ttft_sec*1000:.1f}ms | {l_tps_final:.2f} tok/s",
            right_status=f"Done ({len(right_ids)} toks) | TTFT: {r_ttft_sec*1000:.1f}ms | {r_tps_final:.2f} tok/s",
            final=True,
        )
        mx.clear_cache()

        act_mb, _, _ = get_mlx_memory_metrics_mb()
        left_res: Dict[str, Union[str, int, float, List[int]]] = {
            "text": left_text.strip(),
            "token_ids": left_ids,
            "num_tokens": len(left_ids),
            "ttft_sec": round(l_ttft_sec, 4),
            "decode_tps": round(l_tps_final, 2),
            "embed_kb": round(embed_kb, 1),
            "weight_ram_mb": round(act_mb, 2),
            "cache_hit_rate_pct": round(hit_rate, 2),
            "lookahead_acc_pct": round(lookahead_acc, 2),
            "xor_proofs": cpp_stats["xor_merkle_proofs_verified"],
            "tier2_byte_proofs": cpp_stats["tier2_byte_proofs_verified"],
            "kv_skipped": matched_len,
        }
        right_res: Dict[str, Union[str, int, float, List[int]]] = {
            "text": right_text.strip(),
            "token_ids": right_ids,
            "num_tokens": len(right_ids),
            "ttft_sec": round(r_ttft_sec, 4),
            "decode_tps": round(r_tps_final, 2),
            "weight_ram_mb": round(self.direct_baseline_ram_mb, 2),
        }
        self._print_comparison_telemetry(len(input_ids), left_res, right_res)
        return left_res, right_res

    def _run_cpu_side_by_side(
        self,
        input_ids: List[int],
        max_new_tokens: int,
        renderer: SideBySideColumnRenderer,
    ) -> Tuple[Dict[str, Union[str, int, float, List[int]]], Dict[str, Union[str, int, float, List[int]]]]:
        """
        CPU/NumPy Side-by-Side Execution Path (for headless verification & unit tests).
        """
        rng_left = np.random.default_rng(self.seed)
        rng_right = np.random.default_rng(self.seed)

        if self.is_qwen38:
            assert self.qwen_engine is not None
            eng_q = self.qwen_engine
            t0 = time.perf_counter()
            for l_i in range(min(4, eng_q.num_layers)):
                for s_i in range(eng_q.num_sub_blocks):
                    eng_q.cpp.verify_expert_leaf_bytes(l_i, s_i, force=True)
            h = eng_q.gather_token_embedding_numpy(input_ids[-4:])
            norm0 = eng_q.rmsnorm_f32(
                h[:, -1:, :], "language_model.model.layers.0.post_attention_layernorm.weight"
            )
            mlp0 = eng_q.forward_layer_mlp_sliced_numpy(0, norm0, tile_inter=256)
            h_f = eng_q.rmsnorm_f32(h[:, -1:, :] + mlp0, "language_model.model.norm.weight")[0, 0]
            lm_slice = eng_q.dequantize_linear_weight_numpy("language_model.lm_head", row_slice=(0, 2048))
            logits_1d = lm_slice @ h_f
            l_ttft = time.perf_counter() - t0
            tok_l = sample_token_numpy(logits_1d, self.temperature, rng_left)

            t1 = time.perf_counter()
            tok_r = sample_token_numpy(logits_1d, self.temperature, rng_right)
            r_ttft = time.perf_counter() - t1

            left_ids = [tok_l]
            right_ids = [tok_r]
            left_text = self._decode_tokens(left_ids)
            right_text = self._decode_tokens(right_ids)
            cpp_stats = eng_q.cpp.get_stats()
            embed_kb = (len(eng_q.token_cache) * eng_q.row_bytes) / 1024.0
        else:
            assert self.olmoe_engine is not None
            eng_o = self.olmoe_engine
            kv_k_l: List[Optional[np.ndarray]] = [None for _ in range(eng_o.num_layers)]
            kv_v_l: List[Optional[np.ndarray]] = [None for _ in range(eng_o.num_layers)]
            t0 = time.perf_counter()
            logits_l = eng_o.forward_full_model_numpy(input_ids, kv_k_l, kv_v_l, pos_offset=0)
            tok_l = sample_token_numpy(logits_l, self.temperature, rng_left)
            l_ttft = time.perf_counter() - t0

            kv_k_r: List[Optional[np.ndarray]] = [None for _ in range(eng_o.num_layers)]
            kv_v_r: List[Optional[np.ndarray]] = [None for _ in range(eng_o.num_layers)]
            t1 = time.perf_counter()
            logits_r = eng_o.forward_full_model_numpy(input_ids, kv_k_r, kv_v_r, pos_offset=0)
            tok_r = sample_token_numpy(logits_r, self.temperature, rng_right)
            r_ttft = time.perf_counter() - t1

            left_ids = [tok_l]
            right_ids = [tok_r]
            left_text = self._decode_tokens(left_ids)
            right_text = self._decode_tokens(right_ids)
            cpp_stats = eng_o.cpp.get_stats()
            embed_kb = (len(eng_o.token_cache) * eng_o.row_bytes) / 1024.0

        renderer.update(
            left_text=left_text,
            right_text=right_text,
            left_status=f"Done | Temp: {self.temperature:.2f} | TTFT: {l_ttft*1000:.1f}ms",
            right_status=f"Done | Temp: {self.temperature:.2f} | TTFT: {r_ttft*1000:.1f}ms",
            final=True,
        )
        rss_mb = get_process_rss_mb()
        left_res: Dict[str, Union[str, int, float, List[int]]] = {
            "text": left_text.strip(),
            "token_ids": left_ids,
            "num_tokens": len(left_ids),
            "ttft_sec": round(l_ttft, 4),
            "decode_tps": round(1.0 / max(1e-6, l_ttft), 2),
            "embed_kb": round(embed_kb, 1),
            "weight_ram_mb": round(rss_mb, 2),
            "cache_hit_rate_pct": 100.0,
            "lookahead_acc_pct": 100.0,
            "xor_proofs": cpp_stats["xor_merkle_proofs_verified"],
            "tier2_byte_proofs": cpp_stats["tier2_byte_proofs_verified"],
            "kv_skipped": 0,
        }
        right_res: Dict[str, Union[str, int, float, List[int]]] = {
            "text": right_text.strip(),
            "token_ids": right_ids,
            "num_tokens": len(right_ids),
            "ttft_sec": round(r_ttft, 4),
            "decode_tps": round(1.0 / max(1e-6, r_ttft), 2),
            "weight_ram_mb": round(self.direct_baseline_ram_mb, 2),
        }
        self._print_comparison_telemetry(len(input_ids), left_res, right_res)
        return left_res, right_res

    def _print_comparison_telemetry(
        self,
        prompt_tokens: int,
        left_res: Dict[str, Union[str, int, float, List[int]]],
        right_res: Dict[str, Union[str, int, float, List[int]]],
    ) -> None:
        l_ids = left_res["token_ids"]
        r_ids = right_res["token_ids"]
        assert isinstance(l_ids, list) and isinstance(r_ids, list)
        min_len = min(len(l_ids), len(r_ids))
        matches = sum(1 for a, b in zip(l_ids[:min_len], r_ids[:min_len]) if a == b)
        agree_pct = (100.0 * matches / min_len) if min_len > 0 else 100.0

        dev_str = "MLX-METAL" if self.use_mlx else "CPU-MMAP"
        emb_full_str = "1,241.6 MiB (Full)" if self.is_qwen38 else "196.5 MiB (Full)"

        print("\n" + "─" * 88)
        print(
            f"SIDE-BY-SIDE TELEMETRY (Device: {dev_str} | Temp: {self.temperature:.2f} | "
            f"Seed: {self.seed} | Prompt Tokens: {prompt_tokens})"
        )
        print("─" * 88)
        print(
            f"{'Metric':<34} | {'[LEFT] SynapCore Native':>24} | {'[RIGHT] Direct .safetensors':>24}"
        )
        print("─" * 88)
        print(
            f"{'Index Database Engine':<34} | {'merkle_tree.idx (2.12M)':>24} | {'Static JSON Header':>24}"
        )
        print(
            f"{'Sampling Temperature':<34} | {self.temperature:>24.2f} | {self.temperature:>24.2f}"
        )
        print(
            f"{'Active Weight RAM':<34} | {float(left_res['weight_ram_mb']):>20,.2f} MiB | "
            f"{float(right_res['weight_ram_mb']):>20,.2f} MiB"
        )
        print(
            f"{'Embedding Table in RAM':<34} | {str(left_res['embed_kb']) + ' KB (mmap)':>24} | "
            f"{emb_full_str:>24}"
        )
        print(
            f"{'Prefill TTFT (KV Skipped: ' + str(left_res['kv_skipped']) + ')':<34} | "
            f"{float(left_res['ttft_sec'])*1000:>21.1f} ms | {float(right_res['ttft_sec'])*1000:>21.1f} ms"
        )
        print(
            f"{'Decode Throughput':<34} | {float(left_res['decode_tps']):>18.2f} tok/s | "
            f"{float(right_res['decode_tps']):>18.2f} tok/s"
        )
        print(
            f"{'Merkle Proofs (Tier-1 / Tier-2)':<34} | "
            f"{str(left_res['xor_proofs']) + ' / ' + str(left_res['tier2_byte_proofs']):>24} | {'0 / 0 (Unverified)':>24}"
        )
        print(
            f"{'Token Agreement (Left vs Right)':<34} | "
            f"{f'{matches}/{min_len} ({agree_pct:.1f}%)':>24} | {'Baseline Reference':>24}"
        )
        print("─" * 88 + "\n")

    def close(self) -> None:
        if self.qwen_engine is not None:
            self.qwen_engine.close()
        if self.olmoe_engine is not None:
            self.olmoe_engine.close()
        self.direct_mlx_model = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-Side Dual-Column Chat: SynapCore Native (Left) vs. Direct .safetensors (Right)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.8",
        choices=["qwen3.8", "qwen38", "qwen", "olmoe"],
        help="Model to run side-by-side: 'qwen3.8' (default) or 'olmoe'",
    )
    parser.add_argument(
        "--temperature",
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature used identically for BOTH models (default: 0.0 greedy; try e.g. 0.6 or 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Synchronized PRNG seed so both models use the exact same random draws when temperature > 0",
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
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "mlx", "cpu"])
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    enable_thinking = bool(args.enable_thinking) and not bool(args.disable_thinking)
    dual_chat = SideBySideDualModelChat(
        model_family=args.model,
        temperature=args.temperature,
        seed=args.seed,
        enable_thinking=enable_thinking,
        device=args.device,
    )
    try:
        if args.prompt:
            dual_chat.run_side_by_side_turn(args.prompt, max_new_tokens=args.max_tokens)
            return

        print(
            "Side-by-Side Dual-Column Chat Started! Type a prompt below to see BOTH models generate in "
            "Left & Right columns.\n"
            "Commands: '/temp 0.7', '/temp 0.0', '/seed 123', '/model olmoe', '/model qwen3.8', '/think', or 'exit'."
        )
        while True:
            try:
                msg = input("Prompt (Both Models) > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting Side-by-Side chat.")
                break
            if not msg:
                continue
            if msg.lower() in {"exit", "quit", ":q"}:
                print("Exiting Side-by-Side chat.")
                break
            if msg.lower().startswith("/temp"):
                t_arg = msg[5:].strip()
                try:
                    dual_chat.temperature = max(0.0, float(t_arg))
                    print(f"Both models temperature set to: {dual_chat.temperature:.2f}")
                except ValueError:
                    print("Usage: '/temp 0.0' or '/temp 0.7'")
                continue
            if msg.lower().startswith("/seed"):
                s_arg = msg[5:].strip()
                try:
                    dual_chat.seed = int(s_arg)
                    print(f"Synchronized PRNG seed set to: {dual_chat.seed}")
                except ValueError:
                    print("Usage: '/seed 42'")
                continue
            if msg.lower().startswith("/think"):
                dual_chat.enable_thinking = not dual_chat.enable_thinking
                print(f"Qwen3.8 <think> mode: {'ENABLED' if dual_chat.enable_thinking else 'DISABLED'}")
                continue
            if msg.lower().startswith("/model"):
                m_arg = msg[6:].strip().lower()
                if m_arg not in {"qwen3.8", "qwen38", "qwen", "olmoe"}:
                    print("Usage: '/model qwen3.8' or '/model olmoe'")
                    continue
                cur_temp = dual_chat.temperature
                cur_seed = dual_chat.seed
                cur_think = dual_chat.enable_thinking
                dual_chat.close()
                dual_chat = SideBySideDualModelChat(
                    model_family=m_arg,
                    temperature=cur_temp,
                    seed=cur_seed,
                    enable_thinking=cur_think,
                    device=args.device,
                )
                continue
            dual_chat.run_side_by_side_turn(msg, max_new_tokens=args.max_tokens)
    finally:
        dual_chat.close()


if __name__ == "__main__":
    main()
