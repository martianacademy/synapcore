#!/usr/bin/env python3
"""
Side-by-Side Interactive Chat: Direct .safetensors Model vs. Merkle-MoE System (Metal MPS + CPU)
================================================================================================
For every user prompt, generates two live-streamed responses:
  1. [DIRECT MODEL] Standard HuggingFace `OlmoeForCausalLM` loaded from flat `.safetensors`
     shards (13,197.25 MiB / 13.84 GB full model resident in RAM/VRAM).
  2. [MERKLE-MoE]   Hybrid Merkle / B+ Tree Engine (`olmoe_merkle_store`) with Pinned Shared
     Core (516.25 MiB) + INT8 LM Head (98.35 MiB) bound to Apple Metal (`mps`), 4 KB/token
     Embedding Trie lookup, on-demand 12 MiB Contiguous Expert Leaves, AND automatic
     `<tool_call>` detection + live execution + final response synthesis!
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import DynamicCache, OlmoeForCausalLM

from merkle_engine import MerkleMoEEngine, MerklePrefixKVCache, resolve_device
from synapcore_tools import (
    SYNAPCORE_TOOLS_SCHEMA,
    execute_tool_call,
    extract_tool_calls,
    synthesize_tool_answer,
)


class SideBySideComparator:
    def __init__(
        self,
        orig_model_dir: Path,
        merkle_store_dir: Path,
        expert_cache_size: int = 128,
        device: str = "auto",
    ):
        self.orig_model_dir = Path(orig_model_dir)
        self.merkle_store_dir = Path(merkle_store_dir)
        self.expert_cache_size = expert_cache_size
        self.device = resolve_device(device)

        print("=" * 78)
        print(f"INITIALIZING DUAL INFERENCE ENGINES ON DEVICE: [{str(self.device).upper()}]")
        print("=" * 78)

        # 1. Initialize Merkle-MoE Engine (with Apple Metal MPS Unified Memory Binding)
        t_m0 = time.perf_counter()
        self.merkle_engine = MerkleMoEEngine(
            self.merkle_store_dir,
            expert_cache_size=expert_cache_size,
            enable_lookahead=True,
            enable_merkle_kv=True,
            verify_merkle_on_load=True,
            evict_cold_mmap_pages=(expert_cache_size == 0),
            device=str(self.device),
        )
        self.merkle_init_sec = time.perf_counter() - t_m0
        pinned_mb = (
            self.merkle_engine.manifest["shared_core_bytes"]
            + self.merkle_engine.manifest["lm_head_int8_bytes"]
        ) / (1024**2)
        print(
            f"[1/2] Merkle-MoE Ready in {self.merkle_init_sec:.2f}s on {str(self.device).upper()} "
            f"(Pinned Core+Head: {pinned_mb:.2f} MiB | CoW Leaves Loaded: {self.merkle_engine.cow_leaves_loaded} | "
            f"Root: {self.merkle_engine.active_merkle_root[:12]}...)"
        )

        # 2. Initialize Direct Standard OlmoeForCausalLM from .safetensors shards
        t_d0 = time.perf_counter()
        print("[2/2] Loading Full Direct Model (OlmoeForCausalLM from 3 x .safetensors shards)...")
        self.direct_model = OlmoeForCausalLM.from_pretrained(
            str(self.orig_model_dir),
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        if self.device.type == "mps":
            self.direct_model.to(self.device)
        self.direct_model.eval()
        self.direct_init_sec = time.perf_counter() - t_d0
        self.direct_ram_mb = sum(
            p.numel() * p.element_size() for p in self.direct_model.parameters()
        ) / (1024**2)
        print(
            f"      Direct Model Ready in {self.direct_init_sec:.2f}s on {str(self.device).upper()} "
            f"(Full Weight RAM/VRAM: {self.direct_ram_mb:,.2f} MiB across 6.92B parameters)\n"
        )

    def format_prompt(self, prompt: str) -> List[int]:
        formatted = f"<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n"
        return self.merkle_engine.tokenizer.encode(formatted, add_special_tokens=False).ids

    @torch.inference_mode()
    def generate_direct_stream(self, input_ids: List[int], max_new_tokens: int = 128) -> dict:
        eos_id = self.merkle_engine.config.get("eos_token_id", 50279)
        stop_ids = {eos_id, 0}

        print("\n┌────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ [1] DIRECT STANDARD MODEL (.safetensors Full 13.20 GiB on {str(self.device).upper():<4})            │")
        print("└────────────────────────────────────────────────────────────────────────────┘")
        sys.stdout.write("Assistant (Direct): ")
        sys.stdout.flush()

        t0 = time.perf_counter()
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        past_kv = DynamicCache()
        out = self.direct_model(input_ids=input_tensor, past_key_values=past_kv, use_cache=True)
        if self.device.type == "mps":
            torch.mps.synchronize()
        ttft_sec = time.perf_counter() - t0

        next_token = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
        generated_ids: List[int] = []
        prev_text = ""

        if next_token not in stop_ids:
            generated_ids.append(next_token)
            curr_text = self.merkle_engine.tokenizer.decode(generated_ids)
            sys.stdout.write(curr_text[len(prev_text) :])
            sys.stdout.flush()
            prev_text = curr_text

        t_dec0 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if next_token in stop_ids:
                break
            step_in = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            out = self.direct_model(input_ids=step_in, past_key_values=past_kv, use_cache=True)
            next_token = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
            if next_token in stop_ids:
                break
            generated_ids.append(next_token)
            curr_text = self.merkle_engine.tokenizer.decode(generated_ids)
            sys.stdout.write(curr_text[len(prev_text) :])
            sys.stdout.flush()
            prev_text = curr_text

        if self.device.type == "mps":
            torch.mps.synchronize()
        decode_sec = max(1e-6, time.perf_counter() - t_dec0)
        sys.stdout.write("\n")
        sys.stdout.flush()

        dec_tokens = max(1, len(generated_ids) - 1)
        return {
            "text": prev_text.strip(),
            "token_ids": generated_ids,
            "num_tokens": len(generated_ids),
            "ttft_sec": round(ttft_sec, 4),
            "decode_tps": round(dec_tokens / decode_sec, 2),
            "weight_ram_mb": round(self.direct_ram_mb, 2),
        }

    @torch.inference_mode()
    def generate_merkle_stream(
        self, input_ids: List[int], raw_prompt: str = "", max_new_tokens: int = 128
    ) -> dict:
        eos_id = self.merkle_engine.config.get("eos_token_id", 50279)
        stop_ids = {eos_id, 0}

        print("\n┌────────────────────────────────────────────────────────────────────────────┐")
        print(f"│ [2] MERKLE-MoE ENGINE (Hybrid Merkle Store + {str(self.device).upper():<4} Unified Memory Binding)  │")
        print("└────────────────────────────────────────────────────────────────────────────┘")
        sys.stdout.write("Assistant (Merkle): ")
        sys.stdout.flush()

        self.merkle_engine.reset_telemetry()
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.merkle_engine.num_layers

        t0 = time.perf_counter()
        matched_len = 0
        if self.merkle_engine.enable_merkle_kv:
            matched_len, loaded_kv, _ = self.merkle_engine.kv_prefix_cache.match_longest_prefix(input_ids)
            if matched_len > 0:
                for l_idx in range(self.merkle_engine.num_layers):
                    kv_cache[l_idx] = loaded_kv[l_idx]

        rem_tokens = input_ids[matched_len:]
        logits = self.merkle_engine.forward_tokens(rem_tokens, kv_cache=kv_cache, start_pos=matched_len)

        if self.merkle_engine.enable_merkle_kv and len(input_ids) >= MerklePrefixKVCache.BLOCK_SIZE + 1:
            self.merkle_engine.kv_prefix_cache.save_prefix_blocks(input_ids, kv_cache)

        if self.device.type == "mps":
            torch.mps.synchronize()
        ttft_sec = time.perf_counter() - t0
        next_token = int(torch.argmax(logits, dim=-1).item())
        generated_ids: List[int] = []
        prev_text = ""

        if next_token not in stop_ids:
            generated_ids.append(next_token)
            curr_text = self.merkle_engine.tokenizer.decode(generated_ids)
            sys.stdout.write(curr_text[len(prev_text) :])
            sys.stdout.flush()
            prev_text = curr_text

        curr_pos = len(input_ids)
        t_dec0 = time.perf_counter()

        for _ in range(max_new_tokens - 1):
            if next_token in stop_ids:
                break
            logits = self.merkle_engine.forward_tokens([next_token], kv_cache=kv_cache, start_pos=curr_pos)
            next_token = int(torch.argmax(logits, dim=-1).item())
            curr_pos += 1
            if next_token in stop_ids:
                break
            generated_ids.append(next_token)
            curr_text = self.merkle_engine.tokenizer.decode(generated_ids)
            sys.stdout.write(curr_text[len(prev_text) :])
            sys.stdout.flush()
            prev_text = curr_text

        if self.device.type == "mps":
            torch.mps.synchronize()
        decode_sec = max(1e-6, time.perf_counter() - t_dec0)
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Check if SynapCore emitted a structured <tool_call> and execute it live!
        tool_calls = extract_tool_calls(prev_text)
        tool_results = []
        if tool_calls:
            for tc in tool_calls:
                res_obj = execute_tool_call(tc["name"], tc["arguments"], merkle_engine=self.merkle_engine)
                tool_results.append(res_obj)
                final_ans = synthesize_tool_answer(raw_prompt, res_obj)
                print(
                    f"⚡ [SYNAPCORE TOOL CALL EXECUTED] -> {tc['name']}({json.dumps(tc['arguments'])})\n"
                    f"   Result : {json.dumps(res_obj)}\n"
                    f"💬 Assistant (Merkle Final): {final_ans}"
                )

        dec_tokens = max(1, len(generated_ids) - 1)
        pinned_ram_mb = (
            self.merkle_engine.manifest["shared_core_bytes"]
            + self.merkle_engine.manifest["lm_head_int8_bytes"]
        ) / (1024**2)
        hot_cache_mb = self.merkle_engine.expert_cache.memory_mb()
        embed_kb = (
            len(self.merkle_engine.embed_store.token_cache) * self.merkle_engine.embed_store.row_bytes
        ) / 1024.0
        staging_mb = 12.0 if self.device.type == "mps" else 0.0
        total_ram_mb = pinned_ram_mb + hot_cache_mb + (embed_kb / 1024.0) + staging_mb

        total_reqs = self.merkle_engine.expert_cache.hits + self.merkle_engine.expert_cache.misses
        hit_rate = (100.0 * self.merkle_engine.expert_cache.hits / total_reqs) if total_reqs else 0.0
        lookahead_acc = (
            100.0
            * self.merkle_engine.lsh_predictor.lookahead_correct_experts
            / self.merkle_engine.lsh_predictor.lookahead_predictions
            if self.merkle_engine.lsh_predictor.lookahead_predictions
            else 0.0
        )

        return {
            "text": prev_text.strip(),
            "token_ids": generated_ids,
            "num_tokens": len(generated_ids),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "ttft_sec": round(ttft_sec, 4),
            "decode_tps": round(dec_tokens / decode_sec, 2),
            "pinned_ram_mb": round(pinned_ram_mb, 2),
            "hot_cache_mb": round(hot_cache_mb, 2),
            "embed_kb": round(embed_kb, 1),
            "weight_ram_mb": round(total_ram_mb, 2),
            "ram_saved_pct": round(100.0 * (1.0 - total_ram_mb / self.direct_ram_mb), 2),
            "cache_hit_rate_pct": round(hit_rate, 2),
            "lookahead_acc_pct": round(lookahead_acc, 2),
            "merkle_kv_matched_tokens": matched_len,
        }

    def run_turn(self, prompt: str, max_new_tokens: int = 128) -> Tuple[dict, dict]:
        input_ids = self.format_prompt(prompt)
        direct_res = self.generate_direct_stream(input_ids, max_new_tokens=max_new_tokens)
        merkle_res = self.generate_merkle_stream(input_ids, raw_prompt=prompt, max_new_tokens=max_new_tokens)

        min_len = min(len(direct_res["token_ids"]), len(merkle_res["token_ids"]))
        matching_tokens = sum(
            1 for a, b in zip(direct_res["token_ids"][:min_len], merkle_res["token_ids"][:min_len]) if a == b
        )
        token_match_pct = (100.0 * matching_tokens / min_len) if min_len > 0 else 100.0

        print("\n" + "─" * 78)
        print(
            f"SIDE-BY-SIDE TELEMETRY COMPARISON (Device: {str(self.device).upper()} | Prompt Tokens: {len(input_ids)})"
        )
        print("─" * 78)
        print(f"{'Metric':<34} | {'Direct (.safetensors)':<20} | {'Merkle-MoE Engine':<20}")
        print("─" * 78)
        print(
            f"{'Weight RAM / VRAM Used':<34} | {direct_res['weight_ram_mb']:>15,.2f} MiB | "
            f"{merkle_res['weight_ram_mb']:>8,.2f} MiB (-{merkle_res['ram_saved_pct']}%)"
        )
        print(
            f"{'Embedding Table in RAM':<34} | {'196.50 MiB (Full)':>20} | "
            f"{merkle_res['embed_kb']:>10.1f} KB (Trie)"
        )
        print(
            f"{'Prefill TTFT':<34} | {direct_res['ttft_sec']*1000:>17.1f} ms | "
            f"{merkle_res['ttft_sec']*1000:>17.1f} ms"
        )
        print(
            f"{'Decode Throughput':<34} | {direct_res['decode_tps']:>14.2f} tok/s | "
            f"{merkle_res['decode_tps']:>14.2f} tok/s"
        )
        print(
            f"{'Generated Tokens':<34} | {direct_res['num_tokens']:>20} | "
            f"{merkle_res['num_tokens']:>20}"
        )
        print(
            f"{'Hot-Expert Cache / Lookahead Acc':<34} | {'N/A (100% Static)':>20} | "
            f"{merkle_res['cache_hit_rate_pct']:>6.1f}% / {merkle_res['lookahead_acc_pct']:>5.1f}%"
        )
        print(
            f"{'Token Agreement (Direct vs Merkle)':<34} | {'Baseline (FP16)':>20} | "
            f"{matching_tokens}/{min_len} ({token_match_pct:.1f}%)"
        )
        print("─" * 78 + "\n")
        return direct_res, merkle_res

    def close(self):
        self.merkle_engine.close()


def print_available_tools():
    print("\n┌────────────────────────────────────────────────────────────────────────────┐")
    print("│ SYNAPCORE REGISTERED TOOLS (<tool_call> Live Sandbox + Open-Meteo API)    │")
    print("└────────────────────────────────────────────────────────────────────────────┘")
    for t in SYNAPCORE_TOOLS_SCHEMA:
        fn = t["function"]
        print(f"  • {fn['name']:<24} : {fn['description']}")
    print("\n  Example Prompts to Try Directly in Chat:")
    print("    - current temprature in chandigarh? sirf temprature batao, in short")
    print("    - What is the weather in Tokyo?")
    print("    - Calculate 1542 * 89 + 312")
    print("    - What is the stock price of NVDA?")
    print("    - Convert 100 USD to INR")
    print("    - what is ur name?\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Side-by-Side Chat: Direct OLMoE vs. Merkle-MoE Engine (Metal MPS + CPU)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path.home() / ".lmstudio/models/mlx-community/OLMoE-1B-7B-0125-Instruct"),
    )
    parser.add_argument(
        "--store-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "olmoe_merkle_store"),
    )
    parser.add_argument(
        "--cache-experts",
        type=int,
        default=128,
        help="Number of 12 MiB hot experts to cache in Metal GPU / RAM (default: 128)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per turn",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Target device ('auto' uses Apple Metal GPU 'mps' in Terminal, fallback 'cpu')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional single prompt to run non-interactively and exit",
    )
    args = parser.parse_args()

    comparator = SideBySideComparator(
        orig_model_dir=Path(args.model_dir),
        merkle_store_dir=Path(args.store_dir),
        expert_cache_size=args.cache_experts,
        device=args.device,
    )

    try:
        if args.prompt:
            comparator.run_turn(args.prompt, max_new_tokens=args.max_tokens)
            return

        print_available_tools()
        print("Interactive Dual-Chat Mode Started! Type '/tools' to list available tools, or 'exit' / 'quit' to stop.")
        while True:
            try:
                user_msg = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break
            if not user_msg:
                continue
            if user_msg.lower() in {"exit", "quit", ":q"}:
                print("Exiting chat.")
                break
            if user_msg.lower() == "/tools":
                print_available_tools()
                continue
            comparator.run_turn(user_msg, max_new_tokens=args.max_tokens)
    finally:
        comparator.close()


if __name__ == "__main__":
    main()
