#!/usr/bin/env python3
"""
Unit Tests for SynapCore Qwen 3.8 (`Qwen3.8-27B-MLX-8bit`) Support (`tests/test_qwen38_merkle_engine.py`)
========================================================================================================
Verifies:
  1. Zero-Copy `qwen38_merkle_store/merkle_tree.idx` (`SYNPIDX1`, 2.12 MiB, `64 layers × 16 sub-blocks = 1,024`
     16 KB page-aligned leaves) and C++ `synapcore_native_bin` Tier-1 + Tier-2 byte verification.
  2. `Qwen38HybridAdapter` (`models/adapters.py`) hybrid 48 Linear (`GatedDeltaNet`) + 16 Full GQA (`24 Q / 4 KV`)
     layer classification and slot indexing.
  3. Zero-Copy `.safetensors` Shard `mmap` 8-bit affine dequantization (`U32` packed weights + `BF16` scales/biases)
     for embeddings, RMSNorm, Layer 0 SwiGLU MLP, and `lm_head`.
  4. Dynamic RAM Budget Planner (`resolve_qwen38_ram_budget_plan`) across Full-In-RAM (`>= 28.06 GiB`),
     Hybrid (`16 GiB`), and Low-RAM Layer-Streaming (`4 GiB`) tiers.
  5. End-to-End `StandaloneSynapCoreChat` (`--model qwen3.8`) prompt formatting and streaming turn execution.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.adapters import Qwen38HybridAdapter, create_model_adapter
from native.build_native_db import (
    DEFAULT_QWEN38_SOURCE_DIR,
    QWEN38_STORE_DIR,
    build_qwen38_merkle_store,
)
from native.native_chat import StandaloneSynapCoreChat
from native.qwen38_engine import (
    NativeQwen38MerkleEngine,
    dequantize_mlx_affine_u8_numpy,
    resolve_qwen38_ram_budget_plan,
)


@unittest.skipUnless(
    DEFAULT_QWEN38_SOURCE_DIR.exists(),
    f"Qwen3.8-27B-MLX-8bit model not found at {DEFAULT_QWEN38_SOURCE_DIR}",
)
class TestQwen38MerkleEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_qwen38_merkle_store(DEFAULT_QWEN38_SOURCE_DIR, QWEN38_STORE_DIR)

    def test_1_qwen38_merkle_idx_and_cpp_verification(self) -> None:
        idx_path = QWEN38_STORE_DIR / "merkle_tree.idx"
        self.assertTrue(idx_path.exists())
        self.assertEqual(idx_path.stat().st_size, 2225408)

        # Ensure zero-copy symlinks are used (not 27 GiB file copies!)
        for i in range(1, 7):
            shard_link = QWEN38_STORE_DIR / f"model-0000{i}-of-00006.safetensors"
            self.assertTrue(shard_link.is_symlink(), f"{shard_link} must be a zero-copy symlink")

        bin_path = ROOT_DIR / "native" / "synapcore_native_bin"
        res_align = subprocess.run(
            [str(bin_path), str(QWEN38_STORE_DIR), "--verify-alignment"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("OK: 16KB Page Alignment", res_align.stdout)

        res_scan = subprocess.run(
            [str(bin_path), str(QWEN38_STORE_DIR), "--full-scan"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Verified 1024/1024 leaves", res_scan.stdout)

    def test_2_qwen38_hybrid_adapter_and_ram_budget(self) -> None:
        eng = NativeQwen38MerkleEngine(
            store_dir=QWEN38_STORE_DIR,
            device="cpu",
            load_mlx_weights=False,
        )
        try:
            adapter = create_model_adapter(eng.raw_config)  # type: ignore[arg-type]
            self.assertIsInstance(adapter, Qwen38HybridAdapter)
            self.assertEqual(adapter.spec.num_layers, 64)
            self.assertEqual(adapter.spec.num_experts, 16)
            self.assertEqual(adapter.spec.hidden_size, 5120)
            self.assertEqual(adapter.spec.intermediate_size, 17408)
            self.assertEqual(adapter.spec.vocab_size, 248320)

            # Verify 48 Linear Attention layers + 16 Full GQA layers
            linear_layers = [l for l in range(64) if eng.adapter.is_linear_attention_layer(l)]
            full_layers = [l for l in range(64) if not eng.adapter.is_linear_attention_layer(l)]
            self.assertEqual(len(linear_layers), 48)
            self.assertEqual(len(full_layers), 16)
            self.assertEqual(full_layers[:4], [3, 7, 11, 15])

            # Verify RAM Budget tiers for Qwen 3.8 (28.06 GiB full language model)
            plan_full = resolve_qwen38_ram_budget_plan(max_ram_mb=32768.0)
            self.assertEqual(plan_full["tier_mode"], "full_in_ram")
            self.assertFalse(plan_full["on_demand_streaming"])
            self.assertEqual(plan_full["pinned_layers"], 64)

            plan_hybrid = resolve_qwen38_ram_budget_plan(max_ram_mb=16384.0)
            self.assertEqual(plan_hybrid["tier_mode"], "hybrid_ram_budget")
            self.assertTrue(plan_hybrid["on_demand_streaming"])
            self.assertGreater(plan_hybrid["pinned_layers"], 0)
            self.assertLess(plan_hybrid["pinned_layers"], 64)

            plan_low = resolve_qwen38_ram_budget_plan(max_ram_mb=4096.0)
            self.assertEqual(plan_low["tier_mode"], "low_ram_streaming")
            self.assertTrue(plan_low["on_demand_streaming"])
            self.assertEqual(plan_low["pinned_layers"], 0)
        finally:
            eng.close()

    def test_3_zero_copy_8bit_affine_dequantization_and_layer_forward(self) -> None:
        eng = NativeQwen38MerkleEngine(
            store_dir=QWEN38_STORE_DIR,
            device="cpu",
            load_mlx_weights=False,
        )
        try:
            # 1. Verify zero-copy raw tensor view (`OWNDATA=False`)
            w_u32 = eng.slice_raw_tensor_zero_copy("language_model.model.embed_tokens.weight")
            self.assertFalse(w_u32.flags["OWNDATA"])
            self.assertEqual(w_u32.shape, (248320, 1280))  # 5120 // 4 = 1280

            # 2. Verify token embedding gather + dequantization
            prompt_ids = eng.format_prompt("Hello Qwen 3.8!", enable_thinking=False)
            self.assertGreater(len(prompt_ids), 5)
            h_emb = eng.gather_token_embedding_numpy(prompt_ids[:3])
            self.assertEqual(h_emb.shape, (1, 3, 5120))
            self.assertTrue(np.all(np.isfinite(h_emb)))

            # 3. Verify RMSNorm + sliced 8-bit SwiGLU MLP forward on Layer 0
            norm0 = eng.rmsnorm_f32(
                h_emb[:, -1:, :], "language_model.model.layers.0.post_attention_layernorm.weight"
            )
            mlp_out = eng.forward_layer_mlp_sliced_numpy(0, norm0, tile_inter=128)
            self.assertEqual(mlp_out.shape, (1, 1, 5120))
            self.assertTrue(np.all(np.isfinite(mlp_out)))
            self.assertGreater(float(np.linalg.norm(mlp_out)), 0.0)
        finally:
            eng.close()

    def test_4_standalone_chat_qwen38_cpu_stream(self) -> None:
        chat = StandaloneSynapCoreChat(
            merkle_store_dir=QWEN38_STORE_DIR,
            max_ram_mb=4096.0,
            device="cpu",
            model_family="qwen3.8",
        )
        try:
            self.assertTrue(chat.is_qwen38)
            ids = chat.format_prompt("What is 2+2?")
            res = chat.generate_native_stream(ids, max_new_tokens=2)
            self.assertGreaterEqual(int(res["xor_proofs"]), 1024)
            self.assertGreaterEqual(int(res["tier2_byte_proofs"]), 16)
            self.assertGreaterEqual(int(res["num_tokens"]), 1)
        finally:
            chat.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
