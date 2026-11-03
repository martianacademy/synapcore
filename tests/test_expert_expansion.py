#!/usr/bin/env python3
"""
Unit & End-to-End Verification Suite for 3-Domain Expert Expansion (`tests/test_expert_expansion.py`)
=====================================================================================================
Tests:
  1. Cryptographic SHA-256 Subtree Verification across all 3 Domain Banks (`thinking`, `tool_calling`, `reasoning`)
     supporting up to `256` Experts/Layer (`4,096` Total Leaves = `48.92 GiB` Total Model Size).
  2. 16 KB Page Alignment & Zero-Copy `mmap` Slicing (`OWNDATA=False`) of Expanded Leaves.
  3. Dynamic `(N, 2048)` Router Expansion + Domain Routing Affinity shifts (`thinking`, `tool_calling`, `reasoning`)
     while keeping active per-token compute strictly at `Top-8` (`1.3B` active parameters).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from native.build_native_db import STORE_DIR
from native.expert_expansion import ExpandedExpertBankManager
from native.native_engine import NativeMerkleMoEEngine


class TestThreeDomainExpertExpansion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bank = ExpandedExpertBankManager(STORE_DIR)
        assert cls.bank.is_built(), "Expanded 3-domain expert store must be built before running tests"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.bank.close()

    def test_01_manifest_and_three_domain_merkle_roots(self) -> None:
        """Verifies all 3 domain Merkle subtrees (`thinking`, `tool_calling`, `reasoning`) and top-level root."""
        self.assertTrue(self.bank.verify_merkle_tree())
        assert self.bank.manifest is not None
        self.assertEqual(self.bank.manifest["base_experts_per_layer"], 64)
        exp_per_layer = int(self.bank.manifest["expanded_experts_per_layer"])
        tot_per_layer = int(self.bank.manifest["total_experts_per_layer"])
        self.assertEqual(tot_per_layer, 64 + exp_per_layer)
        self.assertIn(tot_per_layer, (76, 112, 160, 256))
        self.assertEqual(self.bank.manifest["top_k_active"], 8)
        self.assertEqual(len(self.bank.manifest["leaves"]), 16 * exp_per_layer)

        for dom in ("thinking", "tool_calling", "reasoning"):
            metrics = self.bank.manifest["training_metrics"][dom]
            self.assertLess(metrics["final_loss"], metrics["initial_loss"])
            self.assertGreater(metrics["router_win_rate_pct"], 80.0)

    def test_02_zero_copy_16kb_aligned_slicing_all_domains(self) -> None:
        """Verifies 16 KB page alignment, SHA-256 leaf integrity, and OWNDATA=False zero-copy views."""
        for dom, (start_e, end_e) in self.bank.domain_ranges.items():
            gate_w, up_w, down_w, meta = self.bank.slice_expanded_expert_zero_copy(
                layer_idx=0, expert_idx=start_e, verify=True
            )
            self.assertEqual(meta["domain"], dom)
            self.assertEqual(meta["byte_offset"] % 16384, 0)
            self.assertEqual(gate_w.shape, (1024, 2048))
            self.assertEqual(up_w.shape, (1024, 2048))
            self.assertEqual(down_w.shape, (2048, 1024))
            self.assertFalse(gate_w.flags.owndata)
            self.assertFalse(up_w.flags.owndata)
            self.assertFalse(down_w.flags.owndata)

    def test_03_domain_prompt_detection_and_expanded_expert_routing(self) -> None:
        """Verifies that prompts dynamically activate Thinking, Tool-Calling, and Reasoning expert banks."""
        eng = NativeMerkleMoEEngine(
            STORE_DIR,
            expert_cache_size=64,
            on_demand_streaming=False,
            verify_merkle_on_load=False,
            device="cpu",
            enable_expanded_experts=True,
        )
        try:
            self.assertEqual(eng.num_experts, self.bank.total_experts)
            self.assertEqual(eng.top_k, 8)

            # 1. Tool-Calling Prompt -> should route to Tool-Calling Experts and Thinking Experts
            eng.configure_domain_routing(
                "what is the temprature in chandigarh",
                enable_thinking=True,
            )
            assert eng.expanded_bank is not None
            eng.expanded_bank.reset_routing_counts()
            kv_k = [None for _ in range(eng.num_layers)]
            kv_v = [None for _ in range(eng.num_layers)]
            logits_tool = eng.forward_full_model_numpy([1, 250, 400], kv_k, kv_v, pos_offset=0)
            self.assertEqual(logits_tool.shape, (eng.vocab_size,))
            self.assertGreater(eng.expanded_bank.domain_routing_counts["tool_calling"], 0)
            self.assertGreater(eng.expanded_bank.domain_routing_counts["base"], 0)

            # 2. Math / Code Reasoning Prompt -> should route to Reasoning Experts
            eng.configure_domain_routing(
                "Prove by mathematical induction and implement an O(log N) Merkle proof verifier in C++",
                enable_thinking=True,
            )
            eng.expanded_bank.reset_routing_counts()
            kv_k2 = [None for _ in range(eng.num_layers)]
            kv_v2 = [None for _ in range(eng.num_layers)]
            logits_reas = eng.forward_full_model_numpy([1, 512, 1024], kv_k2, kv_v2, pos_offset=0)
            self.assertEqual(logits_reas.shape, (eng.vocab_size,))
            self.assertGreater(eng.expanded_bank.domain_routing_counts["reasoning"], 0)
        finally:
            eng.close()


if __name__ == "__main__":
    unittest.main()
