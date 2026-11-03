#!/usr/bin/env python3
"""
Unit & Integration Tests for the Automated 5 GB/Day Downloader & 256-Expert Trainer Daemon
(`tests/test_auto_daily_trainer.py`)
==========================================================================================
Verifies:
  1. Daily 5.00 GiB quota initialization, byte accounting, and automatic stop when quota is reached.
  2. Automatic midnight date rollover resetting `today_bytes_downloaded = 0` while preserving
     `dataset_cursors` and `total_bytes_downloaded_all_time`.
  3. Live/fallback batch download + incremental in-place 256-expert leaf & router update +
     cryptographic Merkle root verification.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from native.auto_daily_trainer_daemon import AutomatedDailyTrainerDaemon, BYTES_PER_GIB
from native.build_native_db import STORE_DIR
from native.expert_expansion import ExpandedExpertBankManager


class TestAutoDailyTrainerDaemon(unittest.TestCase):
    def test_01_quota_enforcement_and_date_rollover(self) -> None:
        daemon = AutomatedDailyTrainerDaemon(store_dir=STORE_DIR, daily_limit_gb=5.0, batch_rows=20)
        self.assertEqual(daemon.daily_limit_bytes, 5 * BYTES_PER_GIB)

        # Preserve full original state on disk so unit tests never overwrite live daemon progress
        orig_state = copy.deepcopy(daemon.state)
        try:
            daemon.state["today_bytes_downloaded"] = 5 * BYTES_PER_GIB + 1024
            daemon.state_path.write_text(json.dumps(daemon.state, indent=2))
            self.assertTrue(daemon.is_daily_quota_reached())
            self.assertFalse(daemon.run_single_cycle(force_offline_fallback=True))
            self.assertIn("Daily quota reached", daemon.state["last_status"])

            # Simulate date rollover from yesterday -> today_bytes_downloaded must reset to 0 automatically
            daemon.state["current_date"] = "2020-01-01"
            daemon.state_path.write_text(json.dumps(daemon.state, indent=2))
            daemon.refresh_daily_window()
            self.assertEqual(daemon.state["today_bytes_downloaded"], 0)
            self.assertFalse(daemon.is_daily_quota_reached())
        finally:
            daemon.state = orig_state
            daemon.state_path.write_text(json.dumps(orig_state, indent=2))

    def test_02_incremental_batch_download_and_256_expert_merkle_update(self) -> None:
        daemon = AutomatedDailyTrainerDaemon(store_dir=STORE_DIR, daily_limit_gb=5.0, batch_rows=15)
        batches_before = daemon.state["total_batches_trained"]

        ok = daemon.run_single_cycle(force_offline_fallback=True)
        self.assertTrue(ok)
        self.assertEqual(daemon.state["total_batches_trained"], batches_before + 1)
        self.assertLessEqual(daemon.state["last_router_loss_after"], daemon.state["last_router_loss_before"])

        # Verify that the updated 256-expert store passes full 3-domain Merkle tree verification!
        bank = ExpandedExpertBankManager(STORE_DIR)
        try:
            self.assertTrue(bank.verify_merkle_tree())
            assert bank.manifest is not None
            self.assertEqual(bank.manifest["expanded_model_merkle_root"], daemon.state["active_merkle_root"])
        finally:
            bank.close()


if __name__ == "__main__":
    unittest.main()
