"""
Phase 7 tests — operations & living UI backend.

Covers:
- config v3 blocks (MOE / QUANT / SPEED) exist with sane defaults.
- Admin API auth: no token + loopback-bound server -> open for loopback
  clients (personal default); exposed server or configured token -> token
  required, loopback-only unless JINGPT_ALLOW_REMOTE_ADMIN=1.
- /api/checkpoints promote + rollback through the registry.
- /api/learn/trigger validation + background cycle invocation.
- bench.append_history writes a compact record to history.json.

Run:  python -m unittest tests.test_phase7 -v   (stdlib, CPU, no network)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

import config


class TestConfigV3Blocks(unittest.TestCase):
    def test_moe_block_present(self):
        self.assertTrue(hasattr(config, "MOE_N_EXPERTS"))
        self.assertTrue(hasattr(config, "MOE_TOP_K"))
        self.assertTrue(hasattr(config, "MOE_CAPACITY_FACTOR"))
        self.assertTrue(hasattr(config, "MOE_EXPERT_OFFLOAD"))
        self.assertIsInstance(config.MOE_CAPACITY_FACTOR, float)
        self.assertIsInstance(config.MOE_EXPERT_OFFLOAD, bool)

    def test_quant_block_present(self):
        self.assertTrue(hasattr(config, "QUANT_WEIGHT"))
        self.assertTrue(hasattr(config, "QUANT_GROUP"))
        self.assertTrue(hasattr(config, "QUANT_KV_INT8"))
        self.assertIsInstance(config.QUANT_GROUP, int)
        self.assertIsInstance(config.QUANT_KV_INT8, bool)

    def test_speed_block_present(self):
        self.assertTrue(hasattr(config, "SPEED_STATIC_KV"))
        self.assertTrue(hasattr(config, "SPEED_COMPILE"))
        self.assertTrue(hasattr(config, "SPEED_PROMPT_LOOKUP_K"))
        self.assertIsInstance(config.SPEED_PROMPT_LOOKUP_K, int)
        self.assertGreaterEqual(config.SPEED_PROMPT_LOOKUP_K, 0)

    def test_admin_defaults_off(self):
        self.assertEqual(config.ALLOW_REMOTE_ADMIN, False)


class TestBenchHistory(unittest.TestCase):
    def test_append_history_creates_file(self):
        import bench
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = bench.BENCH_DIR
            bench.BENCH_DIR = tmp
            try:
                result = {
                    "timestamp": "2026-09-07T12:00:00",
                    "checkpoint_name": "jingpt_test",
                    "params_m": 151.0,
                    "quality": {"heldout_loss": 2.1, "heldout_ppl": 8.2},
                    "probes": {"pass_rate": 0.42, "mean_lp_gap": 0.9},
                    "fluency": {"avg_score": 0.8},
                    "speed": {"prefill_tok_per_s": 100.0, "decode_tok_per_s": 58.0,
                              "ttft_ms": 210.0, "peak_vram_gb": 3.1},
                    "train": {"train_tok_per_s": 3200.0},
                }
                bench.append_history(result)
                path = os.path.join(tmp, "history.json")
                self.assertTrue(os.path.exists(path))
                with open(path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                self.assertEqual(len(history), 1)
                rec = history[0]
                self.assertEqual(rec["heldout_loss"], 2.1)
                self.assertEqual(rec["probe_pass_rate"], 0.42)
                self.assertEqual(rec["decode_tok_per_s"], 58.0)
                # Append a second run -> 2 records.
                bench.append_history(result)
                with open(path, "r", encoding="utf-8") as f:
                    self.assertEqual(len(json.load(f)), 2)
            finally:
                bench.BENCH_DIR = old_dir

    def test_append_history_survives_corrupt_file(self):
        import bench
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = bench.BENCH_DIR
            bench.BENCH_DIR = tmp
            try:
                with open(os.path.join(tmp, "history.json"), "w",
                          encoding="utf-8") as f:
                    f.write("{not json")
                bench.append_history({"timestamp": "t", "checkpoint_name": "x"})
                with open(os.path.join(tmp, "history.json"), "r",
                          encoding="utf-8") as f:
                    history = json.load(f)
                self.assertEqual(len(history), 1)
            finally:
                bench.BENCH_DIR = old_dir


class AdminApiAuthMixin:
    """Shared helpers: load the Flask app + control config auth knobs."""

    @classmethod
    def setUpClass(cls):
        import app as app_mod
        cls.app_mod = app_mod
        app_mod.app.config["TESTING"] = True
        cls.client = app_mod.app.test_client()
        # Save real knobs so tests never leak state.
        cls._saved = {
            "token": config.SERVER_AUTH_TOKEN,
            "remote": config.ALLOW_REMOTE_ADMIN,
            "registry": config.CHECKPOINT_REGISTRY_PATH,
        }

    def tearDown(self):
        config.SERVER_AUTH_TOKEN = self._saved["token"]
        config.ALLOW_REMOTE_ADMIN = self._saved["remote"]
        config.CHECKPOINT_REGISTRY_PATH = self._saved["registry"]

    def _auth_headers(self, token: str | None):
        return {"Authorization": "Bearer " + token} if token else {}


class TestAdminAuth(AdminApiAuthMixin, unittest.TestCase):
    def test_admin_open_on_loopback_without_a_token(self):
        """Personal model on your own box: no password to switch checkpoints."""
        config.SERVER_AUTH_TOKEN = ""
        with mock.patch.object(config, "SERVER_HOST", "127.0.0.1"):
            r = self.client.get("/api/learn/status")
        self.assertEqual(r.status_code, 200)

    def test_admin_closed_without_token_when_server_is_exposed(self):
        """Bound to 0.0.0.0 (LAN) and no token -> admin stays shut."""
        config.SERVER_AUTH_TOKEN = ""
        with mock.patch.object(config, "SERVER_HOST", "0.0.0.0"):
            r = self.client.get("/api/learn/status")
        self.assertEqual(r.status_code, 403)

    def test_admin_closed_without_token_for_remote_client(self):
        config.SERVER_AUTH_TOKEN = ""
        with mock.patch.object(config, "SERVER_HOST", "127.0.0.1"):
            r = self.client.get("/api/learn/status",
                                environ_overrides={"REMOTE_ADDR": "10.1.2.3"})
        self.assertEqual(r.status_code, 403)

    def test_admin_requires_token(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        r = self.client.get("/api/learn/status")
        self.assertEqual(r.status_code, 403)  # no Authorization header
        r = self.client.get("/api/learn/status",
                            headers=self._auth_headers("wrong"))
        self.assertEqual(r.status_code, 403)

    def test_admin_ok_with_valid_token_on_loopback(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        r = self.client.get("/api/learn/status",
                            headers=self._auth_headers("sekrit"))
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn("budget", body)
        self.assertIn("ledger", body)

    def test_admin_rejected_from_non_loopback(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        config.ALLOW_REMOTE_ADMIN = False
        r = self.client.get(
            "/api/learn/status", headers=self._auth_headers("sekrit"),
            environ_overrides={"REMOTE_ADDR": "10.1.2.3"})
        self.assertEqual(r.status_code, 403)

    def test_admin_allowed_remote_when_explicit(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        config.ALLOW_REMOTE_ADMIN = True
        r = self.client.get(
            "/api/learn/status", headers=self._auth_headers("sekrit"),
            environ_overrides={"REMOTE_ADDR": "10.1.2.3"})
        self.assertEqual(r.status_code, 200)

    def test_bench_history_gated(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        r = self.client.get("/api/bench/history")
        self.assertEqual(r.status_code, 403)
        r = self.client.get("/api/bench/history",
                            headers=self._auth_headers("sekrit"))
        self.assertEqual(r.status_code, 200)
        self.assertIn("runs", r.get_json())


class TestCheckpointsApi(AdminApiAuthMixin, unittest.TestCase):
    def test_promote_and_rollback(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        with tempfile.TemporaryDirectory() as tmp:
            config.CHECKPOINT_REGISTRY_PATH = os.path.join(tmp, "registry.json")
            ckpt = os.path.join(tmp, "jingpt_new.pt")
            with open(ckpt, "w") as f:
                f.write("dummy")
            r = self.client.post(
                "/api/checkpoints/promote",
                json={"path": ckpt, "changelog": {"learned": "topic x"}},
                headers=self._auth_headers("sekrit"))
            self.assertEqual(r.status_code, 200)
            reg = r.get_json()["registry"]
            self.assertEqual(len(reg["checkpoints"]), 1)
            self.assertIsNotNone(reg["golden"])

            # Rollback promotes the older path with action=rollback.
            old = os.path.join(tmp, "jingpt_old.pt")
            with open(old, "w") as f:
                f.write("dummy")
            r = self.client.post(
                "/api/checkpoints/rollback",
                json={"path": old, "reason": "quality dropped"},
                headers=self._auth_headers("sekrit"))
            self.assertEqual(r.status_code, 200)
            reg = r.get_json()["registry"]
            self.assertEqual(reg["checkpoints"][-1]["changelog"]["action"],
                             "rollback")

    def test_promote_missing_file_404(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        with tempfile.TemporaryDirectory() as tmp:
            config.CHECKPOINT_REGISTRY_PATH = os.path.join(tmp, "registry.json")
            r = self.client.post(
                "/api/checkpoints/promote",
                json={"path": os.path.join(tmp, "nope.pt")},
                headers=self._auth_headers("sekrit"))
            self.assertEqual(r.status_code, 404)

    def test_list_returns_files(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        r = self.client.get("/api/checkpoints",
                            headers=self._auth_headers("sekrit"))
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn("registry", body)
        self.assertIn("files", body)


class TestLearnTrigger(AdminApiAuthMixin, unittest.TestCase):
    def test_trigger_requires_topic(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        r = self.client.post("/api/learn/trigger", json={},
                             headers=self._auth_headers("sekrit"))
        self.assertEqual(r.status_code, 400)

    def test_trigger_runs_cycle_in_background(self):
        config.SERVER_AUTH_TOKEN = "sekrit"
        import learn_daemon
        called = []

        def fake_run(topics, cycles=1, dry_run=True, device="cpu"):
            called.append((list(topics), dry_run))

        learn_daemon.run_daemon = fake_run
        try:
            r = self.client.post(
                "/api/learn/trigger",
                json={"topic": "test topic", "dry_run": True},
                headers=self._auth_headers("sekrit"))
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
            self.assertTrue(body["ok"])
            self.assertTrue(body["dry_run"])
            for _ in range(20):
                if called:
                    break
                time.sleep(0.05)
            self.assertEqual(len(called), 1)
            self.assertEqual(called[0][0], ["test topic"])
            self.assertTrue(called[0][1])  # dry_run default True
        finally:
            self.app_mod._LEARN_RUNNING = False


if __name__ == "__main__":
    unittest.main()