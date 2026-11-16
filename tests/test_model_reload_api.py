"""``/api/model/reload`` — A/B two checkpoints without restarting the server.

The handler replaces 1.7GB of live weights from a caller-supplied path, so the
gates matter more than the happy path: on loopback with no token configured it
is open (personal default — the model bar needs that), it closes again as soon
as the server is exposed or a token is configured, it must reject
non-checkpoint paths, and it must drop the cached KV prefixes (they belong to
the previous weights). ``load_model`` is stubbed — the real load is exercised by
every server start.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config

import app as app_module


class TestModelReloadApi(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self.addCleanup(app_module.SESSIONS.clear)

    def test_reload_is_open_on_loopback_without_a_token(self):
        app_module.SESSIONS.clear()
        with mock.patch.object(config, "SERVER_AUTH_TOKEN", ""), \
                mock.patch.object(config, "SERVER_HOST", "127.0.0.1"), \
                mock.patch.object(app_module, "load_model", return_value=True) as fake, \
                mock.patch.object(app_module, "_checkpoint_files", return_value=[]):
            response = self.client.post("/api/model/reload", json={})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(fake.called, "loopback swap must reach load_model")

    def test_reload_is_closed_without_token_when_server_is_exposed(self):
        with mock.patch.object(config, "SERVER_AUTH_TOKEN", ""), \
                mock.patch.object(config, "SERVER_HOST", "0.0.0.0"):
            response = self.client.post("/api/model/reload", json={})
        self.assertEqual(response.status_code, 403)

    def test_reload_is_closed_without_token_for_remote_client(self):
        with mock.patch.object(config, "SERVER_AUTH_TOKEN", ""), \
                mock.patch.object(config, "SERVER_HOST", "127.0.0.1"):
            response = self.client.post("/api/model/reload", json={},
                                        environ_overrides={"REMOTE_ADDR": "10.0.0.5"})
        self.assertEqual(response.status_code, 403)

    def test_rejects_a_non_checkpoint_path(self):
        with mock.patch.object(config, "SERVER_AUTH_TOKEN", "secret"):
            response = self.client.post(
                "/api/model/reload", json={"path": "config.py"},
                headers={"Authorization": "Bearer secret"},
            )
        self.assertEqual(response.status_code, 404)
        self.assertIn("Not a checkpoint file", response.get_json()["error"])

    def test_rejects_a_path_that_does_not_exist(self):
        with mock.patch.object(config, "SERVER_AUTH_TOKEN", "secret"):
            response = self.client.post(
                "/api/model/reload", json={"path": "checkpoints/nope.pt"},
                headers={"Authorization": "Bearer secret"},
            )
        self.assertEqual(response.status_code, 404)

    def test_swap_clears_cached_prefixes(self):
        app_module.SESSIONS["s1"] = {
            "messages": [], "touched": 0.0,
            "prefix_cache": {"prompt_ids": [1, 2, 3], "kv_caches": []},
        }
        with mock.patch.object(config, "SERVER_AUTH_TOKEN", "secret"), \
                mock.patch.object(app_module, "load_model", return_value=True) as fake, \
                mock.patch.object(app_module, "_checkpoint_files", return_value=[]):
            response = self.client.post(
                "/api/model/reload", json={},
                headers={"Authorization": "Bearer secret"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(fake.called, "reload must go through load_model")
        self.assertIsNone(app_module.SESSIONS["s1"]["prefix_cache"])

    def test_failed_load_surfaces_as_500(self):
        with mock.patch.object(config, "SERVER_AUTH_TOKEN", "secret"), \
                mock.patch.object(app_module, "load_model", return_value=False):
            response = self.client.post(
                "/api/model/reload", json={},
                headers={"Authorization": "Bearer secret"},
            )
        self.assertEqual(response.status_code, 500)


class TestStatusPayload(unittest.TestCase):
    """The UI (Model Lab) reads these keys — keep them in the contract."""

    def test_status_exposes_lab_fields(self):
        client = app_module.app.test_client()
        data = client.get("/api/status").get_json()
        for key in ("device", "model_params", "context_window", "max_positions",
                    "defaults", "uptime_s", "requests_served", "checkpoints"):
            self.assertIn(key, data)
        self.assertIn("no_repeat_ngram_size", data["defaults"])
        self.assertIn("stop_on_repeat", data["defaults"])


if __name__ == "__main__":
    unittest.main()
