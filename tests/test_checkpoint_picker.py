"""Checkpoint picker: roles, resolution order, header probe, /api/model/checkpoints.

Three things went wrong in practice and these tests pin all three:

* the Model Lab list was empty because a *stale server process* served an
  ``/api/status`` payload without the ``checkpoints`` key — the endpoint test
  below asserts the JSON contract the UI depends on;
* ``jingpt_best.pt`` / ``jingpt_latest.pt`` sound like the checkpoints to use,
  but they are pretrain artifacts. ``resolve_checkpoint`` must prefer the chat
  model even though "best" and "latest" exist, and must say *why*;
* training runs on Kaggle land in a folder of their own
  (``checkpoints/kaggle_v4_60020/kaggle_v4_instruct.pt``), so a flat
  ``listdir`` hid the newest weights from both the picker and the resolver.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
import unittest.mock as mock

import torch

import config

import app as app_module


class TestCheckpointRole(unittest.TestCase):
    def test_classifies_chat_against_pretrain(self):
        cases = {
            "jingpt_chat.pt": "chat",
            "jingpt_chat_46k_sft4k.pt": "chat",
            "jingpt_best.pt": "pretrain-best",
            "jingpt_latest.pt": "pretrain-latest",
            "jingpt.pt": "legacy",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(config.checkpoint_role(name), expected)

    def test_classifies_kaggle_v4_runs(self):
        cases = {
            "kaggle_v4_instruct.pt": "instruct",
            "kaggle_v4_instruct_best.pt": "instruct",
            "kaggle_v4_pretrain.pt": "pretrain-run",
            "kaggle_v4_pretrain_best.pt": "pretrain-run",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(config.checkpoint_role(name), expected)

    def test_only_chat_and_instruct_can_hold_a_conversation(self):
        self.assertTrue(config.is_chat_capable("chat"))
        self.assertTrue(config.is_chat_capable("instruct"))
        self.assertFalse(config.is_chat_capable("pretrain-run"))
        self.assertFalse(config.is_chat_capable("pretrain-best"))
        self.assertFalse(config.is_chat_capable("legacy"))


class TestResolveCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        patches = [
            mock.patch.object(config, "CHECKPOINT_DIR", self.root),
            mock.patch.object(config, "CHAT_CHECKPOINT_PATH", self._path("jingpt_chat.pt")),
            mock.patch.object(config, "BEST_CHECKPOINT_PATH", self._path("jingpt_best.pt")),
            mock.patch.object(config, "LATEST_CHECKPOINT_PATH", self._path("jingpt_latest.pt")),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        os.environ.pop(config.CHECKPOINT_ENV, None)

    def _path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def _touch(self, name: str, mtime: float | None = None) -> str:
        path = os.path.join(self.root, *name.split("/"))    # name may carry a folder
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"x")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_chat_checkpoint_wins_over_best_and_latest(self):
        self._touch("jingpt_best.pt")
        self._touch("jingpt_latest.pt")
        self._touch("jingpt_chat.pt")
        path, why = config.resolve_checkpoint()
        self.assertEqual(path, self._path("jingpt_chat.pt"))
        self.assertEqual(why, "chat")

    def test_falls_back_to_newest_chat_snapshot_not_pretrain(self):
        self._touch("jingpt_best.pt")
        self._touch("jingpt_chat_2400.pt", mtime=time.time() - 5000)
        self._touch("jingpt_chat_46k_sft4k.pt", mtime=time.time())
        path, why = config.resolve_checkpoint()
        self.assertEqual(path, self._path("jingpt_chat_46k_sft4k.pt"))
        self.assertEqual(why, "chat-snapshot")

    def test_pretrain_is_only_the_last_resort(self):
        self._touch("jingpt_best.pt")
        path, why = config.resolve_checkpoint()
        self.assertEqual(path, self._path("jingpt_best.pt"))
        self.assertEqual(why, "pretrain-fallback")

    def test_env_override_wins_even_over_chat(self):
        self._touch("jingpt_chat.pt")
        self._touch("jingpt_chat_old.pt")
        os.environ[config.CHECKPOINT_ENV] = self._path("jingpt_chat_old.pt")
        try:
            path, why = config.resolve_checkpoint()
        finally:
            os.environ.pop(config.CHECKPOINT_ENV, None)
        self.assertEqual(path, self._path("jingpt_chat_old.pt"))
        self.assertEqual(why, config.CHECKPOINT_ENV)

    def test_env_override_pointing_nowhere_is_ignored(self):
        self._touch("jingpt_chat.pt")
        os.environ[config.CHECKPOINT_ENV] = self._path("missing.pt")
        try:
            path, why = config.resolve_checkpoint()
        finally:
            os.environ.pop(config.CHECKPOINT_ENV, None)
        self.assertEqual(path, self._path("jingpt_chat.pt"))
        self.assertEqual(why, "chat")

    def test_get_checkpoint_path_keeps_its_string_contract(self):
        self._touch("jingpt_chat.pt")
        self.assertIsInstance(config.get_checkpoint_path(), str)
        self.assertEqual(config.get_checkpoint_path(), self._path("jingpt_chat.pt"))

    def test_newest_instruct_tune_beats_an_older_chat_checkpoint(self):
        self._touch("jingpt_chat.pt", mtime=time.time() - 5000)
        kaggle = self._touch("kaggle_v4_60020/kaggle_v4_instruct.pt", mtime=time.time())
        path, why = config.resolve_checkpoint()
        self.assertEqual(path, kaggle)
        self.assertEqual(why, "instruct")

    def test_newer_chat_snapshot_beats_an_older_instruct_tune(self):
        self._touch("kaggle_v4_60020/kaggle_v4_instruct.pt", mtime=time.time() - 5000)
        fresh = self._touch("jingpt_chat_50k.pt", mtime=time.time())
        path, why = config.resolve_checkpoint()
        self.assertEqual(path, fresh)
        self.assertEqual(why, "chat-snapshot")

    def test_pretrain_run_never_becomes_the_default(self):
        best = self._touch("jingpt_best.pt")
        self._touch("pretrain/general_v1_01/jin/checkpoints/kaggle_v4_pretrain.pt",
                    mtime=time.time())
        path, why = config.resolve_checkpoint()
        self.assertEqual(path, best)
        self.assertEqual(why, "pretrain-fallback")


class TestNestedCheckpointDiscovery(unittest.TestCase):
    """Weights inside a run folder must be visible — and newest first."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        patch = mock.patch.object(config, "CHECKPOINT_DIR", self.root)
        patch.start()
        self.addCleanup(patch.stop)

    def _touch(self, rel: str, mtime: float | None = None) -> str:
        path = os.path.join(self.root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"x")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_finds_weights_in_subfolders_newest_first(self):
        stale = self._touch("jingpt_chat.pt", mtime=time.time() - 5000)
        newest = self._touch("kaggle_v4_60020/kaggle_v4_instruct.pt", mtime=time.time())
        deep = self._touch(
            "pretrain/general_v1_01/jin/checkpoints/kaggle_v4_pretrain.pt",
            mtime=time.time() - 9000,
        )
        self.assertEqual(config.iter_checkpoint_paths(), [newest, stale, deep])

    def test_skips_hidden_and_pycache_folders(self):
        kept = self._touch("run/model.pt", mtime=time.time())
        self._touch(".trash/old.pt")
        self._touch("__pycache__/cached.pt")
        self.assertEqual(config.iter_checkpoint_paths(), [kept])

    def test_depth_cap_stops_the_walk(self):
        self._touch("a/b/c/d/e/f/deep.pt")
        self.assertEqual(config.iter_checkpoint_paths(max_depth=3), [])

    def test_missing_checkpoint_dir_is_empty_not_an_error(self):
        with mock.patch.object(config, "CHECKPOINT_DIR",
                               os.path.join(self.root, "nope")):
            self.assertEqual(config.iter_checkpoint_paths(), [])
            self.assertEqual(config.servable_checkpoint_paths(), [])


class TestHeaderProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        app_module._CKPT_META_CACHE.clear()
        self.addCleanup(app_module._CKPT_META_CACHE.clear)

    def _save(self, name: str, **extra) -> str:
        path = os.path.join(self.tmp.name, name)
        torch.save({"iteration": 4000, "val_loss": 2.9957335,
                    "best_loss": 3.0815, "model_state_dict": {}}, path)
        return path

    def test_reads_step_and_val_loss_without_loading_weights(self):
        path = self._save("jingpt_chat.pt")
        meta = app_module._probe_checkpoint(path)
        self.assertEqual(meta["step"], 4000)
        self.assertEqual(meta["val_loss"], 2.9957)
        self.assertEqual(meta["best_loss"], 3.0815)

    def test_second_call_is_served_from_cache(self):
        path = self._save("jingpt_chat.pt")
        app_module._probe_checkpoint(path)
        with mock.patch.object(app_module.torch, "load",
                               side_effect=AssertionError("must not re-read")) as fake:
            meta = app_module._probe_checkpoint(path)
        self.assertFalse(fake.called)
        self.assertEqual(meta["step"], 4000)

    def test_unreadable_file_does_not_raise(self):
        path = os.path.join(self.tmp.name, "broken.pt")
        with open(path, "wb") as handle:
            handle.write(b"not a checkpoint")
        meta = app_module._probe_checkpoint(path)
        self.assertIn("probe_error", meta)

    def test_missing_file_is_empty(self):
        self.assertEqual(app_module._probe_checkpoint("nope.pt"), {})


class TestModelCheckpointsEndpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.client = app_module.app.test_client()
        app_module._CKPT_META_CACHE.clear()
        self.addCleanup(app_module._CKPT_META_CACHE.clear)
        self.chat = self._save("jingpt_chat.pt", 4000, 2.9957)
        self.best = self._save("jingpt_best.pt", 2000, 7.1047)
        patches = [
            mock.patch.object(config, "CHECKPOINT_DIR", self.root),
            mock.patch.object(config, "CHAT_CHECKPOINT_PATH", self.chat),
            mock.patch.object(config, "BEST_CHECKPOINT_PATH", self.best),
            mock.patch.dict(app_module.checkpoint_meta, {"_loaded_from": self.chat}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _save(self, name: str, step: int, val: float) -> str:
        path = os.path.join(self.root, name)
        torch.save({"iteration": step, "val_loss": val, "model_state_dict": {}}, path)
        return path

    def test_picker_lists_every_checkpoint_with_role_and_served_flag(self):
        payload = self.client.get("/api/model/checkpoints").get_json()
        by_name = {f["name"]: f for f in payload["checkpoints"]}
        self.assertEqual(set(by_name), {"jingpt_chat.pt", "jingpt_best.pt"})
        self.assertEqual(by_name["jingpt_chat.pt"]["role"], "chat")
        self.assertTrue(by_name["jingpt_chat.pt"]["current"])
        # "best" exists on disk but must not be the auto-pick.
        self.assertTrue(by_name["jingpt_chat.pt"]["default"])
        self.assertFalse(by_name["jingpt_best.pt"]["default"])
        self.assertEqual(by_name["jingpt_best.pt"]["role"], "pretrain-best")

    def test_picker_probes_real_headers(self):
        payload = self.client.get("/api/model/checkpoints").get_json()
        by_name = {f["name"]: f for f in payload["checkpoints"]}
        self.assertEqual(by_name["jingpt_chat.pt"]["step"], 4000)
        self.assertEqual(by_name["jingpt_chat.pt"]["val_loss"], 2.9957)
        self.assertEqual(by_name["jingpt_best.pt"]["step"], 2000)

    def test_served_checkpoint_is_listed_first(self):
        payload = self.client.get("/api/model/checkpoints").get_json()
        self.assertEqual(payload["checkpoints"][0]["name"], "jingpt_chat.pt")

    def test_nested_instruct_tune_is_listed_and_wins_the_auto_pick(self):
        sub = os.path.join(self.root, "kaggle_v4_60020")
        os.makedirs(sub, exist_ok=True)
        nested = os.path.join(sub, "kaggle_v4_instruct.pt")
        torch.save({"iteration": 60020, "val_loss": 1.5870, "model_state_dict": {}}, nested)
        payload = self.client.get("/api/model/checkpoints").get_json()
        by_name = {f["name"]: f for f in payload["checkpoints"]}
        entry = by_name["kaggle_v4_instruct.pt"]
        self.assertEqual(entry["folder"], "kaggle_v4_60020")
        self.assertEqual(entry["role"], "instruct")
        self.assertEqual(entry["step"], 60020)
        # Newest chat-capable file anywhere under checkpoints/ is the default.
        self.assertTrue(entry["default"])
        self.assertEqual(payload["default"]["name"], "kaggle_v4_instruct.pt")
        self.assertEqual(payload["default"]["why"], "instruct")

    def test_status_lists_nested_checkpoints_without_probing(self):
        sub = os.path.join(self.root, "kaggle_v4_60020")
        os.makedirs(sub, exist_ok=True)
        torch.save({"iteration": 60020, "model_state_dict": {}},
                   os.path.join(sub, "kaggle_v4_instruct.pt"))
        payload = self.client.get("/api/status").get_json()
        by_name = {f["name"]: f for f in payload["checkpoints"]}
        self.assertIn("kaggle_v4_instruct.pt", by_name)
        self.assertNotIn("step", by_name["kaggle_v4_instruct.pt"])

    def test_status_explains_the_auto_pick(self):
        payload = self.client.get("/api/status").get_json()
        default = payload["checkpoint_default"]
        self.assertEqual(default["name"], "jingpt_chat.pt")
        self.assertEqual(default["why"], "chat")
        # The status poll must stay stat-only: no step/val keys without a probe.
        self.assertNotIn("step", payload["checkpoints"][0])


if __name__ == "__main__":
    unittest.main()
