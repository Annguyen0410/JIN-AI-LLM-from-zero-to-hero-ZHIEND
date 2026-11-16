from __future__ import annotations
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import config
import storage


class StorageTests(unittest.TestCase):
    def test_windows_drive_policy(self):
        storage.check_drive(r"D:\data\large.parquet", "D:")
        storage.check_drive(r"d:\data\large.parquet", "D:")
        for path in (r"C:\Temp\large.parquet", r"E:\data\train.txt", r"\\server\share\cache"):
            with self.assertRaisesRegex(ValueError, "must remain"):
                storage.check_drive(path, "D:")

    def test_relative_outputs_anchored_to_project_not_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(storage.output_path("data/new.txt", root=td),
                             str((Path(td)/"data/new.txt").resolve()))

    def test_guard_uses_resolved_junction_destination(self):
        with mock.patch.object(storage, "REQUIRED_DRIVE", "D:"), \
             mock.patch.object(storage.os.path, "realpath", return_value=r"C:\escape\large.txt"):
            with self.assertRaisesRegex(ValueError, "must remain"):
                storage.output_path("cache/junction/large.txt")

    def test_runtime_overrides_c_drive_environment_process_only(self):
        with tempfile.TemporaryDirectory() as td:
            original_temp = tempfile.tempdir
            try:
                with mock.patch.dict(os.environ, {"TEMP": "C:/Temp", "HF_HOME": "C:/Users/cache"}):
                    locations = storage.configure_runtime(td)
                    self.assertEqual(os.environ["TEMP"], str(Path(td)/"cache/tmp"))
                    self.assertEqual(tempfile.gettempdir(), os.environ["TEMP"])
                    for value in locations.values():
                        self.assertTrue(Path(value).is_relative_to(Path(td)))
                        self.assertTrue(Path(value).is_dir())
                    self.assertEqual(os.environ["HF_HOME"], str(Path(td)/"cache/huggingface"))
            finally:
                tempfile.tempdir = original_temp

    def test_default_locations_and_temp_stay_on_project_drive(self):
        for value in config.STORAGE_ENV.values():
            self.assertTrue(Path(value).is_relative_to(Path(config.ROOT_DIR)))
            storage.check_drive(str(Path(value).resolve()))
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(Path(td).is_relative_to(Path(config.CACHE_DIR)/"tmp"))

    @unittest.skipUnless(os.name == "nt", "Windows drive rejection")
    def test_large_writers_reject_c_before_io(self):
        import fetch_datasets
        from data_pipeline import prepare
        from utils import save_checkpoint
        with mock.patch.object(fetch_datasets, "open_url") as network:
            with self.assertRaisesRegex(ValueError, "must remain"):
                fetch_datasets.download_with_progress("https://example.org/data", Path("C:/jin-test/file.parquet"))
            network.assert_not_called()
        with self.assertRaisesRegex(ValueError, "must remain"):
            prepare(Path(config.ROOT_DIR), Path("C:/jin-test/prepared"))
        with self.assertRaisesRegex(ValueError, "must remain"):
            save_checkpoint(None, None, 0, 0, "C:/jin-test/checkpoint.pt")


if __name__ == "__main__":
    unittest.main()
