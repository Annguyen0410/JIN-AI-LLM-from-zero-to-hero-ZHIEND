from __future__ import annotations
import json
import io
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from scripts.download_training_data import choose_files, execute, download_partial, DownloadPaused


class DownloadBudgetTests(unittest.TestCase):
    def entry(self, origin="human_documented"):
        return {"id": "fixture", "origin": origin, "hf_repo": "owner/repo", "resolved_revision": "a"*40,
                "license": "fixture only", "dataset_card": "https://example.org/card",
                "quality_evidence": "fixture", "domain": "knowledge", "text_columns": ["text"],
                "resolved_files": [{"path": f"train-{i}.parquet", "bytes": size} for i,size in enumerate((8, 7, 20, None))]}

    def test_budget_and_cached_files(self):
        with tempfile.TemporaryDirectory() as td:
            cache=Path(td)
            selected,_=choose_files([self.entry()], {"human_documented": 15}, cache)
            self.assertEqual(sum(i["bytes"] for i in selected), 15)
            (cache/"hf-fixture").mkdir()
            (cache/"hf-fixture/train-0.parquet").write_bytes(b"cached")
            selected,skipped=choose_files([self.entry()], {"human_documented": 8}, cache)
            self.assertEqual(sum(i["bytes"] for i in selected), 7)
            self.assertTrue(any("existing_cache" in i["reason"] for i in skipped))

    def test_separate_origins(self):
        with tempfile.TemporaryDirectory() as td:
            selected,_=choose_files([self.entry(),self.entry("synthetic_documented")],
                                    {"human_documented": 15,"synthetic_documented": 8},Path(td))
            self.assertEqual(sum(i["bytes"] for i in selected if i["origin"]=="synthetic_documented"),8)

    def test_execution_error_stops_and_existing_destination_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/"new"
            selected,_=choose_files([self.entry()], {"human_documented": 15}, root)
            plan={"files":selected,"new_planned_bytes":15}
            with mock.patch("scripts.download_training_data.download_with_progress", side_effect=RuntimeError("offline")) as net:
                report=execute(plan,out)
                self.assertEqual(report["status"],"stopped_on_error")
                self.assertEqual(net.call_count,1)
                self.assertEqual(len(report["completed"]),0)
            with self.assertRaisesRegex(ValueError,"Output exists"):
                execute(plan,out)

    def test_paused_plan_resumes_without_redownload(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/"new"
            selected,_=choose_files([self.entry()], {"human_documented":15}, root)
            plan={"files":selected,"new_planned_bytes":15}
            def download(url,path,limit_bytes):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b"x"*limit_bytes)
            with mock.patch("scripts.download_training_data.download_with_progress",side_effect=download) as net:
                report=execute(plan,out,max_files=1)
                self.assertEqual(report["status"],"paused_more_files_pending")
                self.assertEqual(net.call_count,1)
                report=execute(plan,out,resume=True)
                self.assertEqual(net.call_count,2)
                self.assertEqual(len(report["completed"]),2)
                self.assertEqual(report["status"],"downloaded_not_prepared_not_trained")
            (out/"human_documented/fixture/train-0.parquet").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError,"changed"):
                execute(plan,out,resume=True)

    def test_partial_resume_range_and_checksum(self):
        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range":"bytes 3-7/8", "Content-Length":"5"}
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/"file.parquet"
            part=dest.with_suffix(".parquet.part");part.write_bytes(b"123")
            item={"url":"https://example.org/file","bytes":8,"remote_path":"train.parquet",
                  "expected_sha256":hashlib.sha256(b"12345678").hexdigest(),"revision":"pinned"}
            with mock.patch("scripts.download_training_data.urllib.request.urlopen",return_value=Response(b"45678")) as net:
                download_partial(item,dest)
                self.assertEqual(net.call_args.args[0].get_header("Range"),"bytes=3-")
            self.assertEqual(dest.read_bytes(),b"12345678")
            self.assertFalse(part.exists())
            with self.assertRaises(FileExistsError):download_partial(item,dest)

    def test_ignored_range_preserves_partial(self):
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Length":"8"}
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/"file.parquet";part=dest.with_suffix(".parquet.part");part.write_bytes(b"123")
            item={"url":"https://example.org/file","bytes":8,"remote_path":"train.parquet",
                  "expected_sha256":hashlib.sha256(b"12345678").hexdigest(),"revision":"pinned"}
            with mock.patch("scripts.download_training_data.urllib.request.urlopen",return_value=Response(b"12345678")):
                with self.assertRaisesRegex(ValueError,"Range"):download_partial(item,dest)
            self.assertEqual(part.read_bytes(),b"123");self.assertFalse(dest.exists())

    def test_deadline_keeps_partial_and_rejects_bad_hash(self):
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Length":"8"}
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/"file.parquet";part=dest.with_suffix(".parquet.part")
            item={"url":"https://example.org/file","bytes":8,"remote_path":"train.parquet",
                  "expected_sha256":hashlib.sha256(b"12345678").hexdigest(),"revision":"pinned"}
            with mock.patch("scripts.download_training_data.urllib.request.urlopen",return_value=Response(b"12345678")):
                with self.assertRaises(DownloadPaused):download_partial(item,dest,deadline=0)
            self.assertTrue(part.exists());self.assertFalse(dest.exists())
            part.write_bytes(b"xxxxxxxx")
            with self.assertRaisesRegex(ValueError,"SHA256"):download_partial(item,dest)
            self.assertTrue(part.exists());self.assertFalse(dest.exists())

    def test_lock_prevents_parallel_download_and_plan_budget_checked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/"new";lock=root/"new.download.lock"
            lock.write_text("another process")
            with self.assertRaises(FileExistsError):execute({"files":[],"new_planned_bytes":0},out)
            self.assertEqual(lock.read_text(),"another process")
            lock.unlink()
            selected,_=choose_files([self.entry()], {"human_documented":8}, root)
            with self.assertRaisesRegex(ValueError,"exceeds"):
                execute({"files":selected,"new_planned_bytes":8,"total_download_budget_bytes":1},out)
            self.assertFalse(lock.exists())

    def test_checksum_checked_before_marking_success(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/"new"
            selected,_=choose_files([self.entry()], {"human_documented":8}, root)
            selected[0]["expected_sha256"]="bad"
            def download(url,path,limit_bytes):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b"12345678")
            with mock.patch("scripts.download_training_data.download_with_progress",side_effect=download):
                report=execute({"files":selected,"new_planned_bytes":8},out)
                self.assertEqual(report["status"],"stopped_on_error")
                self.assertEqual(report["completed"],[])
                self.assertIn("SHA256",report["failures"][0]["error"])


if __name__=="__main__":
    unittest.main()
