"""Tests cho scripts/fetch_kernel_output.py.

Trọng tâm là đúng cái bug đã xảy ra ngày 16/09: ``kaggle kernels output`` gọi
``out.write(download_response.content)`` nên nạp trọn file vào RAM, và vì file
được mở ở mode ``"wb"`` trước khi đọc, ``MemoryError`` để lại một file 0 byte
trông như đã tải xong. Test ở đây khoá lại hai tính chất:

- tải theo chunk, không bao giờ chạm ``response.content``
- file rỗng/tải hỏng bị cổng kiểm định chặn (và không còn ``.part`` nằm lại)

Run:  python -m unittest tests.test_fetch_kernel_output -v   (stdlib, CPU, no network)
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from scripts.fetch_kernel_output import (
    MIN_PLAUSIBLE_BYTES,
    human_bytes,
    inspect_checkpoint,
    match_files,
    size_gate,
    stream_download,
)


class _FakeOutputFile:
    def __init__(self, file_name: str, url: str = "https://example.invalid/f"):
        self.file_name = file_name
        self.url = url


class _FakeResponse:
    """Response giả: cố tình nổ nếu ai đó đọc ``.content``."""

    def __init__(self, chunks: list[bytes], total: int = None):
        self._chunks = chunks
        self.headers = {} if total is None else {"Content-Length": str(total)}
        self.iterated = False

    def raise_for_status(self):
        return None

    @property
    def content(self):  # pragma: no cover - chỉ để bắt hồi quy
        raise AssertionError(
            "stream_download() đã đọc .content — đây chính là bug MemoryError!"
        )

    def iter_content(self, chunk_size=1):
        self.iterated = True
        for chunk in self._chunks:
            yield chunk

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _response_for(payload: bytes) -> _FakeResponse:
    step = 3 * 1024 * 1024
    chunks = [payload[i:i + step] for i in range(0, len(payload), step)] or [b""]
    return _FakeResponse(chunks, total=len(payload))


class TestHelpers(unittest.TestCase):
    def test_human_bytes_units(self):
        self.assertEqual(human_bytes(0), "0 B")
        self.assertEqual(human_bytes(512), "512 B")
        self.assertEqual(human_bytes(1024), "1.00 KiB")
        self.assertEqual(human_bytes(1536), "1.50 KiB")
        self.assertEqual(human_bytes(1024 ** 3), "1.00 GiB")
        self.assertEqual(human_bytes(1_247_475_570), "1.16 GiB")

    def test_size_gate_blocks_the_zero_byte_trap(self):
        self.assertFalse(size_gate(0))
        self.assertFalse(size_gate(1_000_000_000 - 1))
        self.assertTrue(size_gate(MIN_PLAUSIBLE_BYTES))
        self.assertTrue(size_gate(1_247_475_570))

    def test_match_files_filters_by_regex(self):
        files = [
            _FakeOutputFile("jin/checkpoints/jingpt_latest.pt"),
            _FakeOutputFile("jin/checkpoints/jingpt_best.pt"),
            _FakeOutputFile("jin/data/train.txt"),
            _FakeOutputFile("jin/cache/tokens_abc.stream.u16"),
        ]
        picked = [f.file_name for f in match_files(files, r"jingpt_(latest|best)\.pt")]
        self.assertEqual(picked, [
            "jin/checkpoints/jingpt_latest.pt",
            "jin/checkpoints/jingpt_best.pt",
        ])
        self.assertEqual([f.file_name for f in match_files(files, r"\.u16$")],
                         ["jin/cache/tokens_abc.stream.u16"])

    def test_match_files_returns_empty_when_nothing_matches(self):
        self.assertEqual(match_files([_FakeOutputFile("other.txt")], r"jingpt.*\.pt"), [])


@contextlib.contextmanager
def _quiet():
    """Nuốt stdout khi test gọi hàm thư viện trực tiếp.

    Console Windows mặc định mã hoá cp1252 nên không in được tiếng Việt; CLI thì
    đã tự ``enable_utf8_stdout()`` trong ``main()``, còn test gọi thẳng hàm nên
    chỉ cần tách khỏi console là đủ.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class TestUtf8Stdout(unittest.TestCase):
    def test_enable_utf8_stdout_is_safe_on_any_stream(self):
        from scripts.fetch_kernel_output import enable_utf8_stdout

        with contextlib.redirect_stdout(io.StringIO()):
            enable_utf8_stdout()          # StringIO không có reconfigure -> không được nổ
        enable_utf8_stdout()
        print("đã in được tiếng Việt")


class TestInspectCheckpoint(unittest.TestCase):
    def test_zero_byte_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "jingpt_latest.pt")
            open(path, "wb").close()          # đúng ca 0-byte ngày 16/09
            self.assertEqual(os.path.getsize(path), 0)
            with _quiet(), self.assertRaises(ValueError) as ctx:
                inspect_checkpoint(path)
            self.assertIn("dưới ngưỡng", str(ctx.exception))

    def test_truncated_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "jingpt_latest.pt")
            with open(path, "wb") as fh:
                fh.write(b"\0" * 1024)
            with _quiet(), self.assertRaises(ValueError):
                inspect_checkpoint(path)


class TestStreamDownload(unittest.TestCase):
    def test_writes_chunks_and_leaves_no_part_file(self):
        payload = os.urandom(7 * 1024 * 1024 + 123)
        with tempfile.TemporaryDirectory() as td:
            dest = os.path.join(td, "jin", "checkpoints", "jingpt_latest.pt")
            with mock.patch("requests.get", return_value=_response_for(payload)) as get, _quiet():
                written = stream_download("https://example.invalid/f", dest)
            self.assertEqual(written, len(payload))
            self.assertFalse(os.path.exists(dest + ".part"))
            with open(dest, "rb") as fh:
                self.assertEqual(fh.read(), payload)
            get.assert_called_once()

    def test_skips_when_destination_already_complete(self):
        payload = b"x" * (5 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as td:
            dest = os.path.join(td, "jingpt_latest.pt")
            with open(dest, "wb") as fh:
                fh.write(payload)
            resp = _response_for(payload)
            with mock.patch("requests.get", return_value=resp), _quiet():
                result = stream_download("https://example.invalid/f", dest, force=False)
            self.assertIsNone(result)
            self.assertFalse(resp.iterated, "đã tải lại dù file đích đã đủ")

    def test_short_body_raises_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as td:
            dest = os.path.join(td, "jingpt_latest.pt")
            resp = _FakeResponse([b"a" * 1024], total=999_999)
            with mock.patch("requests.get", return_value=resp), _quiet():
                with self.assertRaises(RuntimeError) as ctx:
                    stream_download("https://example.invalid/f", dest)
            self.assertIn("tải thiếu", str(ctx.exception))
            self.assertFalse(os.path.exists(dest))
            self.assertFalse(os.path.exists(dest + ".part"))


if __name__ == "__main__":
    unittest.main()
