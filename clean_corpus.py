"""
Text cleaning helpers for JIN AI corpus building.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTIBLANK_RE = re.compile(r"\n{3,}")
SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _-]{0,40}):\s+")
GUTENBERG_START_LINE_RE = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
)
GUTENBERG_END_LINE_RE = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE
)
XML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class FileStats:
    path: str
    input_chars: int
    output_chars: int
    kept_lines: int
    skipped_lines: int


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


# ── Streaming readers (constant memory — handles multi-GB files) ────────

def detect_encoding(path: Path, sample_bytes: int = 1 << 16) -> str:
    """Guess the file encoding from a small prefix (no full read).

    Note: utf-16 is only chosen when a real BOM is present —
    ``bytes.decode("utf-16")`` silently accepts BOM-less data (implicit LE)
    while the streaming ``utf-16`` codec raises
    "UTF-16 stream does not start with BOM".
    """
    import codecs

    with open(path, "rb") as handle:
        sample = handle.read(sample_bytes)
    if sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if sample.startswith(codecs.BOM_UTF16_LE) or sample.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeError:
            continue
    return "cp1252"


def read_text_prefix(path: Path, max_chars: int) -> str:
    """Stream-read at most ``max_chars`` characters from the start of a file."""
    if max_chars <= 0:
        return ""
    encoding = detect_encoding(path)
    parts: list[str] = []
    total = 0
    with _safe_text_open(path, encoding) as handle:
        for line in handle:
            parts.append(line)
            total += len(line)
            if total >= max_chars:
                break
    return "".join(parts)[:max_chars]


def _safe_text_open(path: Path, encoding: str):
    """Open for text streaming; on a mid-stream UnicodeError retry with
    cp1252 + replace so one weird file never kills the whole build."""
    class _RetryHandle:
        def __init__(self, p: Path, enc: str):
            self._path = p
            self._encoding = enc
            self._handle = open(p, "r", encoding=enc, errors="strict")

        def __iter__(self):
            return self

        def __next__(self) -> str:
            try:
                return next(self._handle)
            except UnicodeError:
                # Re-open from the start with a lossy decoder.
                self._handle.close()
                self._handle = open(self._path, "r", encoding="cp1252", errors="replace")
                self._encoding = "cp1252(replace)"
                print(f"[Warn] {self._path.name}: stream decode failed -> "
                      f"re-reading with cp1252/replace")
                return next(self._handle)

        def close(self) -> None:
            self._handle.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    return _RetryHandle(path, encoding)


def iter_clean_lines(
    path: Path,
    *,
    min_line_chars: int,
    max_line_chars: int,
    strip_gutenberg: bool = False,
    strip_xml: bool = False,
    max_chars: int | None = None,
    stats: dict | None = None,
):
    """Stream-clean a text file line by line (constant memory).

    Yields cleaned lines. Equivalent semantics to ``read_text_file`` +
    ``clean_text`` for line-oriented content, but never holds the whole
    file in RAM. If ``stats`` is given, fills ``input_chars``, ``kept_lines``
    and ``skipped_lines``. ``max_chars`` stops emission after that many
    characters (used for sampling from huge corpora).
    """
    encoding = detect_encoding(path)

    # Gutenberg marker handling (2-pass: marker scan is cheap and the file
    # is then read from OS cache).
    skip_until_start = False
    if strip_gutenberg:
        with _safe_text_open(path, encoding) as handle:
            for line in handle:
                if GUTENBERG_START_LINE_RE.search(line):
                    skip_until_start = True
                    break

    emitted = 0
    with _safe_text_open(path, encoding) as handle:
        for raw_line in handle:
            if stats is not None:
                stats["input_chars"] += len(raw_line)
            if skip_until_start:
                if GUTENBERG_START_LINE_RE.search(raw_line):
                    skip_until_start = False
                continue
            if strip_gutenberg and GUTENBERG_END_LINE_RE.search(raw_line):
                break
            line = XML_TAG_RE.sub(" ", raw_line) if strip_xml else raw_line
            line = normalize_line(line)
            if is_low_quality(line, min_line_chars=min_line_chars, max_line_chars=max_line_chars):
                if stats is not None:
                    stats["skipped_lines"] += 1
                continue
            if max_chars is not None and emitted + len(line) + 1 > max_chars:
                break
            emitted += len(line) + 1
            if stats is not None:
                stats["kept_lines"] += 1
            yield line


def strip_gutenberg_boilerplate(text: str) -> str:
    start_match = GUTENBERG_START_RE.search(text)
    if start_match:
        text = text[start_match.end() :]
    end_match = GUTENBERG_END_RE.search(text)
    if end_match:
        text = text[: end_match.start()]
    return text


def strip_xml_tags(text: str) -> str:
    return XML_TAG_RE.sub(" ", text)


def normalize_line(line: str) -> str:
    line = html.unescape(line)
    line = unicodedata.normalize("NFKC", line)
    line = CONTROL_RE.sub("", line)
    line = line.replace("\r", "")
    line = MULTISPACE_RE.sub(" ", line).strip()
    return line


def is_low_quality(line: str, min_line_chars: int, max_line_chars: int) -> bool:
    if not line:
        return True
    if len(line) > max_line_chars:
        return True
    alpha = sum(ch.isalpha() for ch in line)
    if alpha == 0:
        return True
    non_space = max(1, sum(not ch.isspace() for ch in line))
    alpha_ratio = alpha / non_space
    if alpha_ratio < 0.55:
        return True
    looks_like_dialogue = bool(SPEAKER_RE.match(line)) or line.startswith(('"', "'", "-"))
    if len(line) < min_line_chars and not looks_like_dialogue:
        return True
    return False


def clean_text(
    text: str,
    min_line_chars: int,
    max_line_chars: int,
    strip_gutenberg: bool = False,
    strip_xml: bool = False,
) -> tuple[str, int, int]:
    if strip_gutenberg:
        text = strip_gutenberg_boilerplate(text)
    if strip_xml:
        text = strip_xml_tags(text)

    kept: list[str] = []
    kept_lines = 0
    skipped_lines = 0

    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if is_low_quality(line, min_line_chars=min_line_chars, max_line_chars=max_line_chars):
            skipped_lines += 1
            continue
        kept.append(line)
        kept_lines += 1

    cleaned = "\n".join(kept)
    cleaned = MULTIBLANK_RE.sub("\n\n", cleaned).strip()
    return cleaned, kept_lines, skipped_lines


def write_report(report_path: Path, stats: list[FileStats]) -> None:
    payload = {
        "file_count": len(stats),
        "total_input_chars": sum(item.input_chars for item in stats),
        "total_output_chars": sum(item.output_chars for item in stats),
        "files": [asdict(item) for item in stats],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
