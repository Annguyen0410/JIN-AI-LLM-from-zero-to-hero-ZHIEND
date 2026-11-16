"""Phase 2 — multi-format corpus ingestion with provenance.

Reads the corpora that actually exist on disk (TXT / JSON / JSONL / CSV /
Parquet) in bounded memory and normalises every record into one schema:

    document_id, source_id, source_name, source_url, license, language,
    domain, text, original_format, original_record_id, content_hash,
    quality_flags, ingestion_timestamp

Nothing is deleted and nothing is overwritten: output goes to
``data/curated_v5/documents/<source_id>.jsonl`` plus ``raw_manifest.json``.

Licensing policy: a source is only eligible for the default training mixture
when its license is explicitly recorded. Missing/unknown license =>
``license_status: "uncertain"`` and the source is written to disk but flagged
ineligible. Parquet support needs ``pyarrow``; without it the reader fails with
an install hint instead of silently skipping data.

Usage
-----
    python scripts/ingest_corpus.py --list
    python scripts/ingest_corpus.py --source wikipedia_en --max-docs 20000
    python scripts/ingest_corpus.py --all --max-docs-per-source 200000
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import time
import zlib
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_ROOT = os.path.join(ROOT, "data", "curated_v5")
DOWNLOAD_REPORT = os.path.join(ROOT, "data", "downloads", "general_v1",
                               "download_report.json")

TEXT_FIELDS = ("text", "content", "body", "document", "raw_content", "article")

# Vietnamese diacritics: used only as a *cheap label*, then verified by ratio.
_VI = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_BOILERPLATE = re.compile(
    r"(cookie policy|accept all cookies|sign in to continue|subscribe to our newsletter|"
    r"javascript is disabled|enable javascript|skip to main content)", re.I)


@dataclass
class Source:
    source_id: str
    source_name: str
    path_glob: str
    fmt: str                     # parquet | jsonl | json | csv | txt
    domain: str
    license: str | None
    license_status: str          # documented | uncertain
    source_url: str | None
    commercial_ok: bool | None
    redistribution_ok: bool | None
    attribution: str | None
    text_field: str | None = None
    language_hint: str | None = None
    notes: str = ""


def _license_record(source_id: str, report: dict) -> dict:
    """Look the source up in the download report (exact dataset id)."""
    for entry in report.get("completed", []):
        if entry.get("dataset_id") == source_id:
            lic = (entry.get("license") or "").strip()
            return {
                "license": lic or None,
                "license_status": "documented" if lic else "uncertain",
                "source_url": entry.get("dataset_card") or entry.get("url"),
                "revision": entry.get("revision"),
                "sha256": entry.get("sha256"),
                "commercial_ok": None if not lic else True,
                "redistribution_ok": None if not lic else True,
                "attribution": ("Required: cite the upstream dataset and its "
                                "license terms." if lic else None),
                "quality_evidence": entry.get("quality_evidence"),
                "text_columns": entry.get("text_columns") or [],
            }
    return {"license": None, "license_status": "uncertain", "source_url": None,
            "revision": None, "sha256": None, "commercial_ok": None,
            "redistribution_ok": None, "attribution": None,
            "quality_evidence": None, "text_columns": []}


def discover_sources(max_report_entries: int = 60) -> list[Source]:
    """Build the source list from what is on disk + the download report."""
    report = {"completed": []}
    if os.path.exists(DOWNLOAD_REPORT):
        with open(DOWNLOAD_REPORT, encoding="utf-8") as fh:
            report = json.load(fh)

    sources: list[Source] = []
    dl_root = os.path.join(ROOT, "data", "downloads", "general_v1")
    for dataset_dir in sorted(glob.glob(os.path.join(dl_root, "*", "*"))):
        if not os.path.isdir(dataset_dir):
            continue
        dataset_id = os.path.basename(dataset_dir)
        parquet = sorted(glob.glob(os.path.join(dataset_dir, "*.parquet")))
        if not parquet:
            continue
        meta = _license_record(dataset_id, report)
        origin = os.path.basename(os.path.dirname(dataset_dir))
        domain = "knowledge" if "wiki" in dataset_id else (
            "educational" if "cosmo" in dataset_id else origin)
        sources.append(Source(
            source_id=dataset_id.replace("-", "_"),
            source_name=dataset_id,
            path_glob=os.path.relpath(os.path.join(dataset_dir, "*.parquet"), ROOT),
            fmt="parquet",
            domain=domain,
            license=meta["license"],
            license_status=meta["license_status"],
            source_url=meta["source_url"],
            commercial_ok=meta["commercial_ok"],
            redistribution_ok=meta["redistribution_ok"],
            attribution=meta["attribution"],
            text_field=(meta["text_columns"] or [None])[0],
            language_hint="en",
            notes=meta.get("quality_evidence") or "",
        ))

    # Project-vetted corpora (prepared_v4 documents its own license policy).
    p4 = os.path.join(ROOT, "data", "prepared_v4", "pretrain")
    if os.path.isdir(p4):
        sources.append(Source(
            source_id="prepared_v4",
            source_name="prepared_v4 pretrain mixture",
            path_glob="data/prepared_v4/pretrain/*.txt",
            fmt="txt", domain="mixed",
            license="Per-source policy: NC and uncertain licenses excluded",
            license_status="documented",
            source_url="data/prepared_v4/manifest.json",
            commercial_ok=True, redistribution_ok=True,
            attribution="See data/prepared_v4/pretrain_sources.jsonl",
            language_hint=None),
        )

    # Local seed conversations (project-authored).
    seed = os.path.join(ROOT, "data", "seed")
    if os.path.isdir(seed):
        sources.append(Source(
            source_id="project_seed",
            source_name="project seed conversations",
            path_glob="data/seed/*.jsonl", fmt="jsonl", domain="conversation",
            license="Project-authored", license_status="documented",
            source_url=None, commercial_ok=True, redistribution_ok=True,
            attribution="Project-authored data", language_hint=None,
        ))
    return sources


def _iter_parquet(path: str, text_field: str | None, batch_size: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:                       # pragma: no cover
        raise SystemExit(
            "Parquet support needs pyarrow:\n"
            "    .venv/Scripts/python.exe -m pip install pyarrow\n"
            f"(missing for {os.path.basename(path)})") from exc
    pf = pq.ParquetFile(path)
    columns = pf.schema_arrow.names
    field_name = text_field if text_field in columns else next(
        (c for c in columns if c in TEXT_FIELDS), None)
    if field_name is None:
        raise SystemExit(f"No text column in {path} (columns={columns})")
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        data = batch.to_pydict()
        n = len(data[field_name])
        for i in range(n):
            yield {"text": data[field_name][i] or "",
                   "id": (data.get("id") or [None] * n)[i],
                   "url": (data.get("url") or [None] * n)[i],
                   "title": (data.get("title") or [None] * n)[i]}


def _iter_jsonl(path: str, text_field: str | None):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, str):
                rec = {"text": rec}
            text = rec.get(text_field) if text_field else None
            if text is None:
                text = next((rec[k] for k in TEXT_FIELDS if rec.get(k)), "")
            if isinstance(rec.get("messages"), list):
                text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}"
                                 for m in rec["messages"])
            yield {"text": text or "", "id": rec.get("id") or line_no,
                   "url": rec.get("url"), "title": rec.get("title")}


def _iter_json(path: str, text_field: str | None):
    with open(path, encoding="utf-8", errors="replace") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload = payload.get("documents") or payload.get("data") or [payload]
    for i, rec in enumerate(payload):
        if isinstance(rec, str):
            rec = {"text": rec}
        text = rec.get(text_field) if text_field else None
        if text is None:
            text = next((rec[k] for k in TEXT_FIELDS if rec.get(k)), "")
        yield {"text": text or "", "id": rec.get("id") or i,
               "url": rec.get("url"), "title": rec.get("title")}


def _iter_csv(path: str, text_field: str | None):
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, rec in enumerate(reader):
            text = rec.get(text_field) if text_field else None
            if text is None:
                text = next((rec[k] for k in TEXT_FIELDS if rec.get(k)), "")
            yield {"text": text or "", "id": i, "url": rec.get("url"), "title": rec.get("title")}


def _iter_txt(path: str, _text_field: str | None):
    """Blank line = document boundary (the convention of the local corpora)."""
    buf: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                buf.append(line)
            elif buf:
                yield {"text": "".join(buf), "id": None, "url": None, "title": None}
                buf = []
    if buf:
        yield {"text": "".join(buf), "id": None, "url": None, "title": None}


READERS = {"parquet": _iter_parquet, "jsonl": _iter_jsonl, "json": _iter_json,
           "csv": _iter_csv, "txt": _iter_txt}


def detect_language(text: str) -> str:
    letters = _LETTER.findall(text[:3000])
    if len(letters) < 20:
        return "unknown"
    vi = sum(1 for ch in letters if ch.lower() in _VI) / len(letters)
    if vi > 0.02:
        return "vi"
    ascii_ratio = sum(1 for ch in letters if ch.isascii()) / len(letters)
    if ascii_ratio > 0.95:
        return "en"
    return "other"


def normalise_text(text: str) -> str:
    """Conservative normalisation: line endings + trailing spaces only.

    No case folding, no punctuation stripping, no NFC/NFD conversion — the raw
    Unicode representation (including Vietnamese diacritics) is preserved
    exactly, and code indentation/newlines survive.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip("\n")


def quality_flags(text: str, min_chars: int, max_repeat_ratio: float) -> list[str]:
    flags = []
    if not text.strip():
        flags.append("empty")
        return flags
    if len(text) < min_chars:
        flags.append("too_short")
    # Repetition must be measured in a length-robust way. "unique word ratio"
    # falls with document length for any natural text (Zipf), and flagged 46%
    # of long Wikipedia articles as repetitive; 5-gram duplication and the
    # compressed size ratio do not have that bias.
    words = text.lower().split()
    if len(words) >= 50:
        grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
        dup_ratio = 1.0 - len(set(grams)) / len(grams)
        if dup_ratio > max_repeat_ratio:
            flags.append("high_repetition")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
        flags.append("repeated_lines")
    if len(text) >= 500:
        ratio = len(zlib.compress(text.encode("utf-8"), 6)) / len(text.encode("utf-8"))
        if ratio < 0.18:
            flags.append("low_information")
    if _BOILERPLATE.search(text[:2000]):
        flags.append("boilerplate")
    if "\ufffd" in text:
        flags.append("replacement_char")
    if text.count("\ufffd") > max(3, len(text) // 5000):
        flags.append("corrupted_encoding")
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.]{2,}", text):
        flags.append("possible_pii")
    return flags


def content_hash(text: str) -> str:
    """Exact-duplicate key: whitespace-collapsed, case-preserved."""
    norm = re.sub(r"[ \t]+", " ", text)
    norm = re.sub(r"\n{2,}", "\n", norm).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def ingest_source(src: Source, max_docs: int, min_chars: int,
                  batch_size: int = 512) -> dict:
    files = sorted(glob.glob(os.path.join(ROOT, src.path_glob)))
    if not files:
        return {"source_id": src.source_id, "present": False}
    out_dir = os.path.join(OUT_ROOT, "documents")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{src.source_id}.jsonl")
    reader = READERS[src.fmt]

    written = 0
    skipped = 0
    flags_count: dict[str, int] = {}
    lang_count: dict[str, int] = {}
    chars = 0
    t0 = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_path + ".tmp", "w", encoding="utf-8") as out:
        for path in files:
            try:
                for rec in reader(path, src.text_field, batch_size) if src.fmt == "parquet" \
                        else reader(path, src.text_field):
                    if written >= max_docs:
                        break
                    text = normalise_text(rec.get("text") or "")
                    if not text:
                        skipped += 1
                        continue
                    flags = quality_flags(text, min_chars, max_repeat_ratio=0.40)
                    if {"empty", "corrupted_encoding"} & set(flags):
                        skipped += 1
                        continue
                    lang = src.language_hint or detect_language(text)
                    if lang is None:
                        lang = detect_language(text)
                    doc = {
                        "document_id": f"{src.source_id}:{written}",
                        "source_id": src.source_id,
                        "source_name": src.source_name,
                        "source_url": rec.get("url") or src.source_url,
                        "license": src.license,
                        "license_status": src.license_status,
                        "language": lang,
                        "domain": src.domain,
                        "text": text,
                        "original_format": src.fmt,
                        "original_record_id": str(rec.get("id")) if rec.get("id") is not None else None,
                        "original_title": rec.get("title"),
                        "content_hash": content_hash(text),
                        "quality_flags": flags,
                        "ingestion_timestamp": stamp,
                    }
                    out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    written += 1
                    chars += len(text)
                    for f in flags:
                        flags_count[f] = flags_count.get(f, 0) + 1
                    lang_count[lang] = lang_count.get(lang, 0) + 1
            except SystemExit:
                raise
            except Exception as exc:                 # corrupt shard: keep going
                print(f"  [warn] {os.path.basename(path)}: {exc}")
                continue
            if written >= max_docs:
                break
    os.replace(out_path + ".tmp", out_path)
    elapsed = max(1e-6, time.time() - t0)
    return {
        "source_id": src.source_id, "present": True, "domain": src.domain,
        "license": src.license, "license_status": src.license_status,
        "commercial_ok": src.commercial_ok, "redistribution_ok": src.redistribution_ok,
        "attribution": src.attribution, "source_url": src.source_url,
        "format": src.fmt, "files": len(files),
        "documents": written, "skipped": skipped, "chars": chars,
        "avg_chars": round(chars / max(1, written), 1),
        "languages": lang_count, "quality_flags": flags_count,
        "output": os.path.relpath(out_path, ROOT),
        "output_sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest()
        if os.path.getsize(out_path) < 200 * 1024 * 1024 else None,
        "docs_per_second": round(written / elapsed, 1),
        "truncated_at_max_docs": written >= max_docs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 corpus ingestion")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--source", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-docs-per-source", type=int, default=200_000)
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    sources = discover_sources()
    if args.list or not (args.source or args.all):
        for s in sources:
            print(f"{s.source_id:28s} {s.fmt:8s} {s.domain:12s} "
                  f"license={'yes' if s.license else 'UNKNOWN'} :: {s.path_glob}")
        return

    selected = sources if args.all else [s for s in sources if s.source_id == args.source]
    if not selected:
        raise SystemExit(f"Unknown source {args.source!r}. Use --list.")

    os.makedirs(OUT_ROOT, exist_ok=True)
    manifest_path = os.path.join(OUT_ROOT, "raw_manifest.json")
    manifest = {"schema_version": 2, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "repo_revision": os.popen("git rev-parse HEAD").read().strip(),
                "sources": [], "policy": {
                    "license_unknown": "excluded from the default training mixture",
                    "normalisation": "CRLF->LF, trailing spaces, document trim only",
                    "unit_separator": "blank line for TXT, one record for JSON/JSONL/CSV/Parquet"}}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["sources"] = [s for s in manifest["sources"]
                               if s.get("source_id") not in
                               {x.source_id for x in selected}]

    for src in selected:
        print(f"\n=== {src.source_id} ({src.fmt}, {src.domain}) ===")
        info = ingest_source(src, args.max_docs_per_source, args.min_chars)
        manifest["sources"].append(info)
        if info.get("present"):
            print(f"  {info['documents']:,} docs | {info['chars']/1e6:.1f}M chars | "
                  f"langs {info['languages']} | {info['docs_per_second']}/s")
        else:
            print("  (absent — skipped)")

    manifest["sources"].sort(key=lambda s: s.get("source_id", ""))
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote {os.path.relpath(manifest_path, ROOT)}")


if __name__ == "__main__":
    main()
