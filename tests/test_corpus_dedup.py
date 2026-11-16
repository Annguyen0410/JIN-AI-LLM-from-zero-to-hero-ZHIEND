"""Tests for exact deduplication (Phase 15B).

These tests exercise the defect that Phase 14 shipped: pass 2 looked up
``seen[hash]`` for *every* occurrence, so duplicates of a retained document were
written again (1,897 such records were measured in the v5 train split). Every
test below therefore audits the **written output**, not internal counters.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def write_documents(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def doc(text: str, source: str, doc_id: str, language: str = "en",
        domain: str = "knowledge", flags: list[str] | None = None) -> dict:
    import hashlib
    norm = " ".join(text.split())
    return {
        "document_id": f"{source}:{doc_id}",
        "source_id": source,
        "source_name": source,
        "source_url": f"https://example.invalid/{doc_id}",
        "license": "project-authored",
        "license_status": "documented",
        "canonical_license": "project-authored",
        "eligible_for_training_mixture": True,
        "language": language,
        "domain": domain,
        "text": text,
        "original_format": "txt",
        "original_record_id": doc_id,
        "content_hash": hashlib.sha256(norm.encode()).hexdigest(),
        "quality_flags": flags or [],
        "ingestion_timestamp": "2026-11-16T00:00:00",
    }


def run_verify(tmp_path, extra_args: list[str] | None = None) -> dict:
    in_root = tmp_path / "curated"
    out_root = tmp_path / "curated"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "verify_corpus.py"),
           "--in-root", str(in_root), "--out-root", str(out_root),
           "--min-chars", "10", "--seed", "7"]
    if extra_args:
        cmd += extra_args
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.load(open(out_root / "reports" / "dedup_quality_splits.json",
                          encoding="utf-8"))


def read_split(root, name: str) -> list[dict]:
    path = root / "splits" / f"{name}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def assert_unique_hashes(records: list[dict]) -> None:
    hashes = [r["content_hash"] for r in records]
    assert len(hashes) == len(set(hashes)), "duplicate content_hash in output"


def test_identical_documents_within_one_source(tmp_path):
    text = "The same paragraph appears twice in one source file. " * 5
    write_documents(str(tmp_path / "curated" / "documents" / "a.jsonl"), [
        doc(text, "src_a", "1"), doc(text, "src_a", "2"), doc(text, "src_a", "3"),
        doc("A different paragraph that is long enough to survive filtering. " * 3,
            "src_a", "4"),
    ])
    report = run_verify(tmp_path)
    total = sum(len(read_split(tmp_path / "curated", s)) for s in ("train", "val", "test"))
    assert report["exact_duplicates_removed"] == 2
    assert total == 2                      # 2 unique documents written, not 4
    for split in ("train", "val", "test"):
        assert_unique_hashes(read_split(tmp_path / "curated", split))


def test_identical_documents_across_sources(tmp_path):
    text = "Cross-source duplication is still duplication, byte for byte. " * 5
    write_documents(str(tmp_path / "curated" / "documents" / "a.jsonl"),
                    [doc(text, "src_a", "1")])
    write_documents(str(tmp_path / "curated" / "documents" / "b.jsonl"),
                    [doc(text, "src_b", "1")])
    report = run_verify(tmp_path)
    total = sum(len(read_split(tmp_path / "curated", s)) for s in ("train", "val", "test"))
    assert report["exact_duplicates_removed"] == 1
    assert total == 1


def test_identical_documents_in_different_files_same_hash(tmp_path):
    text = "Same file contents, different file names, one document. " * 5
    write_documents(str(tmp_path / "curated" / "documents" / "x.jsonl"),
                    [doc(text, "src_a", "1"), doc(text, "src_a", "2")])
    write_documents(str(tmp_path / "curated" / "documents" / "y.jsonl"),
                    [doc(text, "src_a", "3")])
    run_verify(tmp_path)
    total = sum(len(read_split(tmp_path / "curated", s)) for s in ("train", "val", "test"))
    assert total == 1
    assert_unique_hashes(read_split(tmp_path / "curated", "train"))


def test_near_duplicate_cluster_keeps_one_representative(tmp_path):
    # A reformatted copy: identical body, one different closing sentence.
    body = " ".join(f"Sentence number {i} explains a different part of the survey."
                    for i in range(20))
    variants = [f"{body} Closing remark variant {i}." for i in range(4)]
    write_documents(str(tmp_path / "curated" / "documents" / "n.jsonl"),
                    [doc(v, "src_a", str(i)) for i, v in enumerate(variants)])
    report = run_verify(tmp_path, ["--near-dup-threshold", "0.8"])
    total = sum(len(read_split(tmp_path / "curated", s)) for s in ("train", "val", "test"))
    assert total == 1, "near-duplicate cluster must contribute exactly one document"
    assert report["near_duplicate"]["clusters"] == 1
    assert_unique_hashes(read_split(tmp_path / "curated", "train"))


def test_near_duplicate_different_lengths(tmp_path):
    long_text = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 40)
    short_text = long_text[:600] + " Tail sentence differs."
    write_documents(str(tmp_path / "curated" / "documents" / "n.jsonl"),
                    [doc(long_text, "src_a", "long"), doc(short_text, "src_a", "short"),
                     doc("Completely unrelated text about marine biology fieldwork. " * 6,
                         "src_a", "other")])
    report = run_verify(tmp_path)
    total = sum(len(read_split(tmp_path / "curated", s)) for s in ("train", "val", "test"))
    assert total <= 3
    assert report["exact_duplicates_removed"] == 0


def test_repeated_runs_are_deterministic(tmp_path):
    records = [doc(f"Document number {i} with unique content {i * 7919}. " * 4,
                   f"src_{i % 3}", str(i)) for i in range(60)]
    write_documents(str(tmp_path / "curated" / "documents" / "d.jsonl"), records)
    first = run_verify(tmp_path)
    hashes_first = {s: [r["content_hash"] for r in read_split(tmp_path / "curated", s)]
                    for s in ("train", "val", "test")}
    second = run_verify(tmp_path)
    hashes_second = {s: [r["content_hash"] for r in read_split(tmp_path / "curated", s)]
                     for s in ("train", "val", "test")}
    assert first["splits"] == second["splits"]
    assert hashes_first == hashes_second


def test_output_counts_match_unique_documents(tmp_path):
    records = [doc(f"Unique body {i} " + "padding words here " * 20, "src_a", str(i))
               for i in range(30)]
    records += [records[0], records[1], records[2]]          # 3 exact duplicates
    write_documents(str(tmp_path / "curated" / "documents" / "c.jsonl"), records)
    report = run_verify(tmp_path)
    written = sum(report["splits"][s]["chunks"] for s in ("train", "val", "test"))
    assert report["documents_in"] == 33
    assert report["exact_duplicates_removed"] == 3
    assert written == 30
    assert report["output_audit"]["extra_duplicate_records"] == 0


def test_cluster_disjoint_splits(tmp_path):
    records = [doc(f"Cluster member {i} discussing topic {i % 5}. " * 6, "src_a", str(i))
               for i in range(80)]
    write_documents(str(tmp_path / "curated" / "documents" / "s.jsonl"), records)
    report = run_verify(tmp_path)
    audit = report["output_audit"]
    assert audit["train_val_overlap"] == 0
    assert audit["train_test_overlap"] == 0
    assert audit["val_test_overlap"] == 0
    clusters = {}
    for split in ("train", "val", "test"):
        for rec in read_split(tmp_path / "curated", split):
            clusters.setdefault(rec["cluster_id"], set()).add(split)
    assert all(len(v) == 1 for v in clusters.values()), "a cluster straddles splits"


def test_pii_documents_are_excluded(tmp_path):
    clean = "A perfectly ordinary paragraph about rivers and their valleys. " * 4
    with_email = ("Contact the author at someone@example.com for details about the "
                  "river survey carried out last spring. " * 4)
    write_documents(str(tmp_path / "curated" / "documents" / "p.jsonl"),
                    [doc(clean, "src_a", "clean"),
                     doc(with_email, "src_a", "email", flags=["possible_pii"])])
    report = run_verify(tmp_path)
    written = [r for s in ("train", "val", "test")
               for r in read_split(tmp_path / "curated", s)]
    assert len(written) == 1
    assert "email" not in written[0]["text"]
    assert report["pii"]["excluded_documents"] == 1
    assert report["pii"]["policy"] == "exclude"
