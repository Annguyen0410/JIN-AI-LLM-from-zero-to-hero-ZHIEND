"""Phase 3 / 15B / 15C — dedup, license & PII policy, leakage-free splits.

Rewritten in Phase 15 after an independent audit found **2,091 duplicate
records** in the Phase 14 output (1,897 train / 153 val / 41 test). Root cause:
pass 2 looked up ``seen[hash]`` for every occurrence, so every copy of a retained
document resolved to the retained index and was written again.

The fix is not "check the counter, trust the counter": every occurrence is gated
by a ``written`` set, and after writing, the outputs are re-read and audited for
duplicate hashes, cross-split overlap and cluster disjointness. The audit result
is part of the report, so a regression fails loudly instead of quietly.

Pipeline
--------
pass 1  compact metadata only (hash, 64-perm MinHash, length, source, licence
        eligibility, PII findings) — 1.6 GB of text never sits in RAM.
pass 2  cluster, apply policy filters, assign each **cluster** to one split,
        then stream again and write each retained document exactly once.
audit   re-read the written files; assert uniqueness and disjointness.

Usage
-----
    python scripts/verify_corpus.py --in-root data/curated_v5 \
        --out-root data/curated_v6 --pii-policy exclude
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from license_policy import (POLICY_VERSION, decision_from_document,  # noqa: E402
                            detect_pii, redact_pii)

N_HASH = 64
N_BANDS = 8
SHINGLE = 5
MAX_SHINGLES = 192
PROCESSING_VERSION = "phase15.v2"
DEFAULT_SEED = 20261116


def iter_jsonl(path: str):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def shingles(text: str, k: int = SHINGLE, cap: int = MAX_SHINGLES) -> list[str]:
    words = text.lower().split()
    if len(words) < k:
        return [" ".join(words)] if words else []
    n = len(words) - k + 1
    stride = max(1, n // cap)
    return [" ".join(words[i:i + k]) for i in range(0, n, stride)][:cap]


def minhash(sh: list[str], n: int = N_HASH) -> np.ndarray:
    if not sh:
        return np.zeros(n, dtype=np.uint64)
    base = np.fromiter((zlib.crc32(s.encode("utf-8")) for s in sh),
                       dtype=np.uint64, count=len(sh))
    seeds = np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    mixed = base[:, None] ^ seeds[None, :]
    mixed = (mixed * np.uint64(0xBF58476D1CE4E5B9)) ^ (mixed >> np.uint64(27))
    return mixed.min(axis=0)


def lsh_candidates(sig: np.ndarray, band_rows: int = N_HASH // N_BANDS):
    n = sig.shape[0]
    for band in range(N_BANDS):
        block = np.ascontiguousarray(sig[:, band * band_rows:(band + 1) * band_rows])
        order = np.lexsort([block[:, i] for i in range(band_rows - 1, -1, -1)])
        key = [tuple(row) for row in block[order].tolist()]
        start = 0
        for i in range(1, n + 1):
            if i == n or key[i] != key[start]:
                if i - start > 1:
                    yield [int(order[j]) for j in range(start, i)]
                start = i


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _rel(path: str) -> str:
    """relpath that survives a different drive (tests use a C: tmp_path)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return os.path.abspath(path)


def split_bucket(key: str) -> float:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") / 2**64


def audit_outputs(split_paths: dict[str, str]) -> dict:
    """Independent audit of what was actually written to disk."""
    per_split_hashes: dict[str, Counter] = {}
    per_split_clusters: dict[str, set] = {}
    counts: dict[str, int] = {}
    for name, path in split_paths.items():
        hashes: Counter = Counter()
        clusters: set = set()
        n = 0
        if os.path.exists(path):
            for rec in iter_jsonl(path):
                # A document longer than --max-chars is intentionally written as
                # several records, so uniqueness must be judged on the text
                # actually written (text_hash), not on the parent content_hash.
                hashes[rec.get("text_hash") or rec["content_hash"]] += 1
                clusters.add(rec["cluster_id"])
                n += 1
        per_split_hashes[name] = hashes
        per_split_clusters[name] = clusters
        counts[name] = n

    extra_dupes = sum(c - 1 for hc in per_split_hashes.values()
                      for c in hc.values() if c > 1)
    sets = {k: set(v) for k, v in per_split_hashes.items()}
    # Document-level leakage is still judged on content_hash (a chunked parent
    # must not leak either), so it is measured separately from text uniqueness.
    doc_sets = {name: set() for name in split_paths}
    for name, path in split_paths.items():
        if os.path.exists(path):
            for rec in iter_jsonl(path):
                doc_sets[name].add(rec["content_hash"])
    cluster_overlap = {
        "train_val": len(per_split_clusters["train"] & per_split_clusters["val"]),
        "train_test": len(per_split_clusters["train"] & per_split_clusters["test"]),
        "val_test": len(per_split_clusters["val"] & per_split_clusters["test"]),
    }
    return {
        "records_written": counts,
        "unique_hashes_per_split": {k: len(v) for k, v in sets.items()},
        "duplicated_hashes_per_split": {
            k: sum(1 for c in per_split_hashes[k].values() if c > 1) for k in counts},
        "extra_duplicate_records": extra_dupes,
        "train_val_overlap": len(doc_sets["train"] & doc_sets["val"]),
        "train_test_overlap": len(doc_sets["train"] & doc_sets["test"]),
        "val_test_overlap": len(doc_sets["val"] & doc_sets["test"]),
        "text_train_val_overlap": len(sets["train"] & sets["val"]),
        "cluster_overlap_between_splits": cluster_overlap,
        "passed": (extra_dupes == 0
                   and len(doc_sets["train"] & doc_sets["val"]) == 0
                   and len(doc_sets["train"] & doc_sets["test"]) == 0
                   and len(doc_sets["val"] & doc_sets["test"]) == 0
                   and all(v == 0 for v in cluster_overlap.values())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 15 corpus verification")
    ap.add_argument("--in-root", default=os.path.join(ROOT, "data", "curated_v5"))
    ap.add_argument("--out-root", default=None,
                    help="defaults to --in-root (writes splits/reports in place)")
    ap.add_argument("--train", type=float, default=0.90)
    ap.add_argument("--val", type=float, default=0.08)
    ap.add_argument("--test", type=float, default=0.02)
    ap.add_argument("--near-dup-threshold", type=float, default=0.85)
    ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument("--min-chars", type=int, default=100)
    ap.add_argument("--pii-policy", default="exclude", choices=("exclude", "redact", "keep"))
    ap.add_argument("--license-policy", default="enforce", choices=("enforce", "report"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    out_root = args.out_root or args.in_root
    splits_dir = os.path.join(out_root, "splits")
    reports_dir = os.path.join(out_root, "reports")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    t0 = time.time()

    files = sorted(glob.glob(os.path.join(args.in_root, "documents", "*.jsonl")))
    if not files:
        raise SystemExit(f"No documents under {args.in_root}/documents")

    # ── pass 1: compact metadata, dedup by hash ──────────────────────
    hashes: list[str] = []
    lengths: list[int] = []
    meta: list[dict] = []
    sigs: list[np.ndarray] = []
    seen: dict[str, int] = {}
    exact_dup = 0
    duplicate_examples: list[dict] = []
    license_by_source: dict[str, dict] = {}
    pii_docs: list[dict] = []
    pii_kind_counts: Counter = Counter()

    for path in files:
        for doc in iter_jsonl(path):
            text = doc.get("text") or ""
            h = doc.get("content_hash") or hashlib.sha256(text.encode()).hexdigest()
            if h in seen:
                exact_dup += 1
                if len(duplicate_examples) < 20:
                    duplicate_examples.append({"kept_document_id": meta[seen[h]]["document_id"],
                                               "dropped_document_id": doc.get("document_id"),
                                               "source_id": doc.get("source_id")})
                continue
            source_id = doc.get("source_id", "?")
            if source_id not in license_by_source:
                license_by_source[source_id] = decision_from_document(doc).as_dict()
            findings = detect_pii(text)
            if findings:
                for kind, count in findings.items():
                    pii_kind_counts[kind] += count
                pii_docs.append({"document_id": doc.get("document_id"),
                                 "source_id": source_id, "findings": findings})
            seen[h] = len(hashes)
            hashes.append(h)
            lengths.append(len(text))
            meta.append({"document_id": doc.get("document_id"), "source_id": source_id,
                         "language": doc.get("language", "?"),
                         "domain": doc.get("domain", "?"),
                         "quality_flags": doc.get("quality_flags") or [],
                         "pii": findings,
                         "license_status": doc.get("license_status"),
                         "eligible": license_by_source[source_id]["eligible_for_training_mixture"]})
            sigs.append(minhash(shingles(text)))
        print(f"  [pass1] {os.path.basename(path)}: {len(hashes):,} unique so far")

    n = len(hashes)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    sig_array = np.vstack(sigs) if sigs else np.zeros((0, N_HASH), dtype=np.uint64)
    print(f"[pass1] {n:,} unique docs | {exact_dup:,} exact duplicates | {time.time()-t0:.0f}s")

    # ── near-duplicate clustering ────────────────────────────────────
    uf = UnionFind(n)
    merges = pairs = 0
    for members in lsh_candidates(sig_array):
        head = members[0]
        for other in members[1:]:
            pairs += 1
            if float((sig_array[head] == sig_array[other]).mean()) >= args.near_dup_threshold:
                if uf.find(head) != uf.find(other):
                    uf.union(head, other)
                    merges += 1
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    print(f"[near-dup] {pairs:,} pairs | {merges:,} merges | {len(clusters):,} clusters")

    # ── policy filters on cluster representatives ────────────────────
    drops: Counter = Counter()
    keep: dict[int, dict] = {}
    rewrite_text: dict[int, str] = {}
    for root, members in clusters.items():
        rep = max(members, key=lambda i: lengths_arr[i])
        m = meta[rep]
        reasons = []
        if args.license_policy == "enforce" and not m["eligible"]:
            reasons.append("license_ineligible")
        if m["pii"]:
            if args.pii_policy == "exclude":
                reasons.append("pii_excluded")
            elif args.pii_policy == "redact" and "payment_card" in m["pii"]:
                reasons.append("pii_payment_card_excluded")
        if set(m["quality_flags"]) & {"corrupted_encoding", "empty"}:
            reasons.append("corrupted")
        if lengths_arr[rep] < args.min_chars:
            reasons.append("too_short")
        if "high_repetition" in m["quality_flags"]:
            reasons.append("high_repetition")
        if "boilerplate" in m["quality_flags"]:
            reasons.append("boilerplate")
        if reasons:
            for r in reasons:
                drops[r] += len(members)
            continue
        pos = split_bucket(hashes[rep])
        split = ("train" if pos < args.train
                 else "val" if pos < args.train + args.val else "test")
        keep[rep] = {"split": split, "cluster_size": len(members)}
    near_dup_removed = sum(v["cluster_size"] - 1 for v in keep.values())
    print(f"[filter] kept {len(keep):,} clusters | drops {dict(drops)}")

    # ── pass 2: write each retained document exactly once ────────────
    split_paths = {name: os.path.join(splits_dir, f"{name}.jsonl")
                   for name in ("train", "val", "test")}
    handles = {name: open(path, "w", encoding="utf-8") for name, path in split_paths.items()}
    counts: Counter = Counter()
    chars: Counter = Counter()
    per_source: dict[str, Counter] = defaultdict(Counter)
    per_lang: dict[str, Counter] = defaultdict(Counter)
    per_domain: dict[str, Counter] = defaultdict(Counter)
    written: set[str] = set()
    duplicate_suppressed = 0
    redacted = 0

    for path in files:
        for doc in iter_jsonl(path):
            text = doc.get("text") or ""
            h = doc.get("content_hash") or hashlib.sha256(text.encode()).hexdigest()
            idx = seen.get(h)
            if idx is None or idx not in keep:
                continue
            if h in written:                      # the Phase 14 defect
                duplicate_suppressed += 1
                continue
            written.add(h)
            info = keep[idx]
            if args.pii_policy == "redact" and meta[idx]["pii"]:
                text, _ = redact_pii(text)
                redacted += 1
            pieces = ([text] if len(text) <= args.max_chars else
                      [text[i:i + args.max_chars]
                       for i in range(0, len(text), args.max_chars)])
            for k, piece in enumerate(pieces):
                rec = dict(doc)
                rec.update({
                    "text": piece,
                    "text_hash": hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                    "cluster_id": f"c{idx:08d}",
                    "cluster_size": info["cluster_size"],
                    "parent_document_id": doc.get("document_id"),
                    "chunk_index": k,
                    "chunk_count": len(pieces),
                    "split": info["split"],
                    "processing_version": PROCESSING_VERSION,
                    "license_policy_version": POLICY_VERSION,
                    "pii_policy": args.pii_policy,
                })
                handles[info["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[info["split"]] += 1
                chars[info["split"]] += len(piece)
                per_source[info["split"]][doc.get("source_id", "?")] += len(piece)
                per_lang[info["split"]][doc.get("language", "?")] += len(piece)
                per_domain[info["split"]][doc.get("domain", "?")] += len(piece)
    for fh in handles.values():
        fh.close()

    audit = audit_outputs(split_paths)

    # ── reports ──────────────────────────────────────────────────────
    report = {
        "schema_version": 3,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "processing_version": PROCESSING_VERSION,
        "in_root": _rel(args.in_root),
        "out_root": _rel(out_root),
        "inputs": [_rel(f) for f in files],
        "documents_in": n + exact_dup,
        "unique_documents": n,
        "exact_duplicates_removed": exact_dup,
        "exact_duplicates_suppressed_in_pass2": duplicate_suppressed,
        "duplicate_examples": duplicate_examples,
        "near_duplicate": {"hash_permutations": N_HASH, "bands": N_BANDS,
                           "shingle_words": SHINGLE, "max_shingles_per_doc": MAX_SHINGLES,
                           "threshold": args.near_dup_threshold,
                           "candidate_pairs": pairs, "merges": merges,
                           "clusters": len(clusters), "members_removed": near_dup_removed},
        "quality_drops": dict(drops),
        "pii": {"policy": args.pii_policy,
                "documents_flagged": len(pii_docs),
                "finding_counts": dict(pii_kind_counts),
                "excluded_documents": drops.get("pii_excluded", 0)
                + drops.get("pii_payment_card_excluded", 0),
                "redacted_documents": redacted,
                "audit_log": os.path.join(_rel(reports_dir), "pii_audit.jsonl")},
        "license": {"policy": args.license_policy,
                    "sources": license_by_source,
                    "ineligible_documents_removed": drops.get("license_ineligible", 0)},
        "splits": {name: {"chunks": counts[name], "chars": chars[name],
                          "sources": dict(per_source[name]),
                          "languages": dict(per_lang[name]),
                          "domains": dict(per_domain[name])}
                   for name in ("train", "val", "test")},
        "output_audit": audit,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(reports_dir, "dedup_quality_splits.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(reports_dir, "pii_audit.jsonl"), "w", encoding="utf-8") as fh:
        for row in pii_docs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(reports_dir, "license_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(license_by_source, fh, ensure_ascii=False, indent=2)

    for name in ("train", "val", "test"):
        print(f"[split] {name:5s} {counts[name]:>8,} chunks | "
              f"{chars[name]/1e6:>8.1f}M chars | {dict(per_lang[name])}")
    print(f"[pii] policy={args.pii_policy} flagged={len(pii_docs):,} "
          f"excluded={report['pii']['excluded_documents']:,} redacted={redacted:,}")
    print(f"[audit] {audit}")
    print(f"[done] {report['elapsed_seconds']}s -> {_rel(splits_dir)}")
    if not audit["passed"]:
        raise SystemExit("[Fatal] output audit failed: duplicates or split leakage remain")


if __name__ == "__main__":
    main()
