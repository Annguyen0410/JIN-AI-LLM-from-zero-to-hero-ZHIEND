"""Phase 3 — deduplication, quality filtering and leakage-free splits.

Two passes, so 1.6 GB of text never has to sit in RAM at once (this machine has
~2.5 GB free while training runs elsewhere):

Pass 1  read every document, keep only compact metadata: content hash, a
        64-permutation MinHash signature, length, source, language, domain.
Pass 2  cluster, filter, assign each **cluster** to one partition, then stream
        the documents again and write the kept representatives out.

Splitting at cluster level (not document level) is what prevents a
near-duplicate family from straddling train/val/test.

Usage
-----
    python scripts/verify_corpus.py --train 0.90 --val 0.08 --test 0.02
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
V5 = os.path.join(ROOT, "data", "curated_v5")
DOCS = os.path.join(V5, "documents")
SPLITS = os.path.join(V5, "splits")
REPORTS = os.path.join(V5, "reports")

N_HASH = 64
N_BANDS = 8
SHINGLE = 5
MAX_SHINGLES = 192           # evenly spaced sample: bounds cost on long documents
SEED = 20261116


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


def split_bucket(key: str) -> float:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") / 2**64


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 dedup + quality + splits")
    ap.add_argument("--train", type=float, default=0.90)
    ap.add_argument("--val", type=float, default=0.08)
    ap.add_argument("--test", type=float, default=0.02)
    ap.add_argument("--near-dup-threshold", type=float, default=0.85)
    ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(SPLITS, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(DOCS, "*.jsonl")))
    if not files:
        raise SystemExit("No documents — run scripts/ingest_corpus.py first.")

    # ── pass 1: compact metadata only ────────────────────────────────
    hashes: list[str] = []
    lengths = np.zeros(0, dtype=np.int64)
    meta: list[tuple[str, str, str, list[str]]] = []      # source, lang, domain, flags
    file_of: list[int] = []
    sig_list: list[np.ndarray] = []
    exact_dup = 0
    seen: dict[str, int] = {}

    lengths = []
    for fi, path in enumerate(files):
        for doc in iter_jsonl(path):
            text = doc.get("text") or ""
            h = doc.get("content_hash") or hashlib.sha256(text.encode()).hexdigest()
            if h in seen:                                 # exact duplicate
                exact_dup += 1
                continue
            seen[h] = len(hashes)
            hashes.append(h)
            lengths.append(len(text))
            meta.append((doc.get("source_id", "?"), doc.get("language", "?"),
                         doc.get("domain", "?"), doc.get("quality_flags") or []))
            file_of.append(fi)
            sig_list.append(minhash(shingles(text)))
        print(f"  [pass1] {os.path.basename(path)}: {len(hashes):,} unique so far")
    n = len(hashes)
    lengths = np.asarray(lengths, dtype=np.int64)
    sigs = np.vstack(sig_list) if sig_list else np.zeros((0, N_HASH), dtype=np.uint64)
    print(f"[pass1] {n:,} unique docs | {exact_dup:,} exact duplicates | "
          f"{time.time()-t0:.0f}s | sigs {sigs.nbytes/1e6:.0f} MB")

    # ── cluster ──────────────────────────────────────────────────────
    uf = UnionFind(n)
    merges = pairs = 0
    for members in lsh_candidates(sigs):
        head = members[0]
        for other in members[1:]:
            pairs += 1
            if float((sigs[head] == sigs[other]).mean()) >= args.near_dup_threshold:
                if uf.find(head) != uf.find(other):
                    uf.union(head, other)
                    merges += 1
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    print(f"[near-dup] {pairs:,} candidate pairs | {merges:,} merges | "
          f"{len(clusters):,} clusters | {time.time()-t0:.0f}s")

    # ── quality filter + cluster → partition ─────────────────────────
    drops: Counter = Counter()
    keep: dict[int, tuple[str, int]] = {}        # rep index -> (split, chunk_count)
    for root, members in clusters.items():
        rep = max(members, key=lambda i: lengths[i])
        src, lang, domain, flags = meta[rep]
        reasons = []
        if set(flags) & {"corrupted_encoding", "empty"}:
            reasons.append("corrupted")
        if lengths[rep] < args.min_chars:
            reasons.append("too_short")
        if "high_repetition" in flags:
            reasons.append("high_repetition")
        if "boilerplate" in flags:
            reasons.append("boilerplate")
        if reasons:
            for r in reasons:
                drops[r] += len(members)
            continue
        pos = split_bucket(f"{hashes[rep]}")
        split = ("train" if pos < args.train else
                 "val" if pos < args.train + args.val else "test")
        keep[rep] = (split, len(members))
    dropped_near = sum(c for v in keep.values() for c in [v[1] - 1])
    print(f"[quality] kept {len(keep):,} clusters | drops {dict(drops)} | "
          f"near-dup members removed {dropped_near:,}")

    # ── pass 2: stream again and write ───────────────────────────────
    handles = {name: open(os.path.join(SPLITS, f"{name}.jsonl"), "w", encoding="utf-8")
               for name in ("train", "val", "test")}
    counts = Counter()
    chars = Counter()
    per_source: dict[str, Counter] = defaultdict(Counter)
    per_lang: dict[str, Counter] = defaultdict(Counter)
    per_domain: dict[str, Counter] = defaultdict(Counter)
    idx = -1
    written_clusters = 0
    for fi, path in enumerate(files):
        for doc in iter_jsonl(path):
            text = doc.get("text") or ""
            h = doc.get("content_hash") or hashlib.sha256(text.encode()).hexdigest()
            position = seen.get(h)
            if position is None:
                continue
            idx = position
            if idx not in keep:
                continue
            split, cluster_size = keep[idx]
            pieces = ([text] if len(text) <= args.max_chars else
                      [text[i:i + args.max_chars]
                       for i in range(0, len(text), args.max_chars)])
            for k, piece in enumerate(pieces):
                rec = dict(doc)
                rec.update({"cluster_id": f"c{idx:08d}", "cluster_size": cluster_size,
                            "parent_document_id": doc.get("document_id"),
                            "chunk_index": k, "chunk_count": len(pieces),
                            "text": piece, "split": split})
                handles[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[split] += 1
                chars[split] += len(piece)
                per_source[split][doc.get("source_id", "?")] += len(piece)
                per_lang[split][doc.get("language", "?")] += len(piece)
                per_domain[split][doc.get("domain", "?")] += len(piece)
            written_clusters += 1
    for fh in handles.values():
        fh.close()

    # ── leakage check on what was actually written ───────────────────
    hashes_by_split = {}
    for name in ("train", "val", "test"):
        s = set()
        for doc in iter_jsonl(os.path.join(SPLITS, f"{name}.jsonl")):
            s.add(doc["content_hash"])
        hashes_by_split[name] = s
    leakage = {
        "train_val_exact_overlap": len(hashes_by_split["train"] & hashes_by_split["val"]),
        "train_test_exact_overlap": len(hashes_by_split["train"] & hashes_by_split["test"]),
        "val_test_exact_overlap": len(hashes_by_split["val"] & hashes_by_split["test"]),
        "cluster_level_split": True,
        "clusters_written": written_clusters,
    }

    report = {
        "schema_version": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": [os.path.relpath(f, ROOT) for f in files],
        "documents_in": n + exact_dup,
        "unique_documents": n,
        "exact_duplicates_removed": exact_dup,
        "near_duplicate": {"hash_permutations": N_HASH, "bands": N_BANDS,
                           "shingle_words": SHINGLE, "max_shingles_per_doc": MAX_SHINGLES,
                           "threshold": args.near_dup_threshold,
                           "candidate_pairs": pairs, "merges": merges,
                           "clusters": len(clusters),
                           "members_removed": dropped_near},
        "quality_drops": dict(drops),
        "splits": {name: {"chunks": counts[name], "chars": chars[name],
                          "sources": dict(per_source[name]),
                          "languages": dict(per_lang[name]),
                          "domains": dict(per_domain[name])}
                   for name in ("train", "val", "test")},
        "leakage": leakage,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(REPORTS, "dedup_quality_splits.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    for name in ("train", "val", "test"):
        print(f"[split] {name:5s} {counts[name]:>8,} chunks | "
              f"{chars[name]/1e6:>8.1f}M chars | {dict(per_lang[name])}")
    print(f"[leakage] {leakage}")
    print(f"[done] {report['elapsed_seconds']}s -> data/curated_v5/splits/")


if __name__ == "__main__":
    main()
