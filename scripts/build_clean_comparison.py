"""Phase 15E — build a comparison slice that is unseen by BOTH checkpoints.

Why this exists
---------------
Phase 14 compared the pilot and the new checkpoint on v5 validation. v5 contains
`prepared_v4`, which is exactly the corpus the pilot trained on for 1.41 epochs,
so calling that slice "unseen by both models" was wrong.

Neither checkpoint can be evaluated on a source it has never seen *and* the
other has never seen, because the four available sources are split between them:

    pilot      : prepared_v4 only
    phase14    : wikipedia + cosmopedia + prepared_v4 + project_seed

So the strict slice is defined at **document** level, not source level:

    candidate  : v6 test split (document-disjoint from v6 train/val)
    source     : wikipedia or cosmopedia  (never in the pilot's corpus)
    exclusion  : content_hash not present in the v5 train split
                 (never in the Phase 14 checkpoint's training tokens)

Both conditions must hold, so a document in the slice was in neither training
set. The slice is written with its own manifest (hashes, counts, sources) so the
comparison can be re-derived.

Usage
-----
    python scripts/build_clean_comparison.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PILOT_SOURCES = {"prepared_v4"}                      # from the pilot corpus manifest
PHASE14_SOURCES = {"wikipedia_en_20231101", "cosmopedia_v2_sample",
                   "prepared_v4", "project_seed"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hashes(path: str) -> set[str]:
    out: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            out.add(json.loads(line)["content_hash"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the clean unseen-by-both slice")
    ap.add_argument("--v6-test", default=os.path.join(ROOT, "data", "curated_v6",
                                                      "splits", "test.jsonl"))
    ap.add_argument("--v5-train", default=os.path.join(ROOT, "data", "curated_v5",
                                                       "splits", "train.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "curated_v6",
                                                      "benchmark"))
    ap.add_argument("--max-docs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1515)
    args = ap.parse_args()

    t0 = time.time()
    phase14_train_hashes = read_hashes(args.v5_train)
    print(f"[input] v5 train hashes: {len(phase14_train_hashes):,}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "clean_unseen.jsonl")
    skipped_overlap = 0
    skipped_source = 0

    # Read candidates first, then select per source. Reading in file order and
    # stopping at --max-docs produced a slice that was 100 % one source, because
    # the split file groups documents by source: the quota was filled before the
    # second source was ever reached.
    candidates: dict[str, list[dict]] = {}
    with open(args.v6_test, encoding="utf-8", errors="replace") as src:
        for line in src:
            if not line.strip():
                continue
            doc = json.loads(line)
            source = doc.get("source_id")
            if source in PILOT_SOURCES or source not in PHASE14_SOURCES:
                skipped_source += 1
                continue
            if doc["content_hash"] in phase14_train_hashes:
                skipped_overlap += 1
                continue
            candidates.setdefault(source, []).append(doc)

    total_available = sum(len(v) for v in candidates.values())
    if total_available == 0:
        raise SystemExit("[clean slice] no eligible documents")

    # Deterministic order inside each source, then a proportional quota so the
    # slice mirrors the test split's source mix instead of one file block.
    for source, docs in candidates.items():
        docs.sort(key=lambda d: hashlib.sha256(
            f"{args.seed}.{d.get('document_id')}".encode()).digest())

    quotas = {s: args.max_docs * len(d) / total_available for s, d in candidates.items()}
    take = {s: min(len(candidates[s]), int(q)) for s, q in quotas.items()}
    for s in sorted(candidates, key=lambda s: -(quotas[s] - take[s])):
        while sum(take.values()) < args.max_docs and take[s] < len(candidates[s]):
            take[s] += 1
            if sum(take.values()) >= args.max_docs:
                break
    for s in sorted(candidates, key=lambda s: quotas[s] - take[s], reverse=True):
        while sum(take.values()) > args.max_docs and take[s] > 0:
            take[s] -= 1

    selected: list[dict] = []
    for source in sorted(candidates):
        selected.extend(candidates[source][:take[source]])
    selected.sort(key=lambda d: hashlib.sha256(
        f"{args.seed}.{d.get('document_id')}".encode()).digest())

    counts: Counter = Counter()
    languages: Counter = Counter()
    chars = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for doc in selected:
            record = dict(doc)
            record["clean_slice"] = True
            record["unseen_by"] = ["jin_pilot_best_160", "jin_phase14_wiki_best_640"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            chars += len(doc.get("text") or "")
            counts[doc.get("source_id")] += 1
            languages[doc.get("language", "?")] += 1
    kept = len(selected)

    manifest = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "Phase 15E clean comparison: unseen by both checkpoints",
        "definition": {
            "candidates": "data/curated_v6/splits/test.jsonl (document-disjoint)",
            "excluded_sources_for_pilot": sorted(PILOT_SOURCES),
            "excluded_overlap": "content_hash present in data/curated_v5/splits/train.jsonl",
            "requires": "source not used by the pilot AND hash not in the Phase 14 train split",
            "selection": "proportional quota per source, sha256(seed.document_id) order",
            "seed": args.seed,
            "available_per_source": {s: len(d) for s, d in candidates.items()},
            "selected_per_source": dict(counts),
        },
        "documents": kept,
        "characters": chars,
        "sources": dict(counts),
        "languages": dict(languages),
        "skipped_source_filter": skipped_source,
        "skipped_hash_overlap_with_phase14_train": skipped_overlap,
        "phase14_train_hashes": len(phase14_train_hashes),
        "output": os.path.relpath(out_path, ROOT),
        "output_sha256": sha256_file(out_path),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "clean_unseen_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"[clean slice] {kept:,} documents | {chars/1e6:.2f}M chars | "
          f"sources {dict(counts)} | langs {dict(languages)}")
    print(f"[excluded] {skipped_overlap:,} hash-overlap with Phase 14 train, "
          f"{skipped_source:,} by source rule")
    print(f"Wrote {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
