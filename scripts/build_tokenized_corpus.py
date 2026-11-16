"""Phase 6 — turn curated splits into a uint16 token stream for training.

Reads ``data/curated_v5/splits/*.jsonl`` and writes
``data/gpt2_corpus_v5/{train,val,test}.bin`` in the same framing the trainer
already uses: ``<eot> body <eot>`` per document, uint16, memory-mappable.

The manifest records the exact token counts (a full scan — not an estimate),
per-source/split composition and every hash the trainer checks on resume.

Usage
-----
    python scripts/build_tokenized_corpus.py --max-train-tokens 120000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SPLITS = os.path.join(ROOT, "data", "curated_v5", "splits")
OUT = os.path.join(ROOT, "data", "gpt2_corpus_v5")


def log(msg: str) -> None:
    print(msg, flush=True)


def stream_records(path: str):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: str, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def source_char_counts(split_files: list[str]) -> tuple[Counter, int]:
    """Pass 1: characters per source, so the mixture can be thinned by weight
    instead of by file order (the split files are grouped by source, so a plain
    token cap would keep only the alphabetically first source)."""
    per_source: Counter = Counter()
    total = 0
    for path in split_files:
        for rec in stream_records(path):
            n = len(rec.get("text") or "")
            per_source[rec.get("source_id", "?")] += n
            total += n
    return per_source, total


def build(split_files: list[str], out_bin: str, tokenizer, max_tokens: int,
          log_every: int = 20_000, mix: dict[str, float] | None = None,
          tok_per_char: float = 0.27) -> dict:
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    tmp = out_bin + ".tmp"
    keep_prob: dict[str, float] = {}
    if mix and max_tokens:
        chars_by_source, _ = source_char_counts(split_files)
        present = {s: c for s, c in chars_by_source.items() if c > 0}
        weights = {s: mix.get(s, 0.0) for s in present}
        wsum = sum(weights.values())
        if wsum <= 0:
            raise SystemExit(f"--mix names no source present in the splits: {list(present)}")
        for source, chars in present.items():
            share = weights[source] / wsum
            target_tokens = max_tokens * share
            est_tokens = chars * tok_per_char
            keep_prob[source] = min(1.0, target_tokens / max(1.0, est_tokens))
        log(f"[mix] keep probability per source: "
            + ", ".join(f"{s}={p:.2f}" for s, p in sorted(keep_prob.items())))
    docs = tokens = chars = 0
    per_source: Counter = Counter()
    per_lang: Counter = Counter()
    per_domain: Counter = Counter()
    t0 = time.time()
    stop = False

    with open(tmp, "wb") as fh:
        for path in split_files:
            if stop:
                break
            for rec in stream_records(path):
                text = (rec.get("text") or "").strip()
                if not text:
                    continue
                if keep_prob:
                    prob = keep_prob.get(rec.get("source_id", "?"), 0.0)
                    if prob <= 0:
                        continue
                    # deterministic per-document thinning: no order bias
                    h = hashlib.sha256(str(rec.get("document_id")).encode()).digest()
                    if int.from_bytes(h[:8], "little") / 2**64 >= prob:
                        continue
                ids = tokenizer.encode(text)
                if not ids:
                    continue
                seq = [tokenizer.eot_id] + ids + [tokenizer.eot_id]
                np.asarray(seq, dtype=np.uint16).tofile(fh)
                docs += 1
                tokens += len(seq)
                chars += len(text)
                n = len(seq)
                per_source[rec.get("source_id", "?")] += n
                per_lang[rec.get("language", "?")] += n
                per_domain[rec.get("domain", "?")] += n
                if docs % log_every == 0:
                    rate = tokens / max(1e-9, time.time() - t0)
                    log(f"  {docs:,} docs | {tokens:,} tokens | {rate:,.0f} tok/s")
                if max_tokens and tokens >= max_tokens:
                    stop = True
                    break
    os.replace(tmp, out_bin)
    size = os.path.getsize(out_bin)
    return {
        "path": os.path.relpath(out_bin, ROOT),
        "sha256": sha256_file(out_bin),
        "bytes": size,
        "tokens": tokens,
        "documents": docs,
        "source_chars": chars,
        "chars_per_token": round(chars / max(1, tokens), 3),
        "tokens_per_byte": round(tokens / max(1, size), 4),
        "token_composition": dict(per_source),
        "language_composition": dict(per_lang),
        "domain_composition": dict(per_domain),
        "truncated_at_max_tokens": bool(stop),
        "sources": [os.path.relpath(p, ROOT) for p in split_files],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the v5 token stream")
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-train-tokens", type=int, default=120_000_000,
                    help="0 = use every available training token")
    ap.add_argument("--tokenizer", default="gpt2", choices=["gpt2", "jin"])
    ap.add_argument("--mix", default="wikipedia_en_20231101:0.65,cosmopedia_v2_sample:0.25,"
                                       "prepared_v4:0.08,project_seed:0.02",
                   help="comma list source:weight; empty string = no thinning")
    args = ap.parse_args()

    if args.tokenizer == "gpt2":
        from tokenizer_gpt2 import GPT2BPETokenizer
        tokenizer = GPT2BPETokenizer()
    else:
        from utils import tokenizer_from_state
        tokenizer, _ = tokenizer_from_state()
    print(f"[Tokenizer] {tokenizer.kind} vocab={tokenizer.vocab_size}")

    plan = {
        "train": (["train.jsonl"], args.max_train_tokens),
        "val": (["val.jsonl"], 0),
        "test": (["test.jsonl"], 0),
    }
    manifest = {
        "schema_version": 2,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "revision": os.popen("git rev-parse HEAD").read().strip(),
        "tokenizer": {"kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size,
                      "eot_id": int(tokenizer.eot_id),
                      "hash": hashlib.sha256(json.dumps(
                          tokenizer.to_state_dict(), sort_keys=True).encode()).hexdigest()},
        "curated_manifest": os.path.relpath(
            os.path.join(ROOT, "data", "curated_v5", "raw_manifest.json"), ROOT),
        "framing": "<eot> document <eot>, uint16 little-endian",
    }
    missing = []
    for split, (names, cap) in plan.items():
        paths = [os.path.join(args.splits, n) for n in names]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            missing.append(split)
            continue
        print(f"\n=== {split} ({len(paths)} file(s), cap={cap or 'none'}) ===")
        mix = None
        if split == "train" and args.mix:
            mix = {}
            for part in args.mix.split(","):
                name, _, weight = part.partition(":")
                if name.strip():
                    mix[name.strip()] = float(weight or 1.0)
        manifest[split] = build(paths, os.path.join(args.out, f"{split}.bin"),
                                tokenizer, cap, mix=mix)
        print(f"  {manifest[split]['tokens']:,} tokens | "
              f"{manifest[split]['bytes']/1e6:.1f} MB | "
              f"{manifest[split]['tokens_per_byte']:.3f} tok/byte")

    if missing:
        print(f"[warn] missing splits: {missing}")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(out_path):
        old = json.load(open(out_path, encoding="utf-8"))
        for key in ("train", "val", "test"):
            if key in old and key not in manifest:
                manifest[key] = old[key]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
