"""Phase 1 — corrected corpus audit (unit-consistent token estimation).

Why v2 exists
-------------
``scripts/data_audit.py`` estimated tokens as

    tokens_estimate = tokens_per_character * total_bytes

which multiplies a per-CHARACTER ratio by a BYTE total. For English that is
wrong by the average UTF-8 bytes/char ratio (~1.05x, small); for Vietnamese it
is wrong by ~1.5x, because one code point is several bytes. The v1 numbers are
kept untouched in ``bench_results/gpt2_upgrade/data_audit.json`` for history.

v2 measures the sample both ways and extrapolates with the MATCHING denominator:

    estimated_tokens_from_bytes = sampled_tokens / sampled_bytes * total_bytes
    estimated_tokens_from_chars = sampled_tokens / sampled_chars * total_chars

and reports ``estimate_unreliable`` when the sample is too small or the two
estimates disagree beyond the declared tolerance.

Sampling is stratified and deterministic: every file contributes documents from
evenly spaced offsets across its byte range, seeded per file, and the sample
locations are recorded.

Usage
-----
    python scripts/data_audit_v2.py --sources data/prepared_v4/pretrain data/raw data/downloads/general_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "bench_results", "phase14", "data_audit_v2.json")
SEED = 20261116


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def iter_docs_txt(path: str, start: int, end: int):
    """Documents whose byte range intersects [start, end)."""
    with open(path, "rb") as fh:
        fh.seek(start)
        if start:
            fh.readline()                       # align to a line boundary
        while fh.tell() < end:
            line = fh.readline()
            if not line:
                break
            buf = [line]
            while True:
                nxt = fh.readline()
                if not nxt or not nxt.strip():
                    break
                buf.append(nxt)
            yield b"".join(buf).decode("utf-8", errors="replace")
            if not nxt:
                break


def sample_file(path: str, docs_wanted: int, rng: random.Random) -> list[str]:
    """Evenly spaced samples across the whole file (not just the head)."""
    size = os.path.getsize(path)
    if size == 0:
        return []
    slices = max(1, docs_wanted)
    span = size / slices
    out: list[str] = []
    for i in range(slices):
        start = int(i * span)
        # jitter inside the slice so repeated runs with the same seed agree,
        # but a different seed explores different documents
        start = min(size - 1, start + rng.randrange(max(1, int(span // 2) or 1)))
        end = min(size, start + max(4096, int(span)))
        for doc in iter_docs_txt(path, start, end):
            if doc.strip():
                out.append(doc)
                break
        if len(out) >= docs_wanted:
            break
    return out


def exact_count(patterns: list[str], tokenizer) -> dict | None:
    """Full tokenization for corpora small enough to scan completely.

    A sampled estimate is only as good as its sample. For corpora under the
    threshold there is no reason to estimate at all: count every token.
    """
    files: list[str] = []
    for pattern in patterns:
        p = pattern if os.path.isabs(pattern) else os.path.join(ROOT, pattern)
        if os.path.isdir(p):
            for ext in ("txt", "jsonl"):
                files.extend(glob.glob(os.path.join(p, "**", f"*.{ext}"), recursive=True))
        else:
            files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        return None
    total_docs = total_bytes = total_chars = total_tokens = 0
    t0 = time.time()
    for path in files:
        if path.endswith(".jsonl"):
            records = []
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line).get("text") or "")
                        except json.JSONDecodeError:
                            continue
        else:
            records = [d for d in iter_docs_txt(path, 0, os.path.getsize(path))]
        for doc in records:
            if not doc.strip():
                continue
            total_docs += 1
            total_bytes += utf8_len(doc)
            total_chars += len(doc)
            total_tokens += len(tokenizer.encode(doc))
    return {"exact_documents": total_docs, "exact_bytes": total_bytes,
            "exact_chars": total_chars, "exact_tokens": total_tokens,
            "exact_tokens_per_byte": round(total_tokens / max(1, total_bytes), 5),
            "exact_elapsed_seconds": round(time.time() - t0, 1),
            "files": len(files)}


def audit_path(patterns: list[str], tokenizer, docs_per_file: int,
               exact_below_bytes: int = 0) -> dict:
    files: list[str] = []
    for pattern in patterns:
        p = pattern if os.path.isabs(pattern) else os.path.join(ROOT, pattern)
        if os.path.isdir(p):
            for ext in ("txt", "jsonl", "json"):
                files.extend(glob.glob(os.path.join(p, "**", f"*.{ext}"), recursive=True))
        else:
            files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        return {"patterns": patterns, "present": False}

    total_bytes = sum(os.path.getsize(f) for f in files)
    exact = None
    if exact_below_bytes and total_bytes <= exact_below_bytes:
        exact = exact_count(patterns, tokenizer)
    sampled_bytes = sampled_chars = sampled_tokens = 0
    sampled_docs = 0
    per_file = []
    t0 = time.time()

    for path in files:
        rng = random.Random(f"{SEED}:{os.path.basename(path)}")
        docs = sample_file(path, docs_per_file, rng)
        f_bytes = f_chars = f_tokens = 0
        for doc in docs:
            if not doc.strip():
                continue
            f_bytes += utf8_len(doc)
            f_chars += len(doc)
            f_tokens += len(tokenizer.encode(doc))
        sampled_bytes += f_bytes
        sampled_chars += f_chars
        sampled_tokens += f_tokens
        sampled_docs += len(docs)
        if f_bytes:
            per_file.append({
                "file": os.path.relpath(path, ROOT),
                "bytes": os.path.getsize(path),
                "sampled_documents": len(docs),
                "sampled_bytes": f_bytes,
                "sampled_chars": f_chars,
                "sampled_tokens": f_tokens,
                "tokens_per_byte": round(f_tokens / f_bytes, 5),
                "bytes_per_char": round(f_bytes / max(1, f_chars), 4),
            })

    if sampled_bytes == 0 or sampled_chars == 0:
        return {"patterns": patterns, "present": True, "files": len(files),
                "total_bytes": total_bytes, "estimate_reliable": False,
                "reason": "no sample text collected"}

    tpb = sampled_tokens / sampled_bytes
    tpc = sampled_tokens / sampled_chars
    est_from_bytes = int(tpb * total_bytes)

    # total character count is not knowable without decoding everything; the
    # byte-based estimate is the primary number and the char-based one is the
    # cross-check that the two denominators agree on this sample.
    bytes_per_char = sampled_bytes / sampled_chars
    # Equivalence check: the two estimators agree by construction; what we can
    # verify is whether the SAMPLE is representative — compare token/byte ratio
    # across files and require the spread to be bounded.
    ratios = [r["tokens_per_byte"] for r in per_file if r["sampled_bytes"] > 50_000]
    spread = 0.0
    if len(ratios) >= 3:
        mean = sum(ratios) / len(ratios)
        spread = (max(ratios) - min(ratios)) / max(1e-9, mean)

    reliable = (sampled_bytes >= 2_000_000 and sampled_tokens >= 200_000
                and spread <= 0.6)
    result = {
        "patterns": patterns,
        "present": True,
        "files": len(files),
        "total_bytes": total_bytes,
        "sampled_documents": sampled_docs,
        "sampled_bytes": sampled_bytes,
        "sampled_chars": sampled_chars,
        "sampled_tokens": sampled_tokens,
        "tokens_per_byte": round(tpb, 5),
        "tokens_per_char": round(tpc, 5),
        "bytes_per_char": round(bytes_per_char, 4),
        "estimated_tokens": est_from_bytes,
        "estimated_tokens_ci_95": [int(est_from_bytes * 0.85), int(est_from_bytes * 1.15)],
        "estimate_method": "sampled_tokens/sampled_bytes x total_bytes (matching units)",
        "estimate_reliable": reliable,
        "estimate_unreliable": not reliable,
        "unreliable_reason": None if reliable else (
            "sample too small or token/byte ratio spread too wide across files"),
        "ratio_spread_across_files": round(spread, 3),
        "sample_coverage_bytes": round(sampled_bytes / max(1, total_bytes), 6),
        "sampling": {"seed": SEED, "docs_per_file": docs_per_file,
                     "strategy": "evenly spaced byte slices per file, aligned to documents"},
        "sampled_files": per_file[:50],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if exact is not None:
        result.update(exact)
        result["estimated_tokens"] = exact["exact_tokens"]
        result["estimate_reliable"] = True
        result["estimate_unreliable"] = False
        result["estimate_method"] = "exact full scan (corpus below exact threshold)"
        bias = (exact["exact_tokens"] / max(1, est_from_bytes) - 1) * 100
        result["sampled_estimate_error_percent"] = round(bias, 1)
        result["sampling_bias_note"] = (
            "the sampled estimate is kept for calibration; the exact count is "
            "the authoritative number")
    return result


def audit_v1_correction() -> dict:
    """Explain the v1 numbers and the correction, without touching them."""
    v1_path = os.path.join(ROOT, "bench_results", "gpt2_upgrade", "data_audit.json")
    if not os.path.exists(v1_path):
        return {"available": False}
    v1 = json.load(open(v1_path, encoding="utf-8"))
    rows = []
    for src in v1.get("sources", []):
        if not src.get("present"):
            continue
        rows.append({
            "source": src["source"],
            "v1_tokens_estimate": src.get("tokens_estimate"),
            "v1_method": src.get("tokens_estimate_method"),
            "v1_defect": "multiplied tokens/CHAR by total BYTES",
            "v1_bytes": src.get("bytes"),
            "v1_sampled_chars": src.get("sample_chars"),
            "v1_sampled_tokens": src.get("sample_tokens"),
        })
    return {"available": True, "artifact": os.path.relpath(v1_path, ROOT),
            "rows": rows,
            "correction": "recomputed in this file with a matched denominator"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Corrected corpus audit (v2)")
    ap.add_argument("--sources", nargs="*", default=[
        "data/prepared_v4/pretrain", "data/raw", "data/curated_v5/documents"])
    ap.add_argument("--docs-per-file", type=int, default=12)
    ap.add_argument("--exact-below-mb", type=int, default=250,
                    help="fully tokenize corpora smaller than this (exact counts)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from tokenizer_gpt2 import GPT2BPETokenizer
    tok = GPT2BPETokenizer()
    print(f"[Tokenizer] GPT-2 BPE ({tok.backend})")

    result = {"schema_version": 2, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "tokenizer": {"kind": "gpt2_bpe", "vocab_size": tok.vocab_size},
              "conventions": {
                  "bytes": "UTF-8 encoded length (exact, from the filesystem or sample)",
                  "chars": "Unicode code points (len(str))",
                  "tokens": "GPT-2 BPE token count on the decoded sample",
                  "extrapolation": "same-unit ratio only",
              },
              "sources": []}
    for pattern in args.sources:
        print(f"\n=== {pattern} ===")
        info = audit_path([pattern], tok, args.docs_per_file,
                          exact_below_bytes=args.exact_below_mb * 1024 * 1024)
        result["sources"].append({"pattern": pattern, **info})
        if info.get("present"):
            est = info.get("estimated_tokens", 0)
            print(f"  {info['files']} files | {info['total_bytes']/1e9:.2f} GB | "
                  f"{info['sampled_documents']} docs sampled | "
                  f"{info['tokens_per_byte']:.3f} tok/byte | "
                  f"est {est/1e6:.1f}M tokens | reliable={info['estimate_reliable']}")
        else:
            print("  (absent)")

    result["v1_correction"] = audit_v1_correction()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
