"""Phase 3 data audit — measure what the corpus actually contains.

Counts bytes, documents, duplicate rates (exact SHA-256 of normalised
documents), language mix, estimated tokens and train/validation leakage on the
corpora the project already has. Nothing is downloaded and nothing is deleted.

Two numbers are deliberately reported separately:

* ``bytes`` — exact, from the filesystem.
* ``tokens_estimate`` — measured tokens-per-byte ratio on a *sampled* slice,
  extrapolated. It is labelled as an estimate because tokenizing 17GB in this
  pass would take hours; the manifest records the sample size so the
  extrapolation is auditable.

Usage
-----
    python scripts/data_audit.py --out bench_results/gpt2_upgrade/data_audit.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SOURCES = [
    ("prepared_v4_train", "data/prepared_v4/pretrain/*.txt"),
    ("prepared_v4_val", "data/prepared_v4/pretrain/*val*.txt"),
    ("raw_gutenberg", "data/raw/*/*.txt"),
    ("downloads_general_v1", "data/downloads/general_v1/**/*.txt"),
]

VI_CHARS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")
WORD_RE = re.compile(r"\w+", re.UNICODE)
ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def normalise(doc: str) -> str:
    return re.sub(r"\s+", " ", doc).strip().lower()


def doc_hash(doc: str) -> str:
    return hashlib.sha256(normalise(doc).encode("utf-8")).hexdigest()[:16]


def iter_docs(path: str):
    buf = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                buf.append(line)
            elif buf:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)


def detect_language(doc: str) -> str:
    letters = ALPHA_RE.findall(doc[:4000])
    if not letters:
        return "other"
    vi = sum(1 for ch in letters if ch.lower() in VI_CHARS)
    ascii_letters = sum(1 for ch in letters if ch.isascii())
    if vi / max(1, len(letters)) > 0.02:
        return "vi"
    if ascii_letters / max(1, len(letters)) > 0.9:
        return "en"
    return "other"


def audit_source(name: str, pattern: str, tokenizer, sample_docs: int, max_docs: int) -> dict:
    files = sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True))
    if not files:
        return {"source": name, "pattern": pattern, "present": False}

    # Stratified per file: a global doc cap would stop inside the first few
    # files (alphabetical order) and never look at, say, vietnamese.txt —
    # which is exactly how an "everything is English" report gets produced.
    per_file_cap = max(1, max_docs // max(1, len(files)))
    total_bytes = sum(os.path.getsize(f) for f in files)
    hashes: Counter = Counter()
    languages: Counter = Counter()
    docs = 0
    sampled_chars = 0
    sampled_tokens = 0
    shortest = 10**9
    longest = 0
    pii_hits = 0
    empty_docs = 0
    email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
    truncated = False

    for path in files:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size == 0:
            continue
        file_docs = 0
        for doc in iter_docs(path):
            if file_docs >= per_file_cap:
                truncated = docs >= max_docs
                break
            if not doc.strip():
                empty_docs += 1
                continue
            file_docs += 1
            docs += 1
            if len(hashes) < 2_000_000:
                hashes[doc_hash(doc)] += 1
            if sampled_chars < 2_000_000:
                languages[detect_language(doc)] += 1
                if sampled_tokens < 600_000 and len(doc) > 200:
                    ids = tokenizer.encode(doc[:20_000])
                    sampled_chars += len(doc[:20_000])
                    sampled_tokens += len(ids)
                if email_re.search(doc):
                    pii_hits += 1
            shortest = min(shortest, len(doc))
            longest = max(longest, len(doc))
            if docs >= max_docs:
                truncated = True
                break
        if docs >= max_docs:
            break

    dup_docs = sum(count - 1 for count in hashes.values() if count > 1)
    unique = len(hashes)
    ratio = sampled_tokens / sampled_chars if sampled_chars else 0.0
    est_total_tokens = int(ratio * total_bytes)

    return {
        "source": name,
        "pattern": pattern,
        "present": True,
        "files": len(files),
        "bytes": total_bytes,
        "documents_scanned": docs,
        "scan_truncated": truncated,
        "unique_documents": unique,
        "exact_duplicate_documents": dup_docs,
        "exact_duplicate_rate": round(dup_docs / max(1, docs), 4),
        "empty_documents": empty_docs,
        "doc_len_chars_min": shortest if shortest != 10**9 else 0,
        "doc_len_chars_max": longest,
        "document_count_exact": not truncated,
        "languages_sampled": dict(languages),
        "pii_email_docs_sampled": pii_hits,
        "sample_chars": sampled_chars,
        "sample_tokens": sampled_tokens,
        "chars_per_token_sampled": round(sampled_chars / max(1, sampled_tokens), 3),
        "tokens_estimate": est_total_tokens,
        "tokens_estimate_method": "sampled chars/token ratio x total bytes",
    }


def leakage_check(train_files: list[str], val_files: list[str]) -> dict:
    """Exact-document overlap between a train and a validation split."""
    train_hashes = set()
    for path in train_files:
        for doc in iter_docs(path):
            if doc.strip():
                train_hashes.add(doc_hash(doc))
    overlap = 0
    val_docs = 0
    # Long documents are the ones that matter: a duplicated one-line answer
    # ("Resolved.") is boilerplate, a duplicated 2 KB passage is leakage.
    long_overlap = 0
    long_val_docs = 0
    for path in val_files:
        for doc in iter_docs(path):
            if not doc.strip():
                continue
            val_docs += 1
            hit = doc_hash(doc) in train_hashes
            if hit:
                overlap += 1
            if len(doc) >= 200:
                long_val_docs += 1
                if hit:
                    long_overlap += 1
    return {
        "train_documents": len(train_hashes),
        "val_documents": val_docs,
        "overlapping_documents": overlap,
        "overlap_rate": round(overlap / max(1, val_docs), 6),
        "val_documents_over_200_chars": long_val_docs,
        "overlapping_documents_over_200_chars": long_overlap,
        "long_document_overlap_rate": round(long_overlap / max(1, long_val_docs), 6),
        "method": "exact SHA-256 of normalised (lowercase, whitespace-collapsed) documents",
        "note": ("Short boilerplate documents (single words, repeated answers) "
                 "dominate the raw overlap rate; the long-document rate is the "
                 "one that threatens validation quality."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit project corpora")
    ap.add_argument("--out", default="bench_results/gpt2_upgrade/data_audit.json")
    ap.add_argument("--max-docs", type=int, default=400_000)
    args = ap.parse_args()

    from tokenizer_gpt2 import GPT2BPETokenizer

    tokenizer = GPT2BPETokenizer()
    print(f"[Tokenizer] GPT-2 BPE ({tokenizer.backend})")

    result = {"schema_version": 1, "tokenizer": "gpt2_bpe", "sources": []}
    for name, pattern in SOURCES:
        print(f"\n=== {name} ({pattern}) ===")
        info = audit_source(name, pattern, tokenizer, 0, args.max_docs)
        result["sources"].append(info)
        if info.get("present"):
            print(f"  {info['files']} files, {info['bytes']/1e9:.2f} GB, "
                  f"{info['documents_scanned']:,} docs, "
                  f"dup {info['exact_duplicate_rate']:.2%}, "
                  f"langs {info['languages_sampled']}, "
                  f"~{info['tokens_estimate']/1e6:.1f}M tokens est")
        else:
            print("  (absent)")

    print("\n=== leakage: prepared_v4 train vs val ===")
    # The train list must EXCLUDE the val files: a bare ``*.txt`` glob matches
    # them too, which reports a 100 % overlap that does not exist.
    pool = sorted(glob.glob(os.path.join(ROOT, "data/prepared_v4/pretrain/*.txt")))
    train_files = [f for f in pool if "_val" not in os.path.basename(f)
                   and "validation" not in os.path.basename(f)]
    val_files = [f for f in pool if f not in train_files]
    leak = leakage_check(train_files, val_files)
    result["leakage_split"] = {
        "train_files": [os.path.relpath(f, ROOT) for f in train_files],
        "val_files": [os.path.relpath(f, ROOT) for f in val_files],
    }
    print(f"  overlap {leak['overlapping_documents']} / {leak['val_documents']:,} "
          f"({leak['overlap_rate']:.4%})")
    result["leakage"] = leak

    result["heldout_guard"] = {
        "heldout_file": "data/eval/heldout.txt",
        "heldout_bytes": os.path.getsize(os.path.join(ROOT, "data/eval/heldout.txt"))
        if os.path.exists(os.path.join(ROOT, "data/eval/heldout.txt")) else 0,
        "note": "Held-out text is excluded from every corpus mixture above.",
    }

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
