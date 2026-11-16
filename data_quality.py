"""
Data quality pipeline (Phase 3): dedupe, filters, language detection,
and contamination checks. From scratch — numpy + stdlib only.

Pipeline stages (all applied to plain-text documents):

1. :func:`quality_filter` — Gopher/C4-style heuristics: length bounds,
   letter ratio, line repetition, stopword ratio, special-char density,
   boilerplate/menu markers, OCR-garbled text.
2. :func:`dedupe_minhash` — document-level near-duplicate removal with
   MinHash (5-gram shingles, 128 hash bands) + LSH banding; also reports
   the exact-duplicate count.
3. :func:`detect_language` — lightweight n-gram profile (char 3-grams)
   with per-language profiles trained from ``data/lang/`` seed texts
   (fallback: built-in vi/en profiles), so vi/en mixing is controllable.
4. :func:`check_contamination` — removes any document sharing a
   sufficiently long n-gram with ``data/eval/probes.jsonl`` (benchmark
   leakage guard). MUST run in every build_* pipeline.

Every stage emits counts into a report dict; :func:`main` writes a
JSON report next to the output file.

Usage:
    python data_quality.py --input data/train.txt --output data/train_clean.txt
    python data_quality.py --input data/chat_train.txt --output data/chat_train_clean.txt --lang-filter vi,en
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

import config

# ── Quality filter thresholds (Gopher/C4-inspired, reduced) ────────────

MIN_DOC_CHARS = 200
MAX_DOC_CHARS = 200_000
# Documents longer than MAX_DOC_CHARS are split into ~this-size segments
# instead of being dropped (Wikipedia-style dumps produce 1M+ char docs).
SEGMENT_CHARS = 100_000
MIN_LETTER_RATIO = 0.55
MAX_LINE_REPEAT_RATIO = 0.35
MAX_SPECIAL_CHAR_RATIO = 0.30
MAX_BOILERPLATE_HITS = 2
OCR_ERROR_RATIO = 0.08
LONG_LINE_MIN = 300
LONG_LINE_RATIO_LIMIT = 0.25

# Common boilerplate / menu / navigation markers (lowercased substrings).
BOILERPLATE_MARKERS = (
    "sign up",
    "subscribe",
    "cookie policy",
    "privacy policy",
    "terms of service",
    "all rights reserved",
    "menu",
    "navigation",
    "click here",
    "read more",
    "share this",
    "follow us",
    "advertisement",
    "home page",
    "skip to content",
    "log in",
    "create account",
    "newsletter",
)

# Garbage tokens often produced by broken OCR.
OCR_ERROR_PATTERNS = (
    re.compile(r"[Œœ]|\\u[0-9a-fA-F]{4}|Ã[‚€ˆ‰ŠŽ]|â€[™œ]", re.UNICODE),
    re.compile(r"[0-9a-fA-F]{8,}", re.UNICODE),  # long hex runs
    re.compile(r"[|]{4,}"),                       # fence-ish noise
)

EN_STOPWORDS = frozenset(
    "the a an and or but if then else for while of to in on at by with from as is are was were be been being "
    "have has had do does did will would shall should can could may might must not no yes you your we our they "
    "their it its this that these those i me my he him his she her them what which who whom when where why how "
    "all any both each few more most other some such only own same so than too very just about".split()
)
VI_STOPWORDS = frozenset(
    "và của là trong có được cho với các không một người những này đó cũng như khi tại theo để về từ ra vào "
    "nên sẽ đã đang thì còn rất nhiều hơn nhất mỗi mọi cả hai bởi vì nếu hay hoặc do qua trên dưới cùng sau "
    "trước giữa".split()
)

# ── MinHash parameters ──────────────────────────────────────────────────
SHINGLE_SIZE = 5
NUM_HASHES = 128
LSH_BANDS = 16          # 128 / 16 = 8 rows per band
LSH_THRESHOLD = 0.65    # ~(1/16)^(1/8) similarity gate for candidates


def _hash_shingles(text: str) -> list[int]:
    """Return a stable integer hash per 5-gram shingle (word level)."""
    words = re.findall(r"[A-Za-zÀ-ỹ0-9]+", text.lower())
    if len(words) < SHINGLE_SIZE:
        return []
    return [
        int(hashlib.md5(" ".join(words[i : i + SHINGLE_SIZE]).encode("utf-8")).hexdigest()[:8], 16)
        for i in range(len(words) - SHINGLE_SIZE + 1)
    ]


def _minhash_signature(shingles: list[int], num_hashes: int = NUM_HASHES) -> np.ndarray:
    """MinHash signature: for each hash fn, min of (h XOR seed_i) over shingles."""
    if not shingles:
        return np.zeros(num_hashes, dtype=np.uint32)
    arr = np.array(shingles, dtype=np.uint32)
    sig = np.empty(num_hashes, dtype=np.uint32)
    # Deterministic hash family: h_i(x) = (x * a_i + b_i) mod 2^32.
    rng = np.random.RandomState(42)
    a = rng.randint(1, 2**31, size=num_hashes).astype(np.uint32)
    b = rng.randint(0, 2**31, size=num_hashes).astype(np.uint32)
    for i in range(num_hashes):
        vals = arr * a[i] + b[i]
        sig[i] = vals.min()
    return sig


def estimate_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float((sig_a == sig_b).mean())


def dedupe_minhash(
    docs: list[str],
    *,
    threshold: float = LSH_THRESHOLD,
    num_hashes: int = NUM_HASHES,
    bands: int = LSH_BANDS,
) -> tuple[list[str], list[dict]]:
    """Remove near-duplicate documents via MinHash + LSH banding.

    Returns ``(kept_docs, report)``. ``report`` lists each removed doc
    with its nearest kept twin and the estimated similarity. Exact
    duplicates (identical text) are removed first via hashing.
    """
    if not docs:
        return docs, {
            "input_docs": 0, "kept_docs": 0, "removed_docs": 0,
            "exact_removed": 0, "near_dup_removed": 0,
            "removed_ratio": 0.0, "elapsed_s": 0.0,
        }
    rows = bands
    if num_hashes % bands != 0:
        bands = max(1, num_hashes // rows)

    t0 = time.time()
    exact: dict[str, list[int]] = {}
    sigs: list[np.ndarray | None] = []
    for i, doc in enumerate(docs):
        exact.setdefault(doc, []).append(i)
        sigs.append(_minhash_signature(_hash_shingles(doc), num_hashes) if len(doc) > MIN_DOC_CHARS else None)

    kept_flags = [True] * len(docs)
    removed: list[dict] = []

    # 1) exact duplicates: keep the first, drop the rest.
    for dup_idx_list in exact.values():
        if len(dup_idx_list) > 1:
            for idx in dup_idx_list[1:]:
                kept_flags[idx] = False
                removed.append(
                    {"kind": "exact", "index": idx, "twin": dup_idx_list[0],
                     "similarity": 1.0}
                )

    # 2) near-duplicates via LSH candidate pairs.
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, sig in enumerate(sigs):
        if sig is None or not kept_flags[i]:
            continue
        band_len = num_hashes // bands
        for band in range(bands):
            key = (band, int(hashlib.md5(sig[band * band_len : (band + 1) * band_len].tobytes()).hexdigest()[:12], 16))
            buckets.setdefault(key, []).append(i)

    for members in buckets.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            if not kept_flags[members[a]]:
                continue
            for b in range(a + 1, len(members)):
                if not kept_flags[members[b]]:
                    continue
                sim = estimate_similarity(sigs[members[a]], sigs[members[b]])
                if sim >= threshold:
                    # keep the shorter-or-first; drop the later one.
                    drop, keep = members[b], members[a]
                    kept_flags[drop] = False
                    removed.append(
                        {"kind": "near_dup", "index": drop, "twin": keep,
                         "similarity": round(sim, 4)}
                    )

    kept = [d for d, flag in zip(docs, kept_flags) if flag]
    summary = {
        "input_docs": len(docs),
        "kept_docs": len(kept),
        "removed_docs": len(removed),
        "exact_removed": sum(1 for r in removed if r["kind"] == "exact"),
        "near_dup_removed": sum(1 for r in removed if r["kind"] == "near_dup"),
        "removed_ratio": round(len(removed) / max(len(docs), 1), 4),
        "elapsed_s": round(time.time() - t0, 2),
    }
    return kept, summary


# ── Quality filter ─────────────────────────────────────────────────────

def quality_filter(doc: str) -> tuple[bool, dict]:
    """Return ``(keep, reasons)``. ``reasons`` lists which checks failed."""
    reasons: list[str] = []
    if len(doc) < MIN_DOC_CHARS:
        reasons.append("too_short")
    if len(doc) > MAX_DOC_CHARS:
        reasons.append("too_long")

    n = len(doc)
    letters = sum(c.isalpha() for c in doc)
    if n > 0 and letters / n < MIN_LETTER_RATIO:
        reasons.append("low_letter_ratio")

    lines = [ln.strip() for ln in doc.splitlines() if ln.strip()]
    if lines:
        repeated = len(lines) - len(set(lines))
        if repeated / len(lines) > MAX_LINE_REPEAT_RATIO:
            reasons.append("repeated_lines")

    words = doc.split()
    if words:
        en_hits = sum(1 for w in words if w.lower() in EN_STOPWORDS)
        vi_hits = sum(1 for w in words if w.lower() in VI_STOPWORDS)
        if en_hits / len(words) < 0.01 and vi_hits / len(words) < 0.01:
            reasons.append("no_stopwords")

    special = sum(1 for c in doc if not (c.isalnum() or c.isspace()))
    if n > 0 and special / n > MAX_SPECIAL_CHAR_RATIO:
        reasons.append("too_many_special_chars")

    lowered = doc.lower()
    boilerplate_hits = sum(1 for m in BOILERPLATE_MARKERS if m in lowered)
    if boilerplate_hits > MAX_BOILERPLATE_HITS:
        reasons.append("boilerplate")

    # Long single lines are fine (prose paragraphs). Flag only when a
    # substantial fraction of ALL lines are >300 chars (minified code,
    # CSV dumps, copied terminal output).
    long_lines = sum(1 for ln in doc.splitlines() if len(ln) > LONG_LINE_MIN)
    if long_lines >= 2 and lines and long_lines / len(lines) > LONG_LINE_RATIO_LIMIT:
        reasons.append("long_lines")

    ocr_hits = sum(1 for p in OCR_ERROR_PATTERNS if p.search(doc))
    if ocr_hits >= 2:
        reasons.append("ocr_garbled")

    return (not reasons, reasons)


def _segment_long_doc(doc: str, max_chars: int = SEGMENT_CHARS) -> list[str]:
    """Split an oversized document into ~max_chars segments on paragraph
    boundaries (hard-splitting any single mega-paragraph)."""
    if len(doc) <= max_chars:
        return [doc]
    segments: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in re.split(r"\n\s*\n", doc):
        while len(para) > max_chars:
            # hard-split giant paragraphs
            if buf:
                segments.append("\n\n".join(buf))
                buf, buf_len = [], 0
            segments.extend(para[i : i + max_chars] for i in range(0, len(para), max_chars))
            para = ""
        if not para:
            continue
        if buf_len + len(para) + 2 > max_chars and buf:
            segments.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para) + 2
    if buf:
        segments.append("\n\n".join(buf))
    return segments


def filter_docs(docs: list[str]) -> tuple[list[str], dict]:
    expanded: list[str] = []
    segmented_docs = 0
    for doc in docs:
        segs = _segment_long_doc(doc)
        if len(segs) > 1:
            segmented_docs += 1
        expanded.extend(segs)

    kept: list[str] = []
    reasons_counter: dict[str, int] = {}
    for doc in expanded:
        ok, reasons = quality_filter(doc)
        if ok:
            kept.append(doc)
        else:
            for r in reasons:
                reasons_counter[r] = reasons_counter.get(r, 0) + 1
    return kept, {
        "input": len(docs),
        "expanded_segments": len(expanded),
        "segmented_docs": segmented_docs,
        "kept": len(kept),
        "removed": len(expanded) - len(kept),
        "reasons": reasons_counter,
    }


# ── Language detection (char n-gram profiles) ──────────────────────────

def _char_ngrams(text: str, n: int = 3) -> Counter:
    clean = re.sub(r"\s+", " ", text.lower())
    return Counter(clean[i : i + n] for i in range(len(clean) - n + 1))


def train_language_profiles(
    texts: dict[str, list[str]], n: int = 3, top_k: int = 300
) -> dict[str, dict[str, float]]:
    """Train char-3gram profiles ``{lang: {ngram: prob}}`` from seed texts."""
    profiles: dict[str, dict[str, float]] = {}
    for lang, chunks in texts.items():
        counter: Counter = Counter()
        for chunk in chunks:
            counter.update(_char_ngrams(chunk, n))
        total = sum(counter.values())
        top = counter.most_common(top_k)
        profiles[lang] = {g: c / total for g, c in top}
    return profiles


# Built-in fallback profiles (small, deterministic, enough to separate vi/en).
def _default_profiles() -> dict[str, dict[str, float]]:
    en_seed = [
        "the quick brown fox jumps over the lazy dog and then runs away quickly",
        "this is a sample english text with common words like the and of to in",
        "language models are trained on large amounts of text data every day",
        "the government announced a new policy for the people of the nation",
    ]
    vi_seed = [
        "con mèo nhảy qua hàng rào rồi chạy nhanh vào trong vườn",
        "đây là một đoạn văn tiếng việt với những từ phổ biến và của là",
        "mô hình ngôn ngữ được huấn luyện trên một lượng lớn văn bản",
        "chính phủ đã công bố chính sách mới cho người dân của đất nước",
    ]
    return train_language_profiles({"en": en_seed, "vi": vi_seed})


_LANG_PROFILES_CACHE: dict[str, dict[str, float]] | None = None


def detect_language(text: str, top_k: int = 5) -> str:
    """Return the best-guess language ('en' | 'vi' | 'unknown')."""
    global _LANG_PROFILES_CACHE
    if _LANG_PROFILES_CACHE is None:
        _LANG_PROFILES_CACHE = _default_profiles()
    profiles = _LANG_PROFILES_CACHE
    grams = _char_ngrams(text)
    scores = {}
    for lang, prof in profiles.items():
        s = 0.0
        hits = 0
        for g, p in prof.items():
            if g in grams:
                s += grams[g] * p
                hits += 1
        scores[lang] = s / max(hits, 1)
    best = max(scores, key=scores.get)
    return best if scores[best] > 1e-6 else "unknown"


def filter_by_language(
    docs: list[str], allowed: set[str], report: dict | None = None
) -> list[str]:
    kept: list[str] = []
    lang_counter: dict[str, int] = {}
    for doc in docs:
        lang = detect_language(doc)
        lang_counter[lang] = lang_counter.get(lang, 0) + 1
        if lang in allowed:
            kept.append(doc)
    if report is not None:
        report["language_distribution"] = lang_counter
    return kept


# ── Contamination check (benchmark leakage guard) ──────────────────────

def load_probe_texts(probes_path: str = config.PROBES_PATH) -> list[str]:
    """Load question + reference answer strings from probes.jsonl."""
    texts: list[str] = []
    if not os.path.exists(probes_path):
        return texts
    with open(probes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("prompt", "question", "reference", "exact"):
                val = rec.get(key)
                if isinstance(val, str) and val:
                    texts.append(val)
    return texts


def _contamination_ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    words = re.findall(r"[A-Za-zÀ-ỹ0-9]+", text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _contamination_ngrams_adaptive(text: str, n: int = 8) -> set[tuple[str, ...]]:
    """Probe-side n-grams: shrink ``n`` for short probe texts so even a
    6-word probe guards against verbatim leakage in a document."""
    words = re.findall(r"[A-Za-zÀ-ỹ0-9]+", text.lower())
    n_use = min(n, len(words))
    if n_use < 4:
        return set()
    return {tuple(words[i : i + n_use]) for i in range(len(words) - n_use + 1)}


def check_contamination(
    docs: list[str],
    probes_path: str = config.PROBES_PATH,
    n: int = 13,
) -> tuple[list[str], list[dict]]:
    """Drop any document sharing a >=n word n-gram with the probe set."""
    probe_texts = load_probe_texts(probes_path)
    probe_ngrams: set[tuple[str, ...]] = set()
    for pt in probe_texts:
        probe_ngrams |= _contamination_ngrams_adaptive(pt, n)
    if not probe_ngrams:
        return docs, []

    kept: list[str] = []
    removed: list[dict] = []
    for idx, doc in enumerate(docs):
        # A document is contaminated if it contains ANY probe n-gram
        # verbatim (as a contiguous word sequence).
        doc_words = re.findall(r"[A-Za-zÀ-ỹ0-9]+", doc.lower())
        doc_set = {tuple(doc_words[i : i + ng_len])
                   for ng_len in set(len(g) for g in probe_ngrams)
                   for i in range(len(doc_words) - ng_len + 1)}
        if doc_set & probe_ngrams:
            removed.append({"index": idx, "kind": "contaminated"})
        else:
            kept.append(doc)
    return kept, removed


# ── Full pipeline ──────────────────────────────────────────────────────

def split_documents(text: str) -> list[str]:
    """Split a corpus file into documents.

    Priority: ``<doc>`` markers (pretrain corpus), then ``<|endoftext|>``
    (chat corpus — each conversation is one document), then blank-line
    blocks as a last resort.
    """
    if "<doc" in text:
        parts = re.split(r"<doc[^>]*>|</doc>", text)
        return [p.strip() for p in parts if p.strip()]
    if "<|endoftext|>" in text:
        parts = text.split("<|endoftext|>")
        return [p.strip() for p in parts if p.strip()]
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def run_pipeline(
    text: str,
    *,
    lang_filter: str | None = None,
    do_dedupe: bool = True,
    do_contamination: bool = True,
    probes_path: str = config.PROBES_PATH,
) -> tuple[str, dict]:
    """Run all stages over corpus text; return (clean_text, report)."""
    t0 = time.time()
    report: dict = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    docs = split_documents(text)
    report["docs_input"] = len(docs)

    # 1) quality filter
    docs, q_report = filter_docs(docs)
    report["quality"] = q_report

    # 2) language filter (optional)
    if lang_filter:
        allowed = {x.strip().lower() for x in lang_filter.split(",") if x.strip()}
        docs = filter_by_language(docs, allowed, report)
        report["lang_filter"] = lang_filter

    # 3) dedupe
    if do_dedupe:
        docs, d_summary = dedupe_minhash(docs)
        report["dedupe"] = {
            "input": d_summary["input_docs"],
            "removed": d_summary["removed_docs"],
            "exact": d_summary["exact_removed"],
            "near_dup": d_summary["near_dup_removed"],
            "removed_ratio": d_summary["removed_ratio"],
        }

    # 4) contamination
    if do_contamination:
        docs, removed_contam = check_contamination(docs, probes_path=probes_path)
        report["contamination"] = {"removed": len(removed_contam)}

    report["docs_kept"] = len(docs)
    report["elapsed_s"] = round(time.time() - t0, 2)
    out_text = "\n\n".join(docs)
    if out_text:
        out_text += "\n"
    return out_text, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Data quality pipeline (Phase 3)")
    parser.add_argument("--input", required=True, help="Corpus file to clean")
    parser.add_argument("--output", required=True, help="Cleaned corpus file")
    parser.add_argument("--lang-filter", default=None,
                        help="Comma-separated allowed languages, e.g. vi,en")
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--no-contamination", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Run even on very large inputs (dedupe needs RAM).")
    parser.add_argument("--probes", default=config.PROBES_PATH)
    parser.add_argument("--report", default=None, help="JSON report path (default: <output>.report.json)")
    args = parser.parse_args()
    args.output = config.output_path(args.output)
    if args.report:
        args.report = config.output_path(args.report)

    input_size = os.path.getsize(args.input)
    if input_size > 1_000_000_000 and not args.force:
        raise SystemExit(
            "[DataQuality] Input is > 1 GB. MinHash dedupe + n-gram sets need "
            "several GB of RAM. Run this tool on the chat corpus "
            "(data/chat_train.txt) instead, or pass --force to override."
        )

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    clean_text, report = run_pipeline(
        text,
        lang_filter=args.lang_filter,
        do_dedupe=not args.no_dedupe,
        do_contamination=not args.no_contamination,
        probes_path=args.probes,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(clean_text)

    report["chars_input"] = len(text)
    report["chars_output"] = len(clean_text)
    report_path = args.report or (args.output + ".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DataQuality] {report['docs_input']:,} docs -> {report['docs_kept']:,} kept")
    if "dedupe" in report:
        d = report["dedupe"]
        print(f"[DataQuality] dedupe: {d['removed']:,} removed "
              f"({d['exact']:,} exact, {d['near_dup']:,} near-dup)")
    if "contamination" in report:
        print(f"[DataQuality] contamination: {report['contamination']['removed']:,} removed")
    print(f"[DataQuality] chars: {len(text):,} -> {len(clean_text):,}")
    print(f"[DataQuality] report -> {report_path}")


if __name__ == "__main__":
    main()