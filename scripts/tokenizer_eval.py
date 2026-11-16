"""Phase 2 tokenizer research harness.

Measures the existing JIN BPE (vocab 16,384) against the GPT-2 byte-level BPE
(vocab 50,257) on real slices of the project corpus. Numbers are measured, not
estimated: every metric comes from encoding actual text.

Metrics
-------
tokens/char, tokens/word, Vietnamese/English fragmentation (subword tokens per
whitespace word), code token efficiency, unicode round-trip, replacement/unknown
rate and throughput (chars/second).

Usage
-----
    python scripts/tokenizer_eval.py --out bench_results/gpt2_upgrade/tokenizer_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SAMPLES = {
    "vietnamese": ("data/prepared_v4/pretrain/vietnamese.txt", 400_000),
    "english": ("data/prepared_v4/pretrain/knowledge.txt", 400_000),
    "stories": ("data/prepared_v4/pretrain/stories.txt", 300_000),
    "code": ("data/prepared_v4/pretrain/code.txt", 300_000),
    "math": ("data/prepared_v4/pretrain/math.txt", 300_000),
}

PROFILES = ("jin", "gpt2")


def read_sample(rel_path: str, max_chars: int) -> str:
    path = os.path.join(ROOT, rel_path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read(max_chars)


def is_vietnamese_word(word: str) -> bool:
    """A token containing a Latin letter with a diacritic or đ/Đ."""
    if not any(ch.isalpha() for ch in word):
        return False
    if any(ch in "đĐ" for ch in word):
        return True
    return any(unicodedata.combining(ch) or ord(ch) > 0x1E00 for ch in word)


def measure(tokenizer, text: str, name: str) -> dict:
    words = text.split()
    t0 = time.time()
    ids = tokenizer.encode(text)
    elapsed = max(1e-9, time.time() - t0)
    decoded = tokenizer.decode(ids)

    vi_words = [w for w in words if is_vietnamese_word(w)]
    vi_ids = len(tokenizer.encode(" ".join(vi_words))) if vi_words else 0
    en_words = [w for w in words if w.isascii() and any(c.isalpha() for c in w)]
    en_ids = len(tokenizer.encode(" ".join(en_words))) if en_words else 0

    roundtrip = decoded == text
    replacement = decoded.count("\ufffd")

    return {
        "sample": name,
        "chars": len(text),
        "words": len(words),
        "tokens": len(ids),
        "tokens_per_char": round(len(ids) / max(1, len(text)), 4),
        "chars_per_token": round(len(text) / max(1, len(ids)), 3),
        "tokens_per_word": round(len(ids) / max(1, len(words)), 4),
        "vietnamese_words": len(vi_words),
        "vietnamese_tokens_per_word": round(vi_ids / max(1, len(vi_words)), 4) if vi_words else None,
        "english_words": len(en_words),
        "english_tokens_per_word": round(en_ids / max(1, len(en_words)), 4) if en_words else None,
        "roundtrip_exact": roundtrip,
        "replacement_chars": replacement,
        "chars_per_second": int(len(text) / elapsed),
        "vietnamese_diacritics_preserved": roundtrip or "\ufffd" not in decoded,
    }


def roundtrip_stress(tokenizer) -> dict:
    """Unicode / combining-mark / emoji / byte-fallback checks."""
    cases = {
        "vietnamese_lower": "tiếng việt có dấu: ă â đ ê ô ơ ư",
        "vietnamese_upper": "TIẾNG VIỆT HOA: Ă Â Đ Ê Ô Ơ Ư",
        "combining_nfd": unicodedata.normalize("NFD", "Hòa bình"),
        "mixed_scripts": "日本語 中文 한국어 العربية русский",
        "emoji": "hello 🌍🚀 world",
        "code": "def f(x: int) -> list[int]: return [x**2 for x in range(10)]",
        "whitespace": "a\tb\n\nc   d",
        "numbers": "1,234.567 -0.0001 3.14e-9",
    }
    out = {}
    for name, text in cases.items():
        ids = tokenizer.encode(text)
        out[name] = {
            "tokens": len(ids),
            "roundtrip_exact": tokenizer.decode(ids) == text,
            "decoded_preview": tokenizer.decode(ids)[:60],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare JIN BPE vs GPT-2 BPE")
    ap.add_argument("--out", default="bench_results/gpt2_upgrade/tokenizer_eval.json")
    args = ap.parse_args()

    from tokenizer_gpt2 import load_tokenizer

    result: dict = {"profiles": {}, "samples": {}}
    toks = {}
    for profile in PROFILES:
        tok = load_tokenizer(profile)
        toks[profile] = tok
        result["profiles"][profile] = {
            "vocab_size": tok.vocab_size,
            "kind": getattr(tok, "kind", profile),
            "backend": getattr(tok, "backend", "project"),
            "special_tokens": dict(getattr(tok, "special_to_id", {})),
        }
        print(f"[{profile}] vocab={tok.vocab_size:,} backend={getattr(tok, 'backend', 'project')}")

    for name, (rel, limit) in SAMPLES.items():
        text = read_sample(rel, limit)
        print(f"\n=== {name} ({len(text):,} chars from {rel}) ===")
        result["samples"][name] = {"source": rel}
        for profile, tok in toks.items():
            stats = measure(tok, text, name)
            result["samples"][name][profile] = stats
            print(f"  {profile:5s} tok/char {stats['tokens_per_char']:.4f} "
                  f"({stats['chars_per_token']:.2f} c/t)  "
                  f"tok/word {stats['tokens_per_word']:.3f}  "
                  f"VI {stats['vietnamese_tokens_per_word']}  "
                  f"roundtrip {stats['roundtrip_exact']}  "
                  f"{stats['chars_per_second']:,} ch/s")

    result["roundtrip_stress"] = {p: roundtrip_stress(t) for p, t in toks.items()}

    # A single comparable aggregate: cost of encoding the same mixed text.
    mixed = "".join(read_sample(rel, limit // 4) for rel, limit in SAMPLES.values())
    result["aggregate"] = {}
    for profile, tok in toks.items():
        stats = measure(tok, mixed, "mixed")
        result["aggregate"][profile] = stats
        print(f"\n[mixed {len(mixed):,} chars] {profile}: {stats['tokens']:,} tokens "
              f"({stats['chars_per_token']:.2f} chars/token, "
              f"{stats['chars_per_second']:,} ch/s)")

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
