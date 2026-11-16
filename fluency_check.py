"""
Quick "is this model ready to chat?" diagnostic for JIN AI.

Loads a checkpoint, runs a fixed set of probe prompts through the chat
template, scores each reply on three blunt heuristics, and prints a
readable report.

The score is on purpose simple — it cannot tell you the model is *good*,
but it can tell you the model is producing real words instead of token
soup.

Heuristics
----------
1. **Real-word ratio.** Fraction of whitespace tokens in the reply that
   look like real words (length 1-20, mostly letters, vowel + consonant,
   no triple-letter runs like ``thalllll``).
2. **Sentence rhythm.** Are there any ``.``/``!``/``?`` followed by a
   space, capping the reply with a real sentence? Penalises pure ramble.
3. **Echo / collapse.** Does the reply repeat the same 1-2 tokens over
   and over (a classic sign of an undertrained or over-decayed model).

Run::

    python fluency_check.py
    python fluency_check.py --checkpoint checkpoints/jingpt_chat.pt
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

import config
from bpe_tokenizer import BPETokenizer
from chat_template import (
    DEFAULT_SYSTEM_PROMPT,
    END_TAG,
    EOT_TAG,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_TAG,
    format_conversation,
)
from utils import build_inference_bundle


PROBES = [
    "Hi!",
    "Who are you?",
    "What is 2 + 2?",
    "What is the capital of Japan?",
    "Write hello world in Python.",
    "Tell me a one-sentence joke.",
    "How are you?",
    "Bye.",
]


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SENTENCE_RE = re.compile(r"[.!?](?:\s|$)")
_TRIPLE_RUN_RE = re.compile(r"([A-Za-z])\1\1")
_VOWELS = set("aeiouy")


_COMMON_SHORT_WORDS = {
    "i", "a", "is", "it", "in", "to", "of", "or", "on", "at", "as", "an",
    "be", "by", "we", "he", "do", "no", "so", "up", "us", "if", "my",
    "the", "and", "you", "are", "for", "but", "not", "can", "yes",
    "all", "out", "any", "one", "how", "who", "why", "now", "see",
}


def _word_looks_real(word: str) -> bool:
    low = word.lower()
    if not 1 <= len(low) <= 20:
        return False
    if _TRIPLE_RUN_RE.search(low):
        return False
    if not (set(low) & _VOWELS):
        return False
    if not any(c not in _VOWELS for c in low):
        return False
    # Single- and two-letter tokens count only if they're a real common word.
    if len(low) <= 2 and low not in _COMMON_SHORT_WORDS:
        return False
    return True


def fluency_score(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"real_word_ratio": 0.0, "has_sentence": False,
                "echo_ratio": 1.0, "avg_word_len": 0.0, "score": 0.0}

    words = _WORD_RE.findall(text)
    if not words:
        return {"real_word_ratio": 0.0, "has_sentence": False,
                "echo_ratio": 1.0, "avg_word_len": 0.0, "score": 0.0}

    real = [w for w in words if _word_looks_real(w)]
    real_ratio = len(real) / len(words)

    has_sentence = bool(_SENTENCE_RE.search(text))
    avg_word_len = sum(len(w) for w in words) / len(words)
    # Real prose averages ~4-5 chars per word; fragmented token soup drops
    # to ~2-3. Map that onto a 0..1 score.
    length_score = max(0.0, min(1.0, (avg_word_len - 2.0) / 2.5))

    counts: dict[str, int] = {}
    for w in words:
        counts[w.lower()] = counts.get(w.lower(), 0) + 1
    top = max(counts.values()) if counts else 1
    echo_ratio = top / len(words)
    echo_score = max(0.0, 1.0 - max(0.0, echo_ratio - 0.15) * 4)

    score = (
        0.50 * real_ratio
        + 0.25 * length_score
        + 0.15 * (1.0 if has_sentence else 0.0)
        + 0.10 * echo_score
    )
    return {
        "real_word_ratio": real_ratio,
        "has_sentence": has_sentence,
        "echo_ratio": echo_ratio,
        "avg_word_len": avg_word_len,
        "score": round(score, 3),
    }


def chat_once(model, tokenizer, prompt: str, *, system: str, max_new_tokens: int) -> str:
    text = format_conversation(
        [{"role": ROLE_SYSTEM, "content": system},
         {"role": ROLE_USER, "content": prompt}],
        add_generation_prompt=True,
        default_system=None,
    )
    if not isinstance(tokenizer, BPETokenizer):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer.encode(text, allow_special=True)
    idx = torch.tensor([ids], dtype=torch.long, device=config.DEVICE)

    stop_ids: set[int] = set()
    if isinstance(tokenizer, BPETokenizer):
        for tag in (END_TAG, EOT_TAG):
            tid = tokenizer.special_to_id.get(tag)
            if tid is not None:
                stop_ids.add(tid)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.15,
        stop_ids=stop_ids if stop_ids else None,
    )
    new = out[0].tolist()[len(ids):]
    if stop_ids and new and new[-1] in stop_ids:
        new = new[:-1]
    reply = tokenizer.decode(new).strip()
    if isinstance(tokenizer, BPETokenizer):
        for tag in ROLE_TAG.values():
            cut = reply.find(tag)
            if cut != -1:
                reply = reply[:cut].strip()
    return reply


def verdict(scores: list[float]) -> tuple[str, str]:
    avg = sum(scores) / max(len(scores), 1)
    if avg >= 0.80:
        return "READY", "Replies look natural. The model is ready for normal chat."
    if avg >= 0.60:
        return "OK", ("Replies are mostly OK but rough. More instruction-tuning "
                      "steps will sharpen them.")
    if avg >= 0.40:
        return "TRAINING", ("Replies are forming but still messy. Run more "
                             "epochs of train.py --mode instruct.")
    return "BROKEN", ("Replies are token soup — the model is severely "
                       "undertrained or trained on the wrong data. Rebuild "
                       "data with build_instruct_dataset.py and run "
                       "train.py --mode instruct for at least a few thousand steps.")


def _resolve_checkpoint(preferred: str) -> str:
    """Pick the first existing file from ``preferred`` and sensible defaults."""

    chat_pt = os.path.join(config.CHECKPOINT_DIR, "jingpt_chat.pt")
    candidates = [
        preferred,
        chat_pt,
        config.BEST_CHECKPOINT_PATH,
        config.LATEST_CHECKPOINT_PATH,
    ]
    seen: set[str] = set()
    for path in candidates:
        ap = os.path.abspath(path)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.exists(path):
            if path != preferred:
                print(f"[fluency] Note: {preferred!r} not found — using {path!r} instead.")
            return path
    raise FileNotFoundError(
        "No checkpoint found. Train first with train.py (--mode instruct for chat tuning)."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Quick fluency check on a JIN AI checkpoint.")
    p.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    p.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="NAME",
        choices=config.DEVICE_CHOICES,
        help="Device override. Default: JINGPT_DEVICE or auto.",
    )
    args = p.parse_args()
    config.set_device(args.device)

    enc = sys.stdout.encoding or "utf-8"
    def _safe(s: str) -> str:
        return s.encode(enc, errors="replace").decode(enc, errors="replace")

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    print(f"[fluency] Loading checkpoint: {ckpt_path}")
    model, tokenizer, meta = build_inference_bundle(ckpt_path)
    print(f"[fluency] iteration {meta.get('iteration', '?')}, "
          f"loss {meta.get('loss', float('nan'))}")
    print()

    scores: list[float] = []
    for prompt in PROBES:
        reply = chat_once(model, tokenizer, prompt,
                          system=args.system, max_new_tokens=args.max_new_tokens)
        s = fluency_score(reply)
        scores.append(s["score"])
        bar = "#" * int(s["score"] * 20)
        print(f"> {prompt}")
        print(f"  reply : {_safe(reply[:160])}")
        print(f"  real  : {s['real_word_ratio'] * 100:5.1f}%   "
              f"sentence: {'Y' if s['has_sentence'] else 'N'}   "
              f"echo: {s['echo_ratio'] * 100:5.1f}%   "
              f"avg-word-len: {s['avg_word_len']:.1f}   "
              f"score: {s['score']:.2f} [{bar:<20}]")
        print()

    label, advice = verdict(scores)
    print("=" * 60)
    avg = sum(scores) / max(len(scores), 1)
    print(f"OVERALL: {label}   (avg score {avg:.2f} across {len(scores)} probes)")
    print(advice)


if __name__ == "__main__":
    main()
