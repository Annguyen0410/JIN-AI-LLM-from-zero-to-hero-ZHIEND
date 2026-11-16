"""
Rejection-sampling self-distillation (Phase 3.5).

The idea (safe self-improvement without external labels):

1. Take a set of prompts (from reasoning data, seed conversations, or a
   plain text file, one prompt per line).
2. For each prompt, let the CURRENT JIN AI checkpoint sample ``k=8``
   candidate answers (temperature ~0.8).
3. Score each candidate with an automatic verifier:
   - **math**: exact-match on the reference answer, or numeric-match
     (extract the last number and compare to the reference).
   - **code**: execute the fenced code and run unit tests in a sandbox.
   - **generic**: heuristic fluency + length + repetition checks (no
     external judge), or keep the candidate with the highest mean
     token logprob when no other signal exists.
4. Keep only candidates that pass; emit SFT-ready ChatML documents
   (``<|user|> ... <|assistant|> ... <|endoftext|>``) into JSONL.

Verifier failures are logged per kind so the report shows exactly how
much was kept and why (no inflated numbers).

Usage:
    python self_distill.py --checkpoint checkpoints/jingpt_chat.pt \
        --prompts data/prompts.jsonl --k 8 --out data/distill.jsonl
    python self_distill.py --prompts data/reasoning.jsonl --k 8  # needs --answers-field
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re

import torch

import config
from bpe_tokenizer import BPETokenizer
from chat_template import ROLE_ASSISTANT, ROLE_USER, format_training_document
from utils import build_inference_bundle

NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_last_number(text: str) -> str | None:
    matches = NUMERIC_RE.findall(text)
    return matches[-1] if matches else None


def extract_fenced_code(text: str) -> str | None:
    """Pull the first ```python ... ``` (or bare ```) block."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m2 = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m2.group(1) if m2 else None


def sandbox_run_code(code: str, tests: list[tuple[str, object]]) -> bool:
    """Execute ``code`` in a fresh namespace; run all ``tests``."""
    from tools import python_exec
    assertions = "\n".join(f"assert ({expr}) == {expected!r}" for expr, expected in tests)
    try:
        result = python_exec(code + "\n" + assertions + "\nprint('JIN_TESTS_PASSED')", timeout=2)
        return result.strip().endswith("JIN_TESTS_PASSED") and not any(
            error in result for error in ("Error:", "blocked", "disabled"))
    except (RuntimeError, ValueError):
        return False


def _generic_score(text: str) -> float:
    """Heuristic quality score (higher = better) with no external model."""
    if len(text) < 10:
        return -1.0
    words = text.split()
    if not words:
        return -1.0
    # Penalize heavy repetition (gibberish loops).
    uniq = len(set(words))
    rep_penalty = 1.0 if uniq / max(len(words), 1) > 0.4 else 0.3
    # Prefer moderate length over one-liners.
    length_ok = 0.5 if 8 <= len(words) <= 400 else 0.2
    return length_ok * rep_penalty


def verify_candidate(
    prompt: str, candidate: str, *, reference: str | None = None,
    kind: str | None = None, tests: list[tuple[str, object]] | None = None,
) -> tuple[bool, str]:
    """Return (pass, reason)."""
    if kind == "math" and reference is not None:
        ref_num = extract_last_number(reference)
        cand_num = extract_last_number(candidate)
        if ref_num is not None and cand_num is not None:
            try:
                return (abs(float(cand_num) - float(ref_num)) < 1e-6, "math_numeric")
            except ValueError:
                pass
        if reference.strip() in candidate:
            return True, "math_exact"
        return False, "math_mismatch"
    if kind == "code" and tests:
        code = extract_fenced_code(candidate)
        if code is None:
            return False, "code_no_fence"
        return (sandbox_run_code(code, tests), "code_tests")
    score = _generic_score(candidate)
    return (score > 0.35, f"generic_{score:.2f}")


def _mean_token_logprob(model, tokenizer, text: str, device: str) -> float:
    ids = tokenizer.encode(text, allow_special=True)
    if len(ids) < 2:
        return -1e9
    idx = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    target = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(idx)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1), reduction="mean"
        )
    return -float(loss.item())


def load_prompts(path: str) -> list[dict]:
    """Load prompts: JSONL records or plain lines.

    Supported record keys: ``prompt`` (required), optional ``reference``,
    ``kind`` (math|code|generic), ``tests`` (list of [expr, expected]).
    """
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "{":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec.get("prompt"), str) and rec["prompt"].strip():
                    # build_reasoning_data.py records use ``answer`` (the
                    # verified ground truth); the math verifier reads
                    # ``reference`` — accept both so reasoning data can be
                    # distilled directly.
                    rec.setdefault("reference", rec.get("answer"))
                    out.append(rec)
        else:
            for line in f:
                line = line.strip()
                if line:
                    out.append({"prompt": line})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejection-sampling self-distillation (Phase 3.5)")
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    parser.add_argument("--prompts", required=True,
                        help="JSONL (prompt/reference/kind/tests) or plain-text prompts")
    parser.add_argument("--out", required=True, help="Output SFT JSONL (chat format)")
    parser.add_argument("--pref-out", default=None,
                        help="Optional preference-pairs JSONL {prompt, chosen, rejected} "
                             "(best vs worst passed candidate) for DPO/ORPO.")
    parser.add_argument("--k", type=int, default=8, help="Candidates per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None, choices=config.DEVICE_CHOICES)
    args = parser.parse_args()
    args.out = config.output_path(args.out)
    if args.pref_out:
        args.pref_out = config.output_path(args.pref_out)

    config.set_device(args.device)
    device = config.DEVICE
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer, ckpt = build_inference_bundle(args.checkpoint)
    is_bpe = isinstance(tokenizer, BPETokenizer)
    print(f"[Distill] Loaded {ckpt.get('_loaded_from', args.checkpoint)} on {device}")

    prompts = load_prompts(args.prompts)
    print(f"[Distill] {len(prompts):,} prompts, k={args.k}")

    kept_records: list[dict] = []
    pref_pairs: list[dict] = []
    stats: dict[str, int] = {}
    total_candidates = 0

    for i, rec in enumerate(prompts):
        prompt = rec["prompt"]
        reference = rec.get("reference")
        kind = rec.get("kind")
        tests = [tuple(t) for t in rec.get("tests", [])] if rec.get("tests") else None

        best: tuple[float, str] | None = None
        worst: tuple[float, str] | None = None
        for _ in range(args.k):
            with torch.no_grad():
                idx = torch.tensor(
                    [tokenizer.encode(prompt, allow_special=True)],
                    dtype=torch.long, device=device,
                )
                out = model.generate(
                    idx, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k if args.top_k and args.top_k > 0 else None,
                    top_p=args.top_p,
                    repetition_penalty=1.1,
                )
            new_ids = out[0].tolist()[idx.size(1):]
            answer = tokenizer.decode(new_ids).strip()
            if not answer:
                continue
            total_candidates += 1
            passed, reason = verify_candidate(
                prompt, answer, reference=reference, kind=kind, tests=tests
            )
            stats[reason] = stats.get(reason, 0) + 1
            if not passed:
                continue
            score = _generic_score(answer)
            if best is None or score > best[0]:
                best = (score, answer)
            if worst is None or score < worst[0]:
                worst = (score, answer)

        if best is None:
            continue
        # Final quality gate: skip near-empty / repetitive best answers.
        if _generic_score(best[1]) <= 0.35:
            continue
        messages = [
            {"role": ROLE_USER, "content": prompt},
            {"role": ROLE_ASSISTANT, "content": best[1]},
        ]
        doc = format_training_document(messages, default_system=None)
        kept_records.append({
            "messages": messages,
            "source": "self_distill",
            "kind": kind or "generic",
            "prompt": prompt,
        })
        # Preference pair for DPO/ORPO: chosen = best verified answer,
        # rejected = worst verified answer (never unverified text).
        if args.pref_out and worst is not None and worst[1] != best[1]:
            pref_pairs.append({
                "prompt": prompt,
                "chosen": best[1],
                "rejected": worst[1],
                "kind": kind or "generic",
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Distill] candidates: {total_candidates:,} | kept prompts: {len(kept_records):,}")
    print(f"[Distill] verifier stats: {dict(sorted(stats.items()))}")
    print(f"[Distill] SFT data -> {args.out}")

    if args.pref_out:
        os.makedirs(os.path.dirname(args.pref_out) or ".", exist_ok=True)
        with open(args.pref_out, "w", encoding="utf-8") as f:
            for rec in pref_pairs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Distill] preference pairs ({len(pref_pairs):,}) -> {args.pref_out}")


if __name__ == "__main__":
    main()