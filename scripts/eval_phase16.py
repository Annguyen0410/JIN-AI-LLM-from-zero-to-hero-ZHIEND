"""Phase 16 §2 — evaluator that scores every token it says it scored.

What Phase 15 got wrong
-----------------------
``document_windows()`` only kept windows where ``start + seq_len + 1 <= len``.
A document shorter than one window therefore contributed **zero** tokens, and a
document's final partial window was discarded — while the report still counted
those documents in ``documents``. The loss itself was clean (no padding), but the
coverage was not: "24 documents" could mean 24 documents, or 7 documents and 17
that contributed nothing.

Protocol v2
-----------
    framed = [EOT] + tokens(document) + [EOT]
    windows = consecutive seq_len+1 chunk(s), plus the final remainder if len > 1
    every window is scored on its own real targets, batch of 1, no padding
    loss(doc) = sum NLL / tokens scored in that document

Coverage is first-class output:

    documents_selected, documents_scored, documents_unscored (+ reason),
    eligible_tokens, scored_tokens, token_coverage

Two aggregations, kept separate:

    macro_loss      mean of per-stratum losses (each stratum weighted equally)
    token_weighted  sum NLL / sum tokens (long documents dominate)

Usage
-----
    python scripts/eval_phase16.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
        --val-jsonl data/curated_v6/benchmark/clean_unseen.jsonl --seq-len 512 \
        --out bench_results/phase16/eval_phase14_s512.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tokenizer_gpt2 import GPT2BPETokenizer   # noqa: E402
from utils import _torch_load, create_model   # noqa: E402

PROTOCOL_VERSION = "phase16.coverage.v2"
LENGTH_BUCKETS = ((0, 128), (128, 256), (256, 512), (512, 1024),
                  (1024, 2048), (2048, 10**9))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def length_bucket(n_tokens: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n_tokens < hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    return "unknown"


def load_documents(jsonl: str, limit_per_stratum: int, seed: int,
                   strata: tuple[str, ...] = ("source_id", "language", "domain")) -> dict:
    """Deterministic stratified sampling; identical rule for every model."""
    buckets: dict[tuple, list[tuple[bytes, dict]]] = defaultdict(list)
    with open(jsonl, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            key = tuple(doc.get(k, "?") for k in strata)
            buckets[key].append((hashlib.sha256(
                f"{seed}.{doc.get('document_id')}".encode()).digest(), doc))
    chosen = {}
    for key, rows in buckets.items():
        rows.sort(key=lambda r: r[0])
        chosen[key] = [doc for _h, doc in rows[:limit_per_stratum]]
    return chosen


def document_windows(ids: list[int], seq_len: int) -> tuple[list[list[int]], int]:
    """All scoreable windows of one framed document, including the tail.

    Returns (windows, dropped_tail_tokens). A single-token remainder has no
    target and is reported as dropped rather than silently ignored.
    """
    windows: list[list[int]] = []
    start = 0
    while start < len(ids):
        chunk = ids[start:start + seq_len + 1]
        if len(chunk) >= 2:                 # a 1-token chunk has no target
            windows.append(chunk)
        start += seq_len                    # windows share their boundary token
    # Targets per window are len(w)-1, and consecutive windows share exactly one
    # token, so every token after the first is a target exactly once.
    covered = sum(len(w) - 1 for w in windows)
    dropped = len(ids) - 1 - covered
    return windows, max(0, dropped)


@torch.no_grad()
def score_document(model, windows: list[list[int]], device) -> tuple[float, int]:
    """Sum NLL over real target tokens of one document. No padding anywhere."""
    total_nll = 0.0
    total_tokens = 0
    for w in windows:
        x = torch.tensor([w[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([w[1:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
            reduction="sum")
        total_nll += float(loss.item())
        total_tokens += int(y.numel())
    return total_nll, total_tokens


def evaluate(model, tokenizer, chosen: dict, seq_len: int, device: str) -> dict:
    eot = tokenizer.eot_id
    strata = {}
    unscored: list[dict] = []
    docs_scored = 0
    eligible_tokens = 0
    scored_tokens = 0

    for key, docs in sorted(chosen.items(), key=lambda kv: str(kv[0])):
        nll_sum = 0.0
        tok_sum = 0
        n_scored = 0
        for doc in docs:
            body = tokenizer.encode(doc.get("text") or "")
            if not body:
                # [EOT][EOT] would be a legal 1-token window and would count an
                # empty document as scored; it must be reported, not counted.
                unscored.append({"document_id": doc.get("document_id"),
                                 "stratum": list(key),
                                 "reason": "empty_document", "tokens": 0})
                continue
            framed = [eot] + body + [eot]
            eligible_tokens += len(body) + 1     # one terminal boundary target
            windows, dropped = document_windows(framed, seq_len)
            if not windows:
                unscored.append({"document_id": doc.get("document_id"),
                                 "stratum": list(key),
                                 "reason": "too_short_for_any_window",
                                 "tokens": len(body)})
                continue
            nll, tokens = score_document(model, windows, device)
            if tokens == 0:
                unscored.append({"document_id": doc.get("document_id"),
                                 "stratum": list(key),
                                 "reason": "no_scorable_targets", "tokens": len(body)})
                continue
            nll_sum += nll
            tok_sum += tokens
            scored_tokens += tokens
            docs_scored += 1
            n_scored += 1
            if dropped:
                unscored.append({"document_id": doc.get("document_id"),
                                 "stratum": list(key),
                                 "reason": f"single_token_tail_dropped:{dropped}",
                                 "tokens": len(body)})

        if n_scored and tok_sum:
            loss = nll_sum / tok_sum
            strata["|".join(str(k) for k in key)] = {
                "loss": round(loss, 4),
                "perplexity": round(math.exp(min(20.0, loss)), 2),
                "documents_selected": len(docs),
                "documents_scored": n_scored,
                "tokens_scored": tok_sum,
                "nll_sum": round(nll_sum, 2),
            }

    per_stratum_losses = [v["loss"] for v in strata.values()]
    total_nll = sum(v["nll_sum"] for v in strata.values())
    token_weighted = round(total_nll / max(1, scored_tokens), 4)
    macro = round(statistics.mean(per_stratum_losses), 4) if per_stratum_losses else None

    return {
        "protocol": PROTOCOL_VERSION,
        "strata": strata,
        "aggregate": {
            "macro_loss": macro,
            "macro_perplexity": round(math.exp(min(20.0, macro)), 2) if macro else None,
            "token_weighted_loss": token_weighted,
            "token_weighted_perplexity": round(math.exp(min(20.0, token_weighted)), 2),
            "documents_selected": sum(len(d) for d in chosen.values()),
            "documents_scored": docs_scored,
            # Documents that contributed no scored token at all. A tail token
            # dropped from an otherwise scored document is not an unscored doc.
            "documents_unscored": sum(
                1 for u in unscored
                if not u["reason"].startswith("single_token_tail")),
            "eligible_tokens": eligible_tokens,
            "tokens_scored": scored_tokens,
            "token_coverage": round(scored_tokens / max(1, eligible_tokens), 4),
            "padding_tokens": 0,
        },
        "document_length_buckets": _length_buckets(chosen, tokenizer),
        "unscored_documents": unscored,
    }


def _length_buckets(chosen: dict, tokenizer) -> dict:
    out: dict[str, int] = defaultdict(int)
    for docs in chosen.values():
        for doc in docs:
            out[length_bucket(len(tokenizer.encode(doc.get("text") or "")))] += 1
    return dict(sorted(out.items()))


def load_jin(checkpoint: str, device: str):
    blob = _torch_load(checkpoint, map_location="cpu")
    vocab = int(blob.get("tokenizer", {}).get("vocab_size", 50257))
    model = create_model(vocab, checkpoint=blob, device=device)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.eval()
    return model, blob


class HFWrapper(torch.nn.Module):
    def __init__(self, model, block_size: int = 1024):
        super().__init__()
        self.inner = model
        self.block_size = block_size
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, doc_mask=None):
        return self.inner(x).logits, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 16 evaluator (coverage v2)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--hf-model", default=None)
    ap.add_argument("--val-jsonl", default=os.path.join(
        ROOT, "data", "curated_v6", "benchmark", "clean_unseen.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--docs-per-stratum", type=int, default=24)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if bool(args.checkpoint) == bool(args.hf_model):
        raise SystemExit("pass exactly one of --checkpoint / --hf-model")

    tokenizer = GPT2BPETokenizer()
    if args.checkpoint:
        model, blob = load_jin(args.checkpoint, args.device)
        identity = {"checkpoint": os.path.relpath(args.checkpoint, ROOT),
                    "sha256": sha256_file(args.checkpoint),
                    "iteration": blob.get("iteration"),
                    "tokens_seen": blob.get("tokens_seen")}
    else:
        from transformers import GPT2LMHeadModel
        hf = GPT2LMHeadModel.from_pretrained(args.hf_model).eval().to(args.device)
        model = HFWrapper(hf, int(hf.config.n_positions))
        identity = {"hf_model": args.hf_model}

    chosen = load_documents(args.val_jsonl, args.docs_per_stratum, args.seed)
    result = evaluate(model, tokenizer, chosen, args.seq_len, args.device)

    report = {
        "protocol": PROTOCOL_VERSION,
        "identity": identity,
        "benchmark": {"file": os.path.relpath(args.val_jsonl, ROOT),
                      "sha256": sha256_file(args.val_jsonl),
                      "candidate_pool_documents": _count(args.val_jsonl),
                      "sampling": {"strata": ["source_id", "language", "domain"],
                                   "docs_per_stratum": args.docs_per_stratum,
                                   "seed": args.seed}},
        "tokenizer": {"kind": "gpt2_bpe", "vocab_size": 50257,
                      "eot_id": tokenizer.eot_id},
        "seq_len": args.seq_len,
        "selected_document_ids": {str(k): [d.get("document_id") for d in v]
                                  for k, v in chosen.items()},
        **result,
    }

    out = args.out or os.path.join(ROOT, "bench_results", "phase16", "eval_clean_v2.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    a = result["aggregate"]
    print(f"[phase16] macro loss {a['macro_loss']} (ppl {a['macro_perplexity']}) | "
          f"token-weighted {a['token_weighted_loss']} (ppl {a['token_weighted_perplexity']})")
    print(f"[coverage] documents {a['documents_scored']}/{a['documents_selected']} scored | "
          f"tokens {a['tokens_scored']}/{a['eligible_tokens']} "
          f"({a['token_coverage'] * 100:.1f}%) | padding {a['padding_tokens']}")
    print(f"Wrote {os.path.relpath(out, ROOT)}")


def _count(path: str) -> int:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


if __name__ == "__main__":
    main()
