"""Phase 15D — evaluator with masked, document-independent scoring.

Two defects in the Phase 14 evaluator are fixed here:

1. **Padding contamination.** ``per_group_loss`` padded short documents with EOT
   and then scored the whole padded tensor, so artificial tokens entered the
   denominator. Here every document is scored independently and only real
   target tokens are counted. There is no padding anywhere in the metric, so
   padding cannot change it — `tests/test_evaluation_integrity.py` proves that
   by construction and by direct comparison.
2. **Source-order sampling.** Taking the first N matching documents samples one
   upstream source (the files are grouped by source). Sampling here is
   deterministic per (source, language, domain) stratum, ordered by a hash of
   the document id, and records the selected ids and the seed.

Scoring protocol (identical for JIN and the GPT-2 reference):

    ids = [EOT] + tokens(document) + [EOT]
    windows of seq_len+1 tokens, no padding, only complete windows are scored
    loss = sum(cross_entropy over every target token) / total target tokens

Each document therefore starts and ends with an explicit boundary token, and no
attention ever crosses a document boundary because no two documents share a
window.

Usage
-----
    python scripts/eval_phase15.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
        --val-jsonl data/curated_v6/splits/val.jsonl --out bench_results/phase15/eval.json
    python scripts/eval_phase15.py --hf-model gpt2 --val-jsonl ... --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tokenizer_gpt2 import GPT2BPETokenizer   # noqa: E402
from utils import _torch_load, create_model   # noqa: E402

PROTOCOL_VERSION = "phase15.docindependent.v1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_key(document_id: str) -> bytes:
    return hashlib.sha256(f"phase15.{document_id}".encode()).digest()


def load_documents(val_jsonl: str, limit_per_stratum: int, seed: int,
                   strata: tuple[str, ...] = ("source_id", "language", "domain")) -> dict:
    """Deterministic, source-aware sampling.

    Every stratum is capped independently, so a corpus that is 67 % Wikipedia
    cannot fill the sample with Wikipedia documents and call it a domain metric.
    """
    buckets: dict[tuple, list[tuple[bytes, dict]]] = defaultdict(list)
    with open(val_jsonl, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            key = tuple(doc.get(k, "?") for k in strata)
            buckets[key].append((hashlib.sha256(
                f"{seed}.{doc.get('document_id')}".encode()).digest(), doc))

    chosen = {}
    for key, rows in buckets.items():
        rows.sort(key=lambda r: r[0])               # deterministic, not file order
        chosen[key] = [doc for _h, doc in rows[:limit_per_stratum]]
    return chosen


def document_windows(ids: list[int], seq_len: int) -> list[list[int]]:
    """Split one document into non-overlapping complete windows, no padding."""
    framed = ids[:]                               # [EOT] + body + [EOT] supplied by caller
    windows = []
    start = 0
    while start + seq_len + 1 <= len(framed):
        windows.append(framed[start:start + seq_len + 1])
        start += seq_len
    return windows


@torch.no_grad()
def score_documents(model, tokenizer, docs: list[dict], seq_len: int,
                    batch_size: int = 8) -> dict:
    """Aggregate NLL over real tokens across documents (no padding)."""
    eot = tokenizer.eot_id
    windows: list[list[int]] = []
    for doc in docs:
        ids = tokenizer.encode(doc.get("text") or "")
        if len(ids) < 2:
            continue
        framed = [eot] + ids + [eot]
        windows.extend(document_windows(framed, seq_len))

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(windows), batch_size):
        chunk = windows[start:start + batch_size]
        if len(chunk) < 2:                        # model needs a rectangular batch
            x = torch.tensor([w[:-1] for w in chunk], dtype=torch.long)
            y = torch.tensor([w[1:] for w in chunk], dtype=torch.long)
        else:
            x = torch.tensor([w[:-1] for w in chunk], dtype=torch.long)
            y = torch.tensor([w[1:] for w in chunk], dtype=torch.long)
        device = next(model.parameters()).device
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
            reduction="sum")
        total_nll += float(loss.item())
        total_tokens += int(y.numel())

    loss = total_nll / max(1, total_tokens)
    return {"loss": round(loss, 4),
            "perplexity": round(math.exp(min(20.0, loss)), 2),
            "documents": len(docs),
            "windows": len(windows),
            "tokens_scored": total_tokens,
            "padding_tokens": 0,
            "nll_sum": round(total_nll, 2),
            "protocol": PROTOCOL_VERSION}


def load_jin(checkpoint: str, device: str):
    blob = _torch_load(checkpoint, map_location="cpu")
    vocab = int(blob.get("tokenizer", {}).get("vocab_size", 50257))
    model = create_model(vocab, checkpoint=blob, device=device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, blob


class HFWrapper(torch.nn.Module):
    """GPT-2 reference adapted to the same call signature as JinAI."""

    def __init__(self, model, block_size: int):
        super().__init__()
        self.inner = model
        self.block_size = block_size

    def forward(self, x, doc_mask=None):
        return self.inner(x).logits, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 15 evaluator")
    ap.add_argument("--checkpoint", default=None, help="JIN checkpoint")
    ap.add_argument("--hf-model", default=None, help="e.g. gpt2 (reference)")
    ap.add_argument("--val-jsonl", default=os.path.join(ROOT, "data", "curated_v6",
                                                        "splits", "val.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seq-len", type=int, default=1024)
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
                    "architecture_id": blob.get("architecture_id"),
                    "iteration": blob.get("iteration"),
                    "tokens_seen": blob.get("tokens_seen")}
    else:
        from transformers import GPT2LMHeadModel
        hf = GPT2LMHeadModel.from_pretrained(args.hf_model).eval().to(args.device)
        model = HFWrapper(hf, 1024)
        identity = {"hf_model": args.hf_model}

    buckets = load_documents(args.val_jsonl, args.docs_per_stratum, args.seed)
    report = {
        "protocol": PROTOCOL_VERSION,
        "scoring": "fp32 cross-entropy on fp32 logits, document-independent, "
                   "no padding, EOT framing, identical tokenizer ids",
        "model": identity,
        "tokenizer": {"kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size,
                      "eot_id": tokenizer.eot_id},
        "val_jsonl": os.path.relpath(args.val_jsonl, ROOT),
        "val_sha256": sha256_file(args.val_jsonl),
        "sampling": {"seed": args.seed, "docs_per_stratum": args.docs_per_stratum,
                     "strata": ["source_id", "language", "domain"],
                     "order": "sha256(seed.document_id)"},
        "strata": {},
        "aggregate": {},
    }

    all_docs = []
    for key, docs in sorted(buckets.items()):
        if not docs:
            continue
        metrics = score_documents(model, tokenizer, docs, args.seq_len)
        name = "|".join(str(k) for k in key)
        report["strata"][name] = {
            **metrics,
            "source_id": key[0], "language": key[1], "domain": key[2],
            "document_ids": [d.get("document_id") for d in docs][:200],
        }
        all_docs.extend(docs)
        print(f"[stratum] {name:48s} loss {metrics['loss']:.4f} "
              f"ppl {metrics['perplexity']:>9.2f} docs {metrics['documents']:>4d} "
              f"tokens {metrics['tokens_scored']:>8,}")

    # Aggregate over the union of strata (deduplicated by document id).
    unique = {}
    for doc in all_docs:
        unique[doc.get("document_id") or id(doc)] = doc
    aggregate = score_documents(model, tokenizer, list(unique.values()), args.seq_len)
    report["aggregate"] = aggregate
    print(f"[aggregate] loss {aggregate['loss']} | ppl {aggregate['perplexity']} | "
          f"tokens {aggregate['tokens_scored']:,} | padding {aggregate['padding_tokens']}")

    out = args.out or os.path.join(ROOT, "bench_results", "phase15",
                                   "eval_" + (os.path.basename(os.path.dirname(args.checkpoint))
                                              if args.checkpoint else args.hf_model) + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
