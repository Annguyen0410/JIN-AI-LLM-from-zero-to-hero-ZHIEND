"""Phase 9 — evaluation that a Phase 14 claim can actually rest on.

Three measurements, all from the model's own weights, all with FP32
cross-entropy on FP32 logits:

1. ``val_bin``      — held-out loss on ``val.bin`` at fixed, evenly spaced
                      window offsets (deterministic: same windows every run).
2. ``per_domain``   — held-out loss split by domain / language, from the
                      document-disjoint ``val.jsonl`` (so a mixture cannot hide
                      a domain it is bad at).
3. ``generation``   — greedy + sampled continuations of held-out prompts from
                      ``test.jsonl`` with repetition statistics and the raw
                      text recorded. No repetition penalty: degenerate output is
                      evidence and must stay visible.

``test.jsonl`` never enters training, so prompts and scores are independent of
the checkpoint's training text.

Usage
-----
    python scripts/eval_phase14.py --checkpoint checkpoints/gpt2_small/<run>/best.pt \
        --out bench_results/phase14/eval_<run>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tokenizer_gpt2 import GPT2BPETokenizer   # noqa: E402
from utils import _torch_load, create_model   # noqa: E402

V5 = os.path.join(ROOT, "data", "gpt2_corpus_v5")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt_path: str, device: str):
    # _torch_load allowlists TorchVersion, which older checkpoints contain
    # (torch.__version__ pickles as a custom global).
    blob = _torch_load(ckpt_path, map_location="cpu")
    vocab = int(blob.get("tokenizer", {}).get("vocab_size", 50257))
    model = create_model(vocab, checkpoint=blob, device=device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, blob


@torch.no_grad()
def nll_windows(model, seq_len: int, windows: np.ndarray, device: str) -> tuple[float, int]:
    """Mean NLL over pre-built (n_windows, seq_len+1) batches."""
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(windows), 8):
        chunk = windows[start:start + 8]
        x = torch.from_numpy(chunk[:, :-1]).long().to(device)
        y = torch.from_numpy(chunk[:, 1:]).long().to(device)
        logits, _ = model(x)
        logits = logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += float(loss.item())
        total_tokens += int(y.numel())
    return total_nll / max(1, total_tokens), total_tokens


def evenly_spaced_windows(data: np.memmap, seq_len: int, n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    span = len(data) - seq_len - 1
    if span <= 0:
        return np.zeros((0, seq_len + 1), dtype=np.int64)
    starts = np.linspace(0, span, num=n, dtype=np.int64)
    return np.stack([data[s:s + seq_len + 1] for s in starts]).astype(np.int64)


def per_group_loss(model, tokenizer, val_jsonl: str, seq_len: int,
                   docs_per_group: int, model_device: str, min_tokens: int = 32) -> dict:
    """Loss grouped by domain and by language on document-disjoint text."""
    groups: dict[str, list] = {}
    counts: dict[str, int] = {}
    with open(val_jsonl, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            for key in (f"domain:{doc.get('domain', '?')}",
                        f"language:{doc.get('language', '?')}"):
                if counts.get(key, 0) >= docs_per_group:
                    continue
                ids = tokenizer.encode(doc["text"])
                if len(ids) < min_tokens + 1:
                    continue
                ids = ids[:seq_len + 1]
                groups.setdefault(key, []).append(ids)
                counts[key] = counts.get(key, 0) + 1
    out = {}
    for key, seqs in sorted(groups.items()):
        use = min(len(seqs), docs_per_group)
        padded = np.zeros((use, seq_len + 1), dtype=np.int64)
        pad_id = tokenizer.eot_id
        for i, ids in enumerate(seqs[:use]):
            padded[i, :] = pad_id
            padded[i, :len(ids)] = ids
            if len(ids) < seq_len + 1:                 # mask the pad region
                padded[i, len(ids):] = pad_id
        # padding tokens would inflate the loss, so score only real tokens:
        # drop sequences shorter than the window instead of padding
        full = [s for s in seqs if len(s) >= seq_len + 1][:docs_per_group]
        if full:
            padded = np.stack([np.asarray(s[:seq_len + 1]) for s in full]).astype(np.int64)
        else:
            width = max(len(s) for s in seqs[:use])
            padded = np.stack([
                np.pad(np.asarray(s), (0, width - len(s)), constant_values=pad_id)
                for s in seqs[:use]]).astype(np.int64)
        loss, tokens = nll_windows(model, seq_len, padded, model_device)
        out[key] = {"loss": round(loss, 4),
                    "perplexity": round(math.exp(min(20.0, loss)), 2),
                    "documents": len(seqs[:use]), "tokens_scored": tokens,
                    "exact_window_padding": not bool(full)}
    return out


def repetition_stats(text: str) -> dict:
    words = text.split()
    if not words:
        return {"words": 0, "distinct_1": 0.0, "distinct_2": 0.0, "repeat_4gram": 0.0}
    d1 = len(set(words)) / len(words)
    bigrams = list(zip(words, words[1:])) or [(words[0],)]
    d2 = len(set(bigrams)) / len(bigrams)
    grams = [tuple(words[i:i + 4]) for i in range(max(1, len(words) - 3))]
    rep4 = 1 - len(set(grams)) / len(grams)
    return {"words": len(words), "distinct_1": round(d1, 3),
            "distinct_2": round(d2, 3), "repeat_4gram": round(rep4, 3)}


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int, device: str,
             temperature: float = 0.0, seed: int = 0) -> str:
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(max_new):
        x_cond = x[:, -model.block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :].float()
        if temperature <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
        x = torch.cat([x, nxt], dim=1)
        if int(nxt) == tokenizer.eot_id:
            break
    return tokenizer.decode(x[0, len(ids):].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 14 evaluation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=V5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--val-windows", type=int, default=64)
    ap.add_argument("--docs-per-group", type=int, default=24)
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = GPT2BPETokenizer()
    model, blob = load_model(args.checkpoint, args.device)
    print(f"[Model] {blob.get('architecture_id')} | step {blob.get('iteration')} | "
          f"tokens {blob.get('tokens_seen', 0):,} | vocab {tokenizer.vocab_size:,}")

    report = {
        "checkpoint": os.path.relpath(args.checkpoint, ROOT),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture_id": blob.get("architecture_id"),
        "iteration": blob.get("iteration"),
        "tokens_seen": blob.get("tokens_seen"),
        "tokenizer": {"kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size},
        "method": {"loss": "fp32 cross-entropy on fp32 logits",
                   "windows": "evenly spaced offsets, fixed seed",
                   "data": "document-disjoint val/test splits of curated_v5"},
    }

    val_bin = os.path.join(args.data_dir, "val.bin")
    if os.path.exists(val_bin):
        data = np.memmap(val_bin, dtype=np.uint16, mode="r")
        windows = evenly_spaced_windows(data, args.seq_len, args.val_windows)
        loss, tokens = nll_windows(model, args.seq_len, windows, args.device)
        report["val_bin"] = {"loss": round(loss, 4),
                             "perplexity": round(math.exp(min(20.0, loss)), 2),
                             "windows": len(windows), "tokens_scored": tokens,
                             "file": os.path.relpath(val_bin, ROOT),
                             "file_sha256": sha256_file(val_bin)}

    val_jsonl = os.path.join(args.data_dir, "..", "curated_v5", "splits", "val.jsonl")
    val_jsonl = os.path.normpath(val_jsonl)
    if os.path.exists(val_jsonl):
        report["per_group"] = per_group_loss(model, tokenizer, val_jsonl,
                                            args.seq_len, args.docs_per_group,
                                            args.device)

    test_jsonl = os.path.normpath(os.path.join(args.data_dir, "..", "curated_v5",
                                               "splits", "test.jsonl"))
    if os.path.exists(test_jsonl) and args.prompts:
        prompts = []
        wanted = {("en", "knowledge"): args.prompts // 2,
                  ("en", "educational"): args.prompts // 4,
                  ("vi", "mixed"): args.prompts // 4}
        taken = {k: 0 for k in wanted}
        with open(test_jsonl, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                key = (doc.get("language"), doc.get("domain"))
                if key in wanted and taken[key] < wanted[key]:
                    text = doc["text"]
                    cut = text[:400]
                    stop = max(cut.rfind(". "), cut.rfind("\n"))
                    prompt = cut[:stop + 1] if stop > 120 else cut[:240]
                    if len(prompt.split()) >= 12:
                        prompts.append({"key": f"{key[0]}/{key[1]}",
                                        "document_id": doc.get("document_id"),
                                        "prompt": prompt.strip()})
                        taken[key] += 1
                if all(taken[k] >= wanted[k] for k in wanted) and prompts:
                    break
        samples = []
        for i, item in enumerate(prompts):
            greedy = generate(model, tokenizer, item["prompt"], args.max_new_tokens,
                              args.device, temperature=0.0)
            sampled = generate(model, tokenizer, item["prompt"], args.max_new_tokens,
                               args.device, temperature=0.8, seed=1234 + i)
            samples.append({**item, "greedy": greedy,
                            "greedy_stats": repetition_stats(greedy),
                            "sample_t0.8": sampled,
                            "sample_stats": repetition_stats(sampled)})
        agg = {}
        for key in sorted({s["key"] for s in samples}):
            rows = [s for s in samples if s["key"] == key]
            agg[key] = {
                "prompts": len(rows),
                "mean_distinct_2": round(sum(r["sample_stats"]["distinct_2"]
                                             for r in rows) / len(rows), 3),
                "mean_repeat_4gram": round(sum(r["sample_stats"]["repeat_4gram"]
                                               for r in rows) / len(rows), 3),
                "mean_words": round(sum(r["sample_stats"]["words"]
                                        for r in rows) / len(rows), 1),
            }
        report["generation"] = {"decoding": "greedy + temperature 0.8, fixed seeds",
                                "summary": agg, "samples": samples}

    out = args.out or os.path.join(ROOT, "bench_results", "phase14",
                                   f"eval_{os.path.basename(os.path.dirname(args.checkpoint))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    if "val_bin" in report:
        print(f"[val] loss {report['val_bin']['loss']} | "
              f"ppl {report['val_bin']['perplexity']} | "
              f"{report['val_bin']['tokens_scored']:,} tokens")
    for key, value in report.get("per_group", {}).items():
        print(f"[group] {key:24s} loss {value['loss']:.4f} | ppl {value['perplexity']:.1f}")
    print(f"Wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
