"""Phase 9 benchmark — model-only comparison of JIN and GPT-2 Small.

Every number here comes from the **neural network alone**: no RAG, no web
search, no tools, no memory, no human-edited answers. Decoding is greedy so the
result is reproducible from the checkpoint hash.

Two families of metrics, kept separate on purpose:

1. **Language modelling** — held-out loss / perplexity on ``data/eval/heldout.txt``
   (a file excluded from every corpus mixture). Comparable across models only
   when the tokenizer is the same; the report records which tokenizer produced
   the number.
2. **Task probes** — ``benchmarks/gpt2_probes.jsonl``. Answer probes are scored
   by whether an accepted string appears in the first tokens of the reply;
   continuation probes are scored for fluency (repetition, distinct-n, length),
   not for a "correct" answer.

Usage
-----
    python scripts/gpt2_bench.py --model jin --checkpoint checkpoints/jingpt_chat.pt
    python scripts/gpt2_bench.py --model gpt2 --out bench_results/gpt2_upgrade/gpt2_reference.json
    python scripts/gpt2_bench.py --compare      # both, writes comparison.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EVAL_VERSION = "gpt2_upgrade.v1"
PROBE_PATH = os.path.join(ROOT, "benchmarks", "gpt2_probes.jsonl")
HELDOUT_PATH = os.path.join(ROOT, "data", "eval", "heldout.txt")
OUT_DIR = os.path.join(ROOT, "bench_results", "gpt2_upgrade")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


VI_CHARS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")


def split_heldout(text: str) -> dict[str, str]:
    """Split the held-out corpus into English / Vietnamese blocks."""
    blocks = text.split('<doc path="')[1:] or [text]
    groups: dict[str, list[str]] = {"english": [], "vietnamese": []}
    for block in blocks:
        body = block.split(">", 1)[1] if ">" in block else block
        letters = [c for c in body[:4000] if c.isalpha()]
        if not letters:
            continue
        vi = sum(1 for c in letters if c.lower() in VI_CHARS) / len(letters)
        groups["vietnamese" if vi > 0.02 else "english"].append(body)
    return {k: "".join(v) for k, v in groups.items()}


def load_probes() -> list[dict]:
    with open(PROBE_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def repetition_metrics(text: str) -> dict:
    words = text.split()
    if not words:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "max_word_repeat": 0, "words": 0}
    unigrams = len(set(w.lower() for w in words)) / len(words)
    bigrams = [tuple(w.lower() for w in words[i:i + 2]) for i in range(len(words) - 1)]
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams))
    counts: dict[str, int] = {}
    for w in words:
        counts[w.lower()] = counts.get(w.lower(), 0) + 1
    return {
        "distinct_1": round(unigrams, 4),
        "distinct_2": round(distinct_2, 4),
        "max_word_repeat": max(counts.values()),
        "words": len(words),
    }


# ── model wrappers ─────────────────────────────────────────────────────

class JinModel:
    """A project checkpoint served through ``model.JinAI``."""

    family = "jin"

    def __init__(self, checkpoint: str, device: str):
        from utils import create_model, load_checkpoint_file, tokenizer_from_state
        blob = load_checkpoint_file(checkpoint, map_location="cpu")
        state = blob.get("tokenizer_state") or blob.get("tokenizer")
        self.tokenizer = tokenizer_from_state(state)
        if getattr(self.tokenizer, "vocab_size", 0) != blob.get("vocab_size", 0):
            raise SystemExit("Checkpoint tokenizer/vocab mismatch — refusing to score.")
        self.model = create_model(vocab_size=self.tokenizer.vocab_size,
                                  checkpoint=blob, device=device).to(device)
        self.model.load_state_dict(blob["model_state_dict"])
        self.model.eval()
        self.device = device
        self.info = {
            "family": "jin",
            "checkpoint": os.path.relpath(checkpoint, ROOT),
            "checkpoint_sha256": sha256_file(checkpoint),
            "architecture_id": blob.get("model_config", {}).get("arch_id", "jin_modern"),
            "model_config": blob.get("model_config"),
            "tokenizer": {"kind": type(self.tokenizer).__name__,
                          "vocab_size": self.tokenizer.vocab_size},
            "iteration": blob.get("iteration"),
            "vocab_size": self.tokenizer.vocab_size,
        }

    def ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, allow_special=False)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 24) -> str:
        ids = self.ids(prompt)
        if not ids:
            return ""
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.0,
                                  top_k=None, repetition_penalty=1.0, stop_on_repeat=False)
        return self.decode(out[0].tolist()[len(ids):])

    @torch.no_grad()
    def token_loss(self, ids: list[int], max_windows: int = 24) -> tuple[float, int]:
        """Cross-entropy computed in fp32 from logits.

        Never use the model's own ``loss`` under fp16 autocast: measured on
        this stack, ``GPT2LMHeadModel(labels=...)`` returned 11.67 while fp32
        cross-entropy on the *same* logits returned 0.83. Vocab-50k softmax in
        fp16 is not trustworthy, and a perplexity built on it is fiction.
        """
        # Windows must fit the checkpoint's own context (legacy 256, modern 1024).
        seq = int(min(1024, getattr(self.model, "block_size", 1024)))
        n = min(max_windows, max(1, (len(ids) - 1) // seq))
        total, tokens = 0.0, 0
        for i in range(n):
            chunk = ids[i * seq:(i + 1) * seq + 1]
            if len(chunk) < 3:
                break
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=self.device)
            with torch.amp.autocast("cuda", enabled=self.device == "cuda"):
                logits, _ = self.model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), y.view(-1))
            total += float(loss.item()) * (len(chunk) - 1)
            tokens += len(chunk) - 1
        return (total / max(1, tokens), tokens)


class GPT2Reference:
    """OpenAI GPT-2 Small, loaded in an isolated benchmark process only."""

    family = "gpt2"

    def __init__(self, device: str, name: str = "gpt2"):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.tokenizer = GPT2TokenizerFast.from_pretrained(name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.name = name
        self.info = {
            "family": "gpt2-reference",
            "checkpoint": name,
            "architecture_id": "gpt2_small",
            "vocab_size": self.tokenizer.vocab_size,
            "tokenizer": {"kind": "GPT2TokenizerFast", "vocab_size": self.tokenizer.vocab_size},
            "params": sum(p.numel() for p in self.model.parameters()),
        }

    def ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 24) -> str:
        ids = self.ids(prompt)
        if not ids:
            return ""
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tokenizer.eos_token_id)
        return self.decode(out[0].tolist()[len(ids):])

    @torch.no_grad()
    def token_loss(self, ids: list[int], max_windows: int = 24) -> tuple[float, int]:
        seq = 1024
        n = min(max_windows, max(1, (len(ids) - 1) // seq))
        total, tokens = 0.0, 0
        for i in range(n):
            chunk = ids[i * seq:(i + 1) * seq + 1]
            if len(chunk) < 3:
                break
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=self.device)
            # fp32 CE from logits — see JinModel.token_loss for the measurement
            # showing the built-in fp16 loss is unusable.
            logits = self.model(x).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), y.view(-1))
            total += float(loss.item()) * (len(chunk) - 1)
            tokens += len(chunk) - 1
        return (total / max(1, tokens), tokens)


# ── scoring ────────────────────────────────────────────────────────────

def score_probe(model, probe: dict, max_new_tokens: int = 24) -> dict:
    text = model.generate(probe["prompt"], max_new_tokens=max_new_tokens)
    rep = repetition_metrics(text)
    result = {
        "id": probe["id"],
        "category": probe["category"],
        "prompt": probe["prompt"],
        "output": text[:400],
        "repetition": rep,
    }
    if probe.get("accept"):
        lowered = text.lower()
        head = lowered[:120]
        hit = any(a.lower() in head for a in probe["accept"])
        result["passed"] = bool(hit)
        result["accepted"] = probe["accept"]
    else:
        # Continuation: "passes" only if it is coherent enough not to loop.
        result["passed"] = (rep["words"] >= 5 and rep["max_word_repeat"] <= max(3, rep["words"] // 4)
                            and rep["distinct_2"] >= 0.55)
        result["scored_by"] = "fluency"
    return result


def run_benchmark(model, max_new_tokens: int) -> dict:
    probes = load_probes()
    results = [score_probe(model, p, max_new_tokens) for p in probes]

    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(bool(r["passed"]))
    categories = {cat: {"passed": sum(v), "total": len(v),
                        "score": round(sum(v) / len(v), 4)}
                  for cat, v in sorted(by_cat.items())}

    lm = None
    if os.path.exists(HELDOUT_PATH):
        text = open(HELDOUT_PATH, encoding="utf-8", errors="replace").read()
        # The held-out file mixes English and Vietnamese document blocks. A
        # single average hides the fact that an English-only model (GPT-2)
        # is near-random on the Vietnamese half, so the loss is reported per
        # language as well as overall.
        groups = split_heldout(text)
        lm = {"heldout_file": "data/eval/heldout.txt",
              "heldout_sha256": sha256_file(HELDOUT_PATH),
              "tokenizer": model.info["tokenizer"]["kind"],
              "per_language": {}}
        for name, body in groups.items():
            if not body.strip():
                continue
            loss, tokens = model.token_loss(model.ids(body)[: 1024 * 8 + 1])
            lm["per_language"][name] = {
                "chars": len(body), "tokens_scored": tokens,
                "loss": round(loss, 4),
                "perplexity": round(math.exp(min(20.0, loss)), 3)}
        overall_ids = model.ids(groups.get("english", "") + "\n" + groups.get("vietnamese", ""))
        loss, tokens = model.token_loss(overall_ids)
        lm.update({"tokens_scored": tokens, "loss": round(loss, 4),
                   "perplexity": round(math.exp(min(20.0, loss)), 3)})

    answer_cats = [c for c in categories if c != "continuation"]
    total_pass = sum(categories[c]["passed"] for c in answer_cats)
    total_n = sum(categories[c]["total"] for c in answer_cats)
    return {
        "eval_version": EVAL_VERSION,
        "prompt_set": {"path": os.path.relpath(PROBE_PATH, ROOT),
                       "sha256": sha256_file(PROBE_PATH)},
        "model": model.info,
        "decoding": {"strategy": "greedy", "max_new_tokens": max_new_tokens},
        "language_modeling": lm,
        "categories": categories,
        "aggregate_task_score": round(total_pass / max(1, total_n), 4),
        "probes": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Model-only GPT-2-class benchmark")
    ap.add_argument("--model", choices=("jin", "gpt2"), default="jin")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="Build comparison.json from the two existing result files "
                         "without re-scoring (no GPU work).")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default=None, choices=("auto", "cuda", "cpu"))
    args = ap.parse_args()

    import config
    config.set_device(args.device)
    device = config.DEVICE
    os.makedirs(OUT_DIR, exist_ok=True)

    def build(kind: str):
        if kind == "gpt2":
            return GPT2Reference(device)
        ckpt = args.checkpoint or config.get_checkpoint_path()
        return JinModel(ckpt, device)

    def write(name: str, payload: dict) -> str:
        path = args.out or os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")
        return path

    if args.merge:
        jin_path = os.path.join(OUT_DIR, "base_model.json")
        ref_path = os.path.join(OUT_DIR, "gpt2_reference.json")
        if not (os.path.exists(jin_path) and os.path.exists(ref_path)):
            raise SystemExit("Need both base_model.json and gpt2_reference.json first.")
        jin = json.load(open(jin_path, encoding="utf-8"))
        ref = json.load(open(ref_path, encoding="utf-8"))
        write_comparison(jin, ref)
        return

    if not args.compare:
        model = build(args.model)
        report = run_benchmark(model, args.max_new_tokens)
        name = "base_model.json" if args.model == "jin" else "gpt2_reference.json"
        write(name, report)
        print(json.dumps({"model": report["model"]["family"],
                          "aggregate": report["aggregate_task_score"],
                          "categories": {k: v["score"] for k, v in report["categories"].items()},
                          "lm": report["language_modeling"]}, indent=2))
        return

    # Compare both models in one process, sequentially (VRAM is 6GB).
    reports = {}
    for kind, filename in (("jin", "base_model.json"), ("gpt2", "gpt2_reference.json")):
        print(f"\n=== scoring {kind} ===")
        try:
            model = build(kind)
        except Exception as exc:                       # network/VRAM failure
            print(f"[skip] {kind}: {exc}")
            reports[kind] = None
            continue
        report = run_benchmark(model, args.max_new_tokens)
        write(filename, report)
        reports[kind] = report
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if reports.get("jin") and reports.get("gpt2"):
        write_comparison(reports["jin"], reports["gpt2"])
    else:
        print("\n[compare] Reference model unavailable — no comparison.json written. "
              "Reporting JIN numbers alone.")


def write_comparison(jin: dict, ref: dict) -> None:
    """Write comparison.json from two already-scored reports."""
    comparison = {
            "eval_version": EVAL_VERSION,
            "note": ("Task scores use identical prompts and greedy decoding. "
                     "Perplexity is only directly comparable when both models use "
                     "the same tokenizer; tokenizer ids are recorded per model."),
        "task_score": {"jin": jin["aggregate_task_score"],
                       "gpt2": ref["aggregate_task_score"]},
        "relative_to_gpt2": round(
            jin["aggregate_task_score"] / max(1e-9, ref["aggregate_task_score"]), 4),
        "per_category": {
            cat: {"jin": jin["categories"].get(cat, {}).get("score"),
                  "gpt2": ref["categories"].get(cat, {}).get("score")}
            for cat in sorted(set(jin["categories"]) | set(ref["categories"]))
        },
        "perplexity": {
            "jin": jin["language_modeling"],
            "gpt2": ref["language_modeling"],
            "comparable": (jin["language_modeling"] or {}).get("tokenizer")
            == (ref["language_modeling"] or {}).get("tokenizer"),
        },
        "gap_analysis": [],
    }
    for cat, row in comparison["per_category"].items():
        j, g = row["jin"], row["gpt2"]
        if j is not None and g is not None and g > 0 and j < g:
            comparison["gap_analysis"].append(
                {"category": cat, "jin": j, "gpt2": g,
                 "gap_points": round(g - j, 4)})
    path = os.path.join(OUT_DIR, "comparison.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")
    print(json.dumps(comparison["task_score"], indent=2))


if __name__ == "__main__":
    main()
