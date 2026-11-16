"""Phase 15I — generation quality suite (no LLM judge, no hidden scorer).

Every metric here is transparent and computable from the raw text, and every raw
sample is kept — including bad ones — so a claim about generation quality can be
checked by reading the output rather than trusting a summary.

Categories
---------
en_continuation, vi_continuation, factual, educational, code, math,
structured_list, long_context, repetition_stress, eos_behavior

Metrics per category
--------------------
continuation_words, premature_eos_rate, repeat_4gram_rate, distinct_1,
distinct_2, replacement_char_rate, vietnamese_diacritic_preserved (vi),
python_parses (code), topic_overlap (content-word overlap prompt↔continuation)

Usage
-----
    python scripts/generation_suite.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
        --out bench_results/phase15/generation_phase14.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tokenizer_gpt2 import GPT2BPETokenizer   # noqa: E402
from utils import _torch_load, create_model    # noqa: E402

VI_DIACRITICS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ"
                    "ĂÂĐÊÔƠƯÀẢÃÁẠẰẲẴẮẶẦẨẪẤẬÈẺẼÉẸỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌỒỔỖỐỘỜỞỠỚỢÙỦŨÚỤỪỬỮỨỰỲỶỸÝỴ")
STOPWORDS = set("""a an the and or but if then than that this these those of in on at to for from by
with without as is are was were be been being it its they them their we our you your he she his her
not no do does did can could should would may might will just also very more most other some such
""".split())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def content_words(text: str) -> set[str]:
    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def metrics(prompt: str, cont: str, max_new: int, language: str | None = None) -> dict:
    words = cont.split()
    n = len(words)
    bigrams = list(zip(words, words[1:])) or ([words[0]] if words else [])
    grams4 = [tuple(words[i:i + 4]) for i in range(max(1, n - 3))]
    prompt_words = content_words(prompt)
    cont_words = content_words(cont)
    out = {
        "continuation_words": n,
        "continuation_chars": len(cont),
        "premature_eos": n < max(3, max_new // 8),
        "repeat_4gram_rate": round(1 - len(set(grams4)) / max(1, len(grams4)), 3),
        "distinct_1": round(len(set(words)) / max(1, n), 3),
        "distinct_2": round(len(set(bigrams)) / max(1, len(bigrams)), 3),
        "replacement_char_rate": round(cont.count("\ufffd") / max(1, len(cont)), 6),
        "topic_overlap": round(len(prompt_words & cont_words) / max(1, len(prompt_words) or 1), 3),
    }
    if language == "vi" and cont:
        diacritics = sum(1 for ch in cont if ch in VI_DIACRITICS)
        out["vietnamese_diacritics_present"] = diacritics > 0
        out["vietnamese_diacritic_ratio"] = round(diacritics / max(1, len(cont)), 4)
        out["unicode_nfc_stable"] = unicodedata.normalize("NFC", cont) == cont
    return out


def python_parses(text: str) -> bool:
    """Only meaningful when the continuation looks like a function body."""
    snippet = text.strip()
    if "def " not in snippet and "return" not in snippet:
        return False
    for candidate in (snippet, "def _f():\n" + "\n".join("    " + l for l in snippet.splitlines())):
        try:
            ast.parse(candidate)
            return True
        except SyntaxError:
            continue
    return False


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int, temperature: float,
             seed: int, device: str) -> str:
    ids = tokenizer.encode(prompt)
    block = getattr(model, "block_size", 1024)
    x = torch.tensor([ids[-block:]], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    generated: list[int] = []
    for _ in range(max_new):
        logits, _ = model(x[:, -block:])
        logits = logits[:, -1, :].float()
        if temperature <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
        token = int(nxt)
        if token == tokenizer.eot_id:
            break
        generated.append(token)
        x = torch.cat([x, nxt], dim=1)
    return tokenizer.decode(generated)


def load_prompts(test_jsonl: str, per_category: int, seed: int) -> dict[str, list[dict]]:
    """Held-out prompts. Documents are picked deterministically by hash."""
    wanted = {
        "en_continuation": lambda d: d.get("language") == "en" and d.get("domain") == "knowledge",
        "educational": lambda d: d.get("domain") == "educational",
        "vi_continuation": lambda d: d.get("language") == "vi",
        "factual": lambda d: d.get("domain") == "knowledge" and len(d.get("text", "")) > 1200,
    }
    picked: dict[str, list[tuple[bytes, dict]]] = {k: [] for k in wanted}
    with open(test_jsonl, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            for name, rule in wanted.items():
                if len(picked[name]) >= per_category * 4:
                    continue
                try:
                    if rule(doc):
                        key = hashlib.sha256(f"{seed}.{doc.get('document_id')}".encode()).digest()
                        picked[name].append((key, doc))
                except Exception:
                    continue
    out = {}
    for name, rows in picked.items():
        rows.sort(key=lambda r: r[0])
        prompts = []
        for _key, doc in rows[:per_category]:
            text = doc["text"]
            cut = text[:600]
            stop = max(cut.rfind(". "), cut.rfind("\n\n"))
            prompt = cut[:stop + 1] if stop > 150 else cut[:400]
            prompts.append({"document_id": doc.get("document_id"),
                            "source_id": doc.get("source_id"),
                            "language": doc.get("language"),
                            "prompt": prompt})
        out[name] = prompts
    return out


CODE_PROMPT = "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
MATH_PROMPT = ("Problem: A train travels 120 km in 2 hours, then 180 km in 3 hours. "
               "What is its average speed for the whole journey?\nSolution: ")
LIST_PROMPT = "Here are the main steps to set up a small home laboratory:\n1."
REPEAT_PROMPT = ("The museum opened in 1902. " * 24)
EOS_PROMPT = "The end."


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 15 generation suite")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-jsonl", default=os.path.join(ROOT, "data", "curated_v6",
                                                          "splits", "test.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1515)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = GPT2BPETokenizer()
    blob = _torch_load(args.checkpoint, map_location="cpu")
    model = create_model(int(blob.get("tokenizer", {}).get("vocab_size", 50257)),
                         checkpoint=blob, device=args.device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    prompts = load_prompts(args.test_jsonl, args.per_category, args.seed) \
        if os.path.exists(args.test_jsonl) else {}
    prompts["code"] = [{"language": "en", "document_id": "synthetic:code",
                        "prompt": CODE_PROMPT}]
    prompts["math"] = [{"language": "en", "document_id": "synthetic:math",
                        "prompt": MATH_PROMPT}]
    prompts["structured_list"] = [{"language": "en", "document_id": "synthetic:list",
                                   "prompt": LIST_PROMPT}]
    prompts["long_context"] = [{"language": "en", "document_id": "synthetic:long",
                                "prompt": REPEAT_PROMPT}]
    prompts["repetition_stress"] = [{"language": "en", "document_id": "synthetic:rep",
                                     "prompt": REPEAT_PROMPT}]
    prompts["eos_behavior"] = [{"language": "en", "document_id": "synthetic:eos",
                                "prompt": EOS_PROMPT}]

    report = {
        "checkpoint": os.path.relpath(args.checkpoint, ROOT),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "iteration": blob.get("iteration"),
        "tokens_seen": blob.get("tokens_seen"),
        "prompt_source": os.path.relpath(args.test_jsonl, ROOT),
        "decoding": {"greedy": "temperature 0", "sampled": "temperature 0.8",
                     "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                     "no_repetition_penalty": True},
        "categories": {},
        "all_raw_samples_kept": True,
    }

    for category, items in sorted(prompts.items()):
        rows = []
        for i, item in enumerate(items):
            greedy = generate(model, tokenizer, item["prompt"], args.max_new_tokens,
                              0.0, args.seed + i, args.device)
            sampled = generate(model, tokenizer, item["prompt"], args.max_new_tokens,
                               0.8, args.seed + 1000 + i, args.device)
            rows.append({
                **{k: item.get(k) for k in ("document_id", "source_id", "language")},
                "prompt": item["prompt"],
                "greedy": greedy, "greedy_metrics": metrics(item["prompt"], greedy,
                                                             args.max_new_tokens,
                                                             item.get("language")),
                "sampled": sampled, "sampled_metrics": metrics(item["prompt"], sampled,
                                                                args.max_new_tokens,
                                                                item.get("language")),
                "python_parses": python_parses(sampled) if category == "code" else None,
            })
        def mean(key: str) -> float:
            vals = [r["sampled_metrics"].get(key) for r in rows
                    if isinstance(r["sampled_metrics"].get(key), (int, float))]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        report["categories"][category] = {
            "prompts": len(rows),
            "mean_continuation_words": mean("continuation_words"),
            "premature_eos_rate": mean("premature_eos"),
            "repeat_4gram_rate": mean("repeat_4gram_rate"),
            "distinct_1": mean("distinct_1"),
            "distinct_2": mean("distinct_2"),
            "topic_overlap": mean("topic_overlap"),
            "replacement_char_rate": mean("replacement_char_rate"),
            "vietnamese_diacritics_present_rate": mean("vietnamese_diacritics_present"),
            "samples": rows,
        }
        c = report["categories"][category]
        print(f"[{category:18s}] words {c['mean_continuation_words']:>6.1f} | "
              f"premature_eos {c['premature_eos_rate']:.2f} | "
              f"repeat4 {c['repeat_4gram_rate']:.2f} | d2 {c['distinct_2']:.2f} | "
              f"topic {c['topic_overlap']:.2f}")

    out = args.out or os.path.join(ROOT, "bench_results", "phase15",
                                   f"generation_{os.path.basename(os.path.dirname(args.checkpoint))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
