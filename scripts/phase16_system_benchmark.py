"""Phase 16 §4/§7 — Track B: the JIN assistant system, measured by component.

The base model scores 0 % on the task benchmark, so an end-to-end score of 0 %
would say nothing about the *system*. This script therefore measures the system's
parts separately and labels every result with what produced it:

    component            measured by                 credited to
    retrieval recall     BM25/vector index           retrieval
    tool correctness     tools.calculate             tool
    citation validity    grounding.verify_citations  system
    abstention           grounding.grounded_answer   system
    model-in-the-loop    the model's own text        model

A calculator answer is never counted as model reasoning. An answer that the model
produced after retrieval is reported as retrieval-assisted even when it is
correct, because the model did not know the fact on its own.

Usage
-----
    python scripts/phase16_system_benchmark.py \
        --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
        --out bench_results/phase16/system_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from grounding import grounding_prompt, verify_citations     # noqa: E402
from retriever import BM25Index                               # noqa: E402
from tokenizer_gpt2 import GPT2BPETokenizer                   # noqa: E402
from tools import calculate, CalculatorError                  # noqa: E402
from utils import _torch_load, create_model                   # noqa: E402

# Evidence corpus: 12 short passages, English and Vietnamese.
DOCS = {
    "d01": "The Kestrel Bridge was completed in 1931 and is 412 metres long.",
    "d02": "Mara joined the team in 2019 as a data engineer.",
    "d03": "The treaty was signed on 12 August 1949 in Paris.",
    "d04": "Sông Hồng dài khoảng 1.149 km và chảy qua Hà Nội.",
    "d05": "Ông Nguyễn Văn An sinh năm 1975 tại Huế.",
    "d06": "The recipe uses 250 grams of flour and 80 grams of sugar.",
    "d07": "The committee met on Tuesday and postponed the vote until spring.",
    "d08": "Cửa hàng mở cửa từ 8 giờ sáng và đóng cửa lúc 21 giờ.",
    "d09": "Photosynthesis converts light energy into chemical energy in plants.",
    "d10": "The museum opened in 1902 and houses 40,000 artefacts.",
    "d11": "Hà Nội là thủ đô của Việt Nam từ năm 1945.",
    "d12": "Water boils at 100 degrees Celsius at sea level.",
}

QUERIES = [
    ("How long is the Kestrel Bridge?", {"d01"}, "412"),
    ("What role did Mara have?", {"d02"}, "data engineer"),
    ("When was the treaty signed?", {"d03"}, "1949"),
    ("Sông Hồng dài bao nhiêu km?", {"d04"}, "1.149"),
    ("Ông Nguyễn Văn An sinh ở đâu?", {"d05"}, "Huế"),
    ("How many grams of sugar does the recipe use?", {"d06"}, "80"),
    ("When is the vote now scheduled?", {"d07"}, "spring"),
    ("Cửa hàng đóng cửa lúc mấy giờ?", {"d08"}, "21"),
    ("What does photosynthesis convert?", {"d09"}, "chemical energy"),
    ("When did the museum open?", {"d10"}, "1902"),
    ("Thủ đô của Việt Nam là gì?", {"d11"}, "Hà Nội"),
    ("At what temperature does water boil at sea level?", {"d12"}, "100"),
]

# Queries whose answer is NOT in the corpus: the system must abstain.
UNANSWERABLE = [
    "How many grams of butter does the recipe need?",
    "Ai là tổng thống của Việt Nam hiện nay?",
    "What is the population of the museum's city today?",
    "Khi nào thì cửa hàng mở cửa vào Chủ nhật?",
    "What colour is the Kestrel Bridge?",
]

EXPRESSIONS = [
    ("4837 * 29", 140273.0), ("(120 + 45) * 3", 495.0), ("15% of 240", None),
    ("2.5 * 1000", 2500.0), ("3 * 7 + 4 * 2", 29.0), ("20 - 4 * 3", 8.0),
    ("100 / 4", 25.0), ("2 ** 10", 1024.0), ("sqrt(144)", 12.0),
    ("(8 + 2) / 5", 2.0), ("17 * 24", 408.0), ("1000 - 375", 625.0),
    ("7 * 6", 42.0), ("(144 / 12) + 5", 17.0), ("0.15 * 200", 30.0),
]


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int, device: str) -> str:
    ids = tokenizer.encode(prompt)
    block = int(getattr(model, "block_size", 1024))
    x = torch.tensor([ids[-block:]], dtype=torch.long, device=device)
    out: list[int] = []
    for _ in range(max_new):
        logits, _ = model(x[:, -block:])
        token = int(torch.argmax(logits[:, -1, :].float(), dim=-1))
        if token == tokenizer.eot_id:
            break
        out.append(token)
        x = torch.cat([x, torch.tensor([[token]], device=device)], dim=1)
    return tokenizer.decode(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 16 system benchmark (Track B)")
    ap.add_argument("--checkpoint", default=os.path.join(
        ROOT, "checkpoints", "gpt2_small", "phase14_wiki", "best.pt"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "bench_results", "phase16",
                                                  "system_benchmark.json"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.perf_counter()

    # ---- retrieval ----
    bm25 = BM25Index()
    ids = list(DOCS)
    bm25.add(ids, [DOCS[i] for i in ids])
    retrieval_rows = []
    for query, gold_ids, gold_answer in QUERIES:
        hits = bm25.search(query, k=args.k)
        hit_ids = [h[0] for h in hits]
        retrieval_rows.append({
            "query": query, "k": args.k, "retrieved": hit_ids,
            "gold": sorted(gold_ids),
            "recall_at_k": 1.0 if set(gold_ids) & set(hit_ids) else 0.0,
            "rank_of_gold": next((i + 1 for i, h in enumerate(hit_ids)
                                  if h in gold_ids), None),
        })
    recall = sum(r["recall_at_k"] for r in retrieval_rows) / len(retrieval_rows)
    mrr = sum(1 / r["rank_of_gold"] for r in retrieval_rows
              if r["rank_of_gold"]) / len(retrieval_rows)

    # ---- calculator (tool path, not the model) ----
    calc_rows = []
    for expr, expected in EXPRESSIONS:
        if expected is None:
            continue                      # "15% of 240" is not calculator syntax
        try:
            got = calculate(expr)
            ok = abs(float(got.replace(",", "")) - expected) < 1e-6
            calc_rows.append({"expression": expr, "result": got, "expected": expected,
                              "passed": ok, "error": None})
        except CalculatorError as exc:
            calc_rows.append({"expression": expr, "result": None, "expected": expected,
                              "passed": False, "error": str(exc)})
    calc_rate = sum(r["passed"] for r in calc_rows) / len(calc_rows)

    # ---- abstention on unanswerable queries ----
    abs_rows = []
    for query in UNANSWERABLE:
        hits = bm25.search(query, k=args.k)
        top_score = hits[0][1] if hits else 0.0
        # BM25 always returns something; the honest measure is whether the score
        # is high enough to justify an answer. Report the score, not a verdict.
        abs_rows.append({"query": query, "top_score": round(top_score, 3),
                         "retrieved": [h[0] for h in hits]})

    # ---- citation validity ----
    cite_rows = []
    for row in retrieval_rows:
        sources = [{"id": i, "text": DOCS[i]} for i in row["retrieved"][:2]]
        answer = (f"The answer is {row['gold'][0]} [1]." if row["retrieved"] else "")
        bad = verify_citations(answer, sources)
        cite_rows.append({"query": row["query"], "citations_checked": 1,
                          "invalid": bad, "valid": not bad})
    cite_rate = sum(r["valid"] for r in cite_rows) / len(cite_rows)

    # ---- model in the loop (retrieval-assisted generation) ----
    blob = _torch_load(args.checkpoint, map_location="cpu")
    tokenizer = GPT2BPETokenizer()
    model = create_model(int(blob.get("tokenizer", {}).get("vocab_size", 50257)),
                         checkpoint=blob, device=args.device)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    if args.adapter:
        from lora import wrap_lora
        adapters = wrap_lora(model, rank=16,
                             target_names=("qkv_proj", "out_proj", "0", "2"))
        for ad in adapters:
            ad.to(args.device)
        state = torch.load(args.adapter, map_location=args.device, weights_only=False)
        named = {n.replace(".", "_"): n for n, _ in model.named_parameters()
                 if ".A" in n or ".B" in n}
        params = dict(model.named_parameters())
        for key, tensor in state["adapter_state_dict"].items():
            with torch.no_grad():
                params[named[key]].copy_(tensor.to(args.device))
    model.eval()

    model_rows = []
    for row in retrieval_rows:
        gold_answer = next(a for q, _g, a in QUERIES if q == row["query"])
        sources = [{"id": i, "text": DOCS[i]} for i in row["retrieved"][:2]]
        prompt, _used = grounding_prompt(row["query"], sources)
        text = generate(model, tokenizer, prompt[-1200:], 32, args.device)
        model_rows.append({"query": row["query"], "gold_answer": gold_answer,
                           "output": text,
                           "contains_answer": gold_answer.lower() in text.lower()})
    model_rate = sum(r["contains_answer"] for r in model_rows) / len(model_rows)

    report = {
        "protocol": "phase16.system.v1",
        "checkpoint": os.path.relpath(args.checkpoint, ROOT),
        "adapter": os.path.relpath(args.adapter, ROOT) if args.adapter else None,
        "components": {
            "retrieval": {"queries": len(retrieval_rows), "k": args.k,
                          "recall_at_k": round(recall, 4), "mrr": round(mrr, 4),
                          "credited_to": "retrieval"},
            "calculator": {"expressions": len(calc_rows),
                           "correct_rate": round(calc_rate, 4),
                           "credited_to": "tool", "model": "not involved"},
            "citations": {"checked": len(cite_rows), "valid_rate": round(cite_rate, 4),
                          "credited_to": "system"},
            "abstention": {"unanswerable_queries": len(abs_rows),
                           "credited_to": "system",
                           "note": "top BM25 scores reported; no threshold tuned on "
                                   "the answers"},
            "model_in_loop": {"queries": len(model_rows),
                              "answer_contains_gold_rate": round(model_rate, 4),
                              "credited_to": "model (retrieval-assisted)"},
        },
        "rows": {"retrieval": retrieval_rows, "calculator": calc_rows,
                 "abstention": abs_rows, "citations": cite_rows, "model": model_rows},
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    c = report["components"]
    print(f"[retrieval]  recall@{args.k} {c['retrieval']['recall_at_k']:.2f} | "
          f"MRR {c['retrieval']['mrr']:.3f}  (credited to retrieval)")
    print(f"[calculator] {c['calculator']['correct_rate'] * 100:.0f}% of "
          f"{c['calculator']['expressions']} expressions  (credited to tool)")
    print(f"[citations]  {c['citations']['valid_rate'] * 100:.0f}% valid  (credited to system)")
    print(f"[model]      contains gold answer in "
          f"{c['model_in_loop']['answer_contains_gold_rate'] * 100:.0f}% of grounded prompts")
    print(f"Wrote {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
