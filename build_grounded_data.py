"""
Build "grounded answering" SFT data: for real questions, fetch live web
search results and emit conversations in EXACTLY the format app.py injects
at inference time. After training on this, Jin learns to read the search
context block and answer using it with [n] citations.

Zero external deps — reuses web_search.web_context (DuckDuckGo, stdlib only).

Usage:
    python build_grounded_data.py --max-questions 1500
    python build_grounded_data.py --max-questions 200 --dry-run
    python build_grounded_data.py            # resumes (skips already-seen queries)

The output lands in data/seed/grounded/*.jsonl which train.py --mode instruct
picks up automatically (uncapped extra dir).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import config
from web_search import web_context
from grounding import grounded_answer, grounding_prompt
from lineage import object_hash
from data_quality import check_contamination, load_probe_texts

HF_SEED_DIR = os.path.join(config.ROOT_DIR, "data", "seed", "hf")
OUT_DIR = os.path.join(config.ROOT_DIR, "data", "seed", "grounded")

# MUST stay character-identical to app.py :: _maybe_run_web_search so that
# training format == inference format.
GROUNDING_TEMPLATE = (
    "{block}\n\n"
    "---\n"
    "User question: {message}\n"
    "Answer the user using the search results above. If they do not "
    "contain the answer, say so honestly. Cite sources by [number]."
)


def _iter_candidate_pairs(hf_dir: str):
    """Yield (query, answer) pairs from existing HF chat shards."""
    root = Path(hf_dir)
    if not root.exists():
        return
    for path in sorted(root.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = record.get("messages")
                if not isinstance(msgs, list) or len(msgs) < 2:
                    continue
                query = None
                answer = None
                for m in msgs:
                    if m.get("role") == "user" and query is None:
                        query = (m.get("content") or "").strip()
                    elif m.get("role") == "assistant":
                        answer = (m.get("content") or "").strip()
                if not query or not answer:
                    continue
                yield query, answer


def _good_query(q: str) -> bool:
    """Keep natural-language factual questions; skip code/math fragments."""
    if not (20 <= len(q) <= 300):
        return False
    lowered = q.lower()
    bad_markers = ("def ", "print(", "```", "import ", "{", "}", "int(", "=>", "&&")
    if any(m in q for m in bad_markers):
        return False
    letters = sum(c.isalpha() for c in q)
    if letters / max(len(q), 1) < 0.55:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Build grounded web-search SFT data")
    ap.add_argument("--hf-dir", default=HF_SEED_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--max-questions", type=int, default=1_500)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Explicit sampling weight; default avoids duplicate evidence examples.")
    ap.add_argument("--shard-size", type=int, default=1_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out_dir = config.output_path(args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    out_root = Path(args.out_dir)

    # Resume support: collect queries already fetched.
    done_queries: set[str] = set()
    for old in sorted(out_root.glob("grounded_*.jsonl")):
        with open(old, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    msgs = rec.get("messages") or []
                    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                    marker = "User question: "
                    if marker in user:
                        done_queries.add(user.split(marker, 1)[1].split("\n")[0].strip())
                except Exception:
                    continue

    rng = random.Random(1337)
    pairs, seen_q = [], set()
    for query, answer in _iter_candidate_pairs(args.hf_dir):
        if len(pairs) >= args.max_questions:
            break
        q = query.strip()
        if q in seen_q or q in done_queries or not _good_query(q):
            continue
        if len(answer) > 1_200:
            answer = answer[:1_200].rsplit(" ", 1)[0] + "..."
        seen_q.add(q)
        pairs.append((q, answer))
    rng.shuffle(pairs)

    print(f"[grounded] {len(pairs)} candidate questions "
          f"({len(done_queries)} already fetched before).")
    if args.dry_run:
        for q, a in pairs[:10]:
            print("  -", q[:90])
        return

    fetched = 0
    shard_idx = 0
    shard_path = out_root / f"grounded_{shard_idx:04d}.jsonl"
    t0 = time.time()
    for i, (query, answer) in enumerate(pairs, 1):
        try:
            block, sources = web_context(query, max_results=3, fetch_top=0)
        except Exception as exc:
            print(f"[grounded] {i}: search error: {exc}")
            continue
        if not sources:
            continue

        # A pre-existing HF answer is NOT proven by whatever web results we
        # happened to find. Never attach decorative citations to that answer.
        evidence = grounded_answer(query, sources)
        user_content, selected = grounding_prompt(query, sources)
        assistant_content = evidence.answer
        clean, _ = check_contamination([query])
        if not clean:
            continue

        record = {"provenance": {"builder":"extractive_grounding_v1",
                    "evidence_hash":object_hash(selected), "sources":selected,
                    "status":evidence.status, "teacher_used":False}, "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]}
        for _ in range(max(1, args.repeat)):
            out_root.mkdir(parents=True, exist_ok=True)
            with open(shard_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        fetched += 1
        if fetched % args.shard_size == 0:
            shard_idx += 1
            shard_path = out_root / f"grounded_{shard_idx:04d}.jsonl"
        if i % 25 == 0 or fetched % 25 == 0:
            rate = fetched / max(time.time() - t0, 1e-6)
            eta = (len(pairs) - i) / max(rate, 1e-6) / 60
            print(f"[grounded] {i}/{len(pairs)} scanned | {fetched} fetched "
                  f"| {rate:.2f} q/s | ETA {eta:.0f} min")

    print(f"[grounded] Done: {fetched} new grounded examples -> {args.out_dir}")
    print("[grounded] They are picked up automatically by train.py --mode instruct "
          "(uncapped extra dir).")


if __name__ == "__main__":
    main()
