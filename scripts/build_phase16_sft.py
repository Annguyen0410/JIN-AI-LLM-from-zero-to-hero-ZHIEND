"""Phase 16 §5 — build a trusted SFT dataset, or refuse and say why.

Gates, in order, each counted in the report:

1. **Licence** — the source must be in the reviewed `SFT_SOURCES` table.
   `data/chat_train.txt` and `smol-smoltalk` are ineligible (no per-record
   provenance / aggregated constituents).
2. **Format** — `messages` list, roles in {system,user,assistant}, at least one
   user turn and one assistant turn, non-empty content, assistant turn last.
3. **Exact dedup** — sha256 of the normalised conversation.
4. **Near dedup** — 5-gram shingles, Jaccard >= 0.8, greedy clustering.
5. **PII** — pattern detectors; a flagged conversation is excluded, counts only.
6. **Answer quality** — length bounds, no degenerate answers (only punctuation,
   repeated single token, refusal boilerplate).
7. **Contamination** — no conversation may overlap the frozen Phase 16 benchmark
   prompts or gold answers.
8. **Split** — cluster-level, by hash, 98 / 1 / 1, so a near-duplicate cannot
   straddle train and test.

Output: `data/phase16_sft/{train,val,test}.jsonl` + `manifest.json` and
`bench_results/phase16/sft_data_quality.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from license_policy import detect_pii                 # noqa: E402
from sft_sources import SFT_SOURCES, decision         # noqa: E402

VALID_ROLES = {"system", "user", "assistant"}
MIN_ANSWER_CHARS = 2
MAX_ANSWER_CHARS = 4000
NEAR_DUP_JACCARD = 0.8
SHINGLE = 5


def normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def conv_hash(messages: list[dict]) -> str:
    payload = "\n".join(f"{m.get('role')}:{normalize(str(m.get('content', '')))}"
                        for m in messages)
    return hashlib.sha256(payload.encode()).hexdigest()


def shingles(text: str) -> set[str]:
    words = normalize(text).lower().split()
    if len(words) < SHINGLE:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def signature(sh: set[str], k: int = 12) -> tuple[int, ...]:
    """Bottom-k signature: the k smallest hashed shingles (LSH bucket key)."""
    hashes = sorted(int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)
                    for s in sh)
    return tuple(hashes[:k])


def benchmark_fingerprints() -> tuple[list[set[str]], list[str], set[str]]:
    """Prompts and gold answers from the frozen benchmark, for contamination."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from phase16_gold import GOLD
    from phase16_task_benchmark import CODE_PROBLEMS, TASKS
    prompts = [shingles(t["prompt"]) for t in TASKS]
    prompts += [shingles(p) for _f, p, _t in CODE_PROBLEMS]
    golds = [normalize(g).lower() for g in GOLD.values()]
    code_prompts = {f for f, _p, _t in CODE_PROBLEMS}
    return prompts, golds, code_prompts


def validate_conversation(rec: dict) -> tuple[bool, str]:
    messages = rec.get("messages")
    if not isinstance(messages, list) or not messages:
        return False, "messages_missing"
    roles = []
    for m in messages:
        if not isinstance(m, dict):
            return False, "message_not_object"
        role = m.get("role")
        if role not in VALID_ROLES:
            return False, f"bad_role:{role}"
        content = normalize(str(m.get("content") or ""))
        if not content:
            return False, "empty_content"
        roles.append(role)
    if "user" not in roles:
        return False, "no_user_turn"
    if "assistant" not in roles:
        return False, "no_assistant_turn"
    if roles[-1] != "assistant":
        return False, "assistant_not_last"
    return True, "ok"


def answer_quality(messages: list[dict]) -> tuple[bool, str]:
    answer = normalize(str(messages[-1].get("content") or ""))
    if len(answer) < MIN_ANSWER_CHARS:
        return False, "answer_too_short"
    if len(answer) > MAX_ANSWER_CHARS:
        return False, "answer_too_long"
    if not any(ch.isalnum() for ch in answer):
        return False, "no_alphanumeric"
    words = answer.lower().split()
    if len(words) >= 8 and len(set(words)) == 1:
        return False, "single_token_repeated"
    if len(set(words)) <= max(1, len(words) // 10) and len(words) > 20:
        return False, "degenerate_repetition"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 16 SFT dataset builder")
    ap.add_argument("--sft-root", default=os.path.join(ROOT, "data",
                                                      "prepared_general_v1", "sft_train"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "data", "phase16_sft"))
    ap.add_argument("--report", default=os.path.join(ROOT, "bench_results", "phase16",
                                                     "sft_data_quality.json"))
    ap.add_argument("--max-records", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    bench_prompts, bench_golds, _ = benchmark_fingerprints()

    files = sorted(f for f in os.listdir(args.sft_root)
                   if os.path.isfile(os.path.join(args.sft_root, f)))
    stats: dict = {"sources": {}, "gates": Counter(), "input_records": 0}
    kept: list[dict] = []

    for fname in files:
        path = os.path.join(args.sft_root, fname)
        src_stats = {"file": fname, "read": 0, "kept": 0, "dropped": Counter()}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                src_stats["read"] += 1
                stats["input_records"] += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    src_stats["dropped"]["malformed_json"] += 1
                    continue
                prov = rec.get("provenance") or {}
                source_id = prov.get("dataset_id") or fname.replace(".jsonl", "")
                src_stats.setdefault("source_id", source_id)
                dec = decision(source_id)
                if not dec["eligible"]:
                    src_stats["dropped"][f"license:{dec['license']}"] += 1
                    continue
                ok, reason = validate_conversation(rec)
                if not ok:
                    src_stats["dropped"][f"format:{reason}"] += 1
                    continue
                messages = rec["messages"]
                ok, reason = answer_quality(messages)
                if not ok:
                    src_stats["dropped"][f"quality:{reason}"] += 1
                    continue
                text = " ".join(normalize(str(m.get("content"))) for m in messages)
                pii = detect_pii(text)
                if pii:
                    src_stats["dropped"]["pii"] += 1
                    continue
                sh = shingles(text)
                if any(jaccard(sh, b) >= NEAR_DUP_JACCARD for b in bench_prompts):
                    src_stats["dropped"]["contamination:benchmark_prompt"] += 1
                    continue
                if any(g in text.lower() and len(g) > 12 for g in bench_golds):
                    src_stats["dropped"]["contamination:benchmark_answer"] += 1
                    continue
                norm = normalize("\n".join(f"{m['role']}:{normalize(str(m['content']))}"
                                            for m in messages))
                if any(g == norm.lower() for g in bench_golds):
                    src_stats["dropped"]["contamination:benchmark_answer"] += 1
                    continue
                kept.append({
                    "messages": messages,
                    "source_id": source_id,
                    "license": dec["license"],
                    "provenance": prov,
                    "content_hash": conv_hash(messages),
                    "shingles": sorted(sh)[:400],
                })
                src_stats["kept"] += 1
                if args.max_records and len(kept) >= args.max_records:
                    break
        stats["sources"][fname] = {**src_stats, "dropped": dict(src_stats["dropped"])}
        if args.max_records and len(kept) >= args.max_records:
            break

    # exact dedup on content_hash
    seen: dict[str, int] = {}
    unique: list[dict] = []
    for rec in kept:
        if rec["content_hash"] in seen:
            stats["gates"]["exact_duplicates"] += 1
            continue
        seen[rec["content_hash"]] = len(unique)
        unique.append(rec)

    # near dedup: bottom-k LSH banding, then exact Jaccard inside candidate buckets.
    # Comparing every pair would be O(n^2) over 50k conversations; banding gives
    # the same answer for the pairs that can possibly be >= 0.8 similar, and the
    # approximation is stated in the report rather than hidden.
    buckets: dict[int, list[int]] = defaultdict(list)
    cluster_reps: list[set[str]] = []
    cluster_ids: list[int] = []
    clustered: list[dict] = []
    compared = 0
    for rec in unique:
        sh = set(rec["shingles"])
        sig = signature(sh)
        candidate_ids: set[int] = set()
        for key in sig:
            for cid in buckets.get(key, ()):
                candidate_ids.add(cid)
        assigned = None
        for cid in sorted(candidate_ids):
            compared += 1
            if jaccard(sh, cluster_reps[cid]) >= NEAR_DUP_JACCARD:
                assigned = cid
                break
        if assigned is None:
            assigned = len(cluster_reps)
            cluster_reps.append(sh)
            cluster_ids.append(assigned)
        else:
            stats["gates"]["near_duplicates"] += 1
        for key in sig:
            buckets[key].append(assigned)
        rec["cluster_id"] = f"sft{assigned:06d}"
        clustered.append(rec)
    stats["gates"]["jaccard_comparisons"] = compared

    # cluster-level split, by hash
    for rec in clustered:
        h = int(hashlib.sha256(f"phase16.{rec['cluster_id']}".encode()).hexdigest()[:8], 16)
        bucket = h % 100
        rec["split"] = "train" if bucket < 98 else ("val" if bucket < 99 else "test")

    os.makedirs(args.out_root, exist_ok=True)
    counts = Counter()
    for split in ("train", "val", "test"):
        rows = [r for r in clustered if r["split"] == split]
        counts[split] = len(rows)
        with open(os.path.join(args.out_root, f"{split}.jsonl"), "w",
                  encoding="utf-8") as fh:
            for r in rows:
                out = {k: v for k, v in r.items() if k != "shingles"}
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    # verification: no cluster straddles splits, no duplicate hashes
    by_cluster = defaultdict(set)
    for rec in clustered:
        by_cluster[rec["cluster_id"]].add(rec["split"])
    straddle = sum(1 for v in by_cluster.values() if len(v) > 1)
    dupes = len(clustered) - len({r["content_hash"] for r in clustered})

    license_table = {sid: {k: v[k] for k in ("license", "eligible", "card", "review")}
                     for sid, v in SFT_SOURCES.items()}
    report = {
        "protocol": "phase16.sft.v1",
        "input_records": stats["input_records"],
        "kept_after_gates": len(kept),
        "unique_after_exact_dedup": len(unique),
        "after_near_dedup": len(clustered),
        "gates": dict(stats["gates"]),
        "splits": dict(counts),
        "clusters": len(by_cluster),
        "clusters_straddling_splits": straddle,
        "duplicate_content_hashes": dupes,
        "near_dup_jaccard": NEAR_DUP_JACCARD,
        "near_dup_method": "bottom-12 shingle-hash LSH banding + exact Jaccard; "
                           "pairs that share no signature bucket are not compared",
        "sources": stats["sources"],
        "license_table": license_table,
        "pii": "pattern-based only: emails, phones, IPv4, card-shaped numbers, SSNs; "
               "this is not comprehensive PII removal",
        "output_root": os.path.relpath(args.out_root, ROOT),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"[input]   {report['input_records']:,} records from {len(files)} files")
    print(f"[gates]   exact dup {report['gates'].get('exact_duplicates', 0):,} | "
          f"near dup {report['gates'].get('near_duplicates', 0):,}")
    print(f"[kept]    {len(clustered):,} conversations | "
          f"train {counts['train']:,} val {counts['val']:,} test {counts['test']:,}")
    print(f"[check]   straddling clusters {straddle} | duplicate hashes {dupes}")
    for fname, s in stats["sources"].items():
        print(f"  {fname:26s} read {s['read']:>6,} kept {s['kept']:>6,} "
              f"dropped {s['dropped']}")
    print(f"Wrote {os.path.relpath(args.report, ROOT)}")


if __name__ == "__main__":
    main()
