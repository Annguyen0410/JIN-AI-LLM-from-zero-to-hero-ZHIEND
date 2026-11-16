# PHASE 15E — CLEAN UNSEEN-BY-BOTH COMPARISON

## 1. Why Phase 14's comparison was tainted

The Phase 14 table compared JIN checkpoints against GPT-2 on validation windows
that JIN had partially seen: the pilot trained on `prepared_v4`, Phase 14 trained
on a mixture that included `prepared_v4` and Wikipedia, and the validation split
contained fragments of the same documents. GPT-2 had seen none of it. Comparing a
model that has seen the text with one that has not is not a controlled comparison.

## 2. The clean slice

`data/curated_v6/benchmark/clean_unseen.jsonl` + manifest:

| Property | Value |
|---|---|
| Candidates | `data/curated_v6/splits/test.jsonl` (document-disjoint from v6 train) |
| Unseen by Phase 14 | `content_hash` not present in `data/curated_v5/splits/train.jsonl` |
| Unseen by the pilot | source `prepared_v4` excluded entirely (the pilot's whole corpus) |
| Selection | proportional quota per source, `sha256(seed.document_id)` order |
| Result | **4,000 documents / 15.88 M chars**, wiki 1,979 + cosmopedia 2,014 + seed 7 |
| Hash overlap with Phase 14 train | **0** |
| Language | English 100 % (the Vietnamese text lives in `prepared_v4`, which the pilot saw) |

The slice is deliberately English-only. A Vietnamese comparison against the pilot
is impossible by construction, and no Vietnamese claim is made from it.

**A defect was found and fixed while building this slice.** The first version read
`test.jsonl` in file order and stopped at 4,000 documents — and because the split
file groups by source, the slice came out 100 % Cosmopedia. It was stratified by
source quota instead, and the manifest now records `available_per_source` and
`selected_per_source` so the mix can be checked.

## 3. Protocol (identical for every model)

`scripts/eval_phase15.py`, sequence length **512** for all models (Phase 15
checkpoints were trained with a 512-token context and cannot be scored at 1024 —
the first attempt failed with `Sequence length 1024 exceeds max_positions 512`,
and re-running everything at 1024 would have compared different conditions).

- document-independent scoring, EOT-framed, **no padding**
- 24 documents per stratum, 48 documents, **23,040 scored tokens** per model
- same tokenizer (GPT-2 BPE) for JIN and for the reference

## 4. Results

| Model | Train tokens | Overall loss | ppl | Cosmopedia | Wikipedia |
|---|---|---|---|---|---|
| GPT-2 (124 M, reference) | 100 B (published) | **3.2497** | **25.78** | 2.9433 | 3.4539 |
| JIN Phase 14 (`phase14_wiki`) | 47.19 M | 5.0271 | 152.50 | 4.7187 | 5.2328 |
| JIN Phase 15 A, seed 2024 | 3.28 M | 6.6793 | 795.74 | 6.6316 | 6.7111 |
| JIN Phase 15 B2 (synthetic-heavy) | 4.10 M | 6.8172 | 913.44 | **6.6658** | 6.9182 |
| JIN Phase 15 B1 (human-heavy) | 4.10 M | 6.9209 | 1013.26 | 7.1654 | **6.7579** |
| JIN pilot | ~2.6 M | 6.9281 | 1020.56 | 6.7751 | 7.0301 |

## 5. What the numbers say

1. **Phase 14 is 1.78 nats behind GPT-2** on text neither model saw in the same
   form (GPT-2 was trained on a web crawl that very likely contains Wikipedia, so
   the comparison still favours the reference). Perplexity: 152.5 vs 25.8, a 5.9×
   gap. That is the honest size of the remaining distance, not "close to GPT-2".
2. **The Phase 15 4.1 M-token runs are diagnostic, not competitive.** Their loss
   (6.68–6.92) is between the pilot and Phase 14, exactly as their token budget
   predicts.
3. **Mixture effects are visible only per domain, and in the expected direction.**
   B1 (80.7 % Wikipedia) is the best of the three 4.1 M runs on Wikipedia (6.7579)
   and the worst on Cosmopedia (7.1654). B2 (39.7 % Cosmopedia) is the reverse:
   best on Cosmopedia (6.6658), worst on Wikipedia (6.9182). Overall differences
   (0.10) stay inside the seed range (0.19), which is why the overall column must
   not be used to pick a mixture.
4. **The pilot → A(seed 2024) improvement is real but small.** 6.9281 → 6.6793
   with less than twice the pilot's tokens, on data the pilot never saw. This is
   the only "JIN got better" claim in Phase 15 that survives the clean-slice test,
   and it is about the pilot, not about Phase 14.

## 6. What must not be claimed

- Not "JIN matches GPT-2": it is 1.78 nats behind on this slice and 5.9× in
  perplexity.
- Not "the Phase 15 checkpoints are the best JIN": they have 8 % of Phase 14's
  tokens and score accordingly.
- Not "mixture B1/B2 is better": the differences are within seed noise overall,
  and the per-domain split is a data-composition effect, not a quality verdict.
- Not "Vietnamese improved": this slice contains no Vietnamese by construction.

## 7. Reproduce

```bash
python scripts/build_clean_comparison.py --max-docs 4000
python scripts/eval_phase15.py --hf-model gpt2 --val-jsonl data/curated_v6/benchmark/clean_unseen.jsonl \
    --seq-len 512 --docs-per-stratum 24 --out bench_results/phase15/s512_gpt2.json
python scripts/eval_phase15.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
    --val-jsonl data/curated_v6/benchmark/clean_unseen.jsonl --seq-len 512 \
    --docs-per-stratum 24 --out bench_results/phase15/s512_phase14.json
```

Every artifact records the checkpoint SHA-256, the token count, the sampling seed,
the strata and `padding_tokens: 0`.
