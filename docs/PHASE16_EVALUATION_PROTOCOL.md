# PHASE 16 §2 — EVALUATION PROTOCOL v2 (coverage)

## 1. The defect in Phase 15's evaluator

`scripts/eval_phase15.py::document_windows()` kept a window only when
`start + seq_len + 1 <= len(framed)`. Consequences:

| Document | Phase 15 | Correct |
|---|---|---|
| shorter than `seq_len + 1` | **0 tokens scored**, still counted in `documents` | all body tokens scored |
| exactly one window | scored | scored |
| longer, final partial window | **discarded** | scored |
| report | "24 documents" regardless | documents + tokens that actually contributed |

The loss itself was clean (no padding), so Phase 15's claim of "no padding in the
denominator" was true. The **coverage** claim was never made and never checked: the
published clean-slice numbers were computed on 11,264 tokens out of the 37,558
available (30 %).

## 2. Protocol v2 (`phase16.coverage.v2`)

```
framed  = [EOT] + tokens(document) + [EOT]
windows = consecutive seq_len+1 chunks; the final remainder is scored when it has ≥ 2 tokens
each window scored on its own targets (batch of 1) — no padding, no cross-document attention
loss(doc) = Σ NLL / tokens scored in that doc
```

Reported per run:

- `macro_loss` — mean over strata, each stratum weighted equally
- `token_weighted_loss` — Σ NLL / Σ tokens (long documents dominate)
- `documents_selected`, `documents_scored`, `documents_unscored` **with reasons**
- `eligible_tokens`, `tokens_scored`, `token_coverage`
- `padding_tokens: 0`
- document-length buckets, per-stratum rows, selected document ids, seed,
  benchmark file hash

## 3. Measured effect

Identical benchmark (`data/curated_v6/benchmark/clean_unseen.jsonl`, 4,000 candidate
documents), identical sampling seed, identical tokenizer, `seq_len = 512`:

| Model | Protocol | Loss (macro) | ppl | Loss (token-weighted) | Tokens scored | Coverage |
|---|---|---|---|---|---|---|
| JIN Phase 14 | Phase 15 | 5.0271 | 152.50 | — | 11,264 | ~30 % |
| JIN Phase 14 | **Phase 16 v2** | **4.6188** | **101.37** | **4.8935** | **37,558** | **100 %** |
| GPT-2 (reference) | Phase 15 | 3.2497 | 25.78 | — | 11,264 | ~30 % |
| GPT-2 (reference) | **Phase 16 v2** | **3.4227** | **30.65** | **3.3203** | **37,558** | **100 %** |

Read carefully:

- Phase 14's macro loss **improved** (5.03 → 4.62) because the previously discarded
  short documents and tails are easy text for it, and they were being excluded.
- GPT-2's macro loss **worsened** (3.25 → 3.42) for the same reason: the discarded
  slice was not a random sample of the benchmark.
- The gap between the two models on the full-token protocol is **1.20 nats macro**
  (4.6188 vs 3.4227) or **1.57 nats token-weighted** (4.8935 vs 3.3203). Phase 15's
  1.78 nats was measured on a non-random 30 % of the benchmark, so **1.57 nats
  token-weighted is the number to quote**; the macro figure answers a different
  question (equal weight per stratum).
- Neither protocol favours one model: both models are scored by the same code and
  the same documents, which is why this is a correction, not a re-ranking.

**The 4,000-document pool is not "4,000 evaluated documents".** The evaluator
samples 55 documents (24 per stratum × 2 strata + 7 seed conversations) and now
scores all 55, covering 37,558 tokens. Both numbers are in every artifact.

## 4. Tests (`tests/test_evaluation_phase16.py`, 11 tests)

| Test | Property |
|---|---|
| short document | scored, not skipped |
| exactly one window | all targets counted |
| just longer than one window | remainder counted |
| short final remainder | counted |
| mixed lengths | coverage 100 % |
| no padding | loss is exactly ln(vocab) with uniform logits, at any length |
| no cross-document leakage | two documents together = sum of their token counts |
| determinism | byte-identical aggregates on repeat |
| empty document | reported with `empty_document`, never scored |
| window arithmetic | Σ(len−1) = tokens−1 for every length 2…39 |
| length buckets | bucket boundaries |

A uniform-logit stub makes any denominator error visible: with no padding in the
metric, the loss must be `ln(vocab)` exactly.

## 5. What v2 still does not do

- It scores **language modelling**, not task ability. Task ability is measured
  separately in `docs/PHASE16_CAPABILITY_BENCHMARK.md`.
- `token_coverage` is coverage of the *sampled* documents, not of the corpus.
- A single-token document cannot be scored (no target) and is reported as
  `empty_document`/`no_scorable_targets` rather than silently dropped.
- Per-stratum samples are still capped at 24; the strata are reported with their
  document counts so a thin stratum is visible.

## 6. Reproduce

```bash
python scripts/eval_phase16.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
    --val-jsonl data/curated_v6/benchmark/clean_unseen.jsonl --seq-len 512 \
    --out bench_results/phase16/eval_phase14_s512.json
python scripts/eval_phase16.py --hf-model gpt2 --val-jsonl ... --seq-len 512 \
    --out bench_results/phase16/eval_gpt2_s512.json
python -m pytest tests/test_evaluation_phase16.py -q
```
