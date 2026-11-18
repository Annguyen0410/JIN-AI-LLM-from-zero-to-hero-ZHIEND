# Phase 18 — Continued Pretraining Report (300 M tokens)

**Headline: measured, in-domain validation loss went from 3.8585 to 3.4839**
(ppl 47.39 → 32.59) after 299,991,040 additional tokens, initialized from the
Phase 17 `best.pt`. This is Wikipedia validation perplexity on `val.bin` of the
same corpus — **not** a capability claim. No downstream task, generation or
tool-use metric was measured on this checkpoint (`NOT MEASURED`).

## 1. What ran

`notebooks/phase18_kaggle_continue.ipynb` (commit `4366fa76`), Stage 1 → smoke →
G → H, one Kaggle session, accelerator GPU T4 x2 (GPU model **not** visible in the
saved log excerpt — `NOT MEASURED` from the log).

| quantity | value |
|---|---|
| init | `phase17_output/run_100m/best.pt`, sha `caac54d4fcbc…`, `init_mode: continued_pretrain` |
| preset / params | `gpt2_small_reference`, 124.44 M |
| geometry | seq 1024 × batch 4 × grad-accum 8 = **32,768 tok/step** |
| steps | **9,155** (`--max-steps`, closed budget) |
| tokens processed by training | **299,991,040** |
| schedule | warmup 300, cosine 1e-4 → 1e-5 (`min_lr_ratio` 0.1) |
| eval | every 500 steps × 20 iters at val batch 2 (40,960 val tokens per eval) |
| saves | every 250 steps, `latest.pt` + `best.pt` with manifests |
| session budget | 560 min; used **500 min of wall clock in one session** (`--session-max-minutes` never triggered) |
| wall clock | 30,000.7 s = **8.33 h** |
| throughput | **10,055 tok/s** |
| peak VRAM | 5.8 GB |
| seed | 17 |

Corpus is `data/gpt2_corpus_v7`, reused from the Phase 17 output without
rebuilding: train `1b9d5738…` (160,920,724 tokens), val `b8a23690…`.

## 2. Result — measured

| quantity | value |
|---|---|
| best val loss | **3.4839** (ppl **32.59**) @ step **8,500** (278,528,000 additional tokens) |
| final val loss | 3.5244 (ppl 33.93) @ step 9,155 |
| final train loss | 3.5174 @ step 9,155 |
| Phase 17 baseline | 3.8585 (ppl 47.39) @ step 2,500 |
| delta | **−0.3746 loss (−9.7 % relative), −31.2 % perplexity** |

The two numbers are directly comparable: same `val.bin`, same tokenizer, same
`--eval-iters 20`. The Phase 17 figure came from the same protocol, so unlike the
Phase 16 → 17 comparison this is not a cross-evaluator comparison.

Validation bottomed at step 8,500 and sat 0.0405 higher at step 9,155 while
training loss kept falling (3.5174). Each eval is only 40,960 tokens, so a 0.04
swing is near the noise band of the measurement — the honest reading is **no
further improvement in the last 655 steps**, on **1.86 passes over the same
160.9 M tokens** (2.49 passes counting Phase 17's 100 M). The last six hours of
GPU time therefore bought memory of the corpus, not new generalisation.
**Conclusion: another 300 M tokens on this corpus is not the next move. New data
is.**

## 3. Checkpoints (artifact + SHA)

| file | sha256 (prefix from log) | step | tokens this run | best val |
|---|---|---|---|---|
| `phase18_output/run_400m/latest.pt` | `4f0555bd09c34dda…` | 9,155 | 299,991,040 | 3.4839 |
| `phase18_output/run_400m/best.pt` | `cc9c4a73575e7ba0…` | 8,500 | 278,528,000 | 3.4839 |

Both manifests: `init_mode: continued_pretrain`,
`parent_sha256: caac54d4fcbc…` (Phase 17 `best.pt`). The notebook's Stage H cell
re-hashed each file against its manifest and passed (`assert sha(path) == man["sha256"]`).

Only the 16-character prefixes appear in the pasted log, and the checkpoints live
in the Kaggle version output — **no copy and no full digest exists on this
machine**. Machine-readable record: `bench_results/phase18/pretrain_300m.json`.

The Phase 17 run's own `latest.pt` was **not** overwritten: Phase 18 wrote into a
separate `run_400m` directory.

## 4. What this does not say

| claim | status |
|---|---|
| Wikipedia validation perplexity | **measured** (3.4839) |
| capability / 73-task benchmark | **NOT MEASURED** — no `scripts/gpt2_bench.py` run on this checkpoint; Phase 17's 1/73 still stands as the last capability number |
| Vietnamese quality | **NOT MEASURED** — corpus is 100 % `en` (`language_composition`), unchanged from Phase 17 |
| instruction following / chat | **NOT MEASURED** — pretraining does not create it |
| test-split loss | **NOT MEASURED** |
| mid-run session pause → resume on Kaggle | **NOT MEASURED** — the run fitted in one session; only resume-from-a-finished-run was exercised (it exited immediately, correctly) |

`[Plan] … budget 2.49B tokens` in the log is the default `n_params × 20` placeholder
printed before `--max-steps` overrides it. The real budget was 9,155 steps =
300 M tokens.

## 5. Next: Phase 19 needs data, not steps

Ranked by loss reduction per GPU hour:

1. **More tokens.** Ingest the full Wikipedia dump (11 GB ≈ 2.2 B tokens) and lift
   `--max-train-tokens` to 800 M–1 B. At 124 M parameters and ~0.5 B tokens seen,
   the model is at ~3.2 tokens/parameter; GPT-2 Small saw ~80.
2. **Mixture.** Add a second source (code/math) to `--mix`. Wiki-only lowers wiki
   perplexity and leaves chat/tool ability at zero, which is what the 1/73
   benchmark already measures.
3. **Bigger model.** Only after ≥20 B tokens. Not now.

Instruction following remains a separate phase (SFT), not a pretraining outcome.
