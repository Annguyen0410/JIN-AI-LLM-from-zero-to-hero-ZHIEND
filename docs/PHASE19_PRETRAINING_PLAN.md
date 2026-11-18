# Phase 19 — More Data, Then Continue (plan + what is verified)

**Headline: nothing here is a measured score yet.** Phase 19 has no validation
number until the notebook runs on Kaggle. This document records what changed, why,
and which parts were actually checked on this machine.

## 1. Why this phase exists

Phase 18 measured `val 3.4839` after 299,991,040 additional tokens, but validation
bottomed at step 8,500 and did not improve over the last 655 steps while training
loss kept falling. The run was making **1.86 passes over the same 160.9 M tokens**
(2.49 passes counting Phase 17). More steps on that corpus buy memorisation, not
generalisation, so the next lever is data.

## 2. What Phase 19 changes

| | Phase 18 | Phase 19 |
|---|---|---|
| corpus | v7, 160.9 M token, wiki-only | v8, **target 800 M token**, wiki 85 % + cosmopedia 15 % |
| val set | v7 val (`b8a23690…`) | **new** v8 val — not comparable with 3.4839 |
| steps | 9,155 | **24,414** (800 M / 32,768 tok/step) |
| init | Phase 17 `best.pt` | Phase 18 `best.pt` (val 3.4839) |
| recipe | cosine 1e-4 → 1e-5, warmup 300, eval 500×20 | same (single change: data) |
| model | 124.44 M | unchanged — a bigger model needs ≥20 B tokens first |

Data sources, both already reviewed entries in `scripts/license_policy.py`, so
`--license-policy enforce` checks the licence instead of trusting a folder name:

| folder in staging | upstream | reviewed key | licence |
|---|---|---|---|
| `20231101.en` | `wikimedia/wikipedia` (41 shards, 11.6 GB, 6.41 M articles) | `wikipedia-en-20231101` | CC-BY-SA-3.0 + GFDL |
| `cosmopedia-v2` | `HuggingFaceTB/smollm-corpus` (104 shards, 1.18 GB each) | `cosmopedia-v2-sample` | ODC-BY 1.0 |

About 5 Wikipedia shards (~782 k articles ≈ 740 M token at ~950 token/article,
measured on corpus v7) plus 1 cosmopedia shard cover the target. File SHA-256 and
row counts of every downloaded shard are written to
`/kaggle/working/phase19_downloads.json` — the upstream is pinned by hash, not by
"main".

## 3. Two defects this phase had to fix first

**Corpus builder dropped the second source.** `scripts/build_tokenized_corpus.py`
enforced `--max-train-tokens` as one global stop while streaming split files in
order. The first source in file order consumed the whole budget, so
`--mix wiki:0.85,cosmopedia:0.15` produced **100 % wiki** — the mixture would have
been fiction. It now computes a per-source token budget and skips a source once it
has had its share. Local check (400 documents, 2 sources, cap 200 k):

```text
before: train composition {'20231101.en': 200445}
after : train composition {'20231101.en': 172221, 'cosmopedia_v2': 28695}  # 14.3 %
```

**Validation is not comparable after the corpus changes.** A new corpus means a new
val.bin, so `3.4839` (v7 val) cannot be subtracted from a v8 number. The notebook
therefore measures an **anchor**: Phase 18 `best.pt` scored on the *v8* val.bin
before training, using `train_gpt2.evaluate` with the trainer's own val batcher
(seed 24, batch 2, 20 iterations = 40,960 tokens). Stage H scores the new `best.pt`
the same way, so the delta is same-val, same-protocol. `3.4839` is printed only as
context, labelled as a different val set.

## 4. What was verified on this machine

| check | result |
|---|---|
| notebook JSON + every code cell compiles | 15 cells / 7 code, OK |
| constants Stage G2/H use are defined in earlier cells (a real `NameError` risk when a session resumes without running G) | asserted in the generator, OK |
| staging dir → license map → ingest per source → `verify --license-policy enforce` → build with `--mix` | end-to-end OK on 300 wiki + 100 cosmopedia documents; splits audit `passed: True`; licences resolved `cc-by-sa-3.0+gfdl` / `odc-by-1.0`; only the 2 intended sources ingested |
| mixture actually lands at 15 % | 14.3 % on the synthetic set |
| anchor-eval cell | ran against a real 124 M checkpoint (`phase14_wiki/best.pt`, sha `069d9da6…`) on a real val.bin: loads weights, `init_mode: continued_pretrain`, scores 2×2×1024 = 4,096 tokens |
| HF shard naming used by the download cell | `list_repo_files` confirms 41 `20231101.en/train-XXXXX-of-00041.parquet` and 104 `cosmopedia-v2/train-XXXXX-of-00104.parquet` |
| corpus/data unit tests after the builder change | `pytest tests/test_phase3.py tests/test_data_pipeline.py tests/test_corpus_dedup.py tests/test_license_pii.py` → 58 passed |

## 5. Not verified — read before trusting a number

- **No Kaggle run yet.** `KAGGLE_ENVIRONMENT_VERIFIED` for Phase 19, any v8
  validation loss, and all capability claims are **NOT MEASURED**.
- The in-notebook HF download was not executed here (only the file listing was).
  If Internet is off or xet download fails, Stage 1 stops with an explicit error
  rather than training on a partial corpus.
- Dedup cost on ~1.1 M documents is **NOT MEASURED**. Phase 17's 170 k documents
  were cheap; 6–7× more will take time, and Stage 1 grows the session.
- Disk: staging parquet (~2.6 GB) and `curated_v8/documents` are deleted after use,
  but the peak (documents + splits + bin) was not measured at this scale. The
  notebook prints `df -h` before training.

## 6. Session plan

800 M token at the measured 10,055 tok/s is ≈ 22 h → **3 sessions** of ~9 h.
Session 1 also builds the corpus, so `--session-max-minutes` is computed from the
remaining wall clock of an 11-hour cap instead of a fixed 560. `total_steps` never
changes; a later session runs Stage 1 (clone + corpus reuse) and G2 (resume).

If validation is still falling at step 24,414, the next step is more data again
(full Wikipedia ≈ 5–6 B token), not more steps on v8. Instruction following remains
an SFT phase, not a pretraining outcome.
