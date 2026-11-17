# Phase 17 — Starting State Audit

Captured at the start of Phase 17, before any code change.

## Repository

| item | value |
|---|---|
| HEAD | `c85af5292c6dfc1aed1da9c7d9f887808eb5a1c2` (Phase 16) |
| Working tree | clean at audit time |
| Previous commits | `64165e64` (Phase 14) → `5b15ba4f` (Phase 15) → `c85af529` (Phase 16) |
| Preflight artifact | `bench_results/phase17/kaggle_preflight.json` |

## Phase 14 checkpoint (the parent)

| item | value |
|---|---|
| path | `checkpoints/gpt2_small/phase14_wiki/best.pt` |
| size | 1,493,474,638 bytes |
| sha256 | `069d9da650a81dc9e2c0fee94d0e20dc46a5c4f7e266a6c381a72140f48266fa` |
| iteration | 640 |
| tokens seen | 47,185,920 |
| dataset sha | `a44913e1…` |
| tokenizer | GPT-2 BPE, 50,257 vocab, state hash `40bbc29d…` |

Architecture read from the checkpoint: 12 layers, hidden 768, 12 heads, 12 KV
heads, block size 1024, vocabulary 50,257 — total **124,439,808** parameters,
verbatim GPT-2 Small geometry.

### Weight tying

The checkpoint's `state_dict` is 163,037,184 values because `head.weight` is
stored as a duplicate copy of `tok_emb.weight`. Phase 16 verified the two are
byte-identical (max abs difference 0.0). Phase 17 preserves the tie on load and
on save: an untied head would silently be a different model, not a storage
optimisation.

## Phase 16 baseline being preserved

| measurement | value | artifact |
|---|---|---|
| clean language modelling | 4.8935 nats (ppl 133.4) | `bench_results/phase16/eval_*.json`, protocol `phase16.coverage.v2`, 100 % coverage |
| GPT-2 Small reference | 3.3203 nats (ppl 27.7) | same artifact; gap 1.57 nats |
| frozen task benchmark | 0/73 base, 2/73 after LoRA SFT | `bench_results/phase16/task_benchmark*.json` |
| retrieval recall@3 | 1.00 on 12 passages | `bench_results/phase16/system_benchmark.json` |
| calculator | 14/14 called directly (no routing) | same |
| citation syntax | 100 % valid | same |
| model answer-in-evidence | 0 % | same |
| test suite | 411 passed, 21 subtests, 0 failed | `pytest tests/` |

The 2/73 LoRA result is **not** treated as a capability gain anywhere in Phase 17:
two successes in 73 binary tasks is inside sampling noise.

## Hardware in this development environment

`bench_results/phase17/kaggle_preflight.json`:

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060, 6.0 GB |
| CUDA / torch | 12.1 / 2.5.1+cu121 |
| CPU / OS | 12 threads, Windows 10 |
| `/kaggle/input` | absent — this is **not** a Kaggle machine |

Kaggle availability, accelerator, VRAM, quota and runtime are therefore
**unknown**, and no number in Phase 17 assumes them. The historical local figure
of 8,000–10,000 tokens/s is a local measurement on this RTX 2060 and is not a
Kaggle estimate.

## What was missing at the start

1. `train_gpt2.py` had `--resume` but no `--init-from`, so continued pretraining
   from Phase 14 could not be expressed at all. `--resume` restores a run; it
   does not transfer weights into a new experiment.
2. No session-level wall-clock budget. `--extend-budget` mutates the cosine
   schedule, so using it as a per-session stop would change the experiment.
3. No Kaggle notebook or preflight artifact.
4. No `data/curated_v7` / `data/gpt2_corpus_v7`: the ~10 GB Wikipedia parquet has
   never been ingested (see `docs/PHASE17_DATA_REPORT.md`).
5. No routing layer: Phase 16's calculator score came from calling the calculator
   directly, which measures arithmetic, not tool selection.
