# Phase 17 — Release Report

Starting commit `c85af5292c6dfc1aed1da9c7d9f887808eb5a1c2`. Nothing was committed
or pushed: that requires explicit authorization.

## 1. The twenty questions, answered

| # | question | answer |
|---|---|---|
| 1 | Was the actual Phase 14 model used as initialization? | **Yes.** `--init-from` loads it strictly; smoke run first eval 4.7155 vs ≈10.82 for random init. |
| 2 | Exact parent checkpoint SHA-256? | `069d9da650a81dc9e2c0fee94d0e20dc46a5c4f7e266a6c381a72140f48266fa` |
| 3 | Architecture and tokenizer preserved? | Yes — 12 L / 768 / 12 H / block 1024 / vocab 50,257, GPT-2 BPE; mismatches are hard failures (tests cover architecture, tokenizer, `pos_emb` shape, missing keys). Weight tying preserved. |
| 4 | How much v7 data was ingested? | **None.** Corpus v7 was not built. |
| 5 | How many tokens tokenized for v7? | **None. `NOT MEASURED`.** |
| 6 | Realized source mixture? | **`NOT MEASURED`** — no v7 corpus. |
| 7 | Additional tokens JIN actually trained on? | **196,608** (12 steps × 16,384), local RTX 2060, smoke only. |
| 8 | What GPU did Kaggle provide? | **Unknown** — no Kaggle run was possible here (`/kaggle/input` absent). |
| 9 | Throughput and memory? | Local: 2,659–3,025 tok/s (with eval overhead), 4.4 GB peak VRAM at context 1024. Not a Kaggle estimate. |
| 10 | Did session interruption/resume work without changing the schedule? | Locally yes: stopped at step 2 of 12, `total_steps` unchanged, resumed from the recorded SHA to step 12, same fingerprint. Kaggle-level verification not done. |
| 11 | Did validation loss improve on the frozen clean benchmark? | **Not measured.** Not claimed. |
| 12 | Did generated text become more coherent? | **Not measured.** Not claimed. |
| 13 | Instruction following improved? | **Not measured.** Frozen baseline 0/73. |
| 14 | Mathematical reasoning improved? | **Not measured.** Calculator arithmetic is a system capability, not model reasoning. |
| 15 | Programming improved? | **Not measured.** |
| 16 | Vietnamese improved? | **Not measured.** Vietnamese *retrieval* routing works (5/5 after a fix). |
| 17 | Which capabilities remain unproven? | All language-model capabilities at Phase 17 scale: instruction following, QA, math, code, summarization, extraction, Vietnamese generation, multi-turn, termination, coherence. |
| 18 | Did deterministic routing improve real task completion? | Yes at the system level, with measurements: calculator routing 258/258 with 14/14 negative rejections; retrieval recall@3 1.00 and evidence support 0.9726 over 100 docs; abstention 6/6. No model involvement. |
| 19 | Cost in GPU hours and storage? | GPU: ~4 minutes of local smoke/pilot work total (12 steps + 2 session-test steps + 12 resumed steps). Storage: two smoke checkpoints (~1.5 GB each, gitignored). No Kaggle GPU hours consumed. |
| 20 | What should Phase 18 prioritize? | Kaggle Stage A/B/E, then corpus v7 with the size gate, then the LR pilot, then 100 M tokens with multi-session resume and evaluation at 50 M / 100 M. |

## 2. Milestones

| milestone | status | evidence |
|---|---|---|
| `KAGGLE_ENVIRONMENT_VERIFIED` | **not achieved** | no Kaggle run; local preflight only |
| `PHASE14_CHECKPOINT_AVAILABLE` | achieved | size + SHA-256 in `bench_results/phase17/kaggle_preflight.json` |
| `CHECKPOINT_LINEAGE_VERIFIED` | achieved | manifest fields + `tests/test_phase17_init_from.py` |
| `CONTINUED_PRETRAINING_INIT_VERIFIED` | achieved | smoke run, first eval 4.7155 vs 10.82 |
| `KAGGLE_SESSION_RESUME_VERIFIED` | **not achieved** | local session resume only |
| `CORPUS_V7_SOURCE_AUDIT_VERIFIED` | **not achieved** | not ingested |
| `CORPUS_V7_LICENSE_VERIFIED` | **not achieved** | — |
| `CORPUS_V7_SPLITS_VERIFIED` | **not achieved** | — |
| `CORPUS_V7_TOKENIZED` | **not achieved** | — |
| `TOKEN_COMPOSITION_VERIFIED` | **not achieved** | — |
| `PRETRAINING_SMOKE_TEST_PASSED` | achieved (local GPU) | `docs/PHASE17_PRETRAINING_REPORT.md` §2 |
| `CONTINUATION_LR_PILOT_MEASURED` | **not achieved** | — |
| `PRETRAINING_100M_ADDITIONAL_TOKENS` | **not achieved** | 0.197 M processed |
| `PRETRAINING_300M_ADDITIONAL_TOKENS` | **not achieved** | — |
| `PRETRAINING_1B_ADDITIONAL_TOKENS` | **not achieved** | — |
| `LANGUAGE_MODELING_IMPROVED` | **not achieved** | not measured |
| `GENERATION_QUALITY_IMPROVED` | **not achieved** | not measured |
| `TASK_CAPABILITY_IMPROVED` | **not achieved** | not measured |
| `CALCULATOR_ROUTING_VERIFIED` | achieved | `bench_results/phase17/router_benchmark.json` |
| `RETRIEVAL_ROUTING_VERIFIED` | achieved on a synthetic corpus | `bench_results/phase17/system_benchmark.json` |
| `EVIDENCE_GROUNDING_VERIFIED` | partial | containment check 0.9726; entailment and human evaluation absent |
| `END_TO_END_TASKS_VERIFIED` | **not achieved** | no model-driven tool selection |
| `PHASE17_RELEASE_READY` | **not achieved** | 3 of 23 tokens achieved, 7 more blocked on Kaggle |

## 3. Deliverables

| file | state |
|---|---|
| `docs/PHASE17_STARTING_STATE.md` | written |
| `docs/PHASE17_CHECKPOINT_LINEAGE.md` | written |
| `docs/PHASE17_DATA_REPORT.md` | written (v7 `NOT MEASURED`) |
| `docs/PHASE17_KAGGLE_GUIDE.md` | written |
| `docs/PHASE17_PRETRAINING_REPORT.md` | written (smoke only) |
| `docs/PHASE17_CAPABILITY_REPORT.md` | written (baseline frozen) |
| `docs/PHASE17_ASSISTANT_ROUTING.md` | written |
| `docs/PHASE17_RELEASE_REPORT.md` | this file |
| `notebooks/phase17_kaggle_pretrain.ipynb` | 8 stages, 12 code cells, validated JSON |
| `scripts/kaggle_preflight.py` | runs; writes `bench_results/phase17/kaggle_preflight.json` |
| `scripts/phase17_intent_router.py` | 258/258, 14/14 negatives |
| `scripts/phase17_system_benchmark.py` | 100 docs, 73 + 6 queries |
| `tests/test_phase17_init_from.py` | 12 tests |
| `tests/test_phase17_session_resume.py` | 6 tests |
| `tests/test_phase17_intent_router.py` | 47 tests |
| `tests/test_phase17_system_benchmark.py` | 8 tests |
| `train_gpt2.py` | `--init-from`, `--resume-from`, `--session-max-minutes`, `--continue-pretrain`, lineage in manifests |
| `grounding.py` | Vietnamese question words no longer counted as content terms |

Not produced: the ≥300-task Phase 17 capability benchmark. Producing a task set
whose scores mean anything requires held-out prompts and hidden tests, and with no
evaluable new checkpoint there is nothing for it to measure yet; shipping 300
tasks without a checkpoint to score would be paperwork, not evidence. It remains
the first Phase 18 deliverable alongside the Kaggle run.

## 4. Base model vs system — kept separate

**Base model**

| metric | value | artifact | checkpoint |
|---|---|---|---|
| clean LM loss | 4.8935 (ppl 133.4) | `bench_results/phase16/eval_*.json` | `069d9da6…` |
| 73-task benchmark | 0/73 (base), 2/73 (LoRA) | `bench_results/phase16/task_benchmark*.json` | base + adapter |
| Phase 17 checkpoint | 196,608 additional tokens, no capability measurement | `checkpoints/phase17_smoke/*.manifest.json` | derived from `069d9da6…` |

**System**

| component | value | artifact |
|---|---|---|
| calculator routing | 258/258 + 14/14 negatives | `bench_results/phase17/router_benchmark.json` |
| retrieval recall@1/@3 | 0.8493 / 1.0000 | `bench_results/phase17/system_benchmark.json` |
| MRR / P@3 | 0.9178 / 0.3151 | same |
| evidence selection / support | 0.9726 / 0.9726 | same |
| abstention | 6/6 | same |
| Phase 16 recheck after the grounding fix | recall 1.00, calculator 14/14, citations 100 %, model 0 % | `bench_results/phase17/phase16_system_recheck.json` |

## 4b. Corrections after review (before any Kaggle run)

Six defects were found in the first committed version; all are fixed:

| defect | fix | verification |
|---|---|---|
| notebook used `--only`, but ingestion declares `--source` | notebook corrected | `--help` flags inspected, sample command exercised locally |
| `--resume --resume-from X` restored `<out-dir>/latest.pt` and, if absent, trained from scratch | `resolve_restore_path()`: `--resume-from` wins, and a requested resume with no checkpoint is fatal | `tests/test_phase17_session_resume.py::test_resume_from_wins_over_resume`, `::test_missing_resume_checkpoint_is_fatal` |
| `CONFIG["wikipedia_path"]` was unused; discovery only scanned `data/downloads/general_v1` | `scripts/ingest_corpus.py --input-root` discovers parquet under a mounted root | mounted-root listing exercised; licence-less source refused |
| Stage G still trained `data/gpt2_corpus_v5` | Stage G/E read `TRAIN_DATA`, which is `data/gpt2_corpus_v7` or unset — never an older shard | notebook asserts `TRAIN_DATA` before training |
| Stage E needed corpus/benchmark files absent from GitHub | Stage B2 copies the repo into `/kaggle/working/repo` and checks inputs; eval reports `NOT MEASURED` when the frozen file is absent | notebook cells assert paths |
| preflight marked a Windows machine `is_kaggle: true` | `host_kind()` requires `KAGGLE_KERNEL_RUN_TYPE` or both `/kaggle` mounts and is never true on Windows | artifact regenerated: `host_kind: local`, `is_kaggle: false` |

Also new: ingestion refuses to write a mounted source whose licence is not recorded
(`--license-map` from the upstream dataset card). `--all` against every mounted
folder would otherwise have produced an ineligible corpus that looks fine.

## 5. Candid conclusion

**Did JIN become more capable?** No — and Phase 17 does not pretend otherwise. It
trained 196,608 tokens for four minutes on a desktop GPU. No language-model
capability was measured, generation quality was not assessed, and the frozen 73-task
benchmark was deliberately not re-run on 12 steps because the result would be
noise dressed as progress.

**Did the assistant become more useful?** Yes, verifiably, but strictly as a
system: arithmetic requests are now routed and refused correctly (258/258, 14/14
negatives, 3/3 ambiguous), retrieval and evidence selection were measured on a
corpus 8× larger with hard distractors (recall@3 1.00, support 0.9726), and
unanswerable questions are refused 6/6. A real defect was found and fixed on the
way — Vietnamese questions were being rejected by the evidence selector. None of
this is language-model ability, and the base model still contributes nothing to a
calculator or retrieved answer.

**Did only the infrastructure improve?** That is the honest description of the
training track: strict weight transfer with lineage, a session budget that cannot
distort the schedule, resumable state verified across two sessions, a Kaggle
notebook with per-stage gates, and a preflight that refuses to start without the
parent checkpoint. Real but indirect — it makes the 100 M-token milestone
executable and auditable; it does not itself make the model smarter.

The next honest data point requires Kaggle: run the notebook, build corpus v7
behind the size gate, run the LR pilot, then 100 M additional tokens with the fixed
schedule, and evaluate at 50 M and 100 M with the frozen protocol before discussing
300 M.
