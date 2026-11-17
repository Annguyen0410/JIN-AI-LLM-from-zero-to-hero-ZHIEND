# Phase 17 — Pretraining Report

**Headline: no substantial continued pretraining happened in this phase.** What
was built and verified is the machinery that makes it possible, plus one 12-step
smoke run proving the weights transfer. Nothing here is a capability claim.

## 1. What was implemented

`train_gpt2.py`:

| flag | behaviour |
|---|---|
| `--init-from PATH` | transfer model weights from a parent checkpoint into a **new** experiment (fresh optimizer, schedule, sampler, step counters) |
| `--continue-pretrain` | marks the run; refuses to start without `--init-from` |
| `--resume-from PATH` | resume from an arbitrary checkpoint path (e.g. a read-only Kaggle input) instead of `<out-dir>/latest.pt` |
| `--session-max-minutes N` | stop at a step boundary after N wall-clock minutes, save full state, exit 0; schedule and fingerprint unchanged |

Lineage fields are written into every checkpoint manifest:
`init_mode`, `parent_checkpoint`, `parent_sha256`, `parent_iteration`,
`parent_tokens_seen`, `additional_tokens_this_run`, `experiment_config_hash`,
`schedule`.

Details and verification procedure: `docs/PHASE17_CHECKPOINT_LINEAGE.md`.

## 2. Smoke test (Stage 0) — measured

Command:

```bash
python train_gpt2.py --preset gpt2_small_reference --data-dir data/gpt2_corpus_v5 \
  --init-from checkpoints/gpt2_small/phase14_wiki/best.pt --continue-pretrain \
  --out-dir checkpoints/phase17_smoke --seq-len 1024 --batch-size 4 --grad-accum 4 \
  --max-steps 12 --warmup 2 --lr 1e-4 --eval-interval 6 --eval-iters 2 \
  --save-interval 6 --seed 17
```

| quantity | value |
|---|---|
| hardware | RTX 2060 6 GB, CUDA 12.1, torch 2.5.1+cu121 (**not** Kaggle) |
| context length | 1024 (unchanged; not reduced to 512) |
| effective batch tokens | 4 × 4 × 1024 = 16,384 per step |
| steps | 12 |
| tokens processed by training | **196,608** |
| throughput | 2,659–3,025 tokens/s (includes eval overhead; local figure, not a Kaggle estimate) |
| peak VRAM | 4.4 GB |
| val loss @ step 6 | 4.7155 (ppl 111.66) |
| val loss @ step 12 | 4.7876 (ppl 120.01) |
| gradient norm | 0.89–0.91, finite throughout |
| parent | `069d9da650a81dc9e2c0fee94d0e20dc46a5c4f7e266a6c381a72140f48266fa`, iter 640, 47,185,920 tokens |
| checkpoint | `checkpoints/phase17_smoke/{best,latest}.pt` + manifests |
| fingerprint | `98b233d4aa6e4a501eabafb8f534eaa4aaca07239d7084c16fbe4158fb1f14b2` |

The first evaluation at **4.7155** is the load-bearing number: random
initialization gives ≈ ln(50,257) = 10.82. The weights came from Phase 14.

The small rise across the 12 steps is not interpreted. Six steps at lr 1e-4 on
`gpt2_corpus_v5` is far too little to say anything about learning, and the 4.8935
Phase 16 figure comes from a different evaluator run and sample, so the two are
not comparable.

## 3. Session stop / resume — measured

Run A (`checkpoints/phase17_session_test`), `--session-max-minutes 0.05`:

```json
{"reason": "session_max_minutes", "session_minutes": 0.06, "global_step": 2,
 "total_steps_unchanged": 12, "tokens_seen_this_run": 32768,
 "checkpoint_sha256": "516cc0401e51499654e68b9323aab3a0da1e6c03a6db7b3b56d24ca3160a8e4b",
 "parent_sha256": "069d9da650a81dc9e2c0fee94d0e20dc46a5c4f7e266a6c381a72140f48266fa"}
```

Run B: `--resume --resume-from checkpoints/phase17_session_test/latest.pt` with the
same `--max-steps 12` completed the remaining 10 steps and saved `best.pt`
(val 4.5045). The global budget stayed 12; the fingerprint was unchanged; the
parent SHA survived the resume.

This is local evidence. `KAGGLE_SESSION_RESUME_VERIFIED` is **not** claimed: the
same cycle has not been executed inside a Kaggle session across two notebooks.

## 4. Token accounting

| quantity | value |
|---|---|
| tokens in the corpus used for the smoke run | ~51 M (`data/gpt2_corpus_v5`) |
| tokens **actually processed** by Phase 17 training | **196,608** |
| tokens in corpus v7 | **NOT MEASURED** (not built) |
| additional tokens vs Phase 14 | 196,608 |
| global budget of the smoke run | 12 steps, closed |

Available corpus size is never reported as trained tokens.

## 5. Milestones

| milestone | status |
|---|---|
| `PRETRAINING_SMOKE_TEST_PASSED` | achieved — 12 steps, local GPU, finite loss/grad, resumable checkpoint |
| `CONTINUATION_LR_PILOT_MEASURED` | **not achieved** — pilot not run |
| `PRETRAINING_100M_ADDITIONAL_TOKENS` | **not achieved** — 0.197 M processed |
| `PRETRAINING_300M_ADDITIONAL_TOKENS` | **not achieved** |
| `PRETRAINING_1B_ADDITIONAL_TOKENS` | **not achieved** |
| `KAGGLE_ENVIRONMENT_VERIFIED` | **not achieved** — no Kaggle run in this environment |

## 6. What Phase 18 must do first

1. Run Stage A/B/E on Kaggle and record the actual accelerator, VRAM, throughput
   and runtime.
2. Build corpus v7 with the projected-size gate (`docs/PHASE17_DATA_REPORT.md`).
3. Run the LR pilot on the real corpus, then the 100 M-token milestone with the
   fixed global schedule and multi-session resume.
4. Evaluate at 50 M and 100 M with the frozen Phase 16 protocol and the frozen
   73-task benchmark before considering 300 M.
