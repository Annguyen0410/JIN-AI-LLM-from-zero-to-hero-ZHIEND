# PHASE 14 — TRAINING REPORT

## 1. What was broken in the trainer

| Bug | Effect | Fix |
|---|---|---|
| Sampler RNG not saved | `save_state` stored Torch CPU/CUDA RNG but batch positions come from a NumPy `Generator` in `ShardBatcher`. A resumed run drew **different batches**: it continued *roughly*, not *exactly*. | `batcher.state_dict()/load_state_dict()` persists the bit-generator state; `numpy_rng_state` and Python `random` state are stored as plain ints (a raw NumPy tuple makes `torch.load(weights_only=True)` refuse the file — that failure was hit and fixed). |
| No resume guards | A resume against a changed corpus, tokenizer, architecture or step budget silently produced an incomparable run. | `verify_resume_compatibility()` refuses on any train/val corpus hash change, architecture change, tokenizer change, or budget change (unless `--extend-budget`). |
| Validation loss computed inside autocast | FP16 softmax on 50 k logits. The repo had already measured 11.6719 (fp16) vs 0.8336 (fp32) on identical logits. | `evaluate()` casts logits to FP32 and computes cross-entropy in FP32; every eval record carries `loss_dtype: fp32_cross_entropy` and `tokens_scored`. |
| Dropout hard-coded to 0 | An untested assumption presented as a default. | `--dropout` flag + a controlled A/B (§3). |

`tests/test_exact_resume.py` proves the first fix rather than asserting it: path A
trains 6 steps continuously; path B trains 3, saves, rebuilds model/optimizer/
batcher from scratch (with a deliberately different RNG state), restores, and
trains 3 more. Loss sequences match to 1e-9, final weights to 1e-6, and the
AdamW moments match too. Three tests pass (plus the existing
`test_training_resume.py`, whose checkpoint round-trip test caught the missing
`schedule` key — `save_state` now uses `meta.get`).

## 2. Budget arithmetic before starting anything long

| Quantity | Value |
|---|---|
| Parameters | 124.44 M (GPT-2 Small geometry, `gpt2_small_reference`) |
| Tokens per step | 4 × 18 × 1024 = **73,728** |
| Planned steps | 680 |
| Planned tokens | **50.1 M** |
| Corpus available | 51.2 M train tokens → **0.98 epochs**, i.e. no deliberate repetition |
| Measured throughput | 8.1–10.0 k tok/s |
| Projected wall clock | 50.1 M / 8.2 k ≈ **1.7 h** |
| Checkpoint size / cadence | 1.49 GB every 100 steps (≈24 min) |
| Disk headroom | 558 GB free |
| Recovery plan | `--resume` restores model + optimizer + scaler + all RNG + sampler position; a crash costs at most 100 steps |

Peak VRAM 4.4 GB of 6.1 GB with activation checkpointing enabled.

## 3. Dropout A/B (Phase 7.1), controlled

Same architecture, same corpus (`data/gpt2_corpus_ab`, 10.0 M tokens), same
optimiser, same LR schedule, same 200 steps (3.28 M tokens), seq 512,
batch 4 × accum 8. Only `--dropout` differs.

| Run | Dropout | Best val loss | Best step | Final train loss |
|---|---|---|---|---|
| `checkpoints/ab_dropout_00` | 0.0 | **6.1811** | 150 | 6.21 |
| `checkpoints/ab_dropout_01` | 0.1 | 6.2840 | 150 | 6.25 |

At this budget dropout 0.1 is **worse by 0.10 nats**. That is consistent with
regularisation being useless when the model sees only 3 M tokens, and it does
*not* prove dropout is wrong for a 40 B-token run. The conclusion drawn here is
narrow and honest: at the scale this machine can reach in a session, dropout 0
is the better default, so the main run uses 0 and stays comparable to the pilot.
Re-testing at ≥1 B tokens is the correct follow-up.

The first attempt at the dropout-0.1 arm died with `CUDA error: out of memory`
at step ~45 on a 6 GB card (fragmentation, not model size: the same config had
been running at 3.2 GB). It was re-run cleanly with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; both arms reported here are
complete runs.

## 4. Main run

`checkpoints/gpt2_small/phase14_wiki/`

```bash
.venv/Scripts/python.exe train_gpt2.py \
  --preset gpt2_small_reference --data-dir data/gpt2_corpus_v5 \
  --out-dir checkpoints/gpt2_small/phase14_wiki \
  --batch-size 4 --grad-accum 18 --seq-len 1024 \
  --max-steps 680 --warmup 100 --lr 6e-4 \
  --eval-interval 40 --eval-iters 8 --save-interval 100 --dropout 0.0
```

Results are in §4 of `docs/PHASE14_RELEASE_REPORT.md` (single source of truth for
numbers) and in `bench_results/phase14/`.

## 5. Reproducibility contract for this run

A reviewer can rebuild the exact run from:

- code revision: `git rev-parse HEAD` (recorded in every checkpoint as
  `training_revision`)
- corpus hash: `data/gpt2_corpus_v5/manifest.json` (`train.sha256`,
  `val.sha256`) — a resume against a different hash is refused
- tokenizer fingerprint: `gpt2_bpe` state hash `40bbc29de16d5ed2…`
- architecture: `arch_id: gpt2_small`, config exported into the checkpoint
- recipe: `training_args` inside the checkpoint (every CLI flag)
- hardware: GPU name, VRAM, torch version inside the checkpoint

## 6. What this training does *not* demonstrate

- It is **50 M tokens**, 0.125 % of GPT-2's 40 B. Held-out loss will improve a
  lot over the 16 M-token pilot; it will not approach GPT-2 Small, and no
  document here claims it does.
- Vietnamese is 3 % of the mixture, so Vietnamese generation is expected to stay
  weak. Any Vietnamese sample shown is representative of 1.5 M training tokens,
  not of a multilingual corpus.
- One seed per configuration. The A/B is a 3 M-token signal; the main run has no
  seed-replicate, so differences below ~0.02 nats should not be read as real.
