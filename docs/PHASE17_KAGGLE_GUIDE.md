# Phase 17 — Kaggle Guide

Everything below is copy-pasteable. Nothing assumes a GPU model, VRAM size,
runtime quota or dataset slug: each stage detects and prints before acting.

## 0. What to attach

| input | why | notes |
|---|---|---|
| this repository | code | clone or attach as a Dataset |
| `best.pt` (Phase 14) | **required parent weights** | not in the repo; 1.49 GB, sha256 `069d9da6…8266fa` |
| Wikipedia parquet | corpus v7 | only needed for Stage D |
| previous Phase 17 `latest.pt` | session 2+ | optional |
| previous corpus v7 output | skip re-ingestion | optional |

`/kaggle/input` is read-only. All outputs go to `/kaggle/working`. Never put
anything you need to keep under `/kaggle/temp`.

## 1. Configuration cell

`notebooks/phase17_kaggle_pretrain.ipynb`, Stage B — edit only this cell:

```python
CONFIG = {
    "repo_path": "/kaggle/input/jin-ai-repo/JIN-AI-LLM-from-zero-to-hero-ZHIEND",
    "checkpoint_path": "/kaggle/input/jin-phase14-checkpoint/best.pt",
    "wikipedia_path": "/kaggle/input/wikipedia-en-parquet",
    "corpus_v7_path": None,
    "phase17_ckpt_path": None,
    "out_dir": "/kaggle/working/phase17_output",
}
```

## 2. Preflight (Stage A)

```bash
nvidia-smi
python scripts/kaggle_preflight.py \
    --checkpoint /kaggle/input/jin-phase14-checkpoint/best.pt \
    --out /kaggle/working/phase17_output/kaggle_preflight.json
```

It records GPU name/count/VRAM, CUDA, torch, Python, CPU, RAM, disk,
`/kaggle/input` presence, repo HEAD, checkpoint size + SHA-256, and corpus
manifests. It **exits non-zero if the checkpoint is missing** — that is the
point: the run must fail rather than silently train from random weights.

`host.host_kind` is `kaggle` only inside a real kernel (`KAGGLE_KERNEL_RUN_TYPE`
or both `/kaggle/input` and `/kaggle/working`). A local run records
`host_kind: local`, so a desktop preflight artifact can never be mistaken for
Kaggle evidence — Stage A asserts this before training.

The committed `bench_results/phase17/kaggle_preflight.json` was generated on the
development machine and says `host_kind: local`. Replace it with the Kaggle run's
output; the two are not interchangeable.

Record the GPU you actually got in the run log. Kaggle may provide P100, T4,
multiple T4s, or something else. The trainer is single-GPU: with two visible GPUs
it still uses device 0, and throughput is not expected to double.

Do not replace the installed PyTorch unless it is incompatible with the repo.

## 3. Corpus (Stages C/D)

The mounted Dataset lives outside the repo, so ingestion needs `--input-root`. The
licence must come from the dataset card via `--license-map`; a folder name is not
permission and ingestion refuses a source with no recorded licence.

```bash
python scripts/ingest_corpus.py --input-root /kaggle/input/wikipedia-en-parquet --list

# {"wikipedia-en-20231101": {"license": "<from the card>",
#                            "source_url": "<dataset card>",
#                            "text_columns": ["text"], "language": "en"}}
python scripts/ingest_corpus.py --input-root /kaggle/input/wikipedia-en-parquet \
    --source <the-reviewed-source-id> --license-map license_map.json \
    --max-docs-per-source 10000 --out-root data/curated_v7_sample
du -sh data/curated_v7_sample
```

Project bytes-per-document and tokens-per-document from that sample. Only then:

```bash
python scripts/ingest_corpus.py --input-root /kaggle/input/wikipedia-en-parquet \
    --source <the-reviewed-source-id> --license-map license_map.json \
    --max-docs-per-source 1000000 --out-root data/curated_v7
python scripts/verify_corpus.py --in-root data/curated_v7 --out-root data/curated_v7 \
    --pii-policy exclude --license-policy enforce
python scripts/build_tokenized_corpus.py --splits data/curated_v7/splits \
    --out data/gpt2_corpus_v7 --max-train-tokens 1000000000 \
    --mix <source>:0.65,<source>:0.25,<source>:0.08,<source>:0.02
```

Read `data/curated_v7/reports/` before training on it. Exit code 0 is not
acceptance.

## 4. Smoke test (Stage E) — 20 steps

```bash
python train_gpt2.py --preset gpt2_small_reference --data-dir data/gpt2_corpus_v7 \
    --init-from $CKPT --continue-pretrain --out-dir $OUT/smoke \
    --seq-len 1024 --batch-size 4 --grad-accum 4 --max-steps 20 \
    --warmup 2 --lr 1e-4 --eval-interval 10 --eval-iters 4 --save-interval 10 --seed 17
```

Accept only if **all** hold:

* first `[eval]` loss is near the parent (~4.7), not ~10.8 (random init);
* loss and gradient norm are finite;
* `latest.pt.manifest.json` has `init_mode: continued_pretrain` and a
  `parent_sha256`;
* VRAM and tokens/s are recorded (they are printed every step);
* resuming that `latest.pt` continues from the same step.

## 5. Learning-rate pilot (Stage F)

Same parent, same corpus, same seed, same geometry; candidates `3e-5`, `1e-4`,
`3e-4`; a few hundred steps each. Choose on stability, clean validation loss and
generation quality — never on the lowest training loss.

## 6. Continued pretraining (Stage G)

Fixed global budget, chosen before the first session:

```
tokens_per_step = batch * grad_accum * seq_len = 4 * 8 * 1024 = 32,768
100M additional tokens -> 3,052 steps   (use the exact computed value)
```

First session:

```bash
python train_gpt2.py --preset gpt2_small_reference --data-dir data/gpt2_corpus_v7 \
    --init-from $CKPT --continue-pretrain --out-dir $OUT/run_100m \
    --seq-len 1024 --batch-size 4 --grad-accum 8 --max-steps 3052 \
    --warmup 100 --lr <chosen> --eval-interval 250 --eval-iters 8 \
    --save-interval 250 --session-max-minutes 500 --seed 17
```

`--session-max-minutes` stops at an optimizer-step boundary, saves
`latest.pt` + optimizer + scaler + RNG + sampler, writes `session_log.json`, and
exits 0. It does **not** alter `total_steps`, warmup, the cosine schedule or the
experiment fingerprint. Do not use `--extend-budget` to move between sessions.

## 7. Next session — resume (Stage G2)

```bash
python train_gpt2.py --preset gpt2_small_reference --data-dir data/gpt2_corpus_v7 \
    --resume-from /kaggle/input/jin-phase17-s1/latest.pt --out-dir $OUT/run_100m \
    --seq-len 1024 --batch-size 4 --grad-accum 8 --max-steps 3052 \
    --warmup 100 --lr <chosen> --eval-interval 250 --eval-iters 8 \
    --save-interval 250 --session-max-minutes 500 --seed 17
```

`--resume-from` alone is enough: it restores that path directly, wherever it is
mounted. `--resume` is only for the default `<out-dir>/latest.pt`, and combining
the two is harmless but unnecessary — `--resume-from` wins. If a resume is
requested and the file does not exist, the trainer **stops** with
"resume requested but no checkpoint at …" instead of starting a fresh run.
`--max-steps` must equal the first session's value; the fingerprint check refuses
anything else.

## 8. Output contract

```
/kaggle/working/phase17_output/
  latest.pt, latest.pt.manifest.json
  best.pt, best.pt.manifest.json
  session_log.json
  kaggle_preflight.json
  benchmark_results/
```

At the end of every session: save, flush logs, compute SHA-256, compare with the
manifest, export results, print the exact paths the next session needs, then
**Save Version** so the output becomes a Dataset you can attach next time. Never
delete the only verified checkpoint.

## 9. Evaluation (Stage H)

```bash
python scripts/eval_phase16.py --checkpoint $OUT/run_100m/best.pt \
    --val-jsonl data/curated_v6/benchmark/clean_unseen.jsonl --seq-len 1024 \
    --out $OUT/benchmark_results/eval_clean.json
python scripts/phase16_task_benchmark.py --checkpoint $OUT/run_100m/best.pt \
    --out $OUT/benchmark_results/task_benchmark.json
```

Same protocol, same benchmark file, same scoring rules as Phase 16 — otherwise
the numbers are not comparable. A lower validation loss is not a capability
claim; report task counts separately.

## 10. Recovery

| symptom | action |
|---|---|
| session died mid-run | previous `latest.pt` is intact (atomic write); rerun Stage G2 with the same `--max-steps` |
| "Training configuration changed" | something in the recipe differs; compare `experiment_config` in both manifests — do not force it |
| "Step budget changed" | you changed `--max-steps`; restore it or start a new documented experiment |
| smoke loss ≈ 10.8 | weights were not transferred; check `--init-from` and the parent SHA |
| CUDA OOM | lower `--batch-size`, raise `--grad-accum` to keep effective batch tokens, and record the change |

## 11. Security note for benchmarks

Do not execute model-generated code on a Kaggle host under the assumption that a
temp directory plus a timeout is a sandbox. Kaggle has no verified OS-level
network/filesystem isolation for that, and `/kaggle/input` may contain private
data plus credentials. Phase 17's own benchmarks execute **no** generated code;
if code execution is reintroduced, it needs a restricted harness with an explicit
threat model, and the limitation must be documented rather than glossed over.
