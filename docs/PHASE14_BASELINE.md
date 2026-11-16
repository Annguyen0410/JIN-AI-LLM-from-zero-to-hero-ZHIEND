# PHASE 14 — VERIFIED BASELINE

Recorded at the start of Phase 14. Every number below was read from the working
tree on this machine, not copied from earlier documentation.

## 1. Source revision

| Item | Value |
|---|---|
| Commit | `f353dc829029813d590a2630470f5261413c3445` |
| Branch | `main` |
| Remote | `zhiend` → `Annguyen0410/JIN-AI-LLM-from-zero-to-hero-ZHIEND` (public) |
| Working tree at start | clean |

Phase 14 additions are **not committed** (execution rule 12: no commit/push
without explicit authorization):

```
 M train_gpt2.py                     (exact resume + fp32 validation)
?? scripts/ingest_corpus.py          (new)
?? scripts/verify_corpus.py          (new)
?? scripts/data_audit_v2.py          (new)
?? scripts/build_tokenized_corpus.py (new)
?? tests/test_exact_resume.py        (new)
?? bench_results/phase14/            (new)
```

## 2. Hardware actually present

| Item | Measured |
|---|---|
| GPU | NVIDIA GeForce RTX 2060, 6144 MiB (6.4 GB), driver 617.42 |
| CUDA | 12.1 (`torch 2.5.1+cu121`), `torch.cuda.is_available() == True` |
| System RAM | 15.8 GB total; ~2.5 GB free during a training run |
| Disk | `D:` 932 GB, 558 GB free |
| OS | Windows |

Consequences that shaped this phase:

- A 1.49 GB checkpoint takes ~3–4 minutes to load from disk before any step.
- FP16 activations plus AdamW moments fit only at `batch 4 × seq 1024` with
  gradient accumulation; VRAM peaked at 5.9/6.0 GB in the pilot.
- Any run longer than ~2 hours needs a written budget and a resume path before
  it starts.

## 3. Checkpoints that exist (verified on disk)

`checkpoints/gpt2_small/pilot/` — architecture `gpt2_small` (GPT-2 Small
geometry, 124,439,808 parameters), tokenizer `gpt2_bpe` (vocab 50,257).

| File | Step | Tokens seen | Best val loss | SHA-256 (prefix) | Bytes |
|---|---|---|---|---|---|
| `best.pt` | 160 | 11,796,480 | **3.2337** | `bc44d10da2183857` | 1,493,464,398 |
| `latest.pt` | 230 | 16,957,440 | 3.2337 | `51e15fa3aeb71591` | 1,493,464,718 |

The pilot trained on `data/gpt2_corpus` (11,990,607 train tokens / 475,032 val
tokens, sha256 `5aaf961a…f55685`) — that is ≈1.41 epochs, so part of the pilot's
final training loss comes from re-reading the same text.

## 4. Datasets that exist (verified on disk, not in git)

| Path | Size | Content |
|---|---|---|
| `data/downloads/general_v1/human_documented/wikipedia-en-20231101` | 11 GB | 41 Parquet shards, `wikimedia/wikipedia` @ `b04c8d1c`, CC-BY-SA-3.0 + GFDL, per-shard SHA-256 and revision recorded in `download_report.json` |
| `data/downloads/general_v1/synthetic_documented/cosmopedia-v2-sample` | 5.5 GB | 5 Parquet shards, Cosmopedia v2 sample (synthetic, model-generated educational text) |
| `data/raw` | 5.49 GB | 80 text files (Gutenberg-style books) |
| `data/prepared_v4/pretrain` | 38.8 MB | 11 document-disjoint text files, exact count 11,990,607 tokens |
| `data/curated/snapshots` | small | earlier curated snapshots |

`pyarrow` is installed; `pandas` and `datasets` are not (the readers are written
to need only `pyarrow`).

## 5. Bugs confirmed before any change

1. **Unit-mixed token estimate** — `scripts/data_audit.py` computed
   `tokens_per_character × total_bytes`. Measured effect:
   `prepared_v4_train` estimated **9,413,546** tokens from 38,785,853 bytes,
   while full tokenization of the same corpus produces **11,990,607** — the v1
   estimate is **21.5 % low**. For `data/raw` (5.49 GB) v1 reported
   **2.248 B** tokens where the corrected v2 estimator gives **1.468 B** —
   **+53 % too high**. Neither direction is acceptable for planning a token
   budget.
2. **FP16 cross-entropy in evaluation.** The model's own loss runs inside
   `torch.amp.autocast`. On identical logits this repo already measured
   `11.6719` (fp16) vs `0.8336` (fp32) — a validation number produced that way
   is not a measurement of anything.
3. **Resume restored sampler-less state.** `save_state` stored Torch CPU RNG,
   CUDA RNG, model, optimizer, scaler — but batch positions come from a NumPy
   `Generator` inside `ShardBatcher`. A resumed run therefore drew **different
   batches** from the un-interrupted run, so "resume" was a roughly-continued
   run, not the same run.
4. **No resume guards.** Nothing stopped a resume against a changed corpus, a
   changed tokenizer, a changed architecture, or a changed step budget — each of
   which silently produces an incomparable run.
5. **Dropout is hard-coded to 0** in the trainer even though the GPT-2 recipe
   uses 0.1. Not a bug by itself; recorded as an untested default.

## 6. What this baseline implies

The pilot reached val loss 3.2337 on a 12 M-token corpus it had already seen
1.41 times. Any Phase 14 claim of improvement must therefore be measured on text
the new model has **never** seen and — to be honest about the pilot — on the
same held-out file for both models.

That is the evaluation design used in `docs/PHASE14_EVALUATION.md`.
