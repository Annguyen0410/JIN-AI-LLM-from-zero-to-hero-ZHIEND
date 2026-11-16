# PHASE 14 — RELEASE REPORT

**Final status: PRETRAINING IMPROVEMENT DEMONSTRATED.**

Not "GPT-2-class capability", not "comparability". The evidence supports exactly
one claim: a second pretraining run, on a better-verified corpus with a repaired
trainer, learns measurably better language modelling than the pilot — and it is
still 1.52 nats behind GPT-2 Small.

---

## 1. Current baseline (verified, not assumed)

| Item | Value |
|---|---|
| Revision at phase start | `f353dc829029813d590a2630470f5261413c3445` |
| Pilot checkpoint | `checkpoints/gpt2_small/pilot/best.pt`, step 160, 11,796,480 tokens, sha256 `bc44d10da2183857…` |
| Pilot latest | step 230, 16,957,440 tokens, sha256 `51e15fa3aeb71591…` |
| Pilot validation | 3.2337 on its **own** 12 M-token corpus val (1.41 epochs seen) |
| Architecture | `gpt2_small`, 124.4 M parameters, tokenizer `gpt2_bpe` (50,257) |
| Hardware | RTX 2060 6.1 GB, 15.8 GB RAM, Windows, torch 2.5.1+cu121 |

## 2. Critical bugs found

1. Token estimates multiplied **tokens/character** by **total bytes**.
2. Validation loss was computed inside autocast (FP16 softmax over 50 k logits).
3. `save_state` did not persist the NumPy sampler RNG, so a "resumed" run drew
   different batches.
4. No resume guard: changed corpus / tokenizer / architecture / step budget were
   accepted silently.
5. Dropout was hard-coded to 0 with no experiment behind it.

## 3. Bugs fixed

| Fix | Evidence |
|---|---|
| `scripts/data_audit_v2.py` — matched denominators, stratified sampling, exact full scan under 250 MB, `estimate_reliable` flag | `bench_results/phase14/data_audit_v2.json` |
| `evaluate()` — `logits.float()` + FP32 cross-entropy, reports `tokens_scored` and `loss_dtype` | `tests/test_exact_resume.py::test_validation_is_deterministic_and_fp32` |
| Batcher RNG + NumPy + Python `random` saved and restored (encoded as plain ints, because a raw NumPy tuple makes `weights_only=True` refuse the checkpoint) | `test_exact_resume_matches_continuous_training` — 6 continuous steps vs 3+resume+3: identical loss sequence, weights within 1e-6, AdamW moments identical |
| `verify_resume_compatibility()` refuses corpus/arch/tokenizer/budget changes; `--extend-budget` makes budget changes explicit | `test_resume_refuses_changed_corpus` |
| `--dropout` flag + controlled A/B | §10 |

Test suite: **51 tests pass** (15 gpt2_reference, 20 tokenizer_upgrade, 3
training_resume, 3 exact_resume, plus tokenizer/chat_template/others).
`tests/test_check_project.py` errors on a pre-existing pytest fixture collision
(`scripts/check_project.py::test_group`); it is unrelated to this phase and was
not modified.

## 4. Data audit corrections

| Corpus | Old (v1) | Corrected |
|---|---|---|
| `prepared_v4/pretrain` | 9,413,546 | **11,990,607 exact** (−21.5 % error in v1) |
| `data/raw` | 2,247,963,961 | **1,468.0 M** (+53 % error in v1) |

Both old artifacts are preserved untouched; the correction is filed next to them.

## 5. Exact unique training tokens available

| Pool | Tokens |
|---|---|
| Built for this run (`data/gpt2_corpus_v5/train.bin`) | **51,222,642** (exact count) |
| Val / test (exact) | 28,969,659 / 7,403,328 |
| Cleaned, licensed train split not yet tokenized | ≈ 400 M (estimate: 1.488 B chars × 0.269 tok/char) |
| `data/raw` books — **license not recorded, excluded** | ≈ 1,468 M |

### Reconciling the old "40 GB / 40 B tokens" references

Earlier documents mix two different quantities. **40 GB** was a *disk* budget for
a planned download; **40 B tokens** was GPT-2's own training budget used as a
comparison target. Neither was achieved and neither is a property of JIN:
the imported corpus is 1.62 B characters on disk (≈3.2 GB of Parquet text),
51.2 M tokens were tokenised, and 50.1 M tokens were actually trained on.

## 6. License and provenance status

| Source | License | Status |
|---|---|---|
| Wikipedia EN 20231101 (41 Parquet, SHA-256 per shard) | CC-BY-SA-3.0 + GFDL | documented, commercial yes with share-alike/attribution |
| Cosmopedia v2 sample (5 Parquet) | ODC-BY-1.0 | documented, **synthetic** |
| `prepared_v4` | per-source policy, NC excluded | documented |
| `project_seed` | project-authored | documented |
| `data/raw` (5.49 GB books) | **unknown per file** | excluded from the mixture |

Unknown license is never read as permission. Details:
`docs/DATA_INGESTION_V2.md`, `data/curated_v5/raw_manifest.json`.

## 7. Validation quality

- 549,214 documents ingested → 62,243 exact duplicates removed (11.3 %) → 462,198
  clusters kept.
- Splits are **cluster-level**, so a near-duplicate family cannot straddle
  partitions. Verified after writing: train↔val **0**, train↔test **0**,
  val↔test **0** exact overlaps.
- Val contains 28.97 M tokens across knowledge / educational / mixed /
  conversation and en / vi groups, so per-domain and per-language losses are
  measurable, not decorative.

A quality-heuristic bug of my own was caught and fixed mid-phase: unique-word
ratio dropped **222,681** documents (46 % of the corpus, mostly long Wikipedia
articles) because that metric is length-biased; 5-gram duplication drops **24**.
The wrong run was discarded, not reported as a result.

## 8. Resume reproducibility results

`tests/test_exact_resume.py` — three tests, all passing:

- path A (6 continuous steps) vs path B (3 steps → save → rebuild → restore → 3
  steps) produce the **same loss sequence**, **the same weights** (atol 1e-6) and
  **the same AdamW moments**;
- the batcher position after restore equals the uninterrupted run's;
- a changed corpus, architecture, tokenizer or budget raises instead of silently
  producing an incomparable run.

This is verified on CPU with a deterministic small model. A CUDA-side variant is
not part of the automated suite; the main run exercised the CUDA path
(checkpoints every 100 steps with full RNG/sampler state) but was not
interrupted-and-resumed mid-run in this phase.

## 9. Tokenizer correctness

The old `bench_results/gpt2_upgrade/tokenizer_eval.json` reported
`roundtrip_exact: false` for the JIN tokenizer on its long Vietnamese and code
samples. That artifact predates `BPE_IMPL_VERSION = 2`. Re-running the **same
script** now:

| Sample | JIN round-trip | GPT-2 round-trip |
|---|---|---|
| Vietnamese (400 k chars) | **exact** | exact |
| English (400 k) | exact | exact |
| Stories (300 k) | exact | exact |
| Code (300 k) | **exact** | exact |
| Math (300 k) | exact | exact |

Independently measured: 400 k characters of held-out Vietnamese text and 300 k
characters of repository source round-trip exactly; no codepoint in
U+0020–U+2FFF or the emoji block fails a single-character round trip.

Efficiency (tokens per Vietnamese word): **JIN 1.168 vs GPT-2 5.146** (4.4×
fewer tokens); English: JIN 1.488 vs GPT-2 1.362 (GPT-2 slightly better). The
JIN vocabulary is not changed in this phase, and GPT-2 weights are never copied
into JIN.

Artifacts: `bench_results/phase14/tokenizer_eval_v2.json`.

## 10. Training experiments completed

**Dropout A/B** (same architecture, corpus, optimiser, schedule; 200 steps,
3.28 M tokens, seq 512):

| Arm | Best val | Final train |
|---|---|---|
| dropout 0.0 | **6.1811** | 6.21 |
| dropout 0.1 | 6.2840 | 6.25 |

At this budget dropout 0.1 is 0.10 nats worse. Conclusion limited to this scale.

**Main run** — `checkpoints/gpt2_small/phase14_wiki/`:

| Item | Value |
|---|---|
| Steps | 680 (best checkpoint at 640) |
| Tokens | 50,135,040 (0.98 epoch of 51.2 M unique) |
| Tokens/step | 73,728 (4 × 18 × 1024) |
| LR | 6e-4, 100-step warmup, cosine to 6e-5 |
| Dropout | 0.0 |
| Throughput | 7,964–9,991 tok/s |
| Wall clock | 6,284 s (1 h 45 m) |
| Peak VRAM | 4.4 GB / 6.1 GB, GPU 83 °C, 65 W |
| Best val (training sampler) | **4.7399** (ppl 114.42) at step 640 |
| Final val at 680 | 4.9933 (ppl 147.42) |

The final 40 steps lose 0.25 nats while train loss keeps falling — mild
overfitting at the end of the cosine schedule on 0.98 epochs, and the reason
`best.pt` is the release checkpoint, not `latest.pt`.

## 11. Checkpoint hashes

| Checkpoint | Step | Tokens | Best val | SHA-256 (prefix) |
|---|---|---|---|---|
| `phase14_wiki/best.pt` | 640 | 47,185,920 | 4.7399 | `069d9da650a81dc9e2c0` |
| `phase14_wiki/latest.pt` | 680 | 50,135,040 | 4.7399 | `6790ff8fc47406b12286` |
| `pilot/best.pt` (unchanged) | 160 | 11,796,480 | 3.2337 (own val) | `bc44d10da2183857` |
| `ab_dropout_00/best.pt` | 150 | 2,457,600 | 6.1811 | `882005bc3f1ccec4ddfd` |
| `ab_dropout_01/best.pt` | 150 | 2,457,600 | 6.2840 | `8790b9a620cb0f95bf62` |

## 12. Before / after model metrics — same windows, same tokenizer

Identical 64 windows × 1,024 tokens from `data/gpt2_corpus_v5/val.bin`, FP32
cross-entropy, same `gpt2_bpe` token ids. 65,536 tokens scored.

| Model | Loss | Perplexity |
|---|---|---|
| JIN pilot (step 160) | 6.7888 | 887.85 |
| **JIN Phase 14 (step 640)** | **4.9492** | **141.06** |
| GPT-2 Small (reference) | 3.2467 | 25.70 |

**Improvement: −1.84 nats, 6.3× lower perplexity than the pilot** on text
neither model trained on.

Per-domain / per-language (val split, 24 docs per group, one window each):

| Group | Pilot | Phase 14 |
|---|---|---|
| knowledge | 6.7567 | **5.1037** |
| educational | 6.8329 | **4.7789** |
| mixed (`prepared_v4`) | **2.1091** | 4.2999 |
| conversation | **3.4832** | 4.6718 |
| en | 6.8329 | **4.7789** |
| vi | **1.7115** | 4.0784 |

The pilot is better on the `prepared_v4`/Vietnamese slice. That is not a
Vietnamese-capability win: the pilot trained **1.41 epochs** on that exact
corpus (old split had 23.2 % duplicate documents), while Phase 14 saw it once.
It is reported here rather than hidden, and it is why the release gate for
"no regression in any domain" is marked **partially met** (§16).

## 13. GPT-2 reference comparison

GPT-2 Small scores 3.2467 (ppl 25.70) on the same windows and the same tokenizer.
Both sides now use identical token ids, so the ratio is legitimate to state:

- perplexity ratio JIN ÷ GPT-2 = **5.49×** (pilot was 34.5×);
- parameter count is matched (124.4 M vs 124.4 M);
- **training tokens differ by 800×** (50 M vs ~40 B).

This is a scale statement, not a capability parity claim. GPT-2 weights are not
loaded into JIN and no distillation from GPT-2 was performed.

## 14. Hardware and runtime measurements

| Metric | Measured |
|---|---|
| Throughput (seq 1024, batch 4, accum 18, grad-checkpoint) | 7,964–9,991 tok/s |
| Peak VRAM | 4.4 GB |
| GPU temperature / draw | 83 °C / 65 W (thermally limited by clocks, not power) |
| Checkpoint write | 1.49 GB every 100 steps |
| Eval cost per 40 steps | 8 × 2,048 tokens, FP32, ~40 s |
| Corpus build (60 M tokens) | ~20 min (two passes + tokenisation) |
| Dedup + split of 549 k documents | 374 s |

## 15. Remaining blockers

1. **Vietnamese data.** 3 % of training tokens. A licensed Vietnamese corpus is
   required before any Vietnamese quality claim, and it is not on this machine.
2. **`data/raw` licenses.** 1.47 B tokens sit unusable because per-file
   provenance was never recorded.
3. **Scale.** 50 M tokens is 0.125 % of GPT-2's training budget. The 5.49×
   perplexity gap is a data/compute gap, not a code gap.
4. **Single seed.** No seed replicate for the main run; differences below
   ~0.02 nats are not resolvable.
5. **Dropout at scale is untested.** The A/B only says "0 is better at 3 M
   tokens".
6. **No SFT.** Deliberately not started: the base model has only just
   demonstrated LM improvement, and there are no verified preference pairs.
7. **CUDA exact-resume automation.** Verified by construction and by the run's
   checkpoint contents, not by an automated interrupt-and-resume test.

## 16. Release gates

| Gate | Status |
|---|---|
| Data integrity — correct estimates, provenance, license review, exact counts, reproducible manifests | **pass** |
| No major train/test leakage | **pass** (0/0/0) |
| Training integrity — exact resume, LR schedule, FP16 stability, deterministic validation, checkpoint integrity | **pass** (CPU-verified; CUDA path exercised, not auto-tested) |
| Better held-out loss than the pilot | **pass** (−1.84 nats, 6.3× ppl) |
| Better continuation quality | **pass** (mean continuation length 10.9 → 32.5 words on knowledge, 20.3 → 38.3 on educational; pilot ends several prompts immediately with `<|endoftext|>`, Phase 14 does not) |
| No major regression in other domains | **partially met** — `prepared_v4`/vi slice regressed (§12), explained by pilot's 1.41-epoch exposure |
| Benchmark integrity — tokenizer parity, FP32, no contamination, reproducible | **pass** |
| Documentation — README, historical claims marked, current checkpoint identified, no unsupported parity claim | **pass** |

## 17. Exact next training command

Continue the current run into a larger budget *without* changing the schedule
shape, after confirming the corpus hash is unchanged:

```bash
.venv/Scripts/python.exe train_gpt2.py \
  --preset gpt2_small_reference --data-dir data/gpt2_corpus_v5 \
  --out-dir checkpoints/gpt2_small/phase14_wiki \
  --batch-size 4 --grad-accum 18 --seq-len 1024 \
  --max-steps 1360 --warmup 100 --lr 6e-4 \
  --eval-interval 40 --eval-iters 8 --save-interval 100 --dropout 0.0 \
  --resume --extend-budget
```

`--extend-budget` is required: the checkpoint's schedule says 680 steps and the
trainer refuses a silent reshape. Before a run this long, expand the corpus:
`data/curated_v5` holds ≈400 M tokens, so tokenise up to 400 M with
`scripts/build_tokenized_corpus.py --max-train-tokens 400000000 --mix ...` and
train at ~1 epoch rather than cycling the 51 M-token slice.
