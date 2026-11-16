# Phase 0 — Repository audit before the GPT-2-class upgrade

Date: 2026-11-15 (local) · Host: Windows, RTX 2060 6 GB, 15.8 GB RAM (2.5 GB free at audit time), 564 GB free disk
Environment: Python 3.11 `.venv`, torch 2.5.1+cu121, CUDA 12.1

This audit separates **verified by execution** from **read from source** from
**not checked**. Nothing below is inferred from a README claim.

---

## 1. Hardware reality (measured)

| Item | Value | Consequence |
|---|---|---|
| GPU | RTX 2060, 6,144 MiB, CUDA 12.1 | 124M params + AdamW fits, but only with activation checkpointing (peak 5.9 GB observed) |
| System RAM | 15.8 GB total, **2.5 GB available** | multi-GB checkpoint loads are the main stability risk; `mmap` loading matters |
| Disk | 564 GB free | not a constraint for a 1–2 B token corpus |
| Measured training throughput | see `docs/GPT2_CLASS_TRAINING_GUIDE.md` | determines what is achievable per hour |

## 2. Architecture — what actually exists

Verified by constructing the models (`tests/test_gpt2_reference.py`, 15 tests):

| Fact | Status |
|---|---|
| `model.JinAI` is a decoder-only transformer with two code paths | verified |
| `modern=True` → RMSNorm + RoPE + GQA + QK-norm + SwiGLU (+ optional MoE / sliding window) | verified |
| `modern=False` → LayerNorm + learned absolute positions + GELU 4× MLP + tied head | verified |
| Old checkpoints store their own `model_config` and load unchanged | verified (checkpoint round-trip test) |
| The shipped `small` preset is 992d / 12L / 16H / 8KV | verified |
| **Before this pass there was no GPT-2-shaped preset** | verified — added as `gpt2_small_reference` |
| The classic path was **not** GPT-2-faithful: exact GELU instead of `GELU(tanh)`, no residual-scaled init | verified — both added behind `gelu_tanh` / `resid_scaled_init` (default off, so old checkpoints are byte-identical in behaviour) |
| `gpt2_small_reference` parameter count | **124,439,808** (exact GPT-2 Small count) |

## 3. Checkpoint inventory (on this machine, not in git)

| File | Size | Role |
|---|---:|---|
| `checkpoints/jingpt.pt` | 58 MB | legacy (512d/8L) |
| `checkpoints/jingpt_best.pt` | 1.25 GB | pretrain-best (small modern) |
| `checkpoints/jingpt_latest.pt` | 1.25 GB | pretrain-latest |
| `checkpoints/jingpt_chat.pt` | 1.77 GB | SFT chat checkpoint (served by default) |
| `checkpoints/jingpt_chat_{seed_v1,10kbase,1600,2000,2400,pre46k,46k_sft4k}.pt` | 1.77 GB each | SFT snapshots |
| `checkpoints/kaggle_v4_60020/` | — | Kaggle v4 pretrain + instruct (+ `.manifest.json`) |
| `checkpoints/pretrain/general_v1_01/` | — | Kaggle pretrain run |

All of these are **excluded from git** by `.gitignore` (correct: a single
checkpoint exceeds GitHub's 100 MB file limit and the repo would be 7 GB).

No checkpoint was overwritten during this pass. New weights go to
`checkpoints/gpt2_small/`.

## 4. Data — measured, not assumed

| Corpus | Bytes | Documents | Notes |
|---|---:|---:|---|
| `data/prepared_v4/pretrain/*.txt` (train) | 35.75 MB | 142,443 | document-disjoint val split already verified by the project (`verification.json`: 0 overlapping hashes) |
| `data/prepared_v4/pretrain/*val*.txt` | 1.42 MB | 6,140 | same |
| `data/train.txt` | 5.0 GB | — | legacy mixed corpus, not document-segmented |
| `data/downloads/general_v1` | 17 GB | — | raw downloads |
| `data/curated/snapshots` | 5.5 GB | — | snapshots |
| `data/eval/heldout.txt` | 1.1 MB | — | held-out reference text |

Exact token counts measured with the GPT-2 BPE for the corpus actually used in
this pass:

* `data/gpt2_corpus/train.bin` — **11,990,607 tokens**, 142,443 documents, 2.98 chars/token
* `data/gpt2_corpus/val.bin` — **475,032 tokens**, 6,140 documents

That is **12.5 M tokens of usable, document-disjoint training data** — i.e.
0.5 % of the 2.5 B-token budget a 124M model needs. This is the single largest
limitation of the project, and no amount of architecture work compensates for
it. The 17 GB of downloads is the only path to a serious corpus, and it is
not yet converted into a clean, deduplicated, document-segmented stream.

## 5. Bugs found by running the code (all fixed in this pass)

| # | Bug | Evidence | Impact |
|---|---|---|---|
| 1 | **`doc_mask` attention crashed for batch > 1** — mask was `(B,T,T)` while SDPA needs `(B,H,T,T)`/`(B,1,T,T)` | `RuntimeError: expanded size (12) must match existing size (2)` at batch 2 | `train.py --pack` never trained packed data with batch > 1; document-aware pretraining was effectively unusable |
| 2 | **JIN BPE dropped every `_`** — no pre-tokenize alternative matched it | `LAST_UPDATED2` → `LASTUPDATED2`; 120k-char VI slice round-trip failed | silent corpus corruption of all snake_case code before training |
| 3 | **Token cache not invalidated when the splitter changes** | cache key = vocab state only | a splitter fix would keep serving tokens built by the old rules |
| 4 | **`train.py` cosine schedule restarts on `--resume`** — `target = start_step + run_iters`, and LR is a function of that target | code read + reasoning | resumed runs silently change the LR curve; runs are not reproducible |
| 5 | **No `tokens_seen` in checkpoints** | `utils.save_checkpoint` payload | "trained on N tokens" claims cannot be audited from the artifact |
| 6 | **Default pretrain batch = 1×2 at block 256** | `config.BATCH_SIZE=1`, `GRAD_ACCUM=2` | 512 tokens/step: a 124M model would need ~5 M steps for 2.5 B tokens |
| 7 | **`split_train_val` splits a token array at 90 %** | `utils.split_train_val` | same document can straddle the boundary (adjacent-block leakage, no exact-doc leakage) |

Bugs 1–3 and 4–5 are fixed in this pass (`model.py`, `bpe_tokenizer.py`,
`utils.py`, and the new `train_gpt2.py`). Bug 7 is avoided rather than changed:
`train_gpt2.py` uses the project's document-disjoint `*val*` files and never
token-splits a single stream.

## 6. Documented-but-unverified claims in the repo

| Claim | Status |
|---|---|
| "13/165 probes pass, held-out loss 5.6904, ppl 296.02" | historical benchmark numbers **not re-run** in this pass; they live in `bench_results/` artifacts |
| "242 tests" vs "285 tests" | the two numbers come from different revisions; this audit counts only what it ran (see below) |
| RAG / memory / web search / tools / self-learning | present in source (`retriever.py`, `memory.py`, `web_search.py`, `tools.py`, `curiosity_engine.py`) but **not audited functionally** here; they are Layer B and are explicitly excluded from every model benchmark below |
| `train_pref.py` (preference optimisation) | present, **not exercised**; no verified preference pairs |
| MoE / xl-moe preset | constructible, **not trained or benchmarked** in this pass |

## 7. Tests actually run in this pass

| Suite | Result |
|---|---|
| `tests/test_gpt2_reference.py` | 15 passed (param count, shapes, causality, tying, doc-mask isolation, batch boundaries, NaN/Inf, gradients, save/load, CUDA, **overfit-a-tiny-dataset**) |
| `tests/test_tokenizer_upgrade.py` | 20 passed (both profiles, UTF-8 round-trip, underscore regression, 120k-char real Vietnamese) |
| Pilot pretraining run | see `docs/GPT2_CLASS_RELEASE_REPORT.md` |

## 8. What limits quality (ranked, with evidence)

1. **Data volume** — 12.5 M tokens vs. the ~2.5 B that a 124M model needs
   (Chinchilla 20×). This is a 200× shortfall, and it dominates everything else.
2. **Text quality of the legacy corpus** — `data/train.txt` is a 5 GB
   concatenation without document boundaries; training on it teaches
   cross-document continuation.
3. **Tokenizer corruption** (bug 2) — every `_` lost from the training stream.
4. **Training recipe** — defaults were 512 tokens/step with an LR curve that
   cannot be resumed reproducibly.
5. **Model size/shape** — the modern preset is fine; the *classic* path was
   missing two GPT-2 details (now available).
6. **Decoding** — repetition penalties and `stop_on_repeat` mask a model that
   has not learned structure; they are not a substitute for training (see
   directive §Phase 5).
