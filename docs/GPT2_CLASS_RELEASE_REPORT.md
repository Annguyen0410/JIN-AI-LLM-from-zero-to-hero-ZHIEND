# GPT-2-class upgrade — release report

> **Status: historical (pre-Phase-14).** Describes the pilot checkpoint and the
> measurements available at that time. Superseded by
> `docs/PHASE14_RELEASE_REPORT.md`; the pilot numbers themselves (val 3.2337 at
> step 160, 16.96 M tokens seen) are still accurate and are re-verified there.

Scope: the 13-phase directive "JIN AI GPT-2-CLASS UPGRADE".
Verdict is stated first because it is the only claim that matters.

---

## 1. Executive summary

**Verdict: `TRAINING PIPELINE VERIFIED`.**

JIN now has a genuine GPT-2-Small-shaped model (exact 124,439,808 parameters),
a fixed tokenizer that no longer corrupts training text, a pretraining pipeline
that packs real documents and can be resumed reproducibly, and a model-only
benchmark that measures JIN against GPT-2 Small under one protocol.

The weights that exist today are **not** GPT-2 class. A pilot checkpoint
trained on 16.2 M tokens scores **0.00** against GPT-2 Small's **0.24** on the
task probes and has **24.9× worse** held-out perplexity (1,247.87 vs 50.19).
That gap is the honest distance between "pipeline works" and "model is good",
and it is dominated by corpus size: 12.5 M usable tokens vs. the ~2.5 B a
124 M-parameter model needs.

## 2. Architecture

| | Before | After |
|---|---|---|
| GPT-2-shaped preset | none | `gpt2_small_reference` — 124,439,808 params |
| Classic path fidelity | GELU exact, no residual-scaled init | `gelu_tanh`, `resid_scaled_init` (default off → old checkpoints unchanged) |
| Architecture identity in checkpoint | absent | `arch_id` in every `model_config` |
| JIN modern profile | 992d/12L/16H/8KV | preserved, plus explicit `jin_small_modern` alias |
| Document-aware mask | crashed at batch > 1 | fixed (`(B,1,T,T)`), regression-tested |

## 3. Data (measured, not estimated)

| | Value |
|---|---|
| Corpus trained on | 11,990,607 train + 475,032 val tokens |
| Documents | 142,443 train / 6,140 val |
| Chars per token (GPT-2 BPE) | 2.98 |
| Exact-duplicate documents, train sample | 23.2 % |
| Validation documents also in train (normalized hash) | 43.1 % raw / **1.24 % for documents > 200 chars** |
| Raw corpora available, audited | `data/raw` 5.49 GB (~2.2 B tokens est.), `data/train.txt` 5.0 GB, `data/downloads` 17 GB (not `*.txt`) |

The 43 % figure is dominated by one-line boilerplate answers; only 17 of 1,375
long validation documents also appear in training. Validation is *mostly*
clean but not perfectly so, and the short-document duplication is a real data
quality problem for SFT mixtures.

**No 500 M-token corpus was built.** Converting 17 GB of downloads into a
deduplicated, document-segmented, license-checked stream is Phase 3 work that
needs hours of I/O plus a licensing review per source. The pipeline to do it is
`train_gpt2.py --build-data`; the corpus is the blocker, not the code.

## 4. Training (executed)

| Field | Value |
|---|---|
| Checkpoint | `checkpoints/gpt2_small/pilot/best.pt` |
| SHA-256 | `bc44d10d…8596ce29` |
| Iterations / tokens | 220 steps / 16,220,160 tokens (best at 160 / 11,796,480) |
| Batch | 73,728 tokens/step (4 × 18 × 1024) |
| Optimizer / schedule | AdamW, lr 6e-4, warmup 20, cosine → 10 %, clip 1.0 |
| Hardware | RTX 2060 6 GB, 5.9 GB peak, ≈ 7.8 k tokens/s, ~29 min |
| Parent | random init (no GPT-2 weights) |
| Resume | verified: continued from step 220 with optimizer, `tokens_seen` and RNG intact |

Curve (val loss): 5.5119 (40) → 3.6847 (80) → 3.7764 (120) → **3.2337 (160)** →
3.4448 (200) → 3.8894 (220). Random-init loss is ln(50,257) = 10.82, so the
model learned a great deal — and then began to overfit its 12 M-token corpus.

## 5. Software reliability

| Suite | Result |
|---|---|
| `tests/test_gpt2_reference.py` | **15 passed** (params, shapes, causality, tying, doc-mask, batch/seq bounds, NaN/Inf, grads, save/load, CUDA, overfit-a-tiny-dataset) |
| `tests/test_tokenizer_upgrade.py` | **20 passed** (both profiles, UTF-8 round-trip, underscore regression, 120k real Vietnamese chars) |
| `tests/test_training_resume.py` | **3 passed** (absolute LR schedule, checkpoint payload, document-boundary masks) |

38 tests, 0 failed, 0 skipped in these suites. **The project's full historical
suite (the "242"/"285" figures) was not re-run**; no claim is made about it.

Bugs found and fixed (all with evidence in the audit):
1. `doc_mask` attention crashed for batch > 1 — `train.py --pack` was unusable.
2. JIN BPE dropped every `_` — code identifiers corrupted before training.
3. Token caches were not invalidated by a splitter change (now versioned).
4. fp16 cross-entropy at vocab 50 k returns garbage (11.67 vs 0.83 fp32) —
   every benchmark loss now goes through fp32 CE.
5. Resume payload stored `TorchVersion`, making `weights_only` load fail.
6. `train_gpt2.py` throughput counter over-reported 10× until reset correctly.

## 6. Inference

| Metric | Value |
|---|---|
| Pretrain throughput | ≈ 7.8 k tokens/s (training, batch 73,728) |
| Peak VRAM | 4.4 GB allocated / 5.9 GB device |
| Decoding | greedy works; cached and uncached paths unchanged |
| Reliability | loops after ~15 tokens; emits EOT on QA-style prompts |

## 7. Security

Unchanged and still enforced by the existing layer: tool allowlists, admin-API
loopback gating (`JINGPT_TOKEN`), request size/rate limits, no code execution in
the agent path. This pass added no new network surface; the only new outbound
calls are one-time tokenizer/BPE downloads and the GPT-2 reference weights,
both inside the benchmark process and both recorded in artifacts.

## 8. Remaining gap to GPT-2 Small

1. **Data: 12.5 M vs ~40 B tokens** — the dominant gap, ~3000×.
2. **Compute: 29 minutes vs. days/weeks on many GPUs.**
3. **No SFT** — a base checkpoint cannot follow instructions; GPT-2's 0.24 task
   score is itself base-model behaviour, not instruction following.
4. **Vietnamese** — 1/6 of one epoch; both models score 0.

## 9. What to do next (in order)

1. Build the 500 M-token corpus from `data/downloads` (~15 h tokenizing once,
   then 15 h of training at 7.8 k tok/s). This is the only step that changes
   the verdict materially.
2. Run 500 M → 1 B tokens with the committed recipe, evaluating every 250 steps.
3. Then SFT (Phase 6) on `data/prepared_v4/sft_train` with assistant-only loss
   masking, and re-run this benchmark.
4. Only after that, consider preference optimisation.

## 10. Artifacts

```
docs/GPT2_CLASS_UPGRADE_AUDIT.md      Phase 0 audit
docs/TOKENIZER_RESEARCH.md            Phase 2 measurements + decision
docs/GPT2_CLASS_TRAINING_GUIDE.md     Phase 4/11 recipe, cost, resume
docs/GPT2_CLASS_BENCHMARK.md          Phase 9 protocol + results
docs/GPT2_CLASS_MODEL_CARD.md         pilot checkpoint card
bench_results/gpt2_upgrade/baseline_audit.json
bench_results/gpt2_upgrade/tokenizer_eval.json
bench_results/gpt2_upgrade/data_audit.json
bench_results/gpt2_upgrade/{base_model,gpt2_reference,comparison}.json
train_gpt2.py                          reproducible trainer
tokenizer_gpt2.py                      GPT-2 BPE profile
scripts/{tokenizer_eval,data_audit,gpt2_bench}.py
tests/{test_gpt2_reference,test_tokenizer_upgrade,test_training_resume}.py
checkpoints/gpt2_small/pilot/{best,latest}.pt(+ .manifest.json)
```

Nothing was committed or pushed. `checkpoints/`, `data/gpt2_corpus/` and
`runs/` stay out of git by `.gitignore`; the new docs, scripts, tests and
`benchmarks/gpt2_probes.jsonl` are untracked working-tree changes awaiting your
review.
