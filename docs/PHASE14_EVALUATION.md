# PHASE 14 — EVALUATION

The rule for this phase: **a number is only a result if it can be reproduced from
a checkpoint hash and a data hash.** This document defines how the numbers in
`docs/PHASE14_RELEASE_REPORT.md` were produced, so they can be checked or
disputed.

## 1. What "comparable" requires

Two models' perplexities may only be compared when:

1. the same tokenizer produced the same token ids (fingerprint equal), and
2. the same text windows were scored, and
3. the loss is a real cross-entropy, not a cast of an FP16 approximation.

The pilot and every Phase 14 run use `gpt2_bpe` (vocab 50,257, state hash
`40bbc29de16d5ed2…`), so (1) holds. Window sampling and (3) are enforced by the
evaluation script:

```bash
.venv/Scripts/python.exe scripts/eval_phase14.py \
    --checkpoint checkpoints/gpt2_small/<run>/best.pt \
    --out bench_results/phase14/eval_<run>.json
```

## 2. Why FP16 loss is not reported anywhere

`GPT2LMHeadModel(x, labels=y)` under autocast returned **11.6719** where an
explicit FP32 cross-entropy on the same logits returned **0.8336** — a 14×
error, measured in this repository during the earlier phase. Casting the final
scalar to FP32 does not repair a softmax that already overflowed in FP16.

Therefore:

- validation inside `train_gpt2.evaluate` computes `logits.float()` then
  `F.cross_entropy` (`loss_dtype: fp32_cross_entropy` in every eval record);
- every evaluation artifact records the exact number of tokens scored;
- the GPT-2 reference model is scored the same way, never with its built-in loss.

## 3. Language-modelling metrics

| Metric | How |
|---|---|
| `val_bin.loss` / `perplexity` | 64 windows of 1,024 tokens at evenly spaced offsets of `data/gpt2_corpus_v5/val.bin` (deterministic, same windows for every checkpoint) |
| `per_group.domain:*` | loss over documents from `curated_v5/splits/val.jsonl` grouped by `domain` (`knowledge`, `educational`, `mixed`, `conversation`), up to 24 documents per group, one window each |
| `per_group.language:*` | same, grouped by `language` (`en`, `vi`) |
| `tokens_scored` | always recorded per metric, so a 3-document group cannot masquerade as a full-corpus number |

The per-group numbers exist for one reason: a single aggregate loss can hide the
fact that a mixture is good at Wikipedia and bad at code. If Vietnamese loss is
reported, it is reported **with** its 3 % exposure share next to it.

## 4. Generation evaluation

- Prompts come from `curated_v5/splits/test.jsonl`, which never enters training.
- Prompt construction: the first ~240–400 characters of a test document, cut at
  a sentence boundary.
- Decoding: greedy (temperature 0) **and** temperature 0.8 with a fixed seed per
  prompt, 48 new tokens, no repetition penalty and no top-k/top-p shaping.
- Metrics: `distinct_1`, `distinct_2`, `repeat_4gram` of the continuation, plus
  the raw text.
- Degenerate output is **kept**. Adding a repetition penalty until the sample
  looks fluent would hide exactly the failure the measurement exists to detect.

## 5. Task probes

`benchmarks/gpt2_probes.jsonl` (37 probes) remains a smoke test, and is treated
as one. It is not evidence of capability, and the phase does not claim otherwise
on the strength of it. Base-model completion quality is what is scored here;
instruction-following scores belong to SFT and are kept in a separate column,
never averaged in.

## 6. GPT-2 Small reference

The reference is `openai-community/gpt2` (124,439,808 parameters), loaded from
its pinned revision, scored with the same FP32 cross-entropy and the same
windows. Its loss is **not** a target and its weights are **not** copied into
JIN — it exists to say where a 124 M-parameter transformer trained on 40 B
tokens lands, so that a 124 M-parameter transformer trained on 50 M tokens can
be placed honestly on that scale.

The earlier comparison artifact reported `comparable: false`; the reason was a
tokenizer/measurement mismatch, not a model difference. Phase 14 keeps the
metric separate unless the fingerprint check passes, and states which of the two
conditions failed when it does not.

## 7. What is deliberately not measured

- No semantic scoring of generations (no external judge, no LLM-as-grader): both
  would smuggle an outside model into the measurement.
- No "intelligence" score. The only claims allowed are about held-out loss,
  generation statistics, and exact-match probes.
- No comparison against a checkpoint trained on different data with a different
  tokenizer, presented as a ratio.
