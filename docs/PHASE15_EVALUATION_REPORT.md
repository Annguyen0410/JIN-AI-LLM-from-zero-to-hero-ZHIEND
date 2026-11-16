# PHASE 15D — EVALUATION VALIDITY REPORT

## 1. Three defects in the Phase 14 evaluator

### 1.1 Padding contaminated domain metrics

`per_group_loss()` padded short documents with `<|endoftext|>` up to a fixed
width, then `nll_windows()` computed cross-entropy over the whole padded tensor.
Every padded position became a scored target, so a group of short documents had
its loss diluted by artificial EOT tokens. The per-domain numbers in
`bench_results/phase14/eval_phase14_wiki.json` are therefore approximate, and
are superseded by the v6 numbers.

**Fix.** `scripts/eval_phase15.py` scores each document independently:

```
ids = [EOT] + tokens(document) + [EOT]
windows of seq_len+1 tokens, stride = seq_len, no padding
loss = sum(CE over every target token) / (number of target tokens)
```

There is no padding step at all, so no padding can enter the denominator. Each
result carries `tokens_scored`, `windows`, `documents`, `nll_sum` and
`padding_tokens: 0`.

**Proof.** `tests/test_evaluation_integrity.py`:
- with uniform logits the loss is exactly `ln(vocab)` per token, so any
  denominator error is visible;
- adding a document too short to form a window leaves loss, token count and
  window count unchanged;
- two documents scored together produce exactly the sum of their token counts and
  the sum of their NLLs.

### 1.2 Sampling followed file order

The old code took the first 24 matching documents. Because the validation file is
grouped by source, one upstream source could dominate a "domain" metric.

**Fix.** Sampling is stratified on `(source_id, language, domain)`, ordered by
`sha256(f"{seed}.{document_id}")`, capped per stratum, and the selected document
ids plus the seed are written into the result. Tests cover determinism, per-source
coverage, cap enforcement and that file order is not followed.

### 1.3 Document boundaries were ignored

Training masks attention across document boundaries (`ShardBatcher` emits an
`<eot>` boundary mask). The old evaluator packed windows without any mask, so a
window could contain the tail of one document and the head of another — a
condition the model never saw in training, and a different condition from the one
the GPT-2 reference was scored under.

**Fix.** Document-independent scoring: a window never spans two documents, each
document is framed with its own EOT at both ends, and the GPT-2 reference is
scored with the same protocol. The scoring rule is identical for both models, so
a comparison is apples-to-apples; the protocol string
(`phase15.docindependent.v1`) is recorded in every artifact.

## 2. Validation / test separation

| Split | Role |
|---|---|
| train | optimisation only |
| val | checkpoint selection, mixture decisions, decoding experiments |
| test | final reporting only; prompts for the generation suite come from here |

The generation suite draws its prompts from `splits/test.jsonl`, which never
enters training, and the decoding configuration is fixed **before** looking at
outputs (greedy and temperature 0.8, 64 new tokens, no repetition penalty, fixed
seeds). No decoding parameter is tuned against test.

## 3. Benchmark manifest

Every evaluation artifact records: checkpoint path and SHA-256, step and tokens
seen, tokenizer kind/vocab/EOT id, validation file and SHA-256, the sampling seed,
the strata, the per-stratum document ids and the exact number of tokens scored.
That is enough to re-derive a reported number or to show it was measured
differently.

## 4. What is not claimed

- These metrics measure language modelling and continuation statistics. They do
  not measure reasoning, instruction following, factual accuracy or "intelligence".
- A lower loss on Wikipedia-like text does not transfer automatically to chat or
  to Vietnamese, and the per-stratum tables are the evidence for that.
- Per-stratum samples are capped (24 documents by default). A stratum with very
  few documents is reported with its document count next to the metric, and is
  not described as representative of the language or domain as a whole.
