# Phase 17 — Capability Report

**No new capability measurement was possible, because no substantial new model
checkpoint exists.** The only new weights are 12 steps from the smoke run. This
document records the frozen baseline, says explicitly what was not measured, and
refuses to convert infrastructure work into a capability claim.

## 1. Frozen base-model baseline (Phase 16, unchanged)

| capability | score | protocol / artifact | checkpoint |
|---|---|---|---|
| language modelling (clean, unseen) | 4.8935 nats, ppl 133.4 | `phase16.coverage.v2`, 100 % coverage, `bench_results/phase16/eval_*.json` | `069d9da6…8266fa` |
| GPT-2 Small reference | 3.3203 nats, ppl 27.7 | same run, same file | external |
| task benchmark (73 tasks) | **0/73** base, 2/73 after LoRA, 1/73 chat-formatted | `bench_results/phase16/task_benchmark*.json` | base + `checkpoints/phase16_sft/lora_r16/adapter.pt` |
| Vietnamese track | 0/N in the 73-task set | same | base |
| generation coherence | poor; degenerate output documented in Phase 16 | same | base |

The 2/73 LoRA result is inside sampling noise and is not a gain.

## 2. Phase 17 measurements

| capability | status |
|---|---|
| language modelling on the new smoke checkpoint | **NOT MEASURED** — deltas after 196,608 tokens are not interpretable |
| generation coherence after continued pretraining | **NOT MEASURED** |
| instruction following | **NOT MEASURED** |
| mathematical reasoning | **NOT MEASURED** |
| programming | **NOT MEASURED** |
| Vietnamese | **NOT MEASURED** |
| multi-turn consistency | **NOT MEASURED** |
| 300-task Phase 17 benchmark | **NOT BUILT** — the task asks for ≥300 independently scored tasks; only the system-level benchmarks below were produced |

Comparability rule: any future number must name its artifact and checkpoint SHA.
The Phase 16 protocol must not change, or scores become incomparable.

## 3. Why the 12-step checkpoint is not evaluated as capability

1. 196,608 tokens is 0.4 % of Phase 14's own 47.19 M and 0.2 % of a 100 M target.
2. Validation loss on 12 steps moves inside evaluator noise; the two evaluations
   in the smoke run (4.7155, 4.7876) differ by more than any training signal could
   explain at that scale.
3. The frozen 73-task benchmark has 0/73 successes at baseline. A single-digit
   token budget cannot plausibly move it, and running it would invite exactly the
   "one extra success = improvement" error Phase 16 warned about.

## 4. System capability (measured — see `docs/PHASE17_ASSISTANT_ROUTING.md`)

These are system-level results, produced by deterministic components. They are
**not** model ability and are reported in a separate table on purpose:

| component | score | artifact |
|---|---|---|
| calculator routing | 258/258 supported cases, 14/14 negative rejections | `bench_results/phase17/router_benchmark.json` |
| retrieval recall@1 / @3 | 0.849 / 1.000 over 100 docs, 73 answerable queries | `bench_results/phase17/system_benchmark.json` |
| MRR / precision@3 | 0.9178 / 0.3151 | same |
| evidence selection | 0.9726 | same |
| evidence support | 0.9726 | same |
| abstention (6 unanswerable) | 6/6 | same |
| model-in-the-loop contribution | **0** — the model was not asked | same |

## 5. Milestones

| milestone | status | reason |
|---|---|---|
| `LANGUAGE_MODELING_IMPROVED` | **not achieved** | not measured on a meaningful checkpoint |
| `GENERATION_QUALITY_IMPROVED` | **not achieved** | not measured |
| `TASK_CAPABILITY_IMPROVED` | **not achieved** | no held-out improvement beyond noise |
| `END_TO_END_TASKS_VERIFIED` | **not achieved** | no model-driven tool selection demonstrated |

## 6. What would justify a capability claim

1. ≥100 M additional tokens from the verified lineage, with a fixed schedule.
2. Evaluation at 50 M and 100 M with the frozen protocol, plus the 73-task
   benchmark, plus raw generations preserved.
3. A ≥300-task Phase 17 benchmark with paraphrases, hard negatives,
   multi-constraint instructions, hidden-test code problems, English and
   Vietnamese tracks, and held-out prompts — with confidence intervals reported,
   so "one extra success" is visibly indistinguishable from noise.
4. Only then: a claim scoped to what moved, e.g. "language modelling improved and
   generation became less repetitive, but task completion did not".
