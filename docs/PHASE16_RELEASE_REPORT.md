# PHASE 16 — RELEASE REPORT

> **Status: `BASELINE_AI_CAPABILITY_MEASURED` (with `MODEL_SIZE_AND_CHECKPOINT_VERIFIED`,
> `EVALUATION_TOKEN_COVERAGE_VERIFIED`, `BASE_MODEL_TASK_BENCHMARK_VERIFIED`,
> `SYSTEM_TASK_COMPLETION_MEASURED`, `SFT_LICENSE_AND_PROVENANCE_VERIFIED`,
> `SFT_PII_AND_DEDUP_VERIFIED`, `SFT_DATASET_READY`, `SFT_ADAPTER_TRAINED`,
> `BASE_MODEL_VS_SFT_EVALUATED`)**
>
> **Not achieved:** `CAPABILITY_GAIN_VERIFIED`, `ASSISTANT_END_TO_END_VERIFIED`,
> `PHASE16_RELEASE_READY`.

## 1. The thirteen questions

### 1. What could the base model actually do before Phase 16?

Almost nothing that a user would call a task. On 73 independently checked tasks it
scored **0/73 (0 %)**. It can continue English documents; its output on the
benchmark is degenerate repetition or whitespace.

### 2. Which tasks did it consistently fail?

All of them, without exception: instruction following 0/8, QA 0/8, maths 0/8,
programming 0/30, Vietnamese 0/6, extraction/summarisation 0/5, multi-turn 0/4,
tool decisions 0/4. Code: `syntax_ok_rate 0.00` — it does not produce one
syntactically valid function in 30 attempts.

### 3. What was the exact architecture and parameter count?

12 layers, width 768, 12 heads, vocab 50,257, context 1,024, GELU-tanh, pre-LN,
learned positions → **124,439,808 parameters**, verified by instantiating the model
and counting. The raw `state_dict` contains 163,037,184 values because the tied
output head is also stored as its own tensor (byte-identical duplicate, 155 MB
wasted per checkpoint). Checkpoint sha256 `069d9da650a81dc9e2c0fee94d0e20dc46a5c4f7e266a6c381a72140f48266fa`,
iteration 640, tokens seen 47,185,920. Inference: 112.6 tok/s, TTFT 0.012 s,
498 MB peak VRAM.

### 4. What was wrong with the previous evaluation or documentation?

| Defect | Evidence |
|---|---|
| Parameter count wrong in Phase 15 docs ("~25 M", "4-layer") | measured 124,439,808 / 12 layers; corrected in both reports |
| Evaluator dropped short documents and final partial windows | Phase 15 clean-slice numbers used **11,264 of 37,558 tokens (30 %)**; protocol v2 scores 100 % |
| Licence could be self-declared | `decision_from_document` trusted `canonical_license`; now the reviewed mapping decides and conflicts are rejected |
| Clean comparison called "4,000 documents" | it was a 4,000-document **pool**; 55 documents were actually scored |
| Capability implied from perplexity / 1 code prompt / diacritics | replaced by 73 checked tasks, executed code, and a separate Vietnamese score |

### 5. Which SFT datasets passed provenance, licence, PII and quality checks?

dolly-15k (CC-BY-SA-3.0), gsm8k-train (MIT), metamathqa (MIT), oasst2
(Apache-2.0), opencodeinstruct (CC-BY-4.0) → **44,631 conversations**, split
43,706 / 467 / 458 at cluster level, 0 straddling clusters, 0 duplicate hashes.
Rejected: `smol-smoltalk` (aggregated constituents unreviewed) and
`data/chat_train.txt` (no per-record provenance).

### 6. Was a new adapter or checkpoint trained?

Yes: a LoRA adapter (rank 16, 2,359,296 params, 1.9 % of the model) on the frozen
Phase 14 checkpoint — 600 steps, 615,212 assistant tokens, 10 minutes, 972 MB peak
VRAM, val loss 4.2733 → 3.9588. Artifact:
`checkpoints/phase16_sft/lora_r16/adapter.pt` (9.47 MB). The base checkpoint was
not modified.

### 7. Did SFT actually improve held-out task performance?

**No — not verifiably.** Base 0/73, adapter 2/73 (raw prompt) and 1/73 (chat
template). Two extra successes on 73 tasks against a zero baseline are consistent
with chance (p ≈ 0.13). `CAPABILITY_GAIN_VERIFIED` is **not** claimed.

### 8. Which abilities improved, which did not, and which regressed?

| Ability | Change |
|---|---|
| assistant-formatted output | improved (val loss −0.31 nats; produces `<|assistant|>`-shaped text) |
| instruction following | 0/8 → 0/8 (no change) |
| QA, maths, code, Vietnamese, extraction, multi-turn, tool | 0 → 0 (no change) |
| document-continuation loss | not re-measured after SFT — reported as **NOT MEASURED** to avoid an unverified claim |
| nothing regressed measurably, but the benchmark cannot detect small regressions |

### 9. How much of the assistant's success came from the model, retrieval, or tools?

Measured separately, never merged:

| Component | Result | Credited to |
|---|---|---|
| BM25 retrieval, 12 queries / 12 passages, k=3 | recall@3 **1.00**, MRR 1.000 | retrieval |
| calculator, 14 expressions | **100 %** correct | tool |
| citation validation (grounding.verify_citations) | **100 %** valid | system |
| model reading grounded evidence (12 prompts) | **0 %** contained the gold answer | model |

The system's demonstrable competence lives entirely in its retrieval and
calculator; the language model contributes nothing measurable. Retrieval numbers
come from a 12-document corpus with distinctive terms — a smoke test, not a
retrieval-quality benchmark.

### 10. Can JIN now complete real tasks reliably?

**No.** It can compute arithmetic correctly when the expression is handed to the
calculator, and it can retrieve the right passage. It cannot decide on its own to
use either, cannot answer a question, cannot write code, and cannot follow an
instruction. There is no end-to-end task it completes reliably.

### 11. What are its remaining failure modes?

Degenerate repetition under greedy decoding; immediate EOS / empty output on
instruction prompts; byte corruption in Vietnamese; no `def` line in any of 30
code attempts; confident wrong numbers (`"<<10="` for 37 + 58); no abstention
behaviour.

### 12. Accuracy, latency, VRAM, practical value?

Accuracy 0 % on 73 tasks (2.7 % with the adapter, not significant). Latency 0.012 s
TTFT, 112.6 tok/s, 28 ms per 32 tokens on the RTX 2060. VRAM 498 MB inference,
972 MB for LoRA training, 4.4 GB for the Phase 14 recipe. Storage: 1.49 GB base
checkpoint + 9.47 MB adapter. Practical value today: a working retrieval +
calculator system with a language model that cannot be used for tasks.

### 13. What should be prioritised in Phase 17?

1. **Capability before alignment**: continue pretraining (≥1 B tokens) before
   another SFT run — 615 k instruction tokens cannot install arithmetic.
2. **Route, don't reason**: teach the system to call the 100 %-accurate calculator
   and the 100 %-recall retriever through a deterministic intent router with an
   explicit tool trace, instead of asking the model to compute.
3. **Scale the benchmark** to ~300 tasks and add confidence intervals before
   claiming any single-digit improvement.
4. **Vietnamese**: a licensed Vietnamese corpus is still missing; 0.25 % of v6
   characters cannot support Vietnamese ability.

## 2. Scorecard

| Dimension | Score | Evidence |
|---|---|---|
| base-model instruction following | **0 / 8 (0 %)** | `task_benchmark_phase14.json` |
| factual question answering | **0 / 8 (0 %)** | same |
| reasoning and mathematics | **0 / 8 (0 %)** | same |
| programming | **0 / 30 (0 %)**, syntax 0 % | same, sandboxed tests |
| summarisation and extraction | **0 / 5 (0 %)** | same |
| Vietnamese | **0 / 6 (0 %)** | same |
| multi-turn consistency | **0 / 4 (0 %)** | same |
| retrieval grounding | **recall@3 1.00** (12 docs), model answer-in-evidence **0 %** | `system_benchmark_base.json` |
| tool-use reliability | **calculator 100 %** (14 expressions); model tool selection **NOT MEASURED** | same |
| end-to-end task completion | **0 %** (no task completed by the model) | Track A |

`NOT MEASURED` is used where a number does not exist; nothing is scored 0 for
being untested.

## 3. Verdict

**Is JIN AI more useful to a real user after Phase 16, or did the repository
merely become more sophisticated?**

Mostly the second, and the first only in a narrow sense:

- The **repository** became more honest and more verifiable: a real task
  benchmark that a correct answer passes (73/73 oracle), an evaluator that scores
  every token it claims, a licence system that cannot be spoofed by a record, and
  a documented parameter count.
- The **product** gained one thing a user can rely on: a calculator and a
  retriever that are correct when used — but nothing that decides to use them.
- The **model** is not measurably better. It is 124.4 M parameters trained on
  47 M tokens, and at that scale it cannot answer questions. Instruction tuning
  changed its style, not its ability.

A user asking JIN a question today is still better served by reading Wikipedia.
The honest path forward is scaling pretraining and routing to tools — not another
SFT run on a model that cannot yet read.

## 4. Artifacts

| Path | Contents |
|---|---|
| `scripts/phase16_inspect_model.py` | architecture/parameter/latency measurement |
| `scripts/eval_phase16.py` + `tests/test_evaluation_phase16.py` | coverage-aware evaluator, 11 tests |
| `scripts/phase16_task_benchmark.py` + `scripts/phase16_gold.py` + `tests/test_phase16_benchmark.py` | 73-task benchmark, oracle self-test, sandbox, 14 tests |
| `scripts/phase16_system_benchmark.py` | Track B component measurements |
| `scripts/build_phase16_sft.py` + `scripts/sft_sources.py` + `tests/test_license_integrity_phase16.py` | SFT gates, 8 tests |
| `scripts/phase16_sft_train.py` | LoRA SFT trainer |
| `data/phase16_sft/` | 44,631 audited conversations |
| `checkpoints/phase16_sft/lora_r16/` | adapter + manifest |
| `bench_results/phase16/` | all measurements, raw outputs, task-level rows |
| `docs/PHASE16_MODEL_BASELINE.md`, `PHASE16_EVALUATION_PROTOCOL.md`, `PHASE16_CAPABILITY_BENCHMARK.md`, `PHASE16_SFT_DATA_AUDIT.md`, `PHASE16_SFT_TRAINING_REPORT.md` | this phase's reports |

## 5. Verification actually run

- `pytest tests/` → see final message; Phase 16 suites: 11 + 14 + 8 = 33 new tests.
- Benchmark oracle: `73/73 gold answers pass` (a wrong checker would fail this).
- 3 benchmark runs (base, adapter raw, adapter chat), 2 eval-protocol runs, 1
  system benchmark, 1 SFT training run, 1 SFT dataset build — all artifacts kept.
- Resource use: GPU 76–80 °C, ≤ 972 MB VRAM for training, ~10 min per run; no
  multi-hour job.

## 6. Not done, and why

- **No new pretraining run.** 47 M tokens is the capability ceiling; another
  50 M tokens on this hardware (~1 h, 83 °C) would move loss, not task success.
- **No full fine-tuning.** LoRA is reversible and fits the VRAM budget; full FT
  of 124 M params on 6 GB at seq 512 was not attempted.
- **No human evaluation.** No native-speaker review of Vietnamese output was
  available; Vietnamese is scored only by deterministic checks, and marked as
  such.
- **No commit or push.** Phase 16 changes are uncommitted pending explicit
  authorization.
