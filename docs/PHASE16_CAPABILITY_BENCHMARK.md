# PHASE 16 §3/§4 — AI CAPABILITY BENCHMARK

## 1. What this benchmark is

`scripts/phase16_task_benchmark.py` — **73 independently scoreable tasks**, every
one with a deterministic checker:

| Category | Tasks | What is required to pass |
|---|---|---|
| instruction following | 8 | every listed constraint (length, format, forbidden words, JSON shape, ask-instead-of-invent) |
| question answering | 8 | the answer, checked by substring/number/abstention rules |
| mathematics | 8 | the correct number, or a correct "insufficient information" |
| programming | 30 | generated function passes **all** reference tests in a subprocess |
| Vietnamese | 6 | Vietnamese answer, diacritics where required, abstention |
| extraction/summarisation | 5 | valid JSON with required keys / fact preserved / contradiction named |
| multi-turn | 4 | retained fact, corrected value, stated preference, missing context |
| tool decisions | 4 | correct arithmetic result, no-tool answer, correct abstention |

Track A = base model alone (no retrieval, no tools). Track B (system) is measured
separately in `scripts/phase16_system_benchmark.py` and never merged with Track A.

## 2. Benchmark validity — the oracle test

A benchmark that cannot be passed is not a benchmark. `--self-test` runs **gold
answers** (`scripts/phase16_gold.py`, including 30 reference implementations)
through the exact same checkers:

```
[self-test] 73/73 gold answers pass
```

Validated by `tests/test_phase16_benchmark.py` (14 tests), which additionally proves:

- empty output never passes (this caught a real false positive: a whitespace-only
  answer satisfied "no forbidden word");
- a **parseable but wrong** function fails — syntax is never success;
- crashing code, missing functions and infinite loops are reported distinctly
  (timeout, runtime_error, function_not_defined), not as passes;
- the sandbox cannot open a network connection;
- fenced code blocks are extracted before execution.

## 3. Track A results — the base model (`phase14_wiki`, 47.19 M tokens)

| Category | Passed | Rate |
|---|---|---|
| instruction following | 0 / 8 | 0 % |
| question answering | 0 / 8 | 0 % |
| mathematics | 0 / 8 | 0 % |
| programming | 0 / 30 | 0 % |
| Vietnamese | 0 / 6 | 0 % |
| extraction / summarisation | 0 / 5 | 0 % |
| multi-turn | 0 / 4 | 0 % |
| tool decisions | 0 / 4 | 0 % |
| **overall** | **0 / 73** | **0 %** |

Code sub-metrics, reported because they show *where* it fails:
`syntax_ok_rate 0.00`, `ran_rate 0.00`, `mean_hidden_test_pass_rate 0.00`,
`timeout_rate 0.00`. The model does not even produce a syntactically valid
function, so the failures are not near-misses.

### What the outputs look like (verbatim, first 120 characters)

| Task | Output |
|---|---|
| "What is the capital of France?" | `" The first time of the country is the country.\nA person who is the"` |
| "Compute: 37 + 58 =" | `" <<10=<<<<<<<<<<<<<<<<<<<<<<<<<<“` |
| Vietnamese comprehension | `" độiội năm động đ�"` |
| "Write one sentence about the ocean…" | `"                        "` (whitespace only) |

The model is a document continuator trained for 47 M tokens. It has no notion of
a question, an instruction, or a program. **0 % is the honest baseline.**

## 4. Track A after LoRA instruction tuning

| Evaluation mode | Passed | Rate | Which tasks |
|---|---|---|---|
| base model | 0 / 73 | 0.0 % | — |
| LoRA, raw prompt | 2 / 73 | 2.7 % | `math_algebra`, `mt_correction` |
| LoRA, chat template | 1 / 73 | 1.4 % | `math_algebra` |

**This is not a capability gain.** Two successes out of 73 against a 0/73 baseline
is consistent with chance (exact binomial, one-sided, p ≈ 0.13 for the raw-prompt
case under a 0.5 % true rate; the two passing tasks are a one-line algebra answer
and an answer containing the digit "3"). The correct statement is: **instruction
tuning did not produce measurable task ability at this scale.**

It did change something measurable: SFT validation loss fell from 4.27 to 3.9588
and the model now produces assistant-shaped text instead of document continuation.
Lower SFT loss is not task ability, and it is not counted as one here.

## 5. Failure modes, categorised

| Failure | Frequency in outputs | Example |
|---|---|---|
| degenerate repetition | dominant | `"the term the term the term …"` |
| immediate EOS / empty output | common on instruction prompts | whitespace-only answer |
| byte corruption | occasional | `"đ�"` in Vietnamese output |
| format non-compliance | always | prose instead of JSON |
| unparsable code | 30/30 | no `def` line produced |
| confident wrong numbers | common | `"<<10="` for 37 + 58 |

## 6. Reproduce

```bash
python scripts/phase16_task_benchmark.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
    --self-test                                    # 73/73 — benchmark validity
python scripts/phase16_task_benchmark.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
    --out bench_results/phase16/task_benchmark_phase14.json
python scripts/phase16_task_benchmark.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
    --adapter checkpoints/phase16_sft/lora_r16/adapter.pt \
    --out bench_results/phase16/task_benchmark_sft_raw.json
```

Every artifact keeps the raw prompt, raw output, per-constraint failures, decoding
settings, seed and SHA-256 of the checkpoint and adapter.

## 7. Limits of this benchmark

- 73 tasks is enough to distinguish 0 % from 40 %, not 2 % from 6 %. Confidence
  intervals are wide and are reported with the counts.
- The checkers are strict but shallow: a correct-by-accident answer counts as
  correct. The gold-answer self-test bounds *false negatives*, not false positives.
- Programming tasks are single-function and small; passing them would prove
  basic code synthesis, not software engineering.
- There is no human-rated dimension here at all. Subjective quality is reported
  separately with raw samples, and is not scored.
