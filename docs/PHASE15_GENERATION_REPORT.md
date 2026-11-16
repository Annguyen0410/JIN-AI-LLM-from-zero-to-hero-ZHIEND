# PHASE 15I — GENERATION SUITE REPORT

## 1. Method

`scripts/generation_suite.py`, 10 categories, prompts drawn from the held-out
`test.jsonl` for the text categories and fixed synthetic prompts for the
capability categories (code, math, list, long context, repetition, EOS).

- Decoding: greedy (temperature 0) **and** sampled (temperature 0.8), 64 new
  tokens, fixed seeds, **no repetition penalty** (a penalty would hide the
  defect this suite is meant to expose).
- Metrics are computed from the text only: continuation length, premature EOS,
  repeated 4-gram rate, distinct-1/distinct-2, U+FFFD rate, Vietnamese diacritic
  presence, content-word overlap between prompt and continuation, and whether a
  Python continuation parses.
- **No LLM judge.** Every raw sample is stored in
  `bench_results/phase15/generation_*.json`, including the bad ones.

## 2. Results — Phase 14 checkpoint (`phase14_wiki`, 47.19 M tokens)

| Category | Greedy words | Greedy repeat-4 | Sampled repeat-4 | Sampled distinct-2 | Topic overlap |
|---|---|---|---|---|---|
| en_continuation | 41.2 | **0.54** | 0.00 | 0.99 | 0.05 |
| educational | 53.7 | **0.43** | 0.00 | 0.97 | 0.06 |
| factual | 41.2 | **0.54** | 0.00 | 0.99 | 0.05 |
| structured_list | 44.0 | **0.56** | 0.00 | 0.93 | 0.14 |
| long_context | 48.0 | **0.87** | 0.00 | 0.94 | 1.00* |
| repetition_stress | 48.0 | **0.87** | 0.00 | 0.94 | 1.00* |
| eos_behavior | 64.0 | **0.93** | 0.00 | 0.84 | 0.00 |
| code | 1.0 | 0.00 | — | 1.00 | 0.00 |
| math | 0.0 | — | — | — | — |
| vi_continuation | 3.8 | 0.02 | 0.00 | 0.83 | 0.01 |

\* the prompt itself is a repeated sentence, so overlap is trivially 1.0 — that
metric is not informative for this category and is not used as evidence.

## 3. What the samples actually look like

Greedy decoding from the Phase 14 model enters a loop almost immediately:

> prompt: "Here are the main steps to set up a small home laboratory: 1."
> greedy: " **F**: A small, a small, a small, and a small, a small, and a small, …"

> prompt: "The end." (EOS-style prompt)
> greedy: 64 tokens with a 0.93 repeated-4-gram rate

The sampled continuations are less repetitive (repeat-4 ≈ 0.00) but incoherent:
they change topic mid-sentence and do not follow the instruction implied by the
prompt. Code continuation does not parse (`python_parses: false` for every code
sample). The math prompt produced **zero** continuation tokens for the Phase 14
model at greedy decoding.

## 4. Vietnamese

The Phase 14 model does produce Vietnamese diacritics: 5 of 6 sampled Vietnamese
continuations contain them, and the greedy continuation of a Vietnamese prompt
starts in Vietnamese before terminating. That is evidence of orthographic
learning, **not** of fluency: continuations average 11.5 words sampled / 3.8
greedy, and topic overlap with the prompt is 0.01.

## 5. Verdict

**Generation quality is poor, and this report says so.**

| Claim | Status |
|---|---|
| "Produces grammatical English sentences" | **not supported** — greedy collapses into repetition, sampled output is incoherent |
| "Can write code" | **not supported** — 0/1 code continuations parse; the sample is token soup |
| "Can solve math word problems" | **not supported** — zero-token output |
| "Understands instructions" | **not supported** — no instruction-following training has been done |
| "Has learned Vietnamese orthography" | supported (diacritics present) with the limits in §4 |
| "Does not emit garbage characters" | supported — U+FFFD rate 0.000 in every category |

The Phase 15 4.1 M-token checkpoint (seed 2024) is not better on these axes: its
greedy continuations are **shorter** (0 tokens on two text categories, i.e. it
emits EOS immediately) and its Vietnamese output degrades (2 of 6 sampled
continuations contain diacritics). It has 8 % of Phase 14's training tokens; that
is the explanation, and it is not an excuse to hide the result.

## 6. What this means for "make JIN smarter"

Generation quality is capped by two things this suite can measure:

1. **Model and budget.** A ~25 M-parameter transformer trained on 47 M tokens
   cannot produce coherent multi-sentence text; GPT-2 needed 100 B tokens for
   that. Scaling tokens is the lever, not decoding tricks.
2. **No instruction data in pretraining.** The model is a document continuator.
   Nothing in its training taught it to answer a question or follow a list marker.

Adding a repetition penalty would make the samples look better in a table while
leaving the model unchanged — it is deliberately not used here.
