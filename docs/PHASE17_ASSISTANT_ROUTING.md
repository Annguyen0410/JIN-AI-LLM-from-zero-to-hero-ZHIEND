# Phase 17 — Assistant Routing, Retrieval and Grounding

Phase 16 measured components in isolation: the calculator was called directly
(14/14) and retrieval was scored on 12 synthetic passages (recall@3 = 1.00). Those
numbers said nothing about whether the assistant can *choose* a tool, and
`verify_citations()` only checked that `[n]` referred to something.

Phase 17 adds a deterministic router, a larger retrieval benchmark, and an
evidence-support check that is stricter than citation syntax.

## 1. Calculator routing — `scripts/phase17_intent_router.py`

Deterministic, no `eval`/`exec`:

* expressions are parsed to an AST; only numeric constants, `+ - * / // % **` and
  unary sign are evaluated; exponent base and result are bounded;
* normalisation handles `× ÷ − ^`, thousands separators, and framing words
  (`what is`, `calculate`, `tính`, `kết quả`…);
* dates (`12/03/2024`), phone numbers (`090-123-4567`), versions (`v1.2.3`) are
  classified and **named** in the trace instead of being computed;
* percentages need a base: `15% of 240` → 36, `20% off 250` → 200, `30% của 400` →
  120, while `50 + 10%` is reported `ambiguous` rather than guessed;
* unit conversions: length, mass, volume, time, and temperature;
* every result is a structured trace: original request, intent, tool, arguments,
  status, value, explanation, executor, latency.

**Measured** (`bench_results/phase17/router_benchmark.json`):

| suite | result |
|---|---|
| supported expressions (arithmetic, precedence, parentheses, powers, modulo, percentages, conversions, EN + VI framing) | **258/258** |
| router accuracy | 1.0000 |
| arithmetic accuracy | 1.0000 |
| negatives (dates, phones, versions, prose questions, identifiers) | **14/14** rejected |
| ambiguous (`50 + 10%`, `20% more than 30`, `what is 5`) | 3/3 handled without inventing a value |
| invalid (`1/0`, `5 +`, `((3+4)`, `sqrt(9)`, `2 ^ 99999`, `3 + * 4`) | 6/6 refused or errored cleanly |
| credited to the model | 0 |

Read this correctly: the suite is **self-authored**, so 100 % means the
implemented grammar is self-consistent and its negatives are handled. It is not
evidence that the router understands arbitrary language.

## 2. Retrieval and grounded answers — `scripts/phase17_system_benchmark.py`

Corpus: 100 documents — 30 fabricated fact passages, 5 Vietnamese, 10 hard
distractors that reuse the same names and numbers with a different claim, and 55
unrelated filler documents. Queries: 73 answerable (36 direct, 30 paraphrase, 5
Vietnamese, 2 multi-document) plus 6 unanswerable.

| metric | value |
|---|---|
| recall@1 / @3 / @5 | 0.8493 / **1.0000** / 1.0000 |
| MRR | 0.9178 |
| precision@3 / @5 | 0.3151 / 0.1808 |
| retrieval latency (mean / p95) | 0.364 ms / 0.587 ms |
| evidence selection accuracy | **0.9726** |
| evidence support rate | **0.9726** |
| abstention on unanswerable | **6/6** |
| credited to the model | 0 |

By track:

| track | n | recall@3 | evidence selection | evidence support |
|---|---|---|---|---|
| direct | 36 | 1.000 | 1.000 | 1.000 |
| paraphrase | 30 | 1.000 | 0.967 | 0.967 |
| Vietnamese | 5 | 1.000 | 1.000 | 1.000 |
| multi-document | 2 | 1.000 | 0.500 | 0.500 |

`precision@3` is low by construction: most queries have exactly one gold passage,
so 1/3 is the maximum reachable value.

## 3. A real defect found and fixed

First run: Vietnamese evidence selection was **0/5** while recall@3 was 1.00 — the
gold passage was retrieved but then discarded. Cause: `grounding.select_evidence`
requires ≥50 % overlap between query terms and the passage, and Vietnamese
question words (`bao`, `nhiêu`) counted as content terms, so
`sông Lam có số liệu bao nhiêu?` overlapped only 2 of 6 terms.

Fix: `bao`, `nhiêu`, `mấy`, `sao` added to the function-word list in
`grounding.py`. After the fix: Vietnamese selection 5/5, overall selection
0.9041 → **0.9726**, support 0.5068 → 0.9726.

The second jump came from the measurement itself: answers were compared as raw
strings, so `1240000` never matched the passage's `1,240,000`. Numeric comparison
now strips separators. That was a benchmark bug, and it is named here rather than
quietly corrected.

Phase 16 was re-run after the `grounding.py` change to a Phase 17 path
(`bench_results/phase17/phase16_system_recheck.json`): recall@3 1.00, calculator
100 % of 14, citation syntax 100 %, model-in-evidence 0 % — unchanged. The Phase 16
artifact was not overwritten.

## 4. What is still not proven

* **No model-driven tool selection.** The router is deterministic code. The base
  model has never chosen a tool; giving it credit for a calculator result would be
  false attribution.
* **Evidence support is containment, not entailment.** The check asks whether the
  answer string appears in the selected passage. It cannot detect a passage that
  mentions the number while meaning something else — which is exactly what the
  hard distractors are designed to test for retrieval, not for a generated claim.
* **The corpus is synthetic.** 100 project-authored documents is 8× Phase 16's 12
  but still not a web corpus; recall will drop on real text.
* **Vietnamese is 5 queries.** One query is 20 % of that track.
* **No human evaluation** of answer quality; outputs for subjective cases were not
  scored by anyone.

## 5. Tool trace contract

Every assisted result exposes:

```
request → intent → tool → validated args → status → value/result → evidence
        → explanation → executor ("calculator" | "retrieval" | "router")
```

The original user request is preserved verbatim, and `executor` states which
component produced the answer, so no tool result can be mistaken for model
reasoning.
