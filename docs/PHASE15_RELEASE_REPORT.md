# PHASE 15 — RELEASE REPORT

> **Status: `MULTI_SEED_VARIANCE_MEASURED`** — the highest milestone supported by
> evidence in this repository. Not `PHASE15_RELEASE_READY`.

## 1. Milestone ladder (achieved / not achieved)

| Milestone | Status | Evidence |
|---|---|---|
| `EXACT_DEDUP_VERIFIED` | ✅ | v6 audit: 415,197 / 37,233 / 9,431 records, **0** duplicate hashes, **0** split overlap, `passed: true` |
| `LICENSE_AND_PII_ENFORCED` | ✅ | reviewed source table, `license_ineligible` 0 by policy, 388 PII documents excluded, audit log without content |
| `EVALUATION_INTEGRITY_VERIFIED` | ✅ | document-independent scoring, `padding_tokens: 0`, stratified selection, 7 tests |
| `EXACT_RESUME_VERIFIED` | ✅ | fingerprint + gate + 16 tests |
| `CLEAN_COMPARISON_MEASURED` | ✅ | 4,000 unseen-by-both documents, 23,040 tokens per model, 6 models |
| `MULTI_SEED_VARIANCE_MEASURED` | ✅ | 3 seeds, sd 0.0956, range 0.1882 nats |
| `MIXTURE_COMPARISON_POWERED` | ❌ | 1 seed per mixture; effect (0.069) < seed noise (0.188) |
| `GENERATION_QUALITY_ACCEPTABLE` | ❌ | greedy repetition 0.43–0.93, code 0/1 parses, math 0 tokens |
| `SFT_READY` | ❌ | SFT data never gated; see `PHASE15_SFT_READINESS.md` |
| `PHASE15_RELEASE_READY` | ❌ | generation quality poor, Experiment C not run, nothing committed |

## 2. What changed in the code

| File | Change |
|---|---|
| `scripts/verify_corpus.py` | write-once dedup (`written` set), independent output audit that fails the run, `text_hash` per record, licence/PII policy enforcement, `--in-root/--out-root/--pii-policy/--license-policy` |
| `scripts/license_policy.py` | **new** — `REVIEWED_SOURCES`, `LICENSE_POLICIES`, `classify_source`, `decision_from_document`, `detect_pii`, `redact_pii` (POLICY_VERSION 2) |
| `scripts/ingest_corpus.py` | `--out-root`, licence decision on every record, `pii_findings`, `--max-docs-per-source`, stable `policy_key` |
| `scripts/eval_phase15.py` | **new** — document-independent scoring, no padding, stratified sampling, provenance hashes, `--hf-model` reference |
| `scripts/build_clean_comparison.py` | **new** — unseen-by-both slice, stratified quota per source |
| `scripts/build_tokenized_corpus.py` | provenance sidecar per split, `provenance_sha256`, manifest records the actual curated source |
| `scripts/generation_suite.py` | **new** — 10 categories, transparent metrics, all raw samples kept |
| `scripts/phase15_experiments.sh`, `scripts/phase15_eval.sh` | **new** — reproducible experiment and evaluation drivers |
| `train_gpt2.py` | `experiment_fingerprint`, `compare_fingerprints`, resume gate, `SAMPLE_IMPL_VERSION`, `DATA_FRAMING`, `--data-dir/--dropout/--extend-budget` |
| `tests/test_corpus_dedup.py`, `tests/test_license_pii.py`, `tests/test_evaluation_integrity.py` | **new** (9 + 12 + 7 tests) |
| `tests/test_exact_resume.py` | extended to 16 tests |
| `tests/test_check_project.py` | fixture collision fixed (28 tests in that file pass) |

Test status: the Phase 15 suites total **44 new tests + 16 resume tests**, all
passing. Full-suite status is reported in §6.

## 3. Data

`data/curated_v6` (new; `curated_v5` untouched):

| Quantity | Value |
|---|---|
| Documents ingested | 549,214 |
| Exact duplicates suppressed | 62,243 (11.3 % of raw) |
| Near-duplicate merges | 337 |
| Clusters kept | 461,816 |
| Dropped: too short / high repetition / PII | 24,477 / 24 / 388 |
| Train / val / test records | 415,197 / 37,233 / 9,431 |
| Characters (train / val / test) | 1,483.1 M / 132.4 M / 33.7 M |
| Vietnamese share (train) | 3.75 M chars (0.25 % of characters) |
| Hash overlap between splits | **0** |
| Output audit | **passed** |

Token streams built from v6: `gpt2_corpus_v6a` (10.45 M train tokens),
`v6b1` (7.56 M), `v6b2` (6.50 M) — all smaller than v5's 51.2 M because v6 caps
each source at 200 k documents. `data/gpt2_corpus_v5` remains the stream used by
the Phase 14 checkpoint.

## 4. Evaluation (clean slice, identical protocol for all models)

| Model | Loss | ppl |
|---|---|---|
| GPT-2 (reference) | 3.2497 | 25.78 |
| JIN Phase 14 | 5.0271 | 152.50 |
| JIN Phase 15 A (3.28 M tokens) | 6.6793 | 795.74 |
| JIN Phase 15 B2 | 6.8172 | 913.44 |
| JIN Phase 15 B1 | 6.9209 | 1013.26 |
| JIN pilot | 6.9281 | 1020.56 |

Phase 14 is **1.78 nats / 5.9× perplexity** behind GPT-2 on text neither model
saw in the same form. That is the measured gap; no "GPT-2 class" claim survives it.

## 5. Corrections to Phase 14 claims

| Phase 14 claim | Correction |
|---|---|
| "Train↔val/test duplicate = 0" | leakage was 0, but **2,091 duplicate records existed inside the splits**. Fixed and re-verified in v6. |
| "License recorded per source" | recorded, but `commercial_ok` was inferred from a non-empty string. Replaced with a reviewed table. |
| No PII handling | detectors + `exclude` policy + audit log added; 388 documents excluded |
| Evaluation on padded windows | padding inflated/biased per-domain loss; replaced with document-independent scoring (`padding_tokens: 0`) |
| Single run, single seed | three seeds measured: sd 0.0956 nats, range 0.188 nats |
| "Giảm 1,84 nats so với pilot" | true on the Phase 14 validation windows; on the clean slice Phase 14 → pilot is 5.0271 vs 6.9281 = **1.90 nats**, consistent in direction and size |

## 6. Verification actually run

- `pytest tests/test_corpus_dedup.py tests/test_license_pii.py tests/test_evaluation_integrity.py tests/test_exact_resume.py tests/test_check_project.py` → **47 passed**
- `python scripts/verify_corpus.py … --license-policy enforce` → `[audit] passed: True`
- 3 corpus builds, 5 training runs, 6 clean-slice evaluations, 2 generation suites — all artifacts in `bench_results/phase15/` and `runs/`
- **Full suite:** `python -m pytest tests/ -q` → **379 passed, 21 subtests passed, 0 failed**. The failure that existed at the start of Phase 15 (`tests/test_checkpoint_picker.py::…test_nested_instruct_tune_is_listed_and_wins_the_auto_pick`) disappeared once the fixture collision in `tests/test_check_project.py` was fixed: the two files shared a temporary directory, so the picker test sometimes read a project directory left behind by the other. It was never an `app.py` bug, and no application code was changed to make it pass.
- **Not run:** Experiment C (100 M tokens), any commit or push.

## 7. Reproduction

```bash
python scripts/ingest_corpus.py --all --max-docs-per-source 200000 --out-root data/curated_v6
python scripts/verify_corpus.py --in-root data/curated_v6 --out-root data/curated_v6 \
    --pii-policy exclude --license-policy enforce
python scripts/build_tokenized_corpus.py --splits data/curated_v6/splits \
    --out data/gpt2_corpus_v6a --max-train-tokens 12000000 \
    --mix wikipedia_en_20231101:0.65,cosmopedia_v2_sample:0.25,prepared_v4:0.08,project_seed:0.02
bash scripts/phase15_experiments.sh --skip-build
bash scripts/phase15_eval.sh
```

## 8. Known limits, stated once

- Vietnamese is 3 % of v5 training tokens and 0.25 % of v6 characters; no
  Vietnamese capability claim is made.
- Generation is poor (see `PHASE15_GENERATION_REPORT.md`); repetition under greedy
  decoding is the dominant defect.
- Mixture comparison is underpowered: 1 seed per mixture.
- Experiment C (100 M tokens) was not run — heat/time constraint, explicitly
  documented rather than estimated.
- `data/raw` (1.47 B tokens of books) remains excluded for missing licences.
- Nothing is committed or pushed.
