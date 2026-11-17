# PHASE 16 §5 — SFT DATA AUDIT

**Verdict: `SFT_DATASET_READY` for five sources; `smol-smoltalk` and
`data/chat_train.txt` rejected.**

## 1. The licence-integrity defect from Phase 15, fixed

`license_policy.decision_from_document()` used to trust the record's own
`canonical_license` field:

```python
if canonical in LICENSE_POLICIES:      # <- a record could declare itself eligible
```

Any file could ship records labelled `project-authored` or `cc-by-sa-3.0+gfdl`
and they became eligible, no matter where they came from. The reviewed mapping is
now authoritative:

| Input | Old behaviour | New behaviour |
|---|---|---|
| `source_id` unknown, record claims `project-authored` | **eligible** | ineligible (`unknown`) |
| `source_id` = wikipedia, record claims `project-authored` | **eligible** | ineligible (`license_conflict`) |
| `source_id` = wikipedia, record claims its real licence | eligible | eligible |
| id spelling `wikipedia_en_20231101` vs table `wikipedia-en-20231101` | unknown | resolved to the same entry |

Tests for all four rows are in `tests/test_license_pii.py`.

## 2. Reviewed SFT sources (`scripts/sft_sources.py`)

| Source | Cards read | Licence | Commercial | Eligible | Note |
|---|---|---|---|---|---|
| `dolly-15k` | huggingface.co/datasets/databricks/databricks-dolly-15k | CC-BY-SA-3.0 | yes | ✅ | attribution + share-alike follow the data |
| `gsm8k-train` | huggingface.co/datasets/openai/gsm8k | MIT | yes | ✅ | train split only |
| `metamathqa` | huggingface.co/datasets/meta-math/MetaMathQA | MIT | yes | ✅ | augmented from GSM8K/MATH **train** only, per card |
| `oasst2` | huggingface.co/datasets/OpenAssistant/oasst2 | Apache-2.0 | yes | ✅ | crowd-written → PII gate mandatory |
| `opencodeinstruct` | huggingface.co/datasets/nvidia/OpenCodeInstruct | CC-BY-4.0 | yes | ✅ | synthetic code instructions |
| `smol-smoltalk` | huggingface.co/datasets/HuggingFaceTB/smol-smoltalk | Apache-2.0 | **no** | ❌ | aggregated: its licence cannot cover unreviewed constituents |
| `data/chat_train.txt` | — | unknown | no | ❌ | no per-record provenance (source, author, licence all unknown) |

## 3. Gates and results (`scripts/build_phase16_sft.py`)

| Gate | Rule | Dropped |
|---|---|---|
| Licence | source must be in the reviewed table | 9,262 (smol-smoltalk) |
| Format | `messages` list, valid roles, user + assistant turns, assistant last | 0 |
| Answer quality | 2–4,000 chars, alphanumeric, no degenerate repetition | 118 |
| PII (patterns) | email/phone/IPv4/card/SSN | 78 |
| Contamination | overlap ≥ 0.8 Jaccard with benchmark prompts, or benchmark gold answer present | 3 |
| Exact dedup | sha256 of normalised conversation | 1 |
| Near dedup | 5-gram shingles, Jaccard ≥ 0.8, bottom-12 LSH banding | 280 |

Result:

| Quantity | Value |
|---|---|
| Input conversations | 54,091 |
| Kept | **44,631** |
| Clusters | 44,351 |
| Train / val / test | **43,706 / 467 / 458** |
| Clusters straddling splits | **0** |
| Duplicate content hashes in output | **0** |

Per source: dolly 9,468 · gsm8k 7,061 · metamathqa 9,476 · oasst2 9,443 ·
opencodeinstruct 9,184.

## 4. What the gates do **not** do

- **PII is pattern-based.** Emails, phones, IPv4, card-shaped numbers and SSNs
  only. Names, addresses, dates of birth and free-text identifiers are not
  detected. This is not comprehensive PII removal and is not claimed to be.
- **Near-duplicate detection is approximate.** LSH banding compares pairs that
  share a signature bucket (2,091,009 comparisons for 44,631 records); pairs that
  share no bucket are never compared, so a small number of near-duplicates can
  survive. Exact duplicates cannot.
- **Answer quality is heuristics**, not a judgement of correctness. A wrong but
  well-formed answer passes this gate; that is why the benchmark, not the data
  audit, decides whether the model learned anything.
- **Contamination is measured against the 73-task benchmark only.** Other
  evaluation sets would need their own check.
- Datasets were reviewed from their published cards on 2026-11-16. Share-alike
  obligations for Dolly and CC-BY attribution for OpenCodeInstruct follow the data;
  redistributing weights trained on them carries the usual attribution duties.

## 5. Reproduce

```bash
python scripts/build_phase16_sft.py
python -m pytest tests/test_license_pii.py -q
```

Artifacts: `data/phase16_sft/{train,val,test}.jsonl`,
`bench_results/phase16/sft_data_quality.json` (per-gate counts, per-source table,
licence table with card URLs).
