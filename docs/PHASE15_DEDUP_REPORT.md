# PHASE 15B — EXACT DEDUPLICATION REPORT

## 1. The defect

Phase 14's `verify_corpus.py` deduplicated twice and got it wrong the second time.

Pass 1 built `seen: hash -> first_index` and counted later copies as removed.
Pass 2 re-read the files and, for **every** occurrence of a hash, looked up
`seen[hash]`. Since all copies share the first index, every copy passed the
`keep` test and was written. The counter said "removed"; the output said
"written again".

### Evidence (independent audit of the written files)

Auditing the output by counting content hashes — not by trusting the script's
own counters:

| Split | Records | Unique hashes | Duplicated hashes | Extra records |
|---|---|---|---|---|
| train | 417,394 | 415,497 | 1,422 | **1,897** |
| val | 37,419 | 37,266 | 135 | **153** |
| test | 9,476 | 9,435 | 37 | **41** |

Total: **2,091** records that the report described as removed.

Note what this did *not* do: no duplicate crossed a split boundary (the leakage
check compared sets, which is why it reported 0 and was still right about
leakage). The damage was double-counting inside each split and an inflated
"unique documents" claim.

## 2. The fix

```python
written: set[str] = set()
...
if h in written:                 # the Phase 14 defect
    duplicate_suppressed += 1
    continue
written.add(h)
```

Three properties make this more than a one-line patch:

1. **Write-once is enforced at the write site**, so any future refactor that
   re-introduces a lookup cannot re-introduce the bug silently.
2. **The output is audited after writing** (`audit_outputs()`): duplicate hashes
   per split, cross-split hash overlap, cluster overlap, record counts. The run
   raises `SystemExit` if any check fails, so a regression breaks the pipeline
   instead of producing a corpus.
3. **The counters and the audit are separate code paths.** Pass 2 counts
   `duplicate_suppressed`; the audit re-reads the files. Agreement between them
   is evidence; a single counter is not.

## 3. Tests (`tests/test_corpus_dedup.py`, 9 tests, all passing)

| Test | What it pins |
|---|---|
| identical documents within one source | 3 copies + 1 unique → 2 records written, 2 counted as duplicates |
| identical documents across sources | duplicates are content-based, not source-based |
| identical documents in different files | same hash in another file must not be re-written |
| near-duplicate cluster | a cluster contributes exactly one document |
| near-duplicate, different lengths | cluster representative is the longest member |
| repeated runs | byte-identical split contents across two runs |
| output counts vs input counts | `documents_in - duplicates == records written` |
| cluster disjointness | no cluster id appears in two splits |
| PII exclusion | flagged document is not written, and the report counts it |

Every test asserts on the **written files** (`read_split()` + `assert_unique_hashes`),
not on the internal counters.

## 4. Result on the real corpus (v6)

See `docs/PHASE15_RELEASE_REPORT.md` for the v6 numbers. The relevant summary:

- `exact_duplicates_suppressed_in_pass2` > 0 proves the gate is active on real
  data (it would be 0 only if the corpus had no duplicates at all).
- `output_audit.extra_duplicate_records == 0` on all three splits.
- `output_audit.passed == true`, otherwise the pipeline exits non-zero.

## 5. What this changes about Phase 14's claims

- "Zero exact duplicate records in the output" — **false for Phase 14**, true for
  v6. The Phase 14 statement is corrected in `docs/PHASE15_BASELINE_AUDIT.md` §5.
- "Unique documents 462,198" — the count of clusters was right; the number of
  unique *hashes written* was not.
- The trained checkpoint is not invalidated: 1,897 duplicate records were 0.45 %
  of one epoch and do not change the reported loss materially. Retraining was
  not required to keep Phase 14's numbers honest; re-stating them was.
