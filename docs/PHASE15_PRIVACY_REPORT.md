# PHASE 15C — PRIVACY / PII REPORT

## 1. The defect

No PII inspection existed. Every document was written and trained on as-is. A
web-derived corpus (Wikipedia, synthetic educational text, a mixed project
mixture) will contain emails, IP addresses, phone numbers and occasionally
payment-card-shaped strings; the pipeline neither detected nor recorded them, and
nothing in the reports said so.

## 2. Policy

`scripts/license_policy.py` (POLICY_VERSION 2):

| Kind | Detector | Notes |
|---|---|---|
| `email` | RFC-shaped address | |
| `phone` | country code or parentheses **and** 9–15 digits | v2 fix: a year range like "1902-1955" is not a phone |
| `ipv4` | four 0–255 octets | |
| `payment_card` | 15–16 digits, prefix 3–6, Luhn valid | v2 fix: Luhn alone passes ~10 % of arbitrary numbers |
| `ssn` | `ddd-dd-dddd` | |

Policy modes: **`exclude` (default)** or `redact`. `exclude` removes the whole
document from the training corpus. `redact` replaces findings with typed
placeholders and keeps the document; the run records which mode was used, so a
corpus produced with `redact` can never be mistaken for one produced with
`exclude`.

## 3. Audit log

`data/curated_v6/reports/pii_audit.jsonl` — one line per flagged document:

```json
{"document_id": ..., "source_id": ..., "kinds": {"email": 1}, "excluded": true}
```

It records **document id and counts only** — never the matched text, never the
surrounding sentence. A privacy log that stores the private data it is supposed to
protect is a second leak.

## 4. Measured effect on v6

| Quantity | Value |
|---|---|
| Documents scanned | 549,214 (ingest) / 486,971 unique (verified) |
| Documents flagged | **388** |
| Documents excluded (policy `exclude`) | 388 |
| Documents redacted | 0 |
| Findings: `ipv4` | 779 |
| Findings: `email` | 189 |
| Findings: `phone` | 149 |
| Findings: `payment_card` | 9 |
| Findings: `ssn` | 12 |

Excluded documents: 388 of 486,971 = **0.08 %**. Twenty-four thousand documents
were dropped as too short and 24 for high repetition, so PII exclusion is a small,
deliberate cost rather than a filtering side-effect.

### Why the counts changed between runs

The first v6 run (POLICY_VERSION 1) excluded 1,032 documents, inflated by phone
matches on year ranges and by Luhn false positives on arbitrary digit runs. v2
requires a plausible phone shape and a plausible card prefix, which cut the PII
exclusions to 388. Both runs are kept: the earlier number is what a naive
detector costs in discarded data.

## 5. Limits — stated, not hidden

- These are **pattern detectors**. Names, postal addresses, dates of birth,
  medical details and free-text identifiers are not detected at all.
- No NER model or external service is used: the directive forbids external model
  calls, and a local detector is auditable.
- Excluding a document removes its text, not the fact that similar text exists
  elsewhere in the corpus (dedup handles exact/near duplicates, not paraphrase).
- Wikipedia text is public by construction; the aggregate PII counts here are
  therefore expected to be small. The policy matters for future sources, where it
  is applied **before** anything reaches training.
- The audit log lists document ids. Anyone with the raw corpus could join those
  ids back to text; the log itself does not contain the sensitive strings.
