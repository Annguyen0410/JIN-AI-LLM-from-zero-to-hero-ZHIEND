# PHASE 15C — LICENSE AND PROVENANCE REPORT

## 1. The defect

`_license_record()` in `scripts/ingest_corpus.py` (Phase 14) set:

```python
"commercial_ok": None if not lic else True,
"redistribution_ok": None if not lic else True,
```

i.e. **any non-empty license string was treated as commercial permission**. That
is an inference, not a review, and it would have granted training rights to a
source whose license text said "non-commercial only" or "research use".

## 2. The replacement: explicit policy tables

`scripts/license_policy.py` contains two reviewed tables and no inference:

- `LICENSE_POLICIES` — per canonical license: commercial, redistribution,
  attribution, share-alike, NC restriction, eligibility, obligations, review note.
- `REVIEWED_SOURCES` — the **only** place a source becomes eligible, mapping
  source id → canonical license → upstream id → dataset card.

`classify_source()` returns `unknown` (and therefore ineligible) for anything not
present in `REVIEWED_SOURCES`. A folder that merely exists on disk is **not** a
source with rights.

### Recorded decisions

| Source | Canonical license | Commercial | Redistribution | Attribution | Share-alike | Eligible |
|---|---|---|---|---|---|---|
| `wikipedia-en-20231101` | `cc-by-sa-3.0+gfdl` | yes | yes | **yes** | **yes** | yes |
| `cosmopedia-v2-sample` | `odc-by-1.0` | yes | yes | **yes** | no | yes (synthetic, labelled) |
| `prepared_v4` | `project-vetted-mixture` | yes | yes | yes | no | yes |
| `project_seed` | `project-authored` | yes | yes | no | no | yes |
| `data/raw` (books) | `unknown` | **no** | **no** | — | — | **no** |
| anything else | `unknown` | no | no | — | — | no |

`data/raw` is the important case: 5.49 GB of books exist on disk and are excluded,
because no per-file license was ever recorded. Existence is not permission.

## 3. Where the decision is enforced

1. `ingest_corpus.py` writes the full decision onto every document
   (`canonical_license`, `commercial_use_permitted`, `redistribution_permitted`,
   `attribution_required`, `share_alike`, `eligible_for_training_mixture`,
   `decision_reason`, `policy_version`) and into `raw_manifest.json`.
2. `verify_corpus.py --license-policy enforce` drops every document whose
   recorded decision is ineligible, and counts them under
   `quality_drops.license_ineligible`.
3. `reports/license_summary.json` records the decision per source for the run.
4. Tests: `tests/test_license_pii.py` (12 tests) cover unknown source with a
   plausible license string, missing license, reviewed sources, attribution /
   share-alike recording, policy-table completeness, and an end-to-end run that
   proves an ineligible document is not written.

## 4. Bug found in the integration (and fixed)

The first v6 verification run dropped **87,218 documents** as
`license_ineligible`. Root cause: ingestion classified the source by its
*display name* (`"prepared_v4 pretrain mixture"`, `"project seed conversations"`)
instead of a stable policy key, so two reviewed sources fell through to
`unknown`. Fixed by adding an explicit `policy_key` to every source and
classifying on that. The run was discarded and repeated; the counted v6 result is
in §5.

This is recorded because it is exactly the failure mode the directive warns
about: a policy table that is right, wired to the wrong key, silently deletes
data. The `license_ineligible` counter is what made it visible.

## 5. v6 license outcome (measured)

`data/curated_v6/reports/license_summary.json` and
`data/curated_v6/reports/dedup_quality_splits.json`:

- all four reviewed sources → `eligible_for_training_mixture: true`;
- `license_ineligible` drops: 0 documents (nothing eligible was excluded, and no
  ineligible source was present in the ingest set);
- `data/raw` was never ingested, so it contributes nothing to any split.

Attribution survives tokenization: every split has a `*.provenance.jsonl`
sidecar mapping each token range to document id, source id, source URL,
dataset revision, license, original record id, language, domain, content hash and
split.

## 6. Limits of this review

- The review is a **project-level engineering decision**, not legal advice. The
  upstream dataset cards are the authority; both are recorded per source.
- Share-alike obligations on Wikipedia text are recorded but **not** discharged by
  this repository: redistributing trained weights plus training text requires
  following CC-BY-SA terms for the text itself.
- Cosmopedia is synthetic: ODC-BY-1.0 permits use, and the manifest labels the
  data as model-generated so no result can present it as human-authored.
- No license scan was performed on `data/raw`; it remains excluded, and per-file
  review is a prerequisite for any future use.
