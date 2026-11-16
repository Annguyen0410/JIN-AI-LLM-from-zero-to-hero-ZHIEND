# PHASE 15G — DATA STRATEGY

## 1. What the corpus actually contains (v5, measured)

| Source | Train tokens | Share | License | Eligible |
|---|---|---|---|---|
| Wikipedia EN 20231101 | 34,374,566 | 67.1 % | CC-BY-SA-3.0 + GFDL | yes (attribution, share-alike) |
| Cosmopedia v2 (synthetic) | 10,884,310 | 21.2 % | ODC-BY-1.0 | yes, labelled synthetic |
| prepared_v4 | 5,930,568 | 11.6 % | project-vetted mixture | yes |
| project_seed | 33,198 | 0.06 % | project-authored | yes |

Language: English 49,664,157 (96.96 %), **Vietnamese 1,553,188 (3.03 %)**, other 5,297.

Two denominators are kept separate on purpose:

- **character mass** before thinning: prepared_v4 is 2.5 % of characters;
- **tokens in the built mixture**: prepared_v4 is 11.6 % of tokens, because the
  mixture thinning kept a larger share of it (its Vietnamese share matters more
  than its size).

Neither number makes this a balanced multilingual corpus. Vietnamese is 3 % of
training tokens and comes from 148 k short fragments, not from a Vietnamese
corpus.

## 2. What is available but *not* used

| Pool | Size | Why not used |
|---|---|---|
| `data/raw` books | 5.49 GB ≈ 1.47 B tokens | **no per-file licence record** → ineligible under the Phase 15C policy |
| remaining Wikipedia Parquet | ~10 GB of the 11 GB download | not ingested yet; documented, licensed, and the obvious next expansion |
| `chat_train.txt` | 168 MB | SFT-shaped, not pretraining text; audited separately in the SFT readiness report |

The largest pool by tokens (books) is blocked by provenance, not by engineering.
Recording licences for those files is a legal/data task, and pretending they are
usable because they sit on disk is exactly what Phase 15C forbids.

## 3. Mixture pipeline (deterministic, measured after tokenisation)

`scripts/build_tokenized_corpus.py`:

1. Pass 1 counts characters per source.
2. Each document is thinned by a hash of its id — `sha256(document_id)` — against
   a per-source keep probability derived from the target weights. Thinning by
   hash instead of by order means no source is sampled only from its first files.
3. Tokens are counted as they are produced, so the **realised composition** is
   what gets reported, never the target.

Realised v5 composition: 67.1 / 21.2 / 11.6 / 0.06 (tokens), which is close to
the 65 / 25 / 8 / 2 target — and the difference is reported as measured, not
assumed.

Every split also gets a `*.provenance.jsonl` sidecar (added in Phase 15C) mapping
each token range to its document id, source, licence, revision, language, domain
and split, so attribution survives tokenisation.

## 4. Quality gate before training

A corpus may be used for a training run only if all of these hold, and each is
machine-checked:

| Gate | Check |
|---|---|
| Dedup | `output_audit.extra_duplicate_records == 0` (fail-fast in the script) |
| Leakage | train/val/test hash overlap and cluster overlap all 0 |
| Licence | every source has a reviewed policy entry; ineligible documents removed |
| PII | policy applied (`exclude` by default) with an audit log |
| Exact counts | token counts come from the tokenised output, not an estimate |
| Composition | per-source / per-language / per-domain token counts recorded |
| Provenance | sidecar written and hashed |

## 5. Planned expansion (in priority order)

1. **Ingest the remaining Wikipedia shards.** Already downloaded, already
   licensed, highest quality per token. Estimated +1.5 B tokens if all shards are
   processed. No licence work needed.
2. **A licensed Vietnamese corpus.** This is the single biggest capability gap
   (3 % exposure). Candidates must be repositories with explicit licences;
   nothing is invented or machine-translated without being labelled.
3. **Documented code and mathematics text.** Only with licence and provenance,
   and kept as a labelled domain rather than mixed into prose.
4. **Books — only after licence review.** 1.47 B tokens are sitting there; the
   blocker is a per-file licence record, not tokenisation.

## 6. What will not be done

- No use of `data/raw` before its licences are recorded.
- No fabricated Vietnamese data, and no machine translation presented as
  human-authored text.
- No external model API to manufacture training data.
- No "balanced multilingual" claim while Vietnamese is 3 % of tokens.
- No repetition of the same 51 M tokens to imitate a larger budget: an epoch
  count is reported with every run, and one epoch of unique data is the default
  target.
