# Phase 17 — Data Report

**Status: corpus v7 was not built in this phase. No v7 number in this document is
measured, and none may be quoted as if it were.**

## What exists today (Phase 14–16 artifacts, untouched)

| corpus | documents/clusters | train tokens | notes |
|---|---|---|---|
| `data/curated_v6` | 461,816 clusters | — | splits 415,197 / 37,233 / 9,431, 0 duplicates, 0 leak, 388 PII-excluded docs |
| `data/gpt2_corpus_v6a` | — | 10.45 M | wiki 66.0 %, cosmopedia 20.6 %, prepared 13.1 %, vi 3.24 % |
| `data/gpt2_corpus_v6b1` | — | 5.19 M | shard B1 |
| `data/gpt2_corpus_v6b2` | — | 5.24 M | shard B2 |
| `data/gpt2_corpus_v5` | — | ~51 M | the corpus Phase 14 actually trained on |

These are Phase 15/16 assets and were not modified.

## Corpus v7 — NOT MEASURED

| item | status |
|---|---|
| ingestion of the ~10 GB Wikipedia parquet | **NOT DONE** |
| `data/curated_v7/raw_manifest.json` | **DOES NOT EXIST** |
| `scripts/verify_corpus.py --pii-policy exclude --license-policy enforce` on v7 | **NOT RUN** |
| `data/gpt2_corpus_v7/manifest.json` | **DOES NOT EXIST** |
| realized source mixture | **NOT MEASURED** |
| actual GPT-2 BPE token count for v7 | **NOT MEASURED** |

Why: this environment is not a Kaggle machine (`/kaggle/input` absent, preflight
records `is_kaggle: false`), the 10 GB parquet is not attached here, and the
project rule is that a data stage with missing inputs stops rather than proceeds
on an estimate. Ingesting ~10 GB and then verifying ~1 M documents in memory was
not attempted on a 6 GB-VRAM desktop whose disk and RAM budget for that job is
unverified.

The 1–1.5 B token figure reported for that dataset is an **upstream estimate**. It
is not a token count, and Phase 17 does not treat it as one.

## The procedure that is ready and must be followed on Kaggle

Stage C and D of `notebooks/phase17_kaggle_pretrain.ipynb`:

1. `python scripts/ingest_corpus.py --list` — enumerate discovered sources.
2. Resolve the exact Wikipedia source id through `license_policy.classify_source`.
   Underscore/hyphen spellings normalize to the same reviewed entry; a folder name
   or a non-empty licence string is not permission.
3. Bounded sample (10,000 documents) to project:
   input bytes, output JSONL bytes, characters/document, documents/second, PII
   hits, language mix, token estimate, projected split sizes.
4. Only if the projection fits RAM/disk: full ingest to `data/curated_v7`.
5. `python scripts/verify_corpus.py --in-root data/curated_v7 --out-root data/curated_v7 --pii-policy exclude --license-policy enforce`,
   then read `reports/*.json` — exit code 0 is not the acceptance criterion.
6. Tokenize with an explicit mixture; the builder's default (wiki 65 %, cosmopedia
   25 %, prepared 8 %, seed 2 %) is a **weight target**, not a realized share. Any
   eligible source absent from the mixture silently gets zero weight, so the mix
   must be stated explicitly and the realized composition reported from
   `data/gpt2_corpus_v7/manifest.json`.
7. The token builder caps only the training stream with `--max-train-tokens`; val
   and test streams are built in full. Disk projections must include both, plus
   provenance files, before anything is written.

## Known ineligible or unresolved sources

* `data/raw` books (~1.47 B tokens): no per-file licence, therefore ineligible
  under `--license-policy enforce`. Unchanged from Phase 16.
* `smol-smoltalk` and `chat_train.txt`: excluded from SFT sources in Phase 16 and
  still excluded.

## What must be true before a v7 token count is quoted

1. `data/curated_v7/raw_manifest.json` exists and records the upstream revision
   and licence per source.
2. `reports/` shows 0 duplicates, 0 split overlap, 0 PII in output.
3. `data/gpt2_corpus_v7/manifest.json` records tokenizer fingerprint, split
   hashes, per-source token counts and truncation status.
4. The realized mixture is reported next to the intended mixture, with the
   difference explained.

Until all four hold, `CORPUS_V7_SOURCE_AUDIT_VERIFIED`, `CORPUS_V7_LICENSE_VERIFIED`,
`CORPUS_V7_SPLITS_VERIFIED`, `CORPUS_V7_TOKENIZED` and `TOKEN_COMPOSITION_VERIFIED`
stay **unachieved**.
