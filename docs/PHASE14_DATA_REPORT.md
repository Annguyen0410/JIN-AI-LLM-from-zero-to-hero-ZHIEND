# PHASE 14 — DATA REPORT

Everything in this report was measured on this machine during Phase 14. Where a
number is an estimate it is labelled as one, with its method; where it is exact
it came from counting every token.

## 1. The audit bug, quantified

`scripts/data_audit.py` (v1, unchanged, kept for history) extrapolated with

```
tokens_estimate = tokens_per_character × total_bytes
```

a per-character ratio times a byte total. `scripts/data_audit_v2.py` measures
both units on the same sample and extrapolates only with the matching
denominator, and counts exactly whenever a corpus is small enough to scan.

| Corpus | v1 estimate | Corrected | Error in v1 |
|---|---|---|---|
| `prepared_v4/pretrain` (train part) | 9,413,546 | **11,990,607 exact** (full scan) | **−21.5 %** |
| `data/raw` (5.49 GB, 80 files) | 2,247,963,961 | **1,468.0 M** (sampled, 0.268 tok/byte) | **+53 %** |

So the old audit was not merely imprecise: it erred in both directions by tens
of percent, which is the difference between "we have a 40 M-token corpus" and
"we have a 60 M-token corpus" when planning a run. The v1 file was not modified;
`bench_results/phase14/data_audit_v2.json` records the correction next to it.

### Audit v2 mechanics

- Stratified, deterministic sampling: every file contributes documents from
  evenly spaced byte slices across its **whole** range (seed recorded, locations
  recorded per file) — not the head of the first file.
- Both denominators are reported: `tokens_per_byte` and `tokens_per_char`, plus
  `bytes_per_char`, so a reader can audit the extrapolation.
- `estimate_reliable` is `false` when the sample is under 2 M bytes / 200 k
  tokens, or when the token/byte ratio varies more than 60 % across files.
- Corpora under 250 MB are **fully tokenized** instead of estimated
  (`--exact-below-mb`), which is how `prepared_v4` got its exact count.

## 2. Corpus built

`data/curated_v5` → `data/gpt2_corpus_v5` (see `docs/DATA_INGESTION_V2.md` and
`docs/DATA_QUALITY_REPORT.md`).

| Split | Tokens (exact) | Bytes | Documents |
|---|---|---|---|
| train | **51,222,642** | 102.4 MB | 88,000+ |
| val | 28,969,659 | 57.9 MB | 28,969 |
| test | 7,403,328 | 14.8 MB | 7,403 |

Framing: `<eot> document <eot>`, uint16. Tokenizer `gpt2_bpe`, vocab 50,257,
state hash `40bbc29de16d5ed2…` (identical to the pilot's tokenizer, so all loss
comparisons below are token-for-token comparable).

### Realised mixture (measured, not targeted)

| Source | Train tokens | Share | Status |
|---|---|---|---|
| `wikipedia_en_20231101` | 34,374,566 | 67.1 % | human-written, CC-BY-SA-3.0 + GFDL |
| `cosmopedia_v2_sample` | 10,884,310 | 21.2 % | **synthetic**, ODC-BY-1.0 |
| `prepared_v4` | 5,930,568 | 11.6 % | project-vetted mixture |
| `project_seed` | 33,198 | 0.06 % | project-authored |

| Language | Train tokens | Share |
|---|---|---|
| English | 49,664,157 | 96.96 % |
| Vietnamese | **1,553,188** | **3.03 %** |
| other / unlabelled | 5,297 | 0.01 % |

## 3. How much unique, licensed data actually exists here

| Pool | Size | Status |
|---|---|---|
| `curated_v5` train split after dedup | 1.488 B characters ≈ **400 M GPT-2 tokens** (estimate, 0.269 tok/char) | license documented, unused remainder still on disk |
| Wikipedia download not yet ingested | ~10 GB of the 11 GB Parquet set | documented, unexplored |
| `data/raw` (books) | 5.49 GB ≈ **1.468 B tokens** (audit v2) | **license not recorded per file → excluded from the default mixture** |

So the honest answer to "how many tokens do we have ready": **≈0.4 B clean,
licensed, deduplicated English tokens**, of which 51 M were tokenized for this
phase's run. The 1.5 B-token book pile exists but cannot enter training until
its per-book licenses are recorded — a data-license task, not a compute task.

## 4. Provenance and license status

| Source | Publisher / id | Revision | License | Commercial | Redistribution |
|---|---|---|---|---|---|
| Wikipedia EN | `wikimedia/wikipedia` `20231101.en` | `b04c8d1c` | CC-BY-SA-3.0 + GFDL | yes (share-alike, attribution) | yes |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus` (sample) | recorded per shard | ODC-BY-1.0 | yes | yes |
| prepared_v4 | project mixture | local manifest | per-source, NC + uncertain excluded | per source | per source |
| project_seed | project-authored | local | project | yes | yes |
| `data/raw` books | unknown per file | — | **not recorded** | unknown | unknown |

Every source's SHA-256 and download URL lives in
`data/downloads/general_v1/download_report.json`; the ingestion copies them into
`data/curated_v5/raw_manifest.json`. Unknown-license material is written to the
documents directory but excluded from the training mixture — a missing license
is never read as permission.

## 5. Validation quality

The validation split is document-disjoint and cluster-disjoint from training,
verified **after** writing: exact content-hash overlap train↔val = **0**,
train↔test = **0**, val↔test = **0**. It contains 29.0 M tokens, which is large
enough that the 16 k-token evaluation window used during training samples it
rather than fitting it. `docs/DATA_QUALITY_REPORT.md` has the full breakdown.

## 6. Known weaknesses in this data foundation

1. **Vietnamese exposure is 3 %** of training tokens, from 148 k short
   `prepared_v4` fragments and 862 seed conversations. Vietnamese quality cannot
   be claimed from this run; it is measured, not assumed.
2. **21 % synthetic.** Cosmopedia is model-generated and its share is recorded in
   the manifest so no result can quietly attribute it to human data.
3. **English Wikipedia is 67 % of the mixture**, so "general-purpose" here means
   "encyclopedic English with an educational supplement", not a balanced web
   corpus.
4. **No quality classifier.** Filtering is rule-based; it removes broken and
   duplicated text, and deliberately does not pretend to rank prose quality.
5. **`data/raw`'s 1.47 B tokens are unusable until licensed** — the single
   largest available pool is blocked on provenance, not on engineering.
