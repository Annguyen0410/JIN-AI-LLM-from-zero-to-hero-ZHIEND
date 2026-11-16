# DATA INGESTION V2

Phase 14 replaced ad-hoc text scraping with a provenance-carrying ingestion
layer. Everything here was produced by the two scripts below on this machine;
the numbers are read back from
`data/curated_v5/raw_manifest.json`, not typed by hand.

```bash
.venv/Scripts/python.exe scripts/ingest_corpus.py --list
.venv/Scripts/python.exe scripts/ingest_corpus.py --all --max-docs-per-source 200000
.venv/Scripts/python.exe scripts/verify_corpus.py --min-chars 100
.venv/Scripts/python.exe scripts/build_tokenized_corpus.py --max-train-tokens 60000000
```

## 1. Sources actually ingested

| `source_id` | Format | Domain | Docs | Chars | Language mix | License (recorded) |
|---|---|---|---|---|---|---|
| `wikipedia_en_20231101` | Parquet ×41 | knowledge | 200,000 | 871.7 M | en 100 % | CC-BY-SA-3.0 + GFDL (`wikimedia/wikipedia` @ `b04c8d1c`) |
| `cosmopedia_v2_sample` | Parquet ×5 | educational | 200,000 | 748.9 M | en 100 % | ODC-BY-1.0 (`HuggingFaceTB/smollm-corpus`) |
| `prepared_v4` | TXT ×11 | mixed | 148,352 | 37.1 M | en 57,763 / vi 28,557 / other 73 / unlabelled 61,959 docs | per-source policy, NC + uncertain excluded |
| `project_seed` | JSONL | conversation | 862 | 0.1 M | en 690 / vi 155 | project-authored |

Total ingested: **549,214 documents, 1.62 billion characters** before
deduplication.

Two properties matter and are recorded per source in the manifest:

- **Provenance** — dataset id, upstream repository, revision, per-shard SHA-256
  and the exact download URL are copied from `download_report.json` rather than
  re-derived here, so the corpus can be reconstructed byte-for-byte.
- **License status** — `documented` or `uncertain`. An uncertain license is
  written to disk but excluded from every training split. Nothing is inferred:
  a missing license is never treated as permission.

Cosmopedia is **synthetic** (model-generated educational text). It is kept in
the manifest and reported as synthetic in the split report so its share of the
training mixture can never be mistaken for human-written data.

## 2. Unified document schema

Every record, whatever the input format, has the same fields:

```json
{
  "document_id": "wikipedia_en_20231101:0",
  "source_id": "wikipedia_en_20231101",
  "source_name": "wikipedia-en-20231101",
  "source_url": "https://en.wikipedia.org/wiki/...",
  "license": "CC-BY-SA-3.0 and GFDL as dataset card; review upstream terms",
  "license_status": "documented",
  "language": "en",
  "domain": "knowledge",
  "text": "...",
  "original_format": "parquet",
  "original_record_id": "12345",
  "original_title": "...",
  "content_hash": "sha256:...",
  "quality_flags": ["possible_pii"],
  "ingestion_timestamp": "2026-11-16T10:22:13"
}
```

`license`, `source_url` and the original record id survive conversion — the
attribution chain is never dropped.

## 3. Document boundaries

| Input | Boundary rule |
|---|---|
| TXT | blank line separates documents (the convention the project's own corpora already use) |
| Parquet / JSON / JSONL / CSV | one record = one document; the record id is preserved as `original_record_id` |
| Oversized document | split at a fixed 200 k-character stride at *verification* time, keeping `parent_document_id`, `chunk_index`, `chunk_count` — chunks are never re-merged across parents |
| Conversation JSONL | `messages[]` flattened with role prefixes, one record = one conversation |

## 4. Normalisation policy (deliberately conservative)

`normalise_text` performs **only**: CRLF→LF, trailing whitespace removal,
leading/trailing newline trim.

It does **not** lowercase, strip punctuation, collapse repeated spaces
*inside* the stored text, fold diacritics, or convert NFC↔NFD. Vietnamese text
keeps its original Unicode representation, and code keeps indentation,
underscores, newlines and identifiers exactly as written.

The duplicate hash separately normalises whitespace (`[ \t]+`→space,
`\n{2,}`→`\n`) because two documents differing only in spacing are duplicates;
that normalisation is used for *comparison only* and never replaces the stored
text.

## 5. Format support and failure behaviour

| Format | Reader | Dependency |
|---|---|---|
| Parquet | `pyarrow.ParquetFile.iter_batches` (bounded memory) | `pyarrow` (installed) |
| JSONL | line streaming | none |
| JSON | array or `{"documents": [...]}` | none |
| CSV | `csv.DictReader` | none |
| TXT | blank-line streaming | none |

If `pyarrow` is missing, the Parquet reader exits with the exact install
command instead of silently skipping the source. A corrupt shard is logged and
skipped — the run continues and the manifest records what was read.

Text column discovery order: the manifest's `text_columns`, then
`text/content/body/document/raw_content/article`.

## 6. Quality flags measured at ingestion

Flags are recorded per document, never used to delete text at this stage:

| Flag | Count (all sources) | Meaning |
|---|---|---|
| `too_short` (< 200 chars) | 113,768 | mostly `prepared_v4` chunks, which are line-level fragments by construction |
| `high_repetition` (> 35 % duplicate words) | 67,578 | mostly Cosmopedia templates |
| `possible_pii` (email-shaped) | 69 | flagged for review, not redacted |
| `replacement_char` | 35 | U+FFFD present |
| `boilerplate` (cookie/nav/JS-off phrases) | 0 | web junk was already stripped upstream |

`possible_pii` documents are excluded by default in the verification step; the
policy is recorded rather than silently applied.

## 7. Outputs

```
data/curated_v5/
  raw_manifest.json        # every source, license, hash, doc/char counts, policy
  documents/*.jsonl        # one normalised record per line (schema above)
  splits/{train,val,test}.jsonl
  reports/dedup_quality_splits.json
```

`data/curated_v5/` is generated data: it is not committed. The scripts and the
manifest are.

## 8. Known limitations, stated plainly

- **English dominance.** The two large documented sources are both English.
  Vietnamese exists only in `prepared_v4` (≈ 28.5 k fragments, 37 M chars total
  including English) and `project_seed`. A genuinely Vietnamese-heavy mixture
  needs a licensed Vietnamese corpus that is not on this machine; the mixture
  report therefore reports Vietnamese as an *evaluation target*, not as a
  balanced training component.
- **`prepared_v4` is chunk-level, not document-level.** Its mean document is
  250 characters, so `min_chars 100` keeps it while `min_chars 200` would have
  deleted almost all of it — including most of the Vietnamese data.
- **Cosmopedia is synthetic** and its `high_repetition` rate is high (61,918 of
  200,000 docs). It is capped in the mixture rather than trusted.
- Ingestion sampled the first 200,000 Wikipedia shard rows; the remaining
  ~10 GB of the download is untouched and still available if a larger corpus is
  needed.
