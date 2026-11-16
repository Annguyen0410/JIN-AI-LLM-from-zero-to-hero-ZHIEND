# DATA QUALITY REPORT — curated_v5

Produced by `scripts/verify_corpus.py` over `data/curated_v5/documents/`
(549,214 ingested documents). Raw report:
`data/curated_v5/reports/dedup_quality_splits.json`.

## 1. Deduplication

| Step | Result |
|---|---|
| Documents in | 549,214 |
| **Exact duplicates removed** (whitespace-normalised SHA-256) | **62,243** (11.3 %) |
| Unique documents | 486,971 |
| Near-duplicate method | MinHash 64 permutations, 5-word shingles, 192 evenly spaced shingles/doc, 8×8 banded LSH, Jaccard ≥ 0.85 |
| LSH candidate pairs | 9,119 |
| Confirmed merges | 337 |
| Clusters formed | 486,634 |
| Additional near-duplicate members removed | 272 |

The exact-duplicate rate (11.3 %) is dominated by Cosmopedia template overlap,
not by copies of the same article. Near-duplicate filtering at 0.85 Jaccard is
deliberately conservative: it removes reformatted copies, and leaves documents
that merely share terminology alone. 272 documents out of 487 k were removed
this way — the honest reading is that this corpus had little near-duplication,
and a more aggressive filter would have deleted legitimate content.

## 2. Quality filtering

| Reason | Documents dropped |
|---|---|
| `too_short` (< 100 characters) | 24,477 |
| `high_repetition` (5-gram duplication > 40 %) | 24 |
| Corrupted encoding / empty | 0 (already excluded at ingestion) |
| Boilerplate | 0 |
| **Clusters kept** | **462,198** |

A correction worth recording: the first pass of this report used
*unique-word ratio* as the repetition test and dropped **222,681** documents —
46 % of the corpus, including most long Wikipedia articles. That test is
length-biased (Zipf: any long natural text repeats words). Replacing it with
5-gram duplication plus compressed-size ratio dropped **24** documents instead.
The first numbers were wrong and were discarded, not published as results.

## 3. Splits (document/cluster-disjoint)

| Split | Chunks | Characters | knowledge | educational | mixed | conversation |
|---|---|---|---|---|---|---|
| train | 417,394 | 1,488.2 M | 784.1 M | 672.5 M | 31.6 M | 0.11 M |
| val | 37,419 | 132.7 M | 69.4 M | 61.0 M | 2.3 M | 0.01 M |
| test | 9,476 | 33.8 M | 17.8 M | 15.3 M | 0.8 M | 0.002 M |

Split ratio 90 / 8 / 2. Splitting is done on **clusters**, and the partition of
a cluster is derived from a SHA-256 of its representative's content hash, so:

- the same corpus always produces the same split (reproducible), and
- a near-duplicate family can never straddle two partitions.

### Leakage verification (measured on the written files)

| Pair | Exact overlap |
|---|---|
| train ↔ val | **0** |
| train ↔ test | **0** |
| val ↔ test | **0** |

The old `data/prepared_v4` audit found 23.2 % duplicate documents in train and a
43.1 % raw val↔train overlap. In `curated_v5` both are zero by construction and
verified after writing, not assumed.

## 4. Language balance — the honest number

| Split | English | Vietnamese | other | unlabelled |
|---|---|---|---|---|
| train | 1,484.3 M chars | **3.87 M chars (0.26 %)** | 0.016 M | 0.002 M |
| val | 132.4 M | 0.30 M (0.23 %) | 0.001 M | 0.0008 M |
| test | 33.7 M | 0.086 M (0.25 %) | 0.0004 M | 0.0001 M |

Vietnamese is **0.26 % of the training text**. Every Vietnamese capability
statement in this phase must be read against that number: there is no licensed,
large Vietnamese corpus on this machine, and inventing one is not an option.
Vietnamese is therefore reported as an *evaluation target with insufficient
training exposure*, not as a balanced component of the mixture.

## 5. Synthetic data share

Cosmopedia v2 is model-generated. It is 45.2 % of the character mass but is
thinned to a smaller share in the built token stream
(`scripts/build_tokenized_corpus.py --mix`, target weights 0.65 Wikipedia /
0.25 Cosmopedia / 0.08 prepared_v4 / 0.02 project seed, applied by deterministic
per-document hashing, not by file order). The actual realised token composition
is recorded in `data/gpt2_corpus_v5/manifest.json` — report the measured value,
never the target.

## 6. What is *not* claimed

- No PII redaction. 69 documents were flagged `possible_pii` (email-shaped
  strings) and were **not** removed; they are visible in the flags for review.
- No language identification beyond a diacritic/ASCII heuristic; 61,959
  `prepared_v4` fragments are `unlabelled` because they are too short or
  code-like for it.
- No quality model. Filtering is rule-based and deliberately weak — it removes
  broken data, not "low-quality" prose it cannot actually judge.
- No claim that 462 k documents are all good. The claim is narrower: they are
  unique, they carry provenance and license, and they are cleanly separated
  across splits.
