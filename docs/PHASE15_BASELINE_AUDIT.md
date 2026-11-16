# PHASE 15A — BASELINE AUDIT OF PHASE 14

Scope: verify every Phase 14 claim against the implementation and the artifacts on
disk, then separate what is confirmed from what was only documented.

Starting revision: `64165e640d9c9b07009be0da067f2f6ef62b6309` — **confirmed** as
`git rev-parse HEAD`, working tree clean at audit start.

---

## 1. Confirmed facts (independently recomputed)

| Claim | Source | Result |
|---|---|---|
| `phase14_wiki/best.pt` sha256 starts `069d9da650a81dc9e2c0` | `docs/PHASE14_RELEASE_REPORT.md` §11 | **confirmed** by `sha256sum` |
| `pilot/best.pt` sha256 starts `bc44d10da2183857` | same | **confirmed** |
| Train tokens 51,222,642 | `data/gpt2_corpus_v5/manifest.json` | **confirmed** (exact count in manifest) |
| Train bin sha256 `a44913e192991a66…` | manifest | **confirmed** (recorded, used by resume guard) |
| Val tokens 28,969,659, sha `9eafe2dc4c554e23…` | manifest | **confirmed** |
| Tokenizer `gpt2_bpe`, vocab 50,257, state hash `40bbc29de16d5ed2…` | manifest, checkpoint | **confirmed** |
| Published mixture 67.1 % / 21.2 % / 11.6 % / 0.06 % | manifest `token_composition` | **confirmed** |
| Vietnamese 3.03 % of train tokens (1,553,188) | manifest `language_composition` | **confirmed** |
| Hardware RTX 2060 6.1 GB, 4.4 GB peak, 7,964–9,991 tok/s | training log | **confirmed** from `runs/phase14_main.log` |
| Best val 4.7399 at step 640, 47,185,920 tokens | checkpoint manifest | **confirmed** |
| "51 tests pass" | Phase 14 report | **reproduced**: 51 passed in the six named suites |

## 2. Reproduced results

- **Exact resume.** `tests/test_exact_resume.py` still passes (16 tests after
  Phase 15F). Continuous vs save-resume weights match at atol 1e-6.
- **FP32 validation.** `evaluate()` casts logits to FP32; the eval artifacts
  carry `loss_dtype: fp32_cross_entropy` and `tokens_scored`.
- **Tokenizer round-trip.** Re-running `scripts/tokenizer_eval.py` reproduces
  exact round-trip on all five samples for both tokenizers.
- **Evaluation numbers.** `eval_pilot_v5.json` and `eval_phase14_wiki.json`
  reproduce their recorded losses when the script is re-run on the same
  checkpoints and val file (same 64 windows, same seed).

## 3. Supported only by existing artifacts (not re-derived here)

- Generation samples in `bench_results/phase14/generation.json` are tied to the
  checkpoint hash in the same file, but were not regenerated token-for-token.
- The GPT-2 reference number (3.2467) was computed once, in a scratch script
  whose output is `gpt2_small_v5_reference.json`; the script itself was not
  committed at the time (now available as `scripts/eval_phase15.py --hf-model`).

## 4. Unverified claims

- **"Zero exact duplicates in the output."** Phase 14 verified overlap *between*
  splits using sets. It never checked duplicates *within* a split. See §5.
- **PII.** `docs/DATA_INGESTION_V2.md` §6 stated that `possible_pii` documents
  are "excluded by default in the verification step". The implementation did not
  read that flag. The statement was false; it is corrected in
  `docs/PHASE15_PRIVACY_REPORT.md`.
- **Commercial/redistribution permission.** `_license_record()` set
  `commercial_ok = True` whenever the license string was non-empty. That is an
  inference, not a review. Replaced by an explicit policy table.

## 5. Newly discovered defects

### D1 — Exact duplicates were written to the output (CRITICAL)

Independent audit of the written files (content-hash counting, not internal
counters):

| Split | Records | Unique hashes | Duplicated hashes | Extra records |
|---|---|---|---|---|
| train | 417,394 | 415,497 | 1,422 | **1,897** |
| val | 37,419 | 37,266 | 135 | **153** |
| test | 9,476 | 9,435 | 37 | **41** |

Root cause: pass 1 mapped a hash to the first document index and counted later
copies as "removed"; pass 2 then looked the hash up for **every** occurrence, so
each copy resolved to the retained index and was written again. The reported
"exact duplicates removed 62,243" was true of the counter and false of the
output. Fixed in `scripts/verify_corpus.py` (a `written` hash set gates every
write, plus a post-write audit that fails the run). Regression tests:
`tests/test_corpus_dedup.py` (9 tests).

Effect on Phase 14 results: 1,897 duplicate records (≈0.45 % of the train split)
were re-read during training. Small, but it invalidates the "unique documents"
claim and slightly inflates the effective repetition of a few documents. The
trained checkpoint is **not** invalidated — nothing needs retraining to keep the
reported loss honest — but the number is corrected here.

### D2 — Padding contaminated per-domain evaluation

`per_group_loss()` padded short documents with EOT and `nll_windows()` scored the
whole padded tensor, so artificial tokens entered domain/language metrics. The
reported per-domain numbers (knowledge 5.10, educational 4.78, mixed 4.30, vi
4.08) are therefore approximate. Fixed by document-independent scoring with no
padding (`scripts/eval_phase15.py`), tested in
`tests/test_evaluation_integrity.py`.

### D3 — Evaluation sampled by file order

`per_group_loss()` took the first 24 matching documents, and the val file is
grouped by source, so a group could be dominated by one upstream source. Replaced
by deterministic stratum sampling on (source, language, domain), ordered by
`sha256(seed.document_id)`, with selected ids recorded.

### D4 — Evaluation ignored document boundaries

Training masks attention across document boundaries; the evaluator packed windows
without any mask. The new protocol scores each document independently with its own
EOT framing, so no window can mix two documents — the boundary policy is now the
same for JIN and the GPT-2 reference.

### D5 — Resume accepted configuration changes

The guard checked corpus hashes, architecture id, tokenizer kind/vocab size and
step budget only: learning rate, warmup, batch geometry, dropout, seed, sampler
version and precision could change silently. Replaced by a full experiment
fingerprint (Phase 15F), with an explicit, logged budget extension.

### D6 — Pre-existing test failures, unrelated to Phase 14/15

- `tests/test_check_project.py` — importing `test_group` by name made pytest
  collect it as a test ("fixture 'test_id' not found"). **Fixed**: the function is
  now reached through the module.
- `tests/test_checkpoint_picker.py::TestModelCheckpointsEndpoint::test_nested_instruct_tune_is_listed_and_wins_the_auto_pick`
  fails on the starting commit: `app.py` and `config.py` are **unchanged** by
  Phase 14/15 (`git diff HEAD -- app.py config.py` is empty), and the test mocks
  the checkpoint directory. It is an application auto-pick logic issue, reported
  as an open defect, not fixed here.

## 6. What Phase 14 got right

The parts that survive audit unchanged: FP32 validation, exact resume of the
sampler RNG, corpus provenance recording, cluster-level splitting, the corrected
token audit, and the pretraining improvement itself (−1.84 nats on identical
windows). The defects above are implementation gaps in verification, not
fabricated results.

## 7. Consequences adopted for Phase 15

1. Rebuild the corpus as **v6** with corrected dedup, explicit license policy and
   enforced PII policy. v5 stays untouched on disk.
2. Re-run every evaluation with the document-independent evaluator before making
   any new comparison claim.
3. Re-run the pilot-vs-new comparison on a **clean, source-aware slice** that
   neither checkpoint trained on (Phase 15E).
4. Treat all Phase 14 per-domain numbers as superseded by the v6 numbers.
