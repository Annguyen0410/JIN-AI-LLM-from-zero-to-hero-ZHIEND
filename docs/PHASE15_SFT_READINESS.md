# PHASE 15J — SFT READINESS

**Verdict: `NOT_READY_FOR_SFT`.**

The base model is a small LM trained on ~50 M tokens of English-heavy text. SFT
is a separate decision, and nothing in Phase 15 makes it a sound one yet.

## 1. What exists

| Artifact | Size | Provenance recorded? | Phase 15 gates applied? |
|---|---|---|---|
| `data/prepared_general_v1/sft_train` | 54,091 records, 6 files | yes (`provenance.dataset_id` per record) | **no** |
| `data/prepared_v1..v4/sft_train` | 11,385 / 14,220 / 17,086 / 17,086 | yes | **no** |
| `data/chat_train.txt` | 2.65 M lines, 169 MB | **no per-record provenance** | **no** |

The six files in `prepared_general_v1` are Dolly-15k, GSM8K, MetaMathQA, OASST2,
OpenCodeInstruct and one further set, each record carrying a `messages` list and
a `provenance` block.

## 2. Why this is not enough

1. **No dedup pass.** Phase 15 fixed exact and near-duplicate detection for the
   pretraining corpus. The SFT files have never been through it. OASST2 and Dolly
   contain near-duplicate instructions, and a duplicated instruction in SFT
   teaches the model to memorise one phrasing, not to follow instructions.
2. **No licence decision per SFT source.** `REVIEWED_SOURCES` covers pretraining
   sources only. Each SFT dataset has its own terms (Dolly: CC-BY-SA; OASST2:
   Apache-2.0 with use restrictions; GSM8K: MIT; MetaMathQA: inherited from
   GSM8K/MATH; OpenCodeInstruct: check the card), and none of them has been
   recorded in a reviewed table. Training on them without that record repeats the
   Phase 14 licence defect, one level up.
3. **No PII pass.** OASST2 conversations contain user-written text; the Phase 15
   detector has not been run over them.
4. **No evaluation that could show SFT helped.** With no model-as-judge allowed
   and no human preference study, the only honest measurements available are
   held-out loss and format compliance (does the model emit the required
   `<|system|>…<|end|>` structure, does it stop at the right token). Loss on SFT
   data measures imitation of the data, not helpfulness.
5. **The base model is too small for a fair SFT test.** A ~4-layer / 25 M-parameter
   model fine-tuned on 54 k instructions will produce the format and little else.
   Reporting that as "instruction following" would be a claim without evidence.

## 3. What would make it ready

| Step | Gate |
|---|---|
| 1 | Add every SFT dataset to a reviewed `SFT_SOURCES` table with licence, terms and attribution |
| 2 | Run the Phase 15 dedup + PII pipeline over each SFT file (same script, same audit) |
| 3 | Define a format-compliance metric computed from text alone (structure, termination, refusal phrasing) |
| 4 | Fix a held-out instruction set **before** training and never touch it during development |
| 5 | Train with the same resume/fingerprint contract so the SFT run is reproducible |
| 6 | Report base vs SFT on the held-out set with both metrics, including samples that got worse |

## 4. What is *not* claimed here

No SFT run was performed in Phase 15, so there is no before/after comparison and
no improvement claim. The existing `jingpt_chat*.pt` checkpoints predate this
audit and are treated as unverified artifacts: their training data was never
gated, so their quality cannot be attributed to a controlled recipe.
