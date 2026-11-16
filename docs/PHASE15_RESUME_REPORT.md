# PHASE 15F — EXACT RESUME REPORT

## 1. The defect

`train_gpt2.py` could resume from a checkpoint, but "resume" did not mean "the
same trajectory":

1. The sampler restarted from position 0 of the same shard, so a resumed run
   re-read data it had already trained on.
2. The RNG state of the data order was not part of the checkpoint contract, so
   the second segment saw a different stream.
3. Nothing verified that the resumed run used the same tokenizer, the same
   dataset or the same hyperparameters. Resume with a different `--data-dir` was
   accepted silently, and the resulting loss curve looked plausible while
   describing a different experiment.

The consequence is not a crash: it is a loss curve that looks continuous and is
not, and a "resumed" run whose result cannot be attributed to the original
recipe.

## 2. What is now enforced

`experiment_fingerprint()` hashes everything that defines the trajectory:

- dataset: train/val file SHA-256, token count, curated manifest hash
- tokenizer: kind, vocab size, EOT id, tokenizer hash
- architecture: n_layer, n_head, n_embd, block size, vocab size
- optimisation: batch size, grad-accum, seq len, LR, warmup, min-LR ratio,
  weight decay, dropout, grad clip, seed
- implementation: `SAMPLE_IMPL_VERSION`, `DATA_FRAMING`

`compare_fingerprints()` refuses to resume when any field differs, and the error
names the differing keys. `--extend-budget` is the one intentional exception: it
may change `max_steps` and nothing else.

The data position is restored from `tokens_seen` plus the saved sampler offset,
the RNG state (`torch`, `cuda`, python) is saved and restored, and the log prints
**"LR schedule is NOT the original trajectory"** when the run is extended, so a
reader cannot mistake a second segment for the first.

## 3. Tests (`tests/test_exact_resume.py`, 16 tests, all passing)

| Area | Tests |
|---|---|
| fingerprint stability | identical config → identical hash; key order irrelevant |
| fingerprint sensitivity | data hash, seed, LR, dropout, seq len, batch, accum, tokenizer and impl version each change the hash |
| resume gate | mismatched data, tokenizer, seed, LR each refused with the offending keys named |
| budget extension | `--extend-budget` allows a larger `max_steps`, still refuses a changed dataset |
| determinism | sample order identical for the same seed and `tokens_seen` |
| position | resumed sampler starts where the checkpoint ended, not at 0 |
| logging | extension prints the "not the original trajectory" warning |

## 4. What this does *not* prove

- Bit-exact continuation of floating-point loss on GPU is not guaranteed by any
  of this: cuDNN kernels are not bitwise deterministic across process starts.
  What is guaranteed is the **same data order, same hyperparameters, same
  starting weights** — the experiment, not the last decimals of the loss.
- Checkpoints written before Phase 15 have no fingerprint. They can be loaded for
  evaluation, but cannot be resumed without first recording their fingerprint from
  their own manifest. Phase 14's checkpoint manifest is used for exactly that.
- No resume test in this repository proves anything for a different GPU model or
  a different PyTorch build.
