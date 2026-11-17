# Phase 17 — Checkpoint Lineage

How a Phase 17 run proves which model it started from, and why `--init-from` is
not `--resume`.

## Two operations that must not be confused

| | `--resume` / `--resume-from` | `--init-from` |
|---|---|---|
| purpose | continue the **same** experiment | start a **new** experiment from a parent model |
| model weights | from the checkpoint | from the checkpoint |
| optimizer / scaler | restored | **new** (AdamW built fresh) |
| LR schedule | restored; `total_steps` must match | new recipe, step 1 |
| sampler / RNG | restored | reset for the new corpus |
| step counter | continues | starts at 0 |
| corpus | must match the fingerprint | any corpus the new run declares |
| allowed together | — | **no**, mutually exclusive |

`--init-from` with `--resume`/`--resume-from` is refused, as is
`--continue-pretrain` without `--init-from`.

`--resume-from PATH` always takes precedence over `--resume`. Earlier code chose
`<out-dir>/latest.pt` whenever `--resume` was also present; on a fresh Kaggle
working directory that file does not exist, the restore block was skipped, and
the run trained from scratch while describing itself as a resume. The resolution
is now `resolve_restore_path()`: a requested resume with no checkpoint is a hard
error, never a silent fresh start.

## Implementation

`train_gpt2.py::init_from_checkpoint(model, tokenizer, path)`:

1. `torch.load(..., weights_only=True, mmap=True)` — no pickle execution.
2. Hard failures, each naming the offending field:
   * `architecture_id` mismatch;
   * tokenizer kind or vocabulary mismatch;
   * `block_size` / `embed_dim` / `n_layers` / `vocab_size` mismatch;
   * `pos_emb.weight` shape mismatch;
   * any missing or unexpected key (`load_state_dict(..., strict=True)` is itself
     wrapped so a key mismatch is a clear refusal, not a stack trace).
3. Re-ties `head.weight` to `tok_emb.weight` if the loader untied them.
4. Returns lineage: `parent_checkpoint`, `parent_sha256`, `parent_iteration`,
   `parent_tokens_seen`, `init_mode="continued_pretrain"`.

`validate_continued_args()` owns the argument guards so they are testable without
building a 124 M model.

## What a saved checkpoint now records

`latest.pt.manifest.json` (and `best.pt.manifest.json`) contain, in addition to
the Phase 14–16 fields:

```
init_mode                 "continued_pretrain" | "resumed" | "random"
parent_checkpoint         path of the parent weights
parent_sha256             SHA-256 of the parent file
parent_iteration          parent's global step
parent_tokens_seen        parent's tokens
additional_tokens_this_run  tokens processed by THIS run only
experiment_config_hash    fingerprint (unchanged by session stops)
schedule                  the global schedule, including total_steps
```

A reader can therefore answer "which model is this, and how much did it see?"
from the artifact alone, without trusting a report.

## Evidence that it works (measured, not asserted)

Smoke run, `checkpoints/phase17_smoke/`, from the Phase 14 parent:

| step | val loss | ppl |
|---|---|---|
| 6 | 4.7155 | 111.66 |
| 12 | 4.7876 | 120.01 |

A randomly initialized model of this architecture scores about
ln(50,257) ≈ **10.82** (ppl ≈ 50,000) at step 0. The first evaluation is 4.72, so
the weights were transferred. The manifest of that run records:

```
init_mode              continued_pretrain
parent_checkpoint      checkpoints/gpt2_small/phase14_wiki/best.pt
parent_sha256          069d9da650a81dc9e2c0fee94d0e20dc46a5c4f7e266a6c381a72140f48266fa
parent_iteration       640
parent_tokens_seen     47185920
additional_tokens_this_run  196608
```

The tiny rise from 4.7155 → 4.7876 over six steps at lr 1e-4 on a different
corpus is expected at this scale and is not evidence of anything except that
training happened. It is not compared with Phase 14's 4.8935: different evaluator
run, different sample, not comparable.

## How to verify lineage yourself

```bash
python - <<'PY'
import json
m = json.load(open("checkpoints/phase17_smoke/latest.pt.manifest.json"))
print(m["init_mode"], m["parent_sha256"], m["additional_tokens_this_run"])
PY
sha256sum checkpoints/gpt2_small/phase14_wiki/best.pt
```

The two SHA-256 values must match. If `init_mode` is `random`, the run was not a
continuation regardless of what it is called.

## Guard rules carried forward

* A missing parent checkpoint is a fatal error, never a fallback to random init.
* Context length stays 1024; the parent's positional embeddings are the reason
  `block_size` is checked at all.
* A larger global budget is a **new** documented experiment with a new parent
  SHA, not `--extend-budget` in the middle of one.
