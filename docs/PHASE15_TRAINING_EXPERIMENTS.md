# PHASE 15H — TRAINING EXPERIMENTS

## 1. Design

All runs use the same model, tokenizer, recipe and step budget. The **only**
differences are the ones under test.

| Setting | Value |
|---|---|
| Architecture | `gpt2_small_reference` preset (JIN GPT-2 small) |
| Sequence length | 512 |
| Batch × grad-accum | 4 × 8 = 16,384 tokens/step |
| Steps | 250 → **4.10 M tokens** per run |
| LR / warmup | 6e-4, 30 steps, cosine, min-LR ratio 0.1 |
| Dropout | 0.0 |
| Eval | every 50 steps, 4 iterations on the val split |
| Output | a new directory per run; nothing overwritten |
| Seed | 1337 / 2024 / 42 (Experiment A), 1337 (Experiment B) |

Corpora (built from the verified v6 splits, `data/gpt2_corpus_v6*`):

| Corpus | Realised train composition (tokens) | Vietnamese |
|---|---|---|
| `v6a` (A) | wiki 66.0 %, cosmo 20.6 %, prepared_v4 13.1 %, seed 0.3 % | 3.24 % |
| `v6b1` (B, human-heavy) | wiki 80.7 %, prepared_v4 16.4 %, seed 0.4 %, cosmo 2.5 % | 3.91 % |
| `v6b2` (B, synthetic) | wiki 55.2 %, cosmo 39.7 %, prepared_v4 4.6 %, seed 0.5 % | 1.89 % |

Token counts are the realised counts from each `manifest.json`, not the targets.

## 2. Experiment A — seed variance (identical data, identical recipe)

| Seed | Best val loss | Perplexity |
|---|---|---|
| 1337 | 6.5590 | 705.6 |
| 2024 | 6.3708 | 584.5 |
| 42 | 6.5227 | 682.7 |
| **mean ± sd** | **6.4842 ± 0.0956** | — |
| **range** | **0.1882** | — |

**This is the most important measurement in Phase 15.** After 4.1 M tokens, two
runs that differ only in the random seed are 0.19 nats apart. Any experiment that
compares two configurations with a single run each, and reports a difference
smaller than ~0.2 nats, is reporting noise.

## 3. Experiment B — data mixture (one seed each)

| Run | Mixture | Best val loss | Inside A's seed range? |
|---|---|---|---|
| B1 | human-heavy (80.7 % wiki) | 6.4874 | yes |
| B2 | synthetic-heavy (39.7 % cosmo) | 6.5565 | yes |

Both mixtures land **inside the seed range** (6.3708 – 6.5590). The spread
between mixtures (0.069) is smaller than the spread between seeds (0.188).

**Conclusion, stated narrowly:** at this budget the mixture choice is not
detectably better or worse than seed noise, on this validation set. That is a
negative result, and it is reported as one. It does **not** show that mixtures do
not matter — it shows this experiment lacks the statistical power to detect a
0.07-nat mixture effect with one seed per mixture. A powered version needs ≥ 3
seeds per mixture (≈ 2 h of GPU for 3 mixtures on this machine).

## 4. Experiment C — not run

The plan included a 100 M-token continuation (~3.5 h of continuous GPU load). It
was **not run**: the machine is the user's daily driver, GPU temperature reached
83 °C during the 8-minute runs, and the directive explicitly asks not to run
multi-hour jobs. No result is claimed for it, and no number in this report depends
on it.

What would be needed to make the claim such a run is supposed to support: a
100 M-token run per variant, 2–3 variants, and an eval protocol identical to
§5 — i.e. ~10 GPU-hours, not a single afternoon.

## 5. All checkpoints produced

| Path | Val loss | Val ppl | Tokens seen |
|---|---|---|---|
| `checkpoints/phase15/A_seed1337/best.pt` | 6.5590 | 705.6 | 4.10 M |
| `checkpoints/phase15/A_seed2024/best.pt` | 6.3708 | 584.5 | 4.10 M |
| `checkpoints/phase15/A_seed42/best.pt` | 6.5227 | 682.7 | 4.10 M |
| `checkpoints/phase15/B_humanheavy/best.pt` | 6.4874 | 654.9 | 4.10 M |
| `checkpoints/phase15/B_synthetic/best.pt` | 6.5565 | 702.7 | 4.10 M |

Every run wrote `best.pt.manifest.json` + `latest.pt.manifest.json` containing the
experiment fingerprint (data hashes, tokenizer hash, hyperparameters, impl
version). None of these checkpoints replaces `phase14_wiki`, which remains the
project's strongest model at 50 M tokens.

## 6. Honest reading

- 4.1 M tokens is 8 % of what Phase 14 trained on. These runs are diagnostic, not
  a new flagship model, and their absolute loss is not comparable to Phase 14's.
- The purpose was to measure variance and mixture sensitivity under a controlled
  budget — that is what the numbers above do.
- A checkpoint that wins by 0.05 nats here would not be a better model; it would
  be a luckier seed.
