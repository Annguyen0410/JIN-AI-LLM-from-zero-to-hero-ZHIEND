# Model card — JIN GPT-2-class pilot checkpoint

> **Status: historical (pre-Phase-14).** The pilot checkpoint still exists at
> `checkpoints/gpt2_small/pilot/` and remains usable, but the current measured
> checkpoint is `checkpoints/gpt2_small/phase14_wiki/best.pt`. Neither one is
> "GPT-2 equivalent": the architecture matches GPT-2 Small, the trained
> capability does not. See `docs/PHASE14_RELEASE_REPORT.md`.

## Identity

| Field | Value |
|---|---|
| Artifact | `checkpoints/gpt2_small/pilot/best.pt` |
| SHA-256 | `bc44d10da2183857f78e5227a480aca1a20af68058e38c297d37dd0b8596ce29` |
| Size | 1,493,464,398 bytes |
| Architecture id | `gpt2_small` (`config.SCALE_PRESETS["gpt2_small_reference"]`) |
| Parameters | **124,439,808** (exact GPT-2 Small layout) |
| Layers / heads / width | 12 / 12 / 768 (head dim 64, 12 KV heads — no GQA) |
| Context | 1,024 tokens |
| FFN | 3,072, GELU(tanh) |
| Positions | learned absolute embeddings |
| Norm / residual | pre-LayerNorm, GPT-2 residual-scaled init (`1/√(2·n_layer)`) |
| Vocabulary | 50,257 (GPT-2 byte-level BPE) |
| Tied | token embedding = LM head |
| Tokenizer | `tokenizer_gpt2.GPT2BPETokenizer` (`gpt2_bpe`), documents as `<|endoftext|> body <|endoftext|>` |

## Training provenance

| Field | Value |
|---|---|
| Iterations | 220 (best at 160) |
| Tokens seen | 16,220,160 (best at 11,796,480) |
| Batch | 4 × grad-accum 18 × 1024 = 73,728 tokens/step |
| Optimizer | AdamW β=(0.9,0.95), wd 0.1 (0 on 1-D), lr 6e-4, warmup 20, cosine to 10 % |
| Precision | fp16 autocast + GradScaler, activation checkpointing |
| Hardware | RTX 2060 6 GB, 5.9 GB peak, ~7.8 k tokens/s |
| Data | `data/gpt2_corpus/train.bin` (SHA-256 `5aaf961a…e306f55685`), 11,990,607 tokens, 142,443 documents |
| Validation | `data/gpt2_corpus/val.bin` (SHA-256 `1826378b…`), 475,032 tokens, document-disjoint |
| Parent checkpoint | none — trained from random init (no GPT-2 weights) |
| Started from real corpus | `data/prepared_v4/pretrain/*.txt` (38 MB, 6 domains) |

## Metrics (measured)

| Metric | Value |
|---|---|
| Val loss, best (step 160) | **3.2337** |
| Val perplexity, best | **25.37** |
| Val loss, final (step 220) | 3.8894 (overfitting after ~1 epoch) |
| Held-out English perplexity | 1,247.87 |
| Task score (model-only probes) | 0.0000 |
| Continuation fluency probes | 4/4 pass (with looping) |

## Intended use

* **Yes**: validating the pretraining pipeline, the tokenizer profile, the
  document-aware packing, checkpoint provenance and the benchmark harness.
* **No**: serving to users, replacing `checkpoints/jingpt_chat.pt`, instruction
  following, factual answering, Vietnamese generation, code generation as a
  product feature. This checkpoint has seen 16 M tokens of 124 M-parameter
  model training — 0.65 % of the Chinchilla budget (20 tokens/param).

## Known limitations

1. Emits `<|endoftext|>` repeatedly on question-style prompts.
2. Repetition loops after ~15 tokens of generation.
3. No instruction-following behaviour (never SFT'd).
4. Vietnamese: trained on part of one epoch only.
5. Factual knowledge: essentially absent (12 M tokens cannot teach facts).
6. Perplexity is not comparable to GPT-2 across tokenizers; the comparison in
   `docs/GPT2_CLASS_BENCHMARK.md` uses identical GPT-2 BPE for both.

## Ethical / licensing notes

Trained only on the in-repo `data/prepared_v4` mixture, whose manifest records
the source policy (`exclude NC and uncertain per-source licenses`) and the
held-out guard hashes. No GPT-2 weights were copied, and no external LLM API
was used to generate any training target.
