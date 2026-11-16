# GPT-2-class training guide (Phase 4 / 6 / 11)

How to pretrain JIN as a GPT-2-Small-shaped model, what each hyperparameter is
for, what it costs on the hardware in this repository, and how to resume an
interrupted run without corrupting the schedule.

Everything here is executed by `train_gpt2.py` (new) and uses the architecture
added in `config.SCALE_PRESETS["gpt2_small_reference"]`.

---

## 1. Presets

| Preset | Params (measured) | Shape | Tokenizer | Notes |
|---|---:|---|---|---|
| `gpt2_small_reference` | **124,439,808** | 12L / 12H / 768d / 3072 FFN / 1024 ctx | GPT-2 BPE 50,257 | exact GPT-2 Small layout; `modern=False`, `gelu_tanh`, `resid_scaled_init` |
| `jin_small_modern` | ~143 M | 12L / 16H / 992d / 8 KV (GQA) | JIN BPE 16,384 | existing JIN profile, unchanged |
| `tiny` | ~35 M | 6L / 6H / 384d / 512 ctx | either | laptop smoke tests |

`gpt2_small_reference` is the profile this guide trains. It is *from scratch*:
no GPT-2 weights are ever loaded into it (that is only done in the isolated
benchmark process, see `docs/GPT2_CLASS_BENCHMARK.md`).

## 2. Build the corpus

```bash
.venv/Scripts/python.exe train_gpt2.py --build-data
```

Document rules:

* Documents are separated by blank lines and encoded as
  `<|endoftext|> body <|endoftext|>`.
* Training and validation come from **different files** (`*_val.txt`), so no
  document can appear on both sides. The project's own
  `data/prepared_v4/verification.json` reports 0 overlapping exact hashes.
* Tokens are written as a uint16 memmap (`data/gpt2_corpus/train.bin`) — constant
  RAM regardless of corpus size.
* `data/gpt2_corpus/manifest.json` records document counts, token counts,
  chars/token, source files and the SHA-256 of each shard.

Measured on this machine:

```
train.bin : 11,990,607 tokens | 142,443 documents | 2.98 chars/token | 22.9 MB
val.bin   :    475,032 tokens |   6,140 documents
```

## 3. Recipe

| Knob | Value | Why |
|---|---|---|
| Optimizer | AdamW, β=(0.9, 0.95), eps 1e-8 | standard GPT-2-class baseline |
| Weight decay | 0.1 on matrices, 0.0 on norms/biases | decay on 1-D params hurts |
| Peak LR | 6e-4 | GPT-2-scale runs use 2.5e-4 at 32k batch; 6e-4 is correct for a 65–75k-token batch |
| Schedule | linear warmup → cosine to 10 % | absolute step budget, so resume continues the same curve |
| Grad clip | 1.0 | prevents the fp16 spikes seen on 6 GB cards |
| Precision | fp16 autocast + GradScaler | bf16 is not offered by this GPU generation |
| Activation checkpointing | on (forced below 9 GB VRAM) | 5.9 GB peak vs. OOM without it |
| Attention mask | document-aware per block | packed sequences never attend across `<|endoftext|>` |
| Dropout | 0.0 | 1 document epoch budget makes dropout harm, not help |

```bash
# one full run (this is the command to use for a serious run)
.venv/Scripts/python.exe train_gpt2.py \
    --batch-size 4 --grad-accum 18 --seq-len 1024 \
    --lr 6e-4 --warmup 2000 --max-steps 8100 \
    --eval-interval 250 --save-interval 500
```

`4 × 18 × 1024 = 73,728` tokens per optimizer step.
`8100 × 73,728 ≈ 597 M` tokens ≈ 4.8 passes over the current 12.5 M-token corpus.

```bash
# resume exactly where it stopped
.venv/Scripts/python.exe train_gpt2.py --resume
```

## 4. Cost on this hardware (measured)

Measured on RTX 2060 6 GB with `--batch-size 4 --grad-accum 18 --seq-len 1024`:

| Metric | Value |
|---|---|
| Throughput | **≈ 9,350 tokens/s** |
| VRAM (device) | 5.9 GB / 6.1 GB |
| Tokens per step | 73,728 |
| Seconds per step | ≈ 7.9 |
| Parameters | 124.44 M |

| Token budget | Wall-clock on this GPU | Meaning |
|---|---:|---|
| 12.5 M (1 epoch of current corpus) | 22 min | smoke/pilot |
| 100 M | 3.0 h | first visibly coherent continuations |
| 500 M | 14.9 h | first serious milestone |
| 2.49 B (Chinchilla 20×) | **74 h** | what "GPT-2 class" actually costs |

The corpus is the binding constraint, not the GPU: 12.5 M tokens is 0.5 % of
the Chinchilla budget for this model size. A 500 M-token run needs
`data/downloads/general_v1` (17 GB) converted into the same clean,
deduplicated, document-segmented stream — that conversion is Phase 3 work and
is **not** part of this pass.

## 5. What a checkpoint contains (reproducibility)

`checkpoints/gpt2_small/pilot/latest.pt` plus `latest.pt.manifest.json`:

* model weights, optimizer state, GradScaler state, CPU + CUDA RNG state
* `iteration`, `tokens_seen`, `best_val_loss`
* `model_config` (including `arch_id`, `gelu_tanh`, `resid_scaled_init`)
* dataset manifest (shard SHA-256, document/token counts, sources)
* tokenizer identity and vocab size
* hardware (GPU name, VRAM, torch version)
* training arguments and a bounded loss history
* manifest sidecar with the checkpoint's own SHA-256

Resume refuses to continue if the corpus SHA-256 changed — silently training on
different data under the same step counter is exactly how "we trained 1 B
tokens" claims become false.

## 6. Model tests that gate a promotion

Before any checkpoint replaces a served one:

```bash
.venv/Scripts/python.exe -m pytest tests/test_gpt2_reference.py tests/test_tokenizer_upgrade.py -q
.venv/Scripts/python.exe scripts/gpt2_bench.py --model jin --checkpoint <candidate>
```

Release gates (`docs/GPT2_CLASS_BENCHMARK.md`): no NaN/Inf, checkpoint reloads,
tokenizer identity unchanged, held-out loss not worse than the current served
model, no benchmark contamination, and task score not regressed.

## 7. Not done in this pass (explicit)

* No 500 M / 2.5 B-token run. 74 h of GPU time is not something to start
  silently inside an engineering task; the recipe, the data pipeline and the
  measured throughput above are what a scheduled run needs.
* No SFT on top of the new checkpoint: SFT of a model trained on 12.5 M tokens
  would mostly teach the chat template to a model that has not learned
  language yet (directive §Phase 6: SFT after the pretraining baseline).
* No preference optimisation (Phase 7) — no verified preference pairs.
* No new tokenizer vocabulary (see `docs/TOKENIZER_RESEARCH.md`).
