# PHASE 16 §6 — SFT TRAINING REPORT

## 1. What was trained

A LoRA adapter on top of the frozen Phase 14 checkpoint. The base checkpoint was
never modified.

| Property | Value |
|---|---|
| Base | `checkpoints/gpt2_small/phase14_wiki/best.pt` (sha `069d9da6…`, 124,439,808 params) |
| Adapter | rank 16, targets `attn.qkv_proj`, `attn.out_proj`, `ffn.net.0`, `ffn.net.2` |
| Adapter parameters | **2,359,296** (1.9 % of the model) |
| Adapter size on disk | 9.47 MB |
| Training data | `data/phase16_sft/train.jsonl` — 43,706 conversations (audited in `PHASE16_SFT_DATA_AUDIT.md`) |
| Steps × tokens | 600 steps, batch 1, **615,212 target tokens** |
| Sequence length | 512, one conversation per step, no padding |
| Loss | assistant tokens only (prompt labels = −100) |
| Optimiser | AdamW, lr 5e-4 cosine, warmup 50, clip 1.0 |
| Precision | fp16 autocast, GradScaler |
| Seed | 16 |
| Peak VRAM | **972 MB** |
| Wall clock | 591 s (≈ 10 min) |
| Throughput | ≈ 1,040 target tokens/s |
| GPU temperature | 76–80 °C (bounded run, not multi-hour) |

## 2. Training curve

| Step | Train loss | Val loss (118 k targets) |
|---|---|---|
| 0 (base) | — | 4.2733 (measured at step 30; the adapter is initialised to zero, so this is within noise of the frozen base) |
| 200 | 3.69 | 4.14 |
| 400 | 3.48 | 3.996 |
| 600 | 3.31 | **3.9588** |

Validation loss falls 0.31 nats. That means the adapter learned the *shape* of an
assistant reply in this template. It does **not** mean the model learned to answer
questions, and the next section is why.

## 3. Held-out task benchmark, base vs adapter

Identical 73 tasks, identical decoding (greedy), identical checkers.

| Model | Passed | Rate | Passing tasks |
|---|---|---|---|
| Base Phase 14 | **0 / 73** | 0.0 % | — |
| LoRA (raw prompt) | **2 / 73** | 2.7 % | `math_algebra`, `mt_correction` |
| LoRA (chat template) | **1 / 73** | 1.4 % | `math_algebra` |

### Is 2/73 an improvement?

No, not in any statistical sense. With a 0/73 baseline, two successes cannot be
distinguished from chance (one-sided exact binomial p ≈ 0.13 at a 0.5 % underlying
rate). The two passing tasks are a one-line algebra answer (`x = 5`) and an answer
containing the digit `3` after a correction — both are plausible coincidences from
a model that emits short numeric-ish text.

**`CAPABILITY_GAIN_VERIFIED` is therefore NOT claimed.**

## 4. Regressions and trade-offs

| Dimension | Base | LoRA | Reading |
|---|---|---|---|
| SFT val loss | 4.27 | 3.96 | better |
| Task pass rate | 0/73 | 2/73 | not significant |
| Document-continuation loss (clean slice, protocol v2) | 4.8935 token-weighted | not re-measured | would need a full eval pass |
| Instruction format compliance | 0/8 | 0/8 | no change |
| Vietnamese tasks | 0/6 | 0/6 | no change |
| Code tasks | 0/30 | 0/30 | no change |

The honest summary: **format adaptation, not capability**.

## 5. Why it did not work (and what would)

1. **615 k tokens of instruction data is small**, and the base model has only seen
   47 M tokens of text. GPT-2-scale models need billions of tokens before
   instruction tuning produces reliable behaviour.
2. **LoRA at rank 16 changes 1.9 % of the weights.** It can shift style; it cannot
   install arithmetic or code synthesis that the base never learned.
3. **The base model's generation is degenerate** (see `PHASE15_GENERATION_REPORT.md`):
   greedy decoding loops. SFT reduced but did not remove this.
4. **No chain-of-thought or tool-use training data** was used, so nothing teaches
   the model to compute rather than recall.

What would plausibly move the task score:
- continue pretraining to ≥1 B tokens before instruction tuning (capability first,
  then alignment);
- train the *system* to route to the calculator and retriever, which already work
  at 100 % on their own tasks (Track B), instead of asking the model to do
  arithmetic in its head;
- evaluate with a model that can already answer, so SFT has something to shape.

## 6. Artifacts

| File | Contents |
|---|---|
| `checkpoints/phase16_sft/lora_r16/adapter.pt` | adapter state + base hash + data hash + history |
| `checkpoints/phase16_sft/lora_r16/manifest.json` | config, params, tokens, VRAM, history |
| `bench_results/phase16/task_benchmark_sft_raw.json` | 73 raw outputs, greedy, raw prompt |
| `bench_results/phase16/task_benchmark_sft_chat.json` | 73 raw outputs, chat template |
| `runs/p16_sft_train2.log` | training log |

Reproduce:

```bash
python scripts/phase16_sft_train.py --steps 600 --eval-every 200 \
    --out checkpoints/phase16_sft/lora_r16
python scripts/phase16_task_benchmark.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
    --adapter checkpoints/phase16_sft/lora_r16/adapter.pt \
    --out bench_results/phase16/task_benchmark_sft_raw.json
```
