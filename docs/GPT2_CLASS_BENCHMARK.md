# Model-only benchmark: JIN pilot vs. GPT-2 Small (Phase 9)

Harness: `scripts/gpt2_bench.py` · Probes: `benchmarks/gpt2_probes.jsonl` (37 probes)
Artifacts: `bench_results/gpt2_upgrade/{base_model,gpt2_reference,comparison}.json`

**Nothing in this benchmark uses retrieval, memory, web search, tools or RAG.**
It is the neural network alone: tokenizer → transformer → LM head → greedy
decoding. Prompts, held-out text, prompt-set SHA-256, checkpoint SHA-256 and
tokenizer identity are recorded in every artifact.

---

## 1. Headline

| | JIN pilot (`gpt2_small_reference`) | GPT-2 Small (reference) |
|---|---:|---:|
| Task score (answer probes) | **0.0000** (0/32) | **0.2424** (8/33) |
| Held-out perplexity (English) | **1,247.87** | **50.19** |
| Held-out loss (English) | 7.1292 | 3.9158 |
| Params | 124,439,808 | 124,439,808 |
| Training tokens seen | 16.2 M | ~40 B (upstream, not comparable) |
| Checkpoint SHA-256 | `bc44d10d…8596ce29` | `gpt2` (HF revision) |

Relative: task score **0 %** of GPT-2's; perplexity **24.9×** worse.

## 2. Per category

| Category | JIN pilot | GPT-2 Small |
|---|---:|---:|
| Coding | 0.00 | 0.75 |
| Continuation (fluency-scored) | **1.00** | 0.75 |
| English grammar/lexical | 0.00 | 0.25 |
| Factual recall | 0.00 | 0.00 |
| Instruction following | 0.00 | 0.50 |
| Math | 0.00 | 0.17 |
| Reasoning | 0.00 | 0.25 |
| Vietnamese | 0.00 | 0.00 |

Two results deserve emphasis because they are *not* flattering to either model:

* **Factual 0/7 for GPT-2.** A raw base model has no instruction/QA format in
  its weights; it continues text instead of answering. This benchmark measures
  base-model capability, and it shows that the FAQ-style probes need SFT to be
  meaningful for both models. It is not evidence that GPT-2 "cannot" know the
  capital of France.
* **Vietnamese 0/4 for both.** GPT-2 was never trained on Vietnamese. The JIN
  pilot saw Vietnamese for ~1/6 of one epoch — nowhere near enough.

## 3. What the JIN pilot did learn

Real continuations from the pilot checkpoint (greedy, 24 new tokens):

```
"The morning sun rose over the hills and"
  -> " the little girl. The little girl was very happy.\nThe little girl was very happy. She was ..."

"Once upon a time there was a small robot who"
  -> " lived a little girl named Lily. She loved to play with her mom. She loved to play with he ..."
```

That is English-looking, locally coherent, on-topic text — a model that has
learned syntax and lexical structure from 12 M tokens — followed immediately by
looping. On question-style prompts the same checkpoint emits `<|endoftext|>`
repeatedly: it treats "Question: … Answer:" as the end of a document. This is
exactly what a 12 M-token model should look like, and it is why the verdict is
`TRAINING PIPELINE VERIFIED`, not `GPT-2-CLASS PROGRESS`.

## 4. Two measurement bugs this pass fixed

Both would have produced confident, false numbers.

1. **fp16 cross-entropy is unusable at vocab 50k.** `GPT2LMHeadModel(x, labels=y)`
   under fp16 returned **11.6719** on a sequence where fp32 cross-entropy on the
   *same logits* returned **0.8336**. The benchmark therefore computes loss as
   `F.cross_entropy(logits.float(), labels)`. Every perplexity in this document
   comes from the fp32 path. (A first run of this benchmark, using the built-in
   loss, reported GPT-2 perplexity 9,660 — fiction.)
2. **A held-out file that mixes languages needs per-language loss.** A single
   average over English + Vietnamese blocks made an English-only model look
   broken; the metric is now split by language (`per_language` in the JSON).

## 5. Protocol (reproducible)

```bash
# JIN checkpoint only
python scripts/gpt2_bench.py --model jin --checkpoint checkpoints/gpt2_small/pilot/best.pt

# GPT-2 SMALL reference (isolated: weights never touch JIN)
python scripts/gpt2_bench.py --model gpt2

# comparison from the two stored results (no GPU work)
python scripts/gpt2_bench.py --merge
```

* Decoding: greedy (`temperature=0`), 24 new tokens, identical prompts.
* Answer probes: pass iff an accepted string appears in the first 120
  characters of the reply. Continuation probes: fluency rule (≥5 words,
  distinct-2 ≥ 0.55, no word repeated more than max(3, n/4) times).
* Held-out: `data/eval/heldout.txt` (SHA-256 in every artifact), 8 windows of
  1024 tokens per language, loss averaged per window.
* GPT-2 is loaded **only** in this benchmark process. No GPT-2 weight is ever
  copied into, or used to initialise, a JIN model.

## 6. Acceptance targets vs. reality

| Directive target | Result |
|---|---|
| ≥ 80 % of GPT-2's aggregate task score | **0 %** |
| Validation perplexity ≤ 1.25× the reference | **24.9×** |
| Repeatable improvement | pretraining curve is repeatable (2 runs, same seed/loss), but the model itself is not close |

Neither target is met, and no benchmark difficulty was reduced to meet them.
The gap is dominated by training data volume (see
`docs/GPT2_CLASS_UPGRADE_AUDIT.md` §8) and, secondarily, by the pilot's small
step budget.
