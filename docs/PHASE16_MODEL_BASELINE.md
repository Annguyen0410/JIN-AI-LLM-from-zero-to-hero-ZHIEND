# PHASE 16 §1 — MODEL BASELINE (measured from the weights)

## 1. Verified repository state

| Item | Value |
|---|---|
| HEAD at start | `5b15ba4fadd0141af38e2e3989c2f73811a25ed5` |
| Baseline checkpoint | `checkpoints/gpt2_small/phase14_wiki/best.pt` |
| Checkpoint SHA-256 | `069d9da650a81dc9e2c0fee94d0e20dc46a5c4f7e266a6c381a72140f48266fa` |
| File size | 1,493,474,638 bytes (1.49 GB) |
| Iteration | 640 |
| Training tokens seen | 47,185,920 |
| Best val loss recorded in checkpoint | 4.7399 |
| Loads `strict=True` | yes — no missing, no unexpected keys |
| Tokenizer | GPT-2 BPE, vocab 50,257, EOT 50,256, fingerprint `40bbc29de16d5ed2…` |
| Training corpus | `data/gpt2_corpus_v5`, dataset manifest sha256 `a44913e1…` |

## 2. The parameter-count discrepancy — resolved

Three different numbers were in circulation. All three are now explained.

| Source | Claim | Truth |
|---|---|---|
| `config.py` preset `gpt2_small_reference` | 124,439,808 | **correct for the model object** |
| Phase 14 checkpoint manifest | 124.44 M (architecture id `gpt2_small`) | correct model, misleading file |
| Phase 15 reports ("~25 M", "4-layer") | 25 M | **wrong** — corrected in §6 |

Measured by loading the checkpoint:

```
parameters (instantiated model)   124,439,808
parameters (raw state_dict)       163,037,184
  embeddings  39,383,808  (tok_emb 38,597,376 + pos_emb 786,432)
  blocks      85,054,464  (12 × 7,088,064)
  head        38,597,376
  ln_f             1,536
```

**Why the two numbers differ.** The model ties `head.weight` to `tok_emb.weight`
(the preset's 124.44 M assumes tying). The checkpoint nonetheless *stores* a
separate `head.weight` tensor. It is byte-identical to `tok_emb.weight`
(`torch.equal → True`, max abs diff `0.0`), so:

- the model is a genuine **124,439,808-parameter, 12-layer, 768-wide, 12-head,
  1024-context GPT-2-Small-shaped transformer**;
- each checkpoint file carries a redundant 38,597,376-parameter copy
  (≈155 MB of the 1.49 GB), which is storage waste, not extra capacity;
- `load_state_dict(strict=True)` is safe here **because the duplicate is
  identical**. If the two had diverged, loading would have silently replaced the
  embeddings with the head copy — that risk is now documented and is why this
  check exists.

Architecture per the checkpoint config: `modern=false`, GELU-tanh, pre-LN,
learned positional embeddings, full MHA (12 KV heads), no RoPE, no QK-norm, no
MoE, dropout 0.0 at training time.

## 3. Measured inference performance (RTX 2060, fp32, greedy, no KV cache)

From `bench_results/phase16/model_baseline.json`:

| Metric | Value |
|---|---|
| Time to first token | 0.012 s |
| End-to-end for 32 new tokens | 0.284 s (mean of 3 runs) |
| Throughput | **112.6 tokens/s** |
| Peak VRAM (inference) | **498 MB** |
| Checkpoint load time (from disk) | ≈ 5 s |
| Model storage | 1.49 GB checkpoint (0.50 GB weights + 0.99 GB optimizer state) |

The generation loop recomputes the full context for every token (no KV cache in
this path), which is why throughput is 113 tok/s and not several hundred. That is
the path the chat application uses today, so it is the number that matters for
latency claims.

## 4. What the checkpoint is *not*

- Not a 25 M-parameter model. Phase 15's generation and SFT-readiness reports said
  so; they were wrong, and the correction is recorded in §6.
- Not a 163 M-parameter model either — that is the file, not the network.
- Not instruction-tuned. It is a document continuator trained on 47.2 M tokens.
- Not a model whose 47 M tokens can be compared to GPT-2's ~100 B. The clean
  comparison (§Phase 15) puts it 1.78 nats / 5.9× perplexity behind GPT-2 at
  equal protocol.

## 5. Resource envelope for Phase 16

| Resource | Measured |
|---|---|
| Inference VRAM | 0.5 GB of 6 GB |
| Training VRAM (Phase 14 recipe, seq 1024, batch 4 × accum 18) | 4.4 GB |
| GPU idle temperature | 54–62 °C |
| GPU temperature at 99 % utilisation | 81–83 °C |

Phase 16 therefore does not start a large pretraining run: the base model is
already trained to the limit of a sensible budget on this hardware, and the
capability question is about tasks and instruction tuning, not about another 50 M
tokens.

## 6. Corrections published against earlier reports

| File | Wrong statement | Correction |
|---|---|---|
| `docs/PHASE15_GENERATION_REPORT.md` §5 | "A ~25 M-parameter transformer trained on 47 M tokens" | 124,439,808 parameters |
| `docs/PHASE15_SFT_READINESS.md` §2.5 | "A ~4-layer / 25 M-parameter model" | 12 layers, 124.44 M parameters |
| `docs/PHASE15_RELEASE_REPORT.md` §2 (implied scale) | — | points to this document |

No result in Phase 15 changes: every Phase 15 number was measured from the same
checkpoint, and the mis-stated size never entered a metric.

## 7. Reproduce

```bash
python scripts/phase16_inspect_model.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
    --out bench_results/phase16/model_baseline.json
```

The script prints both parameter counts, whether the head is tied in the file, the
tokenizer fingerprint, iteration, tokens seen, and measured latency/throughput/VRAM.
