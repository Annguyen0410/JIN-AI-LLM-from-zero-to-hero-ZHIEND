"""Phase 16 §6 — LoRA instruction tuning of the Phase 14 checkpoint.

Design choices, stated because they change what the result means:

* **LoRA, not full fine-tuning.** The base checkpoint is never modified; the
  adapter is a separate artifact that can be merged or discarded.
* **Loss on assistant turns only.** The prompt is rendered with the chat template
  and its labels are set to -100, so the model is trained to *answer*, not to
  reproduce the question.
* **One conversation per step, truncated to `--seq-len`.** No padding enters the
  loss, and no conversation boundary is crossed.
* **Deterministic:** fixed seed, fixed data order from the frozen SFT split.

Usage
-----
    python scripts/phase16_sft_train.py --steps 3000 --out checkpoints/phase16_sft/lora_r16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chat_template import EOT_TAG, format_conversation   # noqa: E402
from lora import freeze_base, trainable_parameters, wrap_lora   # noqa: E402
from tokenizer_gpt2 import GPT2BPETokenizer            # noqa: E402
from utils import _torch_load, create_model            # noqa: E402

IGNORE = -100


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def encode_example(tokenizer, messages: list[dict], seq_len: int) -> tuple[list[int], list[int]] | None:
    """Render a conversation; labels are -100 on the prompt, real ids on replies."""
    full = format_conversation(messages, add_generation_prompt=False)
    prompt_messages = messages[:-1]
    prompt_text = format_conversation(prompt_messages, add_generation_prompt=True)

    ids = tokenizer.encode(full + EOT_TAG)
    prompt_ids = tokenizer.encode(prompt_text)
    if len(ids) <= len(prompt_ids) + 1:
        return None
    ids = ids[:seq_len + 1]
    labels = [IGNORE] * min(len(prompt_ids), len(ids)) + ids[len(prompt_ids):]
    labels = (labels + [IGNORE] * len(ids))[:len(ids)]
    if all(l == IGNORE for l in labels):
        return None
    return ids, labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 16 LoRA SFT")
    ap.add_argument("--checkpoint", default=os.path.join(
        ROOT, "checkpoints", "gpt2_small", "phase14_wiki", "best.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "phase16_sft",
                                                   "train.jsonl"))
    ap.add_argument("--val", default=os.path.join(ROOT, "data", "phase16_sft",
                                                  "val.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "phase16_sft",
                                                  "lora_r16"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = GPT2BPETokenizer()

    blob = _torch_load(args.checkpoint, map_location="cpu")
    model = create_model(int(blob.get("tokenizer", {}).get("vocab_size", 50257)),
                         checkpoint=blob, device=args.device)
    model.load_state_dict(blob["model_state_dict"], strict=True)

    adapters = wrap_lora(model, rank=args.rank, alpha=float(args.rank),
                         target_names=("qkv_proj", "out_proj", "0", "2"))
    for ad in adapters:                      # LoRA params are created on CPU
        ad.to(args.device)
    freeze_base(model)
    params = trainable_parameters(model)
    adapter_params = sum(p.numel() for p in params)
    base_params = sum(p.numel() for p in model.parameters())
    if not adapters:
        raise SystemExit("no LoRA target modules were wrapped — refusing to train")

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device == "cuda")

    records = load_records(args.data)
    val_records = load_records(args.val)
    rng = random.Random(args.seed)

    def batches(records: list[dict]):
        order = list(range(len(records)))
        rng.shuffle(order)
        i = 0
        while True:
            yield records[order[i % len(order)]]
            i += 1

    stream = batches(records)
    model.train()
    t0 = time.perf_counter()
    running = 0.0
    running_n = 0
    history = []
    tokens_seen = 0
    skipped = 0

    for step in range(1, args.steps + 1):
        lr_scale = (step / max(1, args.warmup) if step <= args.warmup else
                    0.5 * (1 + math.cos(math.pi * (step - args.warmup) /
                                        max(1, args.steps - args.warmup))))
        for group in opt.param_groups:
            group["lr"] = args.lr * lr_scale

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        for _ in range(args.accum):
            rec = next(stream)
            encoded = encode_example(tokenizer, rec["messages"], args.seq_len)
            if encoded is None:
                skipped += 1
                continue
            ids, labels = encoded
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=args.device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=args.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=args.device == "cuda"):
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    y.reshape(-1), ignore_index=IGNORE, reduction="sum")
                n_targets = int((y != IGNORE).sum())
                if n_targets == 0:
                    continue
                loss = loss / n_targets
            scaler.scale(loss / args.accum).backward()
            step_loss += float(loss.item())
            step_tokens += n_targets
        if step_tokens == 0:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        tokens_seen += step_tokens
        running += step_loss
        running_n += 1

        if step % 50 == 0:
            print(f"step {step:5d}/{args.steps} | loss {step_loss / args.accum:.4f} | "
                  f"lr {args.lr * lr_scale:.2e} | {tokens_seen:,} tok | "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

        if args.eval_every and step % args.eval_every == 0:
            val_loss, val_tokens = evaluate(model, tokenizer, val_records, args,
                                            limit=150)
            history.append({"step": step, "train_loss": round(step_loss / args.accum, 4),
                            "val_loss": round(val_loss, 4), "val_tokens": val_tokens})
            print(f"  [val] step {step} loss {val_loss:.4f} on {val_tokens:,} targets",
                  flush=True)

    os.makedirs(args.out, exist_ok=True)
    state = {name.replace(".", "_"): p.detach().cpu()
             for name, p in model.named_parameters() if ".A" in name or ".B" in name}
    torch.save({"adapter_state_dict": state,
                "rank": args.rank, "alpha": float(args.rank),
                "target_modules": ["qkv_proj", "out_proj", "ffn.net.0", "ffn.net.2"],
                "base_checkpoint": os.path.relpath(args.checkpoint, ROOT),
                "base_sha256": blob.get("sha256") or sha256_file(args.checkpoint),
                "data_sha256": sha256_file(args.data),
                "steps": args.steps, "seed": args.seed,
                "tokens_trained": tokens_seen,
                "history": history,
                "elapsed_seconds": round(time.perf_counter() - t0, 1)},
               os.path.join(args.out, "adapter.pt"))
    manifest = {
        "adapter": os.path.relpath(os.path.join(args.out, "adapter.pt"), ROOT),
        "adapter_sha256": sha256_file(os.path.join(args.out, "adapter.pt")),
        "base_checkpoint": os.path.relpath(args.checkpoint, ROOT),
        "base_sha256": sha256_file(args.checkpoint),
        "architecture_parameters": base_params,
        "adapter_parameters": adapter_params,
        "adapter_share_of_model": round(adapter_params / base_params, 4),
        "config": vars(args),
        "tokens_trained": tokens_seen,
        "examples_skipped": skipped,
        "train_examples": len(records),
        "val_examples": len(val_records),
        "history": history,
        "peak_vram_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                         if args.device == "cuda" else None),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {os.path.relpath(args.out, ROOT)} | adapter params {adapter_params:,} "
          f"| tokens {tokens_seen:,} | peak VRAM {manifest['peak_vram_mb']} MB")


@torch.no_grad()
def evaluate(model, tokenizer, records: list[dict], args, limit: int = 150) -> tuple[float, int]:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for rec in records[:limit]:
        encoded = encode_example(tokenizer, rec["messages"], args.seq_len)
        if encoded is None:
            continue
        ids, labels = encoded
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=args.device)
        y = torch.tensor([labels[1:]], dtype=torch.long, device=args.device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=IGNORE, reduction="sum")
        total_nll += float(loss.item())
        total_tokens += int((y != IGNORE).sum())
    model.train()
    return (total_nll / max(1, total_tokens)), total_tokens


if __name__ == "__main__":
    main()
