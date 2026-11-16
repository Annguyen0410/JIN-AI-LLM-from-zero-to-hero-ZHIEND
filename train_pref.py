"""
Preference tuning — DPO (default) and ORPO, implemented from scratch
(Phase 3.6). No external RL libraries.

Data format (JSONL, one record per line):

    {"prompt": "...", "chosen": "...", "rejected": "..."}

The prompt is the user turn; chosen/rejected are assistant answers. We
build (prompt, chosen) and (prompt, rejected) sequences with the ChatML
template and the model's tokenizer.

DPO (Rafailov et al. 2023)
--------------------------
    L = -log sigma( beta * ( log p_theta(chosen|prompt)
                            - log p_ref(chosen|prompt)
                            - log p_theta(rejected|prompt)
                            + log p_ref(rejected|prompt) ) )

``p_ref`` is a frozen reference model (typically the SFT checkpoint).
``beta`` controls how hard the model is pushed toward chosen.

ORPO (Hong et al. 2024)
-----------------------
No reference model. Loss = NLL(chosen) + lambda * log-odds ratio:

    L = -log p_theta(chosen) - lambda * log( sigma(
            log p_theta(chosen) - log p_theta(rejected) ) )

Saves the reference model entirely — great for 6GB VRAM.

Usage:
    python train_pref.py --data data/pref.jsonl --method dpo \\
        --checkpoint checkpoints/jingpt_chat.pt \\
        --reference checkpoints/jingpt_chat.pt --steps 200
    python train_pref.py --data data/pref.jsonl --method orpo \\
        --checkpoint checkpoints/jingpt_chat.pt --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

import config
from bpe_tokenizer import BPETokenizer
from chat_template import ROLE_ASSISTANT, ROLE_USER, format_conversation
from utils import (
    build_inference_bundle,
    configure_cuda_performance,
    count_parameters,
    create_model,
    load_checkpoint_file,
    save_checkpoint,
    tokenizer_from_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO / ORPO preference tuning (Phase 3.6)")
    p.add_argument("--data", required=True, help="JSONL with prompt/chosen/rejected")
    p.add_argument("--method", choices=("dpo", "orpo"), default="dpo")
    p.add_argument("--checkpoint", default=config.CHECKPOINT_PATH,
                   help="Base model (DPO: becomes policy; ORPO: trained model)")
    p.add_argument("--reference", default=None,
                   help="DPO reference model (default: same as --checkpoint)")
    p.add_argument("--out", default=os.path.join(config.CHECKPOINT_DIR, "jingpt_pref.pt"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    p.add_argument("--lambda-orpo", type=float, default=0.1, help="ORPO odds weight")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", default=None, choices=config.DEVICE_CHOICES)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def build_sequences(
    tokenizer: BPETokenizer, prompt: str, answer: str, max_length: int
) -> list[int]:
    """ChatML sequence: <|user|> prompt <|end|> <|assistant|> answer."""
    text = format_conversation(
        [
            {"role": ROLE_USER, "content": prompt},
            {"role": ROLE_ASSISTANT, "content": answer},
        ],
        add_generation_prompt=False,
        default_system=None,
    )
    ids = tokenizer.encode(text, allow_special=True)
    return ids[:max_length]


def seq_logprobs(model, ids: list[int], device: str) -> torch.Tensor:
    """Mean logprob over the full sequence (teacher-forced)."""
    if len(ids) < 2:
        return torch.zeros((), device=device)
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    gathered = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return gathered.mean()


def load_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("prompt") and rec.get("chosen") and rec.get("rejected"):
                pairs.append(rec)
    return pairs


def get_lr(step: int, max_steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    decay = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay)))
    return peak * max(0.1, coeff)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args()
    args.out = config.output_path(args.out)
    config.set_device(args.device)
    device = config.DEVICE
    torch.manual_seed(args.seed)
    configure_cuda_performance(device)

    pairs = load_pairs(args.data)
    if not pairs:
        raise SystemExit(f"No valid prompt/chosen/rejected records in {args.data}")
    print(f"[Pref ] {len(pairs):,} preference pairs | method={args.method}")

    model, tokenizer, ckpt = build_inference_bundle(args.checkpoint)
    model = model.to(device)
    model.train()
    print(f"[Pref ] Policy: {count_parameters(model)/1e6:.2f}M params")

    # Reference model: frozen copy of the policy (DPO only).
    ref_model = None
    if args.method == "dpo":
        ref_path = args.reference or args.checkpoint
        if ref_path == args.checkpoint:
            ref_model = create_model(
                vocab_size=tokenizer.vocab_size, checkpoint=ckpt, device=device
            )
            ref_model.load_state_dict(ckpt["model_state_dict"])
        else:
            ref_ckpt = load_checkpoint_file(ref_path, map_location="cpu")
            ref_model = create_model(
                vocab_size=tokenizer.vocab_size, checkpoint=ref_ckpt, device=device
            )
            ref_model.load_state_dict(ref_ckpt["model_state_dict"])
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
        print(f"[Pref ] Reference frozen (DPO).")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    n_batches = len(pairs) // args.batch_size
    if n_batches == 0:
        raise SystemExit("Not enough pairs for one batch — lower --batch-size.")

    print(f"[Pref ] Steps {args.steps} | batch {args.batch_size}x{args.grad_accum} | "
          f"lr {args.lr} | beta {args.beta} | max_len {args.max_length}")

    step = 0
    t0 = time.time()
    while step < args.steps:
        for b in range(n_batches):
            if step >= args.steps:
                break
            batch = pairs[b * args.batch_size : (b + 1) * args.batch_size]
            lr_now = get_lr(step, args.steps, args.lr, args.warmup)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            optim.zero_grad(set_to_none=True)
            loss_total = 0.0
            for pair in batch:
                chosen_ids = build_sequences(tokenizer, pair["prompt"], pair["chosen"], args.max_length)
                rejected_ids = build_sequences(tokenizer, pair["prompt"], pair["rejected"], args.max_length)

                with torch.amp.autocast(device_type="cuda" if use_amp else "cpu",
                                        dtype=torch.float16 if use_amp else torch.float32,
                                        enabled=use_amp):
                    logp_chosen = seq_logprobs(model, chosen_ids, device)
                    logp_rejected = seq_logprobs(model, rejected_ids, device)

                    if args.method == "dpo":
                        with torch.no_grad():
                            ref_chosen = seq_logprobs(ref_model, chosen_ids, device)
                            ref_rejected = seq_logprobs(ref_model, rejected_ids, device)
                        # log-ratio: policy-vs-ref for chosen minus rejected.
                        log_ratio = (logp_chosen - ref_chosen) - (logp_rejected - ref_rejected)
                        loss = -F.logsigmoid(args.beta * log_ratio)
                    else:  # ORPO
                        nll = -logp_chosen
                        odds_ratio = F.logsigmoid(logp_chosen - logp_rejected)
                        loss = nll - args.lambda_orpo * odds_ratio

                loss = loss / len(batch) / args.grad_accum
                (scaler.scale(loss) if use_amp else loss).backward()
                loss_total += float(loss.item())

            if use_amp:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            (scaler.step(optim) if use_amp else optim.step())
            if use_amp:
                scaler.update()

            step += 1
            if step % 25 == 0 or step == args.steps:
                elapsed = time.time() - t0
                print(f"[Pref ] step {step:>4}/{args.steps} | loss {loss_total:6.4f} | "
                      f"lr {lr_now:.2e} | {elapsed:5.1f}s")

    model.eval()
    save_checkpoint(model, optim, step, best_loss=loss_total, path=args.out,
                    tokenizer=tokenizer, scaler=scaler, train_loss=loss_total)
    print(f"[Pref ] Saved -> {args.out}")


if __name__ == "__main__":
    main()