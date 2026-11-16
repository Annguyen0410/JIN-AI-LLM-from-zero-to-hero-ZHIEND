"""
JIN AI unified training script.

Supports two modes:
- ``pretrain``: standard next-token prediction on raw text.
- ``instruct``: instruction tuning with per-token loss masking
  (chat data, assistant-token gradient only).

Examples:
    python train.py                          # pretrain (default)
    python train.py --mode instruct          # instruction tuning
    python train.py --mode instruct --steps 5000
    python train.py --resume
    python train.py --device cpu
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, random_split

import config
from bpe_tokenizer import BPETokenizer
from chat_dataset import build_dataset as build_chat_dataset
from chat_template import format_conversation, ROLE_USER, ROLE_SYSTEM
from utils import (
    AnyTokenizer,
    build_inference_bundle,
    configure_cuda_performance,
    count_parameters,
    create_model,
    get_batch,
    load_checkpoint,
    load_checkpoint_file,
    load_or_create_token_data,
    load_training_text,
    save_checkpoint,
    split_train_val,
    tokenizer_from_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train JIN AI (pretrain or instruct)")
    p.add_argument("--mode", choices=("pretrain", "instruct"), default="pretrain",
                   help="Training mode (default: pretrain)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Where to save/load the latest checkpoint")
    p.add_argument("--resume", action="store_true",
                   help="Resume optimizer, scaler, RNG state, and step from --checkpoint")
    p.add_argument("--run-iters", type=int, default=config.RUN_ITERS,
                   help="How many iterations to add in this run")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Optional absolute target iteration")
    p.add_argument("--depth", type=int, default=None,
                   help="pretrain only: auto-scale the architecture from a single depth "
                        "(config.py --describe --depth N to preview)")
    p.add_argument("--extend-context", type=int, default=None,
                   help="pretrain only: RoPE-scale the context window to N tokens and "
                        "fine-tune on long blocks (e.g. 4096)")
    p.add_argument("--rope-scale-type", choices=("ntk", "yarn"), default="ntk",
                   help="scaling scheme for --extend-context")
    p.add_argument("--rope-scale-factor", type=float, default=4.0,
                   help="scaling factor for --extend-context (target_ctx / original_ctx)")
    p.add_argument("--data", type=str, default=config.DATA_PATH,
                   help="Training corpus path (pretrain mode)")
    p.add_argument("--mixture", type=str, default=None,
                   help="Curriculum mixing spec (data/mixture.json). Overrides "
                        "--data with per-domain weighted sampling.")
    p.add_argument("--device", type=str, default=None, metavar="NAME",
                   choices=config.DEVICE_CHOICES,
                   help="Device override (CUDA/MPS/CPU)")
    # Instruct-specific
    p.add_argument("--init-from", default=None,
                   help="Optional pretrained checkpoint to warm-start instruct tuning")
    p.add_argument("--duplication", type=int, default=40,
                   help="Each chat seed dialogue is duplicated this many times")
    p.add_argument("--hf-seed-dir",
                   default=os.path.join(config.ROOT_DIR, "data", "seed", "hf"),
                   help="Directory of HF chat shards (*.jsonl) used for instruct "
                        "tuning. Pass an empty string to disable.")
    p.add_argument("--sft-val-dir", default=None,
                   help="Explicit disjoint JSONL validation directory; overrides --val-fraction")
    p.add_argument("--max-convs", type=int, default=30_000,
                   help="Cap total HF conversations for instruct tuning "
                        "(0 = no cap). Seeds are always kept in full.")
    p.add_argument("--grounded-dir",
                   default=os.path.join(config.ROOT_DIR, "data", "seed", "grounded"),
                   help="Directory of grounded web-search SFT shards — loaded "
                        "uncapped (teaches Jin to answer from injected context). "
                        "Pass an empty string to disable.")
    p.add_argument("--fast", action="store_true",
                   help="Low-VRAM profile for instruct tuning")
    p.add_argument("--block-size", type=int, default=512,
                   help="Training sequence block size (both modes)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Micro-batch size. Pretrain honors this too (sets config.BATCH_SIZE)")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation. Pretrain honors this too (sets config.GRAD_ACCUM)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--steps", type=int, default=4_000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--no-pretrained-init", action="store_true",
                   help="Never load jingpt_best.pt for instruct tuning")
    p.add_argument("--seed", type=int, default=config.SEED,
                   help="RNG seed for dataset split and weight init (instruct)")
    p.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw",
                   help="optimizer: muon = momentum + orthogonalization for 2D "
                        "hidden matrices (1 momentum buffer instead of Adam's 2)")
    p.add_argument("--low-vram", action="store_true",
                   help="low-VRAM profile: gradient checkpointing + smaller blocks "
                        "(selective: attention-only, every-2nd layer on GPU)")
    p.add_argument("--pack", action="store_true",
                   help="pretrain: pack sequences with a real document mask so "
                        "attention never crosses <|endoftext|> boundaries")
    return p.parse_args()


def resolve_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if config.AMP_DTYPE == "bfloat16" else torch.float16


def get_lr(step: int, max_iters: int, peak: float, min_ratio: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    decay_ratio = (step - warmup) / max(1, max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_ratio)))
    floor = peak * min_ratio
    return floor + coeff * (peak - floor)


def build_tokenizer(raw_text: str, resume_ckpt: dict | None) -> AnyTokenizer:
    if resume_ckpt and resume_ckpt.get("tokenizer_state"):
        tok = tokenizer_from_state(resume_ckpt["tokenizer_state"])
        print(f"[Tokenizer] Restored from checkpoint ({'BPE' if isinstance(tok, BPETokenizer) else 'char'} "
              f"vocab size={tok.vocab_size:,})")
        return tok
    kind = (config.TOKENIZER_KIND or "char").lower()
    if kind == "bpe":
        tok = BPETokenizer()
        if os.path.exists(config.BPE_PATH):
            tok.load(config.BPE_PATH)
            print(f"[Tokenizer] Loaded BPE from {config.BPE_PATH} (size={tok.vocab_size:,})")
        elif raw_text:
            # Fresh build: include the Phase-3 reasoning/tool tokens
            # (<|think|>, <|/think|>, <|tool|>) at the tail of the vocab.
            from bpe_tokenizer import default_special_tokens
            tok = BPETokenizer(default_special_tokens(extended=True))
            print(f"[Tokenizer] Training BPE (vocab_size={config.BPE_VOCAB_SIZE}, "
                  f"extended special tokens)")
            tok.train(raw_text, vocab_size=config.BPE_VOCAB_SIZE)
            tok.save(config.BPE_PATH)
    else:
        from tokenizer import CharTokenizer
        tok = CharTokenizer()
        if os.path.exists(config.VOCAB_PATH):
            tok.load()
        elif raw_text:
            tok.build(raw_text)
    return tok


# ── Pretrain helpers ────────────────────────────────────────────────────

def estimate_loss_pretrain(model, train_data, val_data, use_amp, amp_dtype):
    model.eval()
    losses = {}
    autocast_device = "cuda" if use_amp else "cpu"
    with torch.no_grad():
        for split_name, split_data in (("train", train_data), ("val", val_data)):
            split_losses = []
            for _ in range(config.EVAL_ITERS):
                x, y = get_batch(split_data, block_size=config.TRAIN_BLOCK_SIZE,
                                 batch_size=config.BATCH_SIZE, device=config.DEVICE)
                with torch.amp.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                split_losses.append(float(loss.item()))
            losses[split_name] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def sample_from_model(model, tokenizer: AnyTokenizer):
    print("\n" + "=" * 72)
    print("Sample generation")
    print("=" * 72)
    for prompt in config.SAMPLE_PROMPTS:
        ids = tokenizer.encode(prompt)
        if not ids:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=config.DEVICE)
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=160, temperature=config.TEMPERATURE,
                                    top_k=config.TOP_K, top_p=config.TOP_P,
                                    repetition_penalty=config.REPETITION_PENALTY)
        print(f"\n>>> {prompt}")
        print(tokenizer.decode(output[0].tolist()))


def _infer_vocab_from_checkpoint(ckpt: dict) -> int | None:
    mc = ckpt.get("model_config") or {}
    if mc.get("vocab_size") is not None:
        return int(mc["vocab_size"])
    sd = ckpt.get("model_state_dict") or {}
    w = sd.get("tok_emb.weight")
    return int(w.shape[0]) if w is not None else None


def _pretrained_compatible(ckpt: dict, tokenizer) -> tuple[bool, str]:
    v_ckpt = _infer_vocab_from_checkpoint(ckpt)
    v_now = tokenizer.vocab_size
    if v_ckpt is None:
        return False, "no vocab_size in checkpoint"
    if v_ckpt != v_now:
        return False, f"vocab mismatch ({v_ckpt} vs {v_now})"
    if ckpt.get("tokenizer_state"):
        from lineage import object_hash
        if object_hash(ckpt["tokenizer_state"]) != object_hash(tokenizer.to_state_dict()):
            return False, "tokenizer hashes differ despite equal vocabulary sizes"
    return True, "vocab and tokenizer match"


# ── Instruct helpers ─────────────────────────────────────────────────────

def _adapt_microbatch_for_vram(device: str, args: argparse.Namespace) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        return
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gb >= 11:
        return
    floor_bs = 4 if gb >= 8 else 1
    if args.batch_size <= floor_bs:
        return
    old_bs, old_ga = args.batch_size, args.grad_accum
    while args.batch_size > floor_bs:
        args.batch_size //= 2
        args.grad_accum *= 2
    print(f"[Train] VRAM ~{gb:.1f} GiB — batch {old_bs}->{args.batch_size}, "
          f"grad-accum {old_ga}->{args.grad_accum}")


def _apply_fast_profile(args: argparse.Namespace) -> None:
    if not args.fast:
        return
    args.block_size = min(args.block_size, 256)
    args.duplication = min(args.duplication, 8)
    args.eval_iters = min(args.eval_iters, 5)
    print(f"[Train] Fast profile: block-size={args.block_size}, "
          f"duplication={args.duplication}, eval-iters={args.eval_iters}")


@torch.no_grad()
def estimate_loss_instruct(model, loader, device, n_batches, *, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, y, m = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y, loss_mask=m)
        count = int(m.sum().item())
        if count:
            losses.append((float(loss.item()), count))
    model.train()
    if not losses:
        raise ValueError("Validation has no supervised tokens")
    return sum(loss * count for loss, count in losses) / sum(count for _, count in losses)


@torch.no_grad()
def _quick_chat_sample(model, tokenizer, device, prompt="Hello!"):
    from chat_template import EOT_TAG, END_TAG
    text = format_conversation(
        [{"role": ROLE_SYSTEM, "content": "You are Jin, a friendly small AI."},
         {"role": ROLE_USER, "content": prompt}],
        add_generation_prompt=True, default_system=None,
    )
    ids = tokenizer.encode(text, allow_special=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stop_ids = set()
    for tag in (END_TAG, EOT_TAG):
        tid = tokenizer.special_to_id.get(tag)
        if tid is not None:
            stop_ids.add(tid)
    out = model.generate(idx, max_new_tokens=80, temperature=0.6, top_k=40,
                          top_p=0.9, repetition_penalty=1.15, stop_ids=stop_ids)
    new = out[0].tolist()[len(ids):]
    if new and new[-1] in stop_ids:
        new = new[:-1]
    reply = tokenizer.decode(new).strip()
    print(f"[sample] you: {prompt}")
    print(f"[sample] jin: {reply or '(empty)'}")


# ── Main training dispatch ──────────────────────────────────────────────

def make_optimizer(model, args: argparse.Namespace, lr: float = config.LR):
    """AdamW (default) or Muon (2D hidden matrices orthogonalized)."""
    if args.optimizer == "muon":
        from optim_muon import build_muon_optimizer
        print(f"[Optim] Muon (lr={lr}) — 1 momentum buffer for matrices, AdamW elsewhere")
        return build_muon_optimizer(model, lr=lr, betas=(config.BETA1, config.BETA2),
                                    weight_decay=config.WEIGHT_DECAY)
    return torch.optim.AdamW(model.parameters(), lr=lr,
                             betas=(config.BETA1, config.BETA2),
                             weight_decay=config.WEIGHT_DECAY)


def build_override_config(args: argparse.Namespace, vocab_size: int) -> dict:
    """Assemble the model-config overrides for --depth / --extend-context."""
    override: dict = {}
    if args.depth:
        derived = config.derive_config_from_depth(args.depth, vocab_size)
        override.update({
            "embed_dim": derived["embed_dim"],
            "n_layers": derived["n_layers"],
            "n_heads": derived["n_heads"],
            "n_kv_heads": derived["n_kv_heads"],
            "block_size": derived["block_size"],
            "qk_norm": True,          # v3 default for new models
        })
        print(f"[Config] --depth {args.depth} -> {derived['embed_dim']}d, {derived['n_layers']}L, "
              f"{derived['n_heads']}H, {derived['n_kv_heads']}KV, block {derived['block_size']}, "
              f"lr {derived['lr']}")
    if args.extend_context:
        override.update({
            "rope_scaling": {
                "type": args.rope_scale_type,
                "factor": args.rope_scale_factor,
                "target_ctx": args.extend_context,
            },
        })
        print(f"[Config] --extend-context {args.extend_context} via {args.rope_scale_type} "
              f"(factor {args.rope_scale_factor})")
    if getattr(args, "low_vram", False):
        # Selective activation checkpointing: recompute every 2nd layer
        # (half the activation memory at ~10% throughput cost).
        override["gradient_checkpointing"] = "layer2"
        print("[Config] --low-vram -> selective gradient checkpointing (every 2nd layer)")
    return override


def train_pretrain(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    config.BATCH_SIZE = max(1, args.batch_size)
    config.GRAD_ACCUM = max(1, args.grad_accum)
    config.TRAIN_BLOCK_SIZE = max(16, args.block_size)
    config.LR = args.lr
    config.EVAL_INTERVAL = args.eval_interval
    config.EVAL_ITERS = args.eval_iters
    config.CHECKPOINT_INTERVAL = args.checkpoint_interval
    ckpt_arg = args.checkpoint or config.LATEST_CHECKPOINT_PATH
    best_ckpt = (ckpt_arg[:-3] + "_best.pt") if args.checkpoint else config.BEST_CHECKPOINT_PATH

    resume_ckpt = None
    if args.resume:
        resume_ckpt = load_checkpoint_file(ckpt_arg, map_location="cpu")

    need_raw_text = (
        resume_ckpt is None
        and not os.path.exists(config.BPE_PATH)
        and (config.TOKENIZER_KIND or "char").lower() == "bpe"
    ) or (
        resume_ckpt is None
        and not os.path.exists(config.VOCAB_PATH)
        and (config.TOKENIZER_KIND or "char").lower() != "bpe"
    )
    if need_raw_text:
        if os.path.getsize(args.data) > 20_000_000:
            from clean_corpus import read_text_prefix
            raw_text = read_text_prefix(args.data, 20_000_000)
            print(f"[Data] Loaded 20.0M char prefix (of {os.path.getsize(args.data)/1e6:.0f} MB) "
                  f"for tokenizer building")
        else:
            raw_text = load_training_text(args.data)
            print(f"[Data] Loaded {len(raw_text):,} characters from {args.data}")
    else:
        raw_text = None
    tokenizer = build_tokenizer(raw_text, resume_ckpt)

    override_config = build_override_config(args, tokenizer.vocab_size)
    if args.depth and resume_ckpt is not None:
        raise ValueError("--depth changes weight shapes; do not combine with --resume")
    # Context scaling and activation checkpointing preserve parameter shapes:
    # retain pretrained weights rather than silently discarding learned knowledge.

    mixture = None
    if args.mixture:
        from utils import MixtureSampler
        mixture = MixtureSampler(
            tokenizer, mixture_path=args.mixture,
            block_size=config.TRAIN_BLOCK_SIZE, batch_size=config.BATCH_SIZE,
            device=config.DEVICE, strict=True,
        )
        print(f"[Mix ] Curriculum mixture: {sorted(mixture.domains)}")

    from utils import load_or_create_token_data_from_file

    if mixture is not None and mixture.validation_path:
        validation_path = mixture.validation_path
        if not os.path.isabs(validation_path):
            validation_path = os.path.join(config.ROOT_DIR, validation_path)
        val_data, _, _ = load_or_create_token_data_from_file(validation_path, tokenizer)
        train_data = next(iter(mixture.domains.values()))
        print(f"[Data] Mixture uses explicit document-disjoint validation: {validation_path}")
    else:
        data, skipped, from_cache = load_or_create_token_data_from_file(args.data, tokenizer)
        if skipped:
            print(f"[Data] Skipped {skipped:,} chars not in vocab")
        print(f"[Data] Encoded into {len(data):,} tokens ({'cached' if from_cache else 'fresh'})")
        train_data, val_data = split_train_val(data)
    print(f"[Data] Train: {len(train_data):,} | Val: {len(val_data):,} tokens")

    model_raw = create_model(vocab_size=tokenizer.vocab_size, checkpoint=resume_ckpt,
                             device=config.DEVICE, override_config=override_config)
    print(f"[Model] {count_parameters(model_raw)/1e6:.2f}M params")

    optimizer = make_optimizer(model_raw, args, lr=config.LR)
    use_amp = config.DEVICE == "cuda"
    amp_dtype = resolve_amp_dtype()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_step = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = load_checkpoint(model_raw, optimizer=optimizer, scaler=scaler, path=ckpt_arg, restore_rng=True)
        start_step = int(ckpt.get("iteration", 0))
        best_loss = float(ckpt.get("best_loss", ckpt.get("loss", float("inf"))))
        print(f"[Resume] Continuing from iteration {start_step:,}")

    target_iters = args.max_iters or (start_step + args.run_iters)
    if target_iters <= start_step:
        target_iters = start_step + max(1, args.run_iters)

    model = model_raw
    if config.COMPILE_MODEL and hasattr(torch, "compile") and config.DEVICE == "cuda":
        model = torch.compile(model_raw)
        print("[Model] torch.compile enabled")

    # Long-context adaptation (--extend-context) samples full-length blocks;
    # the depth dial also carries its own peak LR / warmup (nanochat-style).
    train_block = min(args.extend_context or config.TRAIN_BLOCK_SIZE, len(train_data) - 2)
    peak_lr = config.LR
    warmup_iters = config.WARMUP_ITERS
    if args.depth:
        derived = config.derive_config_from_depth(args.depth, tokenizer.vocab_size)
        peak_lr = derived["lr"]
        warmup_iters = derived["warmup"]

    # Honor --batch-size / --grad-accum in PRETRAIN as well: the loop below
    # reads config.BATCH_SIZE / config.GRAD_ACCUM directly, so without this
    # the CLI flags were silently ignored (pretrain always ran 1x2).
    if not args.mixture:
        config.BATCH_SIZE = max(1, args.batch_size)
        config.GRAD_ACCUM = max(1, args.grad_accum)

    print(f"[Train] Mode: pretrain | batch {config.BATCH_SIZE}x{config.GRAD_ACCUM} "
          f"| block {train_block} | start {start_step:,} target {target_iters:,} | lr {peak_lr}")

    from lineage import start_run, attach_run, record_batch
    dataset_paths = [args.data]
    if args.mixture:
        import json
        with open(args.mixture, encoding="utf-8") as handle:
            mixture_spec = json.load(handle)
        dataset_paths = [args.mixture] + [p if os.path.isabs(p) else os.path.join(config.ROOT_DIR,p)
                                         for p in mixture_spec.get("domains", {}).values()]
        if mixture_spec.get("validation_path"):
            dataset_paths.append(mixture_spec["validation_path"])
    run = start_run(model_raw, tokenizer, vars(args), dataset_paths,
                    ckpt_arg if args.resume else None, ckpt_arg)
    attach_run(model_raw, run)
    t0 = time.time()
    model.train()
    for step in range(start_step + 1, target_iters + 1):
        lr = get_lr(step, target_iters, peak_lr, config.MIN_LR_RATIO, warmup_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(config.GRAD_ACCUM):
            record_batch(model_raw, config.BATCH_SIZE * train_block)
            if mixture is not None:
                progress = min(1.0, (step - start_step) / max(1, target_iters - start_step))
                x, y = mixture.batch(progress)
                with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                train_loss += float(loss.item())
                loss = loss / config.GRAD_ACCUM
                (scaler.scale(loss) if use_amp else loss).backward()
                continue
            if args.pack and isinstance(tokenizer, BPETokenizer):
                eot_id = tokenizer.special_to_id.get("<|endoftext|>")
                if eot_id is not None:
                    x, y, doc_mask = get_batch_with_doc_mask(
                        train_data, eot_id, block_size=train_block,
                        batch_size=config.BATCH_SIZE, device=config.DEVICE)
                    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                        _, loss = model(x, y, doc_mask=doc_mask)
                    train_loss += float(loss.item())
                    loss = loss / config.GRAD_ACCUM
                    (scaler.scale(loss) if use_amp else loss).backward()
                    continue
            x, y = get_batch(train_data, block_size=train_block,
                             batch_size=config.BATCH_SIZE, device=config.DEVICE)
            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                train_loss += float(loss.item())
                loss = loss / config.GRAD_ACCUM
            (scaler.scale(loss) if use_amp else loss).backward()

        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_raw.parameters(), config.MAX_GRAD_NORM)
        (scaler.step(optimizer) if use_amp else optimizer.step())
        if use_amp:
            scaler.update()
        train_loss /= config.GRAD_ACCUM

        first_step = step == start_step + 1 and start_step == 0
        should_eval = first_step or step % config.EVAL_INTERVAL == 0 or step == target_iters
        should_save = step % config.CHECKPOINT_INTERVAL == 0 or should_eval

        if should_eval:
            losses = estimate_loss_pretrain(model, train_data, val_data, use_amp, amp_dtype)
            elapsed = time.time() - t0
            print(f"{step:>8}  {losses['train']:>10.4f}  {losses['val']:>10.4f}  {lr:>10.6f}  {elapsed:>8.1f}s")
            next_best = min(best_loss, losses["val"])
            save_checkpoint(model_raw, optimizer, step, best_loss=next_best,
                            path=ckpt_arg, tokenizer=tokenizer, scaler=scaler,
                            train_loss=losses["train"], val_loss=losses["val"])
            if losses["val"] < best_loss:
                best_loss = losses["val"]
                save_checkpoint(model_raw, optimizer, step, best_loss=best_loss,
                                path=best_ckpt, tokenizer=tokenizer, scaler=scaler,
                                train_loss=losses["train"], val_loss=losses["val"])
        elif should_save:
            save_checkpoint(model_raw, optimizer, step, best_loss=best_loss,
                            path=ckpt_arg, tokenizer=tokenizer, scaler=scaler,
                            train_loss=train_loss, val_loss=None)

    if mixture is not None:
        realized = mixture.realized_mix()
        total = sum(realized.values()) or 1
        mix_str = ", ".join(f"{k}: {v/total:.1%}" for k, v in sorted(realized.items()))
        print(f"[Mix ] Realized domain mix: {mix_str}")

    print("\n[Train] Finished.")
    if os.path.exists(best_ckpt):
        model, tokenizer, _ = build_inference_bundle(best_ckpt)
    sample_from_model(model, tokenizer)


def train_instruct(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    device = config.DEVICE
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    configure_cuda_performance(device)
    _apply_fast_profile(args)

    hf_dir = (args.hf_seed_dir or "").strip()
    extra_dirs = (hf_dir,) if hf_dir and os.path.isdir(hf_dir) else ()
    grounded_dir = (args.grounded_dir or "").strip()
    extra_dirs_full = (grounded_dir,) if grounded_dir and os.path.isdir(grounded_dir) else ()
    validation_dirs = (args.sft_val_dir,) if args.sft_val_dir else ()
    if args.sft_val_dir and not os.path.isdir(args.sft_val_dir):
        raise FileNotFoundError(f"Explicit SFT validation directory missing: {args.sft_val_dir}")
    train_ds, val_ds, tokenizer = build_chat_dataset(
        duplication=args.duplication, block_size=args.block_size, seed=args.seed,
        extra_dirs=extra_dirs, max_convs=args.max_convs,
        extra_dirs_full=extra_dirs_full, val_fraction=args.val_fraction,
        validation_dirs=validation_dirs,
    )
    if len(train_ds) < 2 or len(val_ds) < 1:
        raise SystemExit(
            "Not enough chat data. Add JSONL files in data/seed/ (or check "
            "--hf-seed-dir) or lower --block-size."
        )

    _adapt_microbatch_for_vram(device, args)

    # Train/validation split occurs at unique conversation level in the builder,
    # before seed duplication and packing; no duplicate block leakage.

    # For Python 3.9+ we avoid the typing issue with random_split
    from torch.utils.data import Subset, DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── Checkpoint loading ─────────────────────────────────────────────
    ckpt_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "jingpt_chat.pt")
    resume_path = ckpt_path if args.resume and os.path.exists(ckpt_path) else None
    pretrained_path = args.init_from
    if pretrained_path is None and resume_path is None and os.path.exists(config.BEST_CHECKPOINT_PATH):
        pretrained_path = config.BEST_CHECKPOINT_PATH

    weights_ckpt = None
    weights_label = ""
    if args.init_from and not os.path.isfile(args.init_from):
        raise FileNotFoundError(f"Explicit parent checkpoint missing: {args.init_from}")
    if args.resume and resume_path is None:
        raise FileNotFoundError(f"Resume checkpoint missing: {ckpt_path}")
    if resume_path:
        weights_ckpt = load_checkpoint_file(resume_path, map_location="cpu")
        ok, reason = _pretrained_compatible(weights_ckpt, tokenizer)
        if not ok:
            raise SystemExit(f"[Fatal] Cannot resume: {reason}")
        weights_label = f"resume:{resume_path}"
        print(f"[Init] Resuming from {resume_path}")
    elif pretrained_path and os.path.exists(pretrained_path):
        weights_ckpt = load_checkpoint_file(pretrained_path, map_location="cpu")
        ok, reason = _pretrained_compatible(weights_ckpt, tokenizer)
        if args.no_pretrained_init:
            print("[Init] --no-pretrained-init set — training from scratch.")
            weights_ckpt = None
        elif not ok:
            raise ValueError(f"Incompatible parent: {reason}. Use --no-pretrained-init explicitly for random initialization.")
        else:
            weights_label = pretrained_path
            print(f"[Init] Warm-starting from {pretrained_path}")

    model = create_model(vocab_size=tokenizer.vocab_size, checkpoint=weights_ckpt, device=device)
    if weights_ckpt and "model_state_dict" in weights_ckpt:
        model.load_state_dict(weights_ckpt["model_state_dict"], strict=True)
        print(f"[Init] Weights loaded ({weights_label}).")
    print(f"[Model] {count_parameters(model)/1e6:.2f}M params "
          f"({'modern' if model.modern else 'classic'}, "
          f"{model.n_layers}L, {model.n_heads}H, dim {model.embed_dim})")

    optim = make_optimizer(model, args, lr=args.lr)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if config.AMP_DTYPE == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_step = 0
    best_val = float("inf")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = load_checkpoint_file(ckpt_path, map_location="cpu")
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            optim.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = int(ckpt.get("iteration", 0))
        best_val = float(ckpt.get("best_loss", best_val))
        print(f"[Resume] Step {start_step:,} (best val {best_val:.4f})")

    if not len(train_loader):
        raise ValueError("Training split has fewer blocks than microbatch; reduce --batch-size")
    from lineage import start_run, attach_run, record_batch
    from pathlib import Path
    dataset_paths = [str(p) for p in Path(config.ROOT_DIR, "data", "seed").glob("*.jsonl")]
    for directory in extra_dirs + extra_dirs_full + validation_dirs:
        dataset_paths += [str(p) for p in Path(directory).glob("*.jsonl")]
    run = start_run(model, tokenizer, vars(args), dataset_paths,
                    resume_path or (pretrained_path if weights_ckpt else None), ckpt_path)
    run["dataset_split"] = train_ds.split_manifest
    attach_run(model, run)
    target_step = start_step + args.steps
    print(f"[Train] Instruct: step {start_step:,}->{target_step:,} | batch {args.batch_size}x{args.grad_accum}")

    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()
    for step in range(start_step + 1, target_step + 1):
        lr_now = get_lr(step, target_step, args.lr, args.min_lr_ratio, args.warmup)
        for pg in optim.param_groups:
            pg["lr"] = lr_now

        optim.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                x, y, m = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y, m = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            record_batch(model, x.numel(), int(m.sum().item()))
            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y, loss_mask=m)
                running_loss += float(loss.item())
                loss = loss / args.grad_accum
            (scaler.scale(loss) if use_amp else loss).backward()

        if use_amp:
            scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        (scaler.step(optim) if use_amp else optim.step())
        if use_amp:
            scaler.update()
        running_loss /= args.grad_accum

        if step % 25 == 0 or step == target_step:
            elapsed = time.time() - t0
            done = step - start_step
            rate = done / max(elapsed, 1e-6)
            eta = (target_step - step) / max(rate, 1e-6)
            sys.stdout.write(f"\r[step {step:>6}/{target_step:,}] loss {running_loss:7.4f} | lr {lr_now:.2e} | {rate:5.1f} it/s | eta {eta/60:5.1f} min")
            sys.stdout.flush()

        if step % args.eval_interval == 0 or step == target_step:
            print()
            val_loss = estimate_loss_instruct(model, val_loader, device, args.eval_iters, use_amp=use_amp, amp_dtype=amp_dtype)
            print(f"[eval] step {step:,}: train {running_loss:.4f} | val {val_loss:.4f}")
            _quick_chat_sample(model, tokenizer, device)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optim, step, best_loss=best_val, path=ckpt_path[:-3] + "_best.pt",
                                tokenizer=tokenizer, scaler=scaler,
                                train_loss=running_loss, val_loss=val_loss)
        if step % args.checkpoint_interval == 0 or step == target_step or step % args.eval_interval == 0:
            save_checkpoint(model, optim, step, best_loss=best_val, path=ckpt_path,
                            tokenizer=tokenizer, scaler=scaler,
                            train_loss=running_loss,
                            val_loss=val_loss if step % args.eval_interval == 0 or step == target_step else None)

    print(f"\n[Train] Done. Best val loss: {best_val:.4f}")


def main() -> None:
    # Windows consoles default to cp1252 and crash printing Vietnamese sample
    # output (UnicodeEncodeError). Prefer UTF-8 with a graceful fallback.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args()
    if args.checkpoint:
        args.checkpoint = config.output_path(args.checkpoint)
    config.set_device(args.device)
    configure_cuda_performance(config.DEVICE)

    print("=" * 72)
    print(f"JIN AI {'Instruction Tuning' if args.mode == 'instruct' else 'Pretraining'}")
    print("=" * 72)
    print(f"Device: {config.DEVICE}")
    if config.DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:    {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f} GB)")

    if args.mode == "instruct":
        train_instruct(args)
    else:
        train_pretrain(args)


if __name__ == "__main__":
    main()