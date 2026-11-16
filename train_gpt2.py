"""Reproducible GPT-2-class pretraining script (Phase 4 + 6 infrastructure).

This is a *separate* entry point from ``train.py``: it targets the
``gpt2_small_reference`` architecture (124M, 50257 vocab, 1024 ctx) and trades
the old script's flexibility for the things a from-scratch pretraining run
actually needs:

* **Absolute** cosine schedule — the step counter, total steps, tokens seen and
  dataloader position all live in the checkpoint, so ``--resume`` continues the
  same curve instead of restarting it (``train.py`` recomputes
  ``target = start + run_iters``, which silently reshapes the schedule).
* **Document-aware packing** — every document is encoded as
  ``<|endoftext|> body <|endoftext|>``, and attention is masked so a token
  never attends into a previous document inside the same block.
* **Measured manifests** — dataset/tokenizer/config SHA-256 hashes, tokens
  seen, architecture id and hardware are written next to every checkpoint.
  A claim like "trained on 1B tokens" can then be checked against the artifact.
* **Constant-RAM data** — the tokenized corpus is a uint16 memmap, not a
  multi-GB in-process tensor.

Usage
-----
    # tokenize + train from scratch
    python train_gpt2.py --build-data --train-files data/prepared_v4/pretrain/*.txt

    # resume exactly where it stopped
    python train_gpt2.py --resume

    # quick sanity run (20 steps) before committing GPU hours
    python train_gpt2.py --smoke
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

import config
from model import JinAI
from tokenizer_gpt2 import GPT2BPETokenizer

ROOT = config.ROOT_DIR
DATA_DIR = os.path.join(ROOT, "data", "gpt2_corpus")
OUT_DIR = os.path.join(ROOT, "checkpoints", "gpt2_small")


# ── helpers ────────────────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str, payload: dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def log(msg: str) -> None:
    print(msg, flush=True)


# ── data pipeline ──────────────────────────────────────────────────────

def iter_documents(path: str, chunk_lines: int = 400):
    """Yield documents from a text file (blank line = document boundary)."""
    buf: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                buf.append(line)
            elif buf:
                yield "".join(buf)
                buf = []
            if len(buf) >= chunk_lines:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)


def build_shard(paths: list[str], out_bin: str, tokenizer, *, log_every: int = 20) -> dict:
    """Tokenize documents into a uint16 token stream + EOT boundary flags."""
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    tmp_bin = out_bin + ".tmp"
    docs = 0
    tokens = 0
    chars = 0
    t0 = time.time()
    with open(tmp_bin, "wb") as fh:
        for path in paths:
            for doc in iter_documents(path):
                ids = tokenizer.encode(doc)
                if not ids:
                    continue
                # <|endoftext|> body <|endoftext|>: the trailing EOT both
                # teaches end-of-document and marks a boundary for the mask.
                seq = [tokenizer.eot_id] + ids + [tokenizer.eot_id]
                np.asarray(seq, dtype=np.uint16).tofile(fh)
                docs += 1
                tokens += len(seq)
                chars += len(doc)
                if docs % log_every == 0:
                    rate = tokens / max(1e-9, time.time() - t0)
                    log(f"  [{os.path.basename(path)}] {docs:,} docs | "
                        f"{tokens:,} tokens | {rate:,.0f} tok/s")
    os.replace(tmp_bin, out_bin)
    return {
        "path": os.path.relpath(out_bin, ROOT),
        "sha256": sha256_file(out_bin),
        "bytes": os.path.getsize(out_bin),
        "documents": docs,
        "tokens": tokens,
        "source_chars": chars,
        "chars_per_token": round(chars / max(1, tokens), 3),
        "sources": [os.path.relpath(p, ROOT) for p in paths],
    }


def build_dataset(train_files: list[str], val_files: list[str], tokenizer) -> dict:
    log(f"[Data] Tokenizing {len(train_files)} train / {len(val_files)} val files …")
    train = build_shard(train_files, os.path.join(DATA_DIR, "train.bin"), tokenizer)
    val = build_shard(val_files, os.path.join(DATA_DIR, "val.bin"), tokenizer)
    manifest = {
        "schema_version": 1,
        "tokenizer": {"kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size,
                      "hash": hashlib.sha256(json.dumps(tokenizer.to_state_dict(),
                                                        sort_keys=True).encode()).hexdigest()},
        "train": train,
        "val": val,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_json(os.path.join(DATA_DIR, "manifest.json"), manifest)
    log(f"[Data] train {train['tokens']:,} tok / val {val['tokens']:,} tok "
        f"({train['chars_per_token']} chars/token)")
    return manifest


class ShardBatcher:
    """Random windows from a uint16 token memmap, with document masks."""

    def __init__(self, path: str, seq_len: int, batch_size: int, device: str,
                 seed: int, eot_id: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.eot_id = eot_id
        self.rng = np.random.default_rng(seed)
        self.tokens_served = 0

    # Sample-position RNG is part of the training state: without it a resumed
    # run draws different batches and "exact resume" is false.
    def state_dict(self) -> dict:
        raw = self.rng.bit_generator.state
        return {"bit_generator": {"bit_generator": raw["bit_generator"],
                                  "state": {k: int(v) for k, v in raw["state"].items()},
                                  "has_uint32": int(raw.get("has_uint32", 0)),
                                  "uinteger": int(raw.get("uinteger", 0))},
                "tokens_served": int(self.tokens_served)}

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.rng.bit_generator.state = state["bit_generator"]
        self.tokens_served = int(state.get("tokens_served", 0))

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def batch(self):
        if len(self) <= 0:
            raise ValueError("Shard is shorter than seq_len + 1")
        starts = self.rng.integers(0, len(self), size=self.batch_size)
        x = np.stack([self.data[s: s + self.seq_len] for s in starts])
        y = np.stack([self.data[s + 1: s + self.seq_len + 1] for s in starts])
        # doc_mask: 1 at block start and at every token following an EOT.
        prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
        doc_mask = (prev == self.eot_id).astype(np.int64)
        doc_mask[:, 0] = 1
        self.tokens_served += self.batch_size * self.seq_len
        return (
            torch.from_numpy(x.astype(np.int64)).to(self.device, non_blocking=True),
            torch.from_numpy(y.astype(np.int64)).to(self.device, non_blocking=True),
            torch.from_numpy(doc_mask).to(self.device, non_blocking=True),
        )


# ── model / optimizer ──────────────────────────────────────────────────

def build_model(preset: str, vocab_size: int, block_size: int, device: str,
                grad_checkpoint: bool, dropout: float = 0.0) -> JinAI:
    p = config.SCALE_PRESETS[preset]
    cfg = {
        "vocab_size": vocab_size,
        "embed_dim": p["embed_dim"],
        "block_size": block_size,
        "n_layers": p["n_layers"],
        "n_heads": p["n_heads"],
        "n_kv_heads": p["n_kv_heads"],
        # GPT-2 pretraining uses dropout 0.1; 0 was the pilot's default for VRAM
        # reasons. --dropout makes the choice an experiment instead of an
        # assumption (see docs/PHASE14_TRAINING_REPORT.md).
        "dropout": dropout,
        "gradient_checkpointing": grad_checkpoint,
        "modern": p["modern"],
        "arch_id": p["arch_id"],
        "gelu_tanh": p["gelu_tanh"],
        "resid_scaled_init": p["resid_scaled_init"],
        "qk_norm": p.get("qk_norm", False),
        "rope_theta": p.get("rope_theta", 10_000.0),
        "rope_scaling": None,
        "sliding_window": None,
        "local_layer_every": None,
        "moe": None,
        "expert_offload": False,
        "static_kv_cache": False,
        "kv_cache_int8": False,
    }
    model = JinAI.from_config({**cfg, "device": device})
    return model


def lr_at(step: int, total_steps: int, peak: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1.0 - min_ratio) * cosine)


@torch.no_grad()
def evaluate(model, batcher: ShardBatcher, iters: int, amp_dtype, use_amp: bool) -> dict:
    """Validation loss with FP32 cross-entropy on FP32 logits.

    The model's own loss runs inside autocast; at vocab 50k an FP16 softmax is
    not trustworthy (measured elsewhere in this repo: 11.67 fp16 vs 0.83 fp32
    on identical logits). Validation must be correct even if it costs a cast.
    The evaluation batcher is seeded deterministically, so every run scores the
    same windows and the token count is reported.
    """
    model.eval()
    losses = []
    tokens = 0
    for _ in range(iters):
        x, y, m = batcher.batch()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(x, doc_mask=m)
        logits = logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss.item()))
        tokens += int(y.numel())
        del logits
    model.train()
    mean = sum(losses) / max(1, len(losses))
    return {"loss": round(mean, 4), "perplexity": round(math.exp(min(20.0, mean)), 2),
            "iters": iters, "tokens_scored": tokens,
            "loss_dtype": "fp32_cross_entropy"}


# ── checkpointing ──────────────────────────────────────────────────────

# Bump when the batch sampler changes behaviour: a resume across sampler
# versions cannot reproduce the same batch sequence even with identical data.
SAMPLE_IMPL_VERSION = 1
DATA_FRAMING = "<eot> document <eot>, uint16 little-endian"


def _sha_payload(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     default=str).encode()).hexdigest()


def experiment_fingerprint(model, tokenizer, dataset_manifest: dict,
                           args, total_steps: int) -> dict:
    """Everything that must be identical for a resume to reproduce a run.

    The Phase 14 guard compared corpus hashes, arch id, tokenizer kind and step
    budget. That let a resume change learning rate, warmup, batch geometry,
    dropout, seed or precision and still claim to be the same experiment.
    """
    return {
        "architecture": model.export_config(),
        "arch_id": model.arch_id,
        "preset": args.preset,
        "tokenizer": {"kind": tokenizer.kind,
                      "vocab_size": tokenizer.vocab_size,
                      "state_hash": _sha_payload(tokenizer.to_state_dict())},
        "corpus": {"train_sha256": dataset_manifest["train"]["sha256"],
                   "val_sha256": dataset_manifest["val"]["sha256"]},
        "data": {"seq_len": args.seq_len, "batch_size": args.batch_size,
                 "grad_accum": args.grad_accum,
                 "effective_batch_tokens": args.batch_size * args.grad_accum * args.seq_len,
                 "framing": DATA_FRAMING,
                 "sampler": f"numpy-pcg64.v{SAMPLE_IMPL_VERSION}"},
        "optimizer": {"kind": "adamw", "betas": [0.9, 0.95], "eps": 1e-8,
                      "weight_decay": 0.1, "grad_clip": 1.0},
        "schedule": {"kind": "cosine_with_warmup", "total_steps": total_steps,
                     "warmup": args.warmup, "peak_lr": args.lr,
                     "min_lr_ratio": args.min_lr_ratio},
        "precision": {"amp": not args.no_amp, "amp_dtype": "float16"},
        "seed": args.seed,
        "dropout": args.dropout,
        "code_revision": os.popen("git rev-parse HEAD").read().strip() or None,
    }


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = value


def compare_fingerprints(old: dict, new: dict,
                         ignore: tuple[str, ...] = ()) -> list[tuple]:
    """Dotted-key differences, ignoring keys explicitly permitted to change."""
    a, b = {}, {}
    _flatten("", old, a)
    _flatten("", new, b)
    diffs = []
    for key in sorted(set(a) | set(b)):
        if key in ignore:
            continue
        if a.get(key) != b.get(key):
            diffs.append((key, a.get(key), b.get(key)))
    return diffs


def _np_state_to_dict(state) -> dict:
    """NumPy global RNG state as plain ints.

    ``torch.load(weights_only=True)`` rejects the raw tuple because it holds an
    ndarray, so the checkpoint must carry primitives only.
    """
    return {"kind": str(state[0]), "keys": [int(k) for k in state[1]],
            "pos": int(state[2]), "has_gauss": int(state[3]),
            "cached_gaussian": float(state[4])}


def _np_state_from_dict(data):
    if data is None:
        return None
    if isinstance(data, tuple):                       # legacy checkpoints
        return data
    return (data["kind"], np.array(data["keys"], dtype=np.uint32), data["pos"],
            data["has_gauss"], data["cached_gaussian"])


def verify_resume_compatibility(blob: dict, model, tokenizer, dataset_manifest: dict,
                                total_steps: int, extend_budget: bool,
                                log_fn=print, current_fingerprint: dict | None = None) -> None:
    """Refuse a resume that would silently change the experiment."""
    old = blob.get("dataset_manifest", {})
    if old.get("train", {}).get("sha256") != dataset_manifest["train"]["sha256"]:
        raise SystemExit("[Fatal] Training corpus changed since this checkpoint — refusing to resume.")
    if old.get("val", {}).get("sha256") != dataset_manifest["val"]["sha256"]:
        raise SystemExit("[Fatal] Validation corpus changed since this checkpoint — refusing to resume.")
    if blob.get("architecture_id") != model.arch_id:
        raise SystemExit(f"[Fatal] Architecture changed: checkpoint={blob.get('architecture_id')} "
                         f"now={model.arch_id}")
    old_tok = blob.get("tokenizer") or {}
    if old_tok.get("vocab_size") != tokenizer.vocab_size or old_tok.get("kind") != tokenizer.kind:
        raise SystemExit("[Fatal] Tokenizer identity changed since this checkpoint.")
    old_schedule = blob.get("schedule") or {}
    old_total = int(old_schedule.get("total_steps", total_steps))
    if old_total != total_steps and not extend_budget:
        raise SystemExit(
            f"[Fatal] Step budget changed ({old_total} -> {total_steps}). "
            "This would silently reshape the LR schedule. Pass --extend-budget to "
            "intentionally continue with the new budget.")
    if old_total != total_steps:
        log_fn(f"[Resume] extending budget {old_total} -> {total_steps} (--extend-budget)")

    # Full experiment-configuration check (Phase 15F).
    old_fp = blob.get("experiment_config")
    if old_fp and current_fingerprint is not None:
        ignore = ("schedule.total_steps",) if extend_budget else ()
        diffs = compare_fingerprints(old_fp, current_fingerprint, ignore=ignore)
        if diffs:
            detail = "; ".join(f"{k}: {o!r} -> {n!r}" for k, o, n in diffs[:8])
            raise SystemExit(
                "[Fatal] Training configuration changed since this checkpoint: "
                f"{detail}. A different configuration is a different experiment; "
                "start a new --out-dir instead of resuming.")
        if extend_budget and old_total != total_steps:
            log_fn("[Resume] LR schedule is NOT the original trajectory: the cosine "
                   "decay now spans a longer budget. This is explicit, not silent.")


def restore_training_state(blob: dict, model, optimizer, scaler=None,
                           batcher=None, device: str = "cpu") -> None:
    """Restore every piece of state the training loop depends on."""
    import random as _random

    model.load_state_dict(blob["model_state_dict"])
    if optimizer is not None and blob.get("optimizer_state_dict"):
        optimizer.load_state_dict(blob["optimizer_state_dict"])
    if scaler is not None and blob.get("scaler_state_dict"):
        scaler.load_state_dict(blob["scaler_state_dict"])
    torch.set_rng_state(blob["rng_state"])
    if device == "cuda" and blob.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(blob["cuda_rng_state_all"])
    if blob.get("numpy_rng_state") is not None:
        np.random.set_state(_np_state_from_dict(blob["numpy_rng_state"]))
    if blob.get("python_rng_state") is not None:
        _random.setstate(blob["python_rng_state"])
    if batcher is not None and blob.get("batcher_state"):
        batcher.load_state_dict(blob["batcher_state"])


def save_state(path: str, model, optimizer, scaler, step: int, tokens_seen: int,
               best_val: float, history: list, meta: dict,
               batcher: "ShardBatcher | None" = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "format_version": 3,
        "artifact_kind": "training_checkpoint",
        "architecture_id": meta["architecture_id"],
        "iteration": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val,
        "model_config": model.export_config(),
        "vocab_size": model.vocab_size,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": _np_state_to_dict(np.random.get_state()),
        "python_rng_state": __import__("random").getstate(),
        "batcher_state": batcher.state_dict() if batcher is not None else None,
        # meta.get: checkpoints written by other entry points (and the older
        # resume test) do not carry a schedule; a missing schedule must not
        # crash a save.
        "schedule": meta.get("schedule"),
        "effective_batch_tokens": meta.get("effective_batch_tokens"),
        "training_revision": meta.get("revision"),
        "training_args": meta["training_args"],
        "dataset_manifest": meta["dataset_manifest"],
        "experiment_config": meta.get("experiment_config"),
        "experiment_config_hash": meta.get("experiment_config_hash"),
        "tokenizer": meta["tokenizer"],
        # Canonical key read by utils.tokenizer_from_state / app / chat / bench.
        "tokenizer_state": {"kind": meta["tokenizer"]["kind"],
                            "vocab_size": meta["tokenizer"]["vocab_size"],
                            "special_tokens": ["<|endoftext|>"]},
        "hardware": meta["hardware"],
        "history": history[-200:],
    }
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    atomic_json(path + ".manifest.json", {
        "artifact": os.path.relpath(path, ROOT),
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "architecture_id": meta["architecture_id"],
        "iteration": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val,
        "dataset_sha256": meta["dataset_manifest"]["train"]["sha256"],
        "tokenizer": meta["tokenizer"],
        "model_config": model.export_config(),
        "history_tail": history[-10:],
    })


# ── main ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproducible GPT-2-class pretraining")
    p.add_argument("--preset", default="gpt2_small_reference")
    p.add_argument("--data-dir", default=DATA_DIR,
                   help="directory holding manifest.json + train.bin/val.bin "
                        "(relative paths resolve against the repo root)")
    p.add_argument("--train-files", nargs="*", default=None,
                   help="Corpus files (default: data/prepared_v4/pretrain/*.txt minus *_val)")
    p.add_argument("--val-files", nargs="*", default=None,
                   help="Validation files (default: the *_val.txt / validation.txt split)")
    p.add_argument("--build-data", action="store_true",
                   help="(Re)tokenize the corpus into data/gpt2_corpus/")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--total-tokens", type=float, default=None,
                   help="Stop after this many tokens (default: 20x params ≈ 2.5B)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override the derived step budget")
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=8)
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--extend-budget", action="store_true",
                   help="Resume into a DIFFERENT (larger) step budget. Without this "
                        "flag a changed budget is refused, because silently "
                        "reshaping the cosine schedule makes runs incomparable.")
    p.add_argument("--device", default=None, choices=config.DEVICE_CHOICES)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Activation checkpointing (needed below ~8GB VRAM)")
    p.add_argument("--smoke", action="store_true", help="20 steps, small budget")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="residual/attention dropout (GPT-2 recipe uses 0.1)")
    return p.parse_args()


def resolve_files(args) -> tuple[list[str], list[str]]:
    if args.train_files:
        train = [f for pat in args.train_files for f in glob.glob(os.path.join(ROOT, pat))]
    else:
        pool = sorted(glob.glob(os.path.join(ROOT, "data", "prepared_v4", "pretrain", "*.txt")))
        train = [f for f in pool if "_val" not in f and "validation" not in f]
    if args.val_files:
        val = [f for pat in args.val_files for f in glob.glob(os.path.join(ROOT, pat))]
    else:
        val = sorted(glob.glob(os.path.join(ROOT, "data", "prepared_v4", "pretrain", "*val*.txt")))
    if not train:
        raise SystemExit("No training files matched.")
    if not val:
        raise SystemExit("No validation files matched (document-disjoint set required).")
    return train, val


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args()
    global DATA_DIR
    DATA_DIR = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(
        ROOT, args.data_dir)
    config.set_device(args.device)
    device = config.DEVICE
    if args.smoke:
        args.max_steps = args.max_steps or 20
        args.batch_size = min(args.batch_size, 2)
        args.grad_accum = min(args.grad_accum, 2)
        args.eval_interval = 10
        args.save_interval = 20
        args.eval_iters = 2
    if device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 9 * 1024**3:
        args.grad_checkpoint = True          # 6GB card: recompute activations

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = GPT2BPETokenizer()
    log(f"[Tokenizer] {tokenizer.backend} (vocab {tokenizer.vocab_size:,})")

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    if args.build_data or not os.path.exists(manifest_path):
        train_files, val_files = resolve_files(args)
        dataset_manifest = build_dataset(train_files, val_files, tokenizer)
    else:
        with open(manifest_path, encoding="utf-8") as fh:
            dataset_manifest = json.load(fh)
        log(f"[Data] Reusing corpus manifest ({dataset_manifest['train']['tokens']:,} train tokens)")

    train_batcher = ShardBatcher(os.path.join(ROOT, dataset_manifest["train"]["path"]),
                                 args.seq_len, args.batch_size, device,
                                 args.seed, tokenizer.eot_id)
    val_batcher = ShardBatcher(os.path.join(ROOT, dataset_manifest["val"]["path"]),
                               args.seq_len, max(1, args.batch_size // 2), device,
                               args.seed + 7, tokenizer.eot_id)

    model = build_model(args.preset, tokenizer.vocab_size, args.seq_len, device,
                        args.grad_checkpoint, dropout=args.dropout)
    n_params = model.num_parameters()
    tokens_budget = int(args.total_tokens or n_params * 20)
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    total_steps = args.max_steps or max(1, tokens_budget // tokens_per_step)
    log(f"[Plan] {n_params/1e6:.2f}M params | {tokens_per_step:,} tok/step | "
        f"{total_steps:,} steps | budget {tokens_budget/1e9:.2f}B tokens")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    use_amp = device == "cuda" and not args.no_amp
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    fingerprint = experiment_fingerprint(model, tokenizer, dataset_manifest, args,
                                        total_steps)
    meta = {
        "architecture_id": model.arch_id,
        "experiment_config": fingerprint,
        "experiment_config_hash": _sha_payload(fingerprint),
        "training_args": vars(args),
        "dataset_manifest": dataset_manifest,
        "tokenizer": {"kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size,
                      "backend": tokenizer.backend},
        "schedule": {"kind": "cosine_with_warmup", "absolute_step_based": True,
                     "total_steps": total_steps, "warmup": args.warmup,
                     "peak_lr": args.lr, "min_lr_ratio": args.min_lr_ratio,
                     "grad_clip": 1.0},
        "effective_batch_tokens": args.batch_size * args.grad_accum * args.seq_len,
        "revision": os.popen("git rev-parse HEAD").read().strip() or None,
        "hardware": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            if device == "cuda" else None,
            # str(): torch.__version__ is a TorchVersion instance, which pickles
            # as a custom global and makes the checkpoint unloadable with
            # weights_only=True.
            "torch": str(torch.__version__),
        },
    }

    ckpt_path = os.path.join(args.out_dir, "latest.pt")
    start_step = 0
    tokens_seen = 0
    best_val = float("inf")
    history: list = []
    if args.resume and os.path.exists(ckpt_path):
        # Allowlist TorchVersion so checkpoints written before the str() fix
        # still resume; nothing else is added to the safe-global list.
        add_safe = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe is not None:
            try:
                from torch.torch_version import TorchVersion
                add_safe([TorchVersion])
            except Exception:
                pass
        # mmap: a 1.5 GB checkpoint must not be read into a machine that has
        # ~2.5 GB free. Pages are faulted in on demand instead.
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)

        # ── refuse unsafe partial restorations, then restore everything ──
        current_fp = experiment_fingerprint(model, tokenizer, dataset_manifest, args,
                                            total_steps)
        verify_resume_compatibility(blob, model, tokenizer, dataset_manifest,
                                    total_steps, args.extend_budget, log_fn=log,
                                    current_fingerprint=current_fp)
        restore_training_state(blob, model, optimizer, scaler, train_batcher, device)
        start_step = int(blob["iteration"])
        tokens_seen = int(blob["tokens_seen"])
        best_val = float(blob["best_val_loss"])
        history = list(blob.get("history", []))
        log(f"[Resume] step {start_step:,} | tokens {tokens_seen:,} | best val {best_val:.4f} "
            f"| sampler+cpu+cuda RNG restored")

    log(f"[Train] device={device} amp={use_amp} gc={args.grad_checkpoint} "
        f"batch={args.batch_size}x{args.grad_accum} seq={args.seq_len}")

    model.train()
    t0 = time.time()
    tokens_at_start = tokens_seen
    step_tokens = args.batch_size * args.grad_accum * args.seq_len
    run_loss = 0.0
    for step in range(start_step + 1, total_steps + 1):
        lr = lr_at(step, total_steps, args.lr, args.warmup, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        run_loss = 0.0
        for _ in range(args.grad_accum):
            x, y, m = train_batcher.batch()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y, doc_mask=m)
            run_loss += float(loss.item())
            (scaler.scale(loss / args.grad_accum) if use_amp
             else loss / args.grad_accum).backward()
        if use_amp:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        (scaler.step(optimizer) if use_amp else optimizer.step())
        if use_amp:
            scaler.update()
        tokens_seen += step_tokens
        run_loss /= args.grad_accum

        if step % 10 == 0 or step == total_steps:
            elapsed = max(1e-6, time.time() - t0)
            # Throughput from CUMULATIVE tokens, so the number cannot drift
            # with logging cadence (a per-window counter that is only reset at
            # log time over-reports by the log interval).
            tok_s = (tokens_seen - tokens_at_start) / elapsed
            vram = (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else 0.0
            eta_h = (total_steps - step) * step_tokens / max(1e-6, tok_s) / 3600
            log(f"step {step:>6}/{total_steps:,} | loss {run_loss:7.4f} | ppl "
                f"{math.exp(min(20.0, run_loss)):8.1f} | lr {lr:.2e} | gnorm "
                f"{float(grad_norm):.2f} | {tok_s:,.0f} tok/s | {vram:.1f} GB | "
                f"{tokens_seen/1e6:.1f}M tok | eta {eta_h:.2f}h")

        if step % args.eval_interval == 0 or step == total_steps:
            val = evaluate(model, val_batcher, args.eval_iters, amp_dtype, use_amp)
            log(f"  [eval] step {step:,} val loss {val['loss']} ppl {val['perplexity']}")
            entry = {"step": step, "tokens_seen": tokens_seen,
                     "train_loss": round(run_loss, 4), "val_loss": val["loss"],
                     "val_perplexity": val["perplexity"],
                     "lr": round(lr, 7), "elapsed_s": round(time.time() - t0, 1)}
            history.append(entry)
            if val["loss"] < best_val:
                best_val = val["loss"]
                save_state(os.path.join(args.out_dir, "best.pt"), model, optimizer, scaler,
                           step, tokens_seen, best_val, history, meta, batcher=train_batcher)
                log(f"  [save] best.pt (val {best_val:.4f})")

        if step % args.save_interval == 0 or step == total_steps:
            save_state(ckpt_path, model, optimizer, scaler, step, tokens_seen,
                       best_val, history, meta, batcher=train_batcher)
            log(f"  [save] latest.pt (sha256 in latest.pt.manifest.json)")

    log("\n[Done] Pretraining stopped at the configured budget.")
    log(f"[Done] best val loss {best_val:.4f} (ppl {math.exp(min(20.0, best_val)):.2f})")
    log(f"[Done] weights: {os.path.relpath(os.path.join(args.out_dir, 'best.pt'), ROOT)}")
    log("[Done] This is a measured checkpoint, not a capability claim: run "
        "scripts/gpt2_bench.py to compare against GPT-2 Small.")


if __name__ == "__main__":
    main()
