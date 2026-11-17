"""Phase 16 §1 — inspect a checkpoint and measure it, from the weights.

Documentation and presets are claims; this script reads the checkpoint and counts
what is actually there, then measures latency/throughput/VRAM for a fixed
generation workload.

Usage:
    python scripts/phase16_inspect_model.py --checkpoint checkpoints/gpt2_small/phase14_wiki/best.pt \
        --out bench_results/phase16/model_baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tokenizer_gpt2 import GPT2BPETokenizer   # noqa: E402
from utils import _torch_load, create_model    # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def count_from_state_dict(sd: dict) -> dict:
    total = sum(v.numel() for v in sd.values())
    by_prefix: Counter = Counter()
    for k, v in sd.items():
        by_prefix[k.split(".")[0]] += v.numel()
    emb = sum(v.numel() for k, v in sd.items() if "emb" in k)
    head = sum(v.numel() for k, v in sd.items() if k.startswith("head"))
    tied = (sd["head.weight"].data_ptr() == sd["tok_emb.weight"].data_ptr()
            if "head.weight" in sd and "tok_emb.weight" in sd else None)
    return {
        "parameter_count": total,
        "by_prefix": dict(by_prefix),
        "embedding_parameters": emb,
        "output_head_parameters": head,
        "output_head_tied_to_embeddings": bool(tied) if tied is not None else None,
        "non_embedding_parameters": total - emb,
        "tensors": len(sd),
    }


@torch.no_grad()
def measure_generation(model, tokenizer, device: str, prompt: str,
                       new_tokens: int = 32, runs: int = 3) -> dict:
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    block = int(getattr(model, "block_size", 1024))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # warmup (kernel autotune / cuBLAS workspace), not measured
    x = ids[:, -block:]
    for _ in range(3):
        logits, _ = model(x)
        x = torch.cat([x, torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)], dim=1)

    times: list[float] = []
    first_token_times: list[float] = []
    produced = 0
    for _ in range(runs):
        x = ids[:, -block:]
        t0 = time.perf_counter()
        logits, _ = model(x)
        first = int(torch.argmax(logits[:, -1, :], dim=-1))
        if device == "cuda":
            torch.cuda.synchronize()
        t_first = time.perf_counter() - t0
        x = torch.cat([x, torch.tensor([[first]], device=device)], dim=1)
        n = 1
        while n < new_tokens:
            logits, _ = model(x[:, -block:])
            x = torch.cat([x, torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)], dim=1)
            n += 1
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        first_token_times.append(t_first)
        produced = n

    total = sum(times)
    return {
        "prompt_tokens": len(tokenizer.encode(prompt)),
        "new_tokens_per_run": produced,
        "runs": runs,
        "time_to_first_token_seconds": round(sum(first_token_times) / runs, 4),
        "end_to_end_seconds_mean": round(total / runs, 3),
        "tokens_per_second": round(produced * runs / total, 1),
        "peak_vram_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                         if device == "cuda" else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 16 model inspection")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--new-tokens", type=int, default=32)
    args = ap.parse_args()

    t0 = time.perf_counter()
    blob = _torch_load(args.checkpoint, map_location="cpu")
    load_seconds = time.perf_counter() - t0
    sd = blob["model_state_dict"]
    counts = count_from_state_dict(sd)

    cfg = blob.get("model_config") or {}
    tokenizer = GPT2BPETokenizer()
    model = create_model(int(cfg.get("vocab_size", 50257)), checkpoint=blob,
                         device=args.device)
    missing = model.load_state_dict(sd, strict=True)
    model.eval()
    built = sum(p.numel() for p in model.parameters())

    gen = measure_generation(model, tokenizer, args.device,
                             "The history of the printing press begins",
                             new_tokens=args.new_tokens)

    report = {
        "checkpoint": os.path.relpath(args.checkpoint, ROOT),
        "sha256": sha256_file(args.checkpoint),
        "bytes": os.path.getsize(args.checkpoint),
        "load_seconds": round(load_seconds, 2),
        "load_state_dict": {"missing": list(missing.missing_keys),
                            "unexpected": list(missing.unexpected_keys)},
        "state_dict": counts,
        "instantiated_model_parameters": built,
        "parameters_match_state_dict": built == counts["parameter_count"],
        "model_config": cfg,
        "iteration": blob.get("iteration"),
        "tokens_seen": blob.get("tokens_seen"),
        "best_val_loss": blob.get("best_val_loss"),
        "dataset_manifest": blob.get("dataset_manifest"),
        "tokenizer": blob.get("tokenizer"),
        "tokenizer_fingerprint": hashlib.sha256(json.dumps(
            tokenizer.to_state_dict(), sort_keys=True).encode()).hexdigest(),
        "hardware": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "generation": gen,
        "reproduce": ("python scripts/phase16_inspect_model.py --checkpoint "
                      f"{os.path.relpath(args.checkpoint, ROOT)}"),
    }

    out = args.out or os.path.join(ROOT, "bench_results", "phase16", "model_baseline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"parameters (state_dict)   {counts['parameter_count']:,}")
    print(f"parameters (instantiated) {built:,}")
    print(f"  embeddings {counts['embedding_parameters']:,} | "
          f"head {counts['output_head_parameters']:,} "
          f"(tied={counts['output_head_tied_to_embeddings']})")
    print(f"layers {cfg.get('n_layers')} | width {cfg.get('embed_dim')} | "
          f"heads {cfg.get('n_heads')} | block {cfg.get('block_size')} | "
          f"vocab {cfg.get('vocab_size')}")
    print(f"iteration {blob.get('iteration')} | tokens_seen {blob.get('tokens_seen'):,}")
    print(f"tok/s {gen['tokens_per_second']} | ttft {gen['time_to_first_token_seconds']}s | "
          f"peak VRAM {gen['peak_vram_mb']} MB")
    print(f"Wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
