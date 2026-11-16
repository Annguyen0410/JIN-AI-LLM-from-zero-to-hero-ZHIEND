"""
Load the ``xl-moe`` preset (MoE, ~0.78B total / ~0.31B active) with expert
offload and generate a short text on a small-VRAM GPU (6GB).

Usage::

    python scripts/demo_xl_moe.py                 # CUDA if available, else CPU
    python scripts/demo_xl_moe.py --device cuda --max-tokens 48

Prints total vs active parameters and peak VRAM so you can verify the
"params total / params active" accounting and that the offloaded experts
keep the model inside VRAM.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from bpe_tokenizer import BPETokenizer
from model import JinAI


def main() -> None:
    p = argparse.ArgumentParser(description="xl-moe demo (expert offload).")
    p.add_argument("--device", default="auto", choices=config.DEVICE_CHOICES)
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--prompt", default="Once upon a time,")
    args = p.parse_args()

    config.set_device(args.device or "auto")
    device = config.DEVICE
    print(f"device: {device}")

    tok = BPETokenizer()
    if os.path.exists(config.BPE_PATH):
        tok.load(config.BPE_PATH)
    else:
        raise SystemExit(f"BPE vocabulary not found at {config.BPE_PATH}")

    preset = config.SCALE_PRESETS["xl-moe"]
    model = JinAI(
        vocab_size=tok.vocab_size,
        embed_dim=preset["embed_dim"],
        block_size=preset["block_size"],
        n_layers=preset["n_layers"],
        n_heads=preset["n_heads"],
        n_kv_heads=preset["n_kv_heads"],
        dropout=0.1,
        modern=True,
        qk_norm=preset["qk_norm"],
        rope_theta=preset["rope_theta"],
        sliding_window=preset["sliding_window"],
        local_layer_every=preset["local_layer_every"],
        moe=preset["moe"],
        expert_offload=preset["expert_offload"],
        device=device,
    )
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ids = tok.encode(args.prompt, allow_special=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.max_tokens, temperature=0.7,
                             top_k=40, top_p=0.9, repetition_penalty=1.1)
    text = tok.decode(out[0].tolist())
    print(f"\n--- xl-moe generation ({args.max_tokens} new tokens) ---")
    print(text[:400])

    total_m = model.num_parameters() / 1e6
    active_m = model.active_parameters() / 1e6
    print(f"\nparams total : {total_m:.1f}M")
    print(f"params active: {active_m:.1f}M  (per-forward routing)")
    if torch.cuda.is_available():
        print(f"peak VRAM    : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()