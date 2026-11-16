"""
JIN AI text generation script.
"""

from __future__ import annotations

import argparse

import torch

import config
from bpe_tokenizer import BPETokenizer
from utils import build_inference_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with JIN AI")
    parser.add_argument("--prompt", type=str, default="", help="Starting text")
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument("--top_p", type=float, default=config.TOP_P)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=config.REPETITION_PENALTY,
    )
    parser.add_argument("--max_tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained checkpoint "
                             "(default: jingpt_chat.pt, else jingpt_best.pt).")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="NAME",
        choices=config.DEVICE_CHOICES,
        help="Device override. Default: JINGPT_DEVICE or auto (cuda if present).",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=config.WEIGHT_QUANT or "none",
        choices=["none", "int8", "int4"],
        help="Weight-only quantization (int8 per-channel / int4 group-128).",
    )
    parser.add_argument("--static-kv", action="store_true",
                        help="Pre-allocated KV buffers (no per-step cat).")
    parser.add_argument("--int8-kv", action="store_true",
                        help="int8 per-head-scale KV cache (4x less memory).")
    args = parser.parse_args()

    # config.DEVICE is a "cpu" placeholder until set_device() runs.
    config.set_device(args.device)

    model, tokenizer, checkpoint = build_inference_bundle(
        args.checkpoint or config.get_checkpoint_path(),
        quant=None if args.quant == "none" else args.quant,
    )

    if args.prompt:
        ids = tokenizer.encode(args.prompt)
        if not ids:
            raise ValueError("Prompt contains no characters present in the active vocab.")
        idx = torch.tensor([ids], dtype=torch.long, device=config.DEVICE)
    else:
        start_token = tokenizer.special_to_id.get("<|endoftext|>", 0)
        idx = torch.full((1, 1), start_token, dtype=torch.long, device=config.DEVICE)

    output = model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k and args.top_k > 0 else None,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        static_kv=args.static_kv or None,
        kv_int8=args.int8_kv or None,
    )

    print("\n" + "=" * 72)
    print("JIN AI Generated Text")
    print("=" * 72)
    print(f"Checkpoint file: {checkpoint.get('_loaded_from', args.checkpoint)}")
    print(f"Checkpoint step: {checkpoint.get('iteration', '?')}")
    print(tokenizer.decode(output[0].tolist()))
    print("=" * 72)


if __name__ == "__main__":
    main()
