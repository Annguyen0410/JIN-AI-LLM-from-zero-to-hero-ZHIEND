"""
JIN AI interactive chat.

Works with two kinds of checkpoints:

- Legacy char-level checkpoint: plain text prompt continuation, no
  special turn handling. We keep that mode for backward compatibility.
- New BPE chat checkpoint: enforces the ChatML-style turn structure
  defined in :mod:`chat_template` and stops generation at ``<|end|>``
  or ``<|endoftext|>``.
"""

from __future__ import annotations

import argparse

import torch

import config
from bpe_tokenizer import BPETokenizer
from chat_template import (
    DEFAULT_SYSTEM_PROMPT,
    END_TAG,
    EOT_TAG,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TAG,
    ROLE_USER,
    format_conversation,
)
from utils import build_inference_bundle


BANNER = """
+--------------------------------------------------------------+
|                         JIN AI Chat                          |
|   Type 'quit' to exit, 'reset' to clear conversation memory. |
+--------------------------------------------------------------+
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with JIN AI")
    parser.add_argument(
        "--checkpoint",
        default=config.CHECKPOINT_PATH,
        help="Path to a trained checkpoint.",
    )
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    parser.add_argument("--repetition-penalty", type=float,
                        default=config.REPETITION_PENALTY)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--max-turns", type=int, default=12,
                        help="How many past turns to keep in the prompt.")
    parser.add_argument("--ctx", type=int, default=None, metavar="N",
                        help="Serve a longer context than trained via RoPE scaling.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="NAME",
        choices=config.DEVICE_CHOICES,
        help="Device override. Default: JINGPT_DEVICE or auto.",
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
    return parser.parse_args()


def _stop_token_ids(tokenizer) -> set[int]:
    if not isinstance(tokenizer, BPETokenizer):
        return set()
    ids: set[int] = set()
    for tag in (END_TAG, EOT_TAG):
        tok_id = tokenizer.special_to_id.get(tag)
        if tok_id is not None:
            ids.add(tok_id)
    return ids


@torch.no_grad()
def generate_with_stop(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    stop_ids: set[int],
    device: str,
    static_kv: bool = False,
    kv_int8: bool = False,
) -> list[int]:
    import torch.nn.functional as F

    was_training = model.training
    model.eval()

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    kv_caches = model.empty_kv_cache(static=static_kv, int8=kv_int8)

    # First step prefills the whole prompt into the cache; afterwards only
    # the newest token is fed (the cache holds the rest).
    idx_cond = idx[:, -model.max_positions :]
    logits, _ = model(idx_cond, kv_caches=kv_caches)

    for _ in range(max_new_tokens):
        logits = logits[:, -1, :]

        if repetition_penalty and repetition_penalty > 1.0:
            seen_tokens = torch.unique(idx_cond[0]).unsqueeze(0)
            logits = model._apply_repetition_penalty(logits, seen_tokens, repetition_penalty)

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove_mask = cumulative > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                sorted_logits[remove_mask] = float("-inf")
                filled = torch.full_like(logits, float("-inf"))
                filled.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
                logits = filled
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        token_id = int(next_id.item())
        if token_id in stop_ids:
            break
        generated.append(token_id)
        idx = torch.cat([idx, next_id], dim=1)
        model._trim_kv_cache(kv_caches, model.max_positions - 1)
        logits, _ = model(next_id, kv_caches=kv_caches)

    if was_training:
        model.train()
    return generated


def truncate_history(messages: list[dict], max_turns: int) -> list[dict]:
    """Keep the system prompt plus the last ``max_turns`` user/assistant turns."""

    system = [m for m in messages if m["role"] == ROLE_SYSTEM][-1:]
    other = [m for m in messages if m["role"] != ROLE_SYSTEM]
    # Each turn = one user + one assistant message, so keep last 2*max_turns.
    other = other[-(max_turns * 2) :]
    return system + other


def chat_loop(args: argparse.Namespace) -> None:
    model, tokenizer, checkpoint = build_inference_bundle(
        args.checkpoint,
        quant=None if args.quant == "none" else args.quant,
        override_config=config.context_override(args.ctx) if args.ctx else None,
    )

    is_bpe = isinstance(tokenizer, BPETokenizer)
    stop_ids = _stop_token_ids(tokenizer)

    print(BANNER)
    print(
        f"Loaded {checkpoint.get('_loaded_from', args.checkpoint)} "
        f"at step {checkpoint.get('iteration', '?')} "
        f"on {config.DEVICE}."
    )
    print(f"Tokenizer: {'BPE' if is_bpe else 'char'} (vocab {tokenizer.vocab_size:,})")
    if not is_bpe:
        print(
            "[Info] Running in legacy mode (no chat template). "
            "The model will just continue your text."
        )
    print()

    messages: list[dict] = []
    if is_bpe and args.system_prompt:
        messages.append({"role": ROLE_SYSTEM, "content": args.system_prompt})

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            messages = []
            if is_bpe and args.system_prompt:
                messages.append({"role": ROLE_SYSTEM, "content": args.system_prompt})
            print("[Memory cleared]\n")
            continue

        if is_bpe:
            messages.append({"role": ROLE_USER, "content": user_input})
            messages_for_prompt = truncate_history(messages, args.max_turns)
            prompt_text = format_conversation(
                messages_for_prompt,
                add_generation_prompt=True,
                default_system=None,
            )
            prompt_ids = tokenizer.encode(prompt_text, allow_special=True)

            generated_ids = generate_with_stop(
                model,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop_ids=stop_ids,
                device=config.DEVICE,
                static_kv=args.static_kv,
                kv_int8=args.int8_kv,
            )
            reply = tokenizer.decode(generated_ids).strip()
            # If the model emitted another role tag, stop at the first one.
            for tag in ROLE_TAG.values():
                idx = reply.find(tag)
                if idx != -1:
                    reply = reply[:idx].strip()
            if not reply:
                reply = "(no reply)"
            messages.append({"role": ROLE_ASSISTANT, "content": reply})
            print(f"\nJin: {reply}\n")
        else:
            ids = tokenizer.encode(user_input)
            if not ids:
                print("No characters from that prompt exist in the current vocabulary.")
                continue
            prompt_text = tokenizer.decode(ids)
            idx = torch.tensor([ids], dtype=torch.long, device=config.DEVICE)
            output = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            full_text = tokenizer.decode(output[0].tolist())
            print(f"\nJin:{full_text[len(prompt_text):]}\n")


def main() -> None:
    args = parse_args()
    config.set_device(args.device)
    chat_loop(args)


if __name__ == "__main__":
    main()
