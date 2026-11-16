"""
Tokenize chat conversations into (input_ids, target_ids, loss_mask)
triples for instruction tuning.

Why per-token loss masking matters
---------------------------------
A naive language-model loss treats every token equally. For a chat
example like::

    <|user|>\n  what is 2+2 ?  \n<|end|>\n<|assistant|>\n  4  \n<|end|>

the model gets penalised for failing to predict the *user's* tokens
just as much as the assistant's. With a small from-scratch model this
is wasteful: the model spends most of its capacity learning to mimic
how users phrase questions and how the chat tags look, and only a tiny
fraction on the actual replies.

The fix every modern instruction-tuned LM uses is the same: only
back-propagate loss on assistant tokens. That means the gradient is
focused on "given this conversation so far, write the right reply".
With the same compute budget, this can be the difference between a
model that babbles and one that holds a conversation.

This module produces, for each conversation:

    input_ids: [t0, t1, ..., tN-1]
    target_ids: [t1, t2, ..., tN]            # standard next-token shift
    loss_mask:  [m0, m1, ..., mN-1]           # 1 only for assistant content tokens
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

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
)


# ── Building one masked example ───────────────────────────────────────
def build_masked_example(
    messages: list[dict],
    tokenizer: BPETokenizer,
    *,
    train_on_system: bool = False,
    add_default_system: bool = True,
) -> tuple[list[int], list[int]]:
    """Return ``(token_ids, loss_mask)`` for one conversation.

    ``loss_mask`` has length equal to ``token_ids`` and is 1 only on
    tokens we want the model to learn to produce — by default, only the
    body of assistant turns.
    """

    has_system = any(m.get("role") == ROLE_SYSTEM for m in messages)
    if not has_system and add_default_system:
        messages = [{"role": ROLE_SYSTEM, "content": DEFAULT_SYSTEM_PROMPT}] + list(messages)

    token_ids: list[int] = []
    loss_mask: list[int] = []

    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role not in ROLE_TAG or not content:
            continue
        tag = ROLE_TAG[role]
        prefix_text = f"{tag}\n"
        suffix_text = f"\n{END_TAG}\n"

        prefix_ids = tokenizer.encode(prefix_text, allow_special=True)
        content_ids = tokenizer.encode(content, allow_special=False)
        suffix_ids = tokenizer.encode(suffix_text, allow_special=True)

        train_here = (
            (role == ROLE_ASSISTANT)
            or (role == ROLE_SYSTEM and train_on_system)
        )
        token_ids.extend(prefix_ids + content_ids + suffix_ids)
        loss_mask.extend(
            [0] * len(prefix_ids)
            + ([1] * len(content_ids) if train_here else [0] * len(content_ids))
            + ([1] * len(suffix_ids) if train_here else [0] * len(suffix_ids))
        )

    eot_ids = tokenizer.encode(EOT_TAG + "\n", allow_special=True)
    token_ids.extend(eot_ids)
    loss_mask.extend([0] * len(eot_ids))

    return token_ids, loss_mask


# ── Loading conversations from one or more JSONL files ────────────────
def iter_conversations(
    paths: Iterable[str],
    *,
    max_per_file: int | None = None,
    max_total: int | None = None,
) -> Iterable[list[dict]]:
    """Stream conversations from JSONL files.

    ``max_per_file`` stops reading a file after that many conversations
    (so a 1M-row shard like OpenHermes never sits fully in RAM) and
    ``max_total`` stops everything once the global budget is reached.
    """
    total = 0
    for path in paths:
        if not os.path.exists(path):
            continue
        if max_total is not None and total >= max_total:
            break
        per_file = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if max_per_file is not None and per_file >= max_per_file:
                    break
                if max_total is not None and total >= max_total:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = record.get("messages")
                if not isinstance(messages, list):
                    continue
                kept: list[dict] = []
                for m in messages:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role")
                    content = m.get("content")
                    if role in {ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT} and content:
                        kept.append({"role": role, "content": str(content)})
                while kept and kept[-1]["role"] != ROLE_ASSISTANT:
                    kept.pop()
                if any(m["role"] == ROLE_ASSISTANT for m in kept):
                    yield kept
                    total += 1
                    per_file += 1


# ── Pack many examples into fixed-length blocks ───────────────────────
class PackedChatDataset(Dataset):
    """Concatenate many tokenized chat examples and slice into fixed
    blocks. Each slice carries its own loss_mask so block boundaries
    don't accidentally train across unrelated conversations.
    """

    def __init__(
        self,
        token_blocks: torch.Tensor,
        mask_blocks: torch.Tensor,
    ):
        if token_blocks.shape != mask_blocks.shape:
            raise ValueError("token and mask block shapes must match")
        self.tokens = token_blocks
        self.masks = mask_blocks

    @classmethod
    def from_conversations(
        cls,
        conversations: list[list[dict]],
        tokenizer: BPETokenizer,
        block_size: int,
        *,
        seed: int = 1337,
        max_examples: int | None = None,
        verbose: bool = True,
    ) -> "PackedChatDataset":
        rng = random.Random(seed)
        convs = list(conversations)
        rng.shuffle(convs)
        if max_examples:
            convs = convs[:max_examples]

        all_ids: list[int] = []
        all_mask: list[int] = []
        eos = tokenizer.special_to_id.get(EOT_TAG)
        for i, conv in enumerate(convs):
            ids, mask = build_masked_example(conv, tokenizer)
            # Hard separator between conversations so attention can never
            # spill across documents.
            if eos is not None and all_ids and all_ids[-1] != eos:
                all_ids.append(eos)
                all_mask.append(0)
            all_ids.extend(ids)
            all_mask.extend(mask)
            if verbose and (i + 1) % 5_000 == 0:
                print(f"[chat-dataset] tokenized {i + 1:,}/{len(convs):,}")

        n_blocks = max(1, (len(all_ids) - 1) // block_size)
        usable = n_blocks * block_size + 1
        all_ids = all_ids[:usable]
        all_mask = all_mask[:usable]

        token_tensor = torch.tensor(all_ids, dtype=torch.long)
        mask_tensor = torch.tensor(all_mask, dtype=torch.float32)
        # We need pairs (input, target) so slice into block_size + 1
        token_blocks = token_tensor[: n_blocks * (block_size + 1)].view(-1, 1)
        # Easier: store flat and let __getitem__ slice.
        return cls.from_flat(token_tensor, mask_tensor, block_size)

    @classmethod
    def from_flat(
        cls,
        token_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        block_size: int,
    ) -> "PackedChatDataset":
        # Build sliding windows aligned to block_size.
        n_blocks = max(1, (len(token_tensor) - 1) // block_size)
        token_tensor = token_tensor[: n_blocks * block_size + 1]
        mask_tensor = mask_tensor[: n_blocks * block_size + 1]
        # Create non-overlapping training windows.
        starts = list(range(0, len(token_tensor) - block_size, block_size))
        if not starts:
            raise ValueError("Not enough tokens for one complete packed block")
        token_blocks = torch.stack([token_tensor[s : s + block_size + 1] for s in starts])
        mask_blocks = torch.stack([mask_tensor[s : s + block_size + 1] for s in starts])
        # Empty assistant-loss windows return a misleading zero validation
        # loss and waste whole optimizer updates. Keep supervised windows only.
        supervised = mask_blocks[:, 1:].sum(dim=1) > 0
        if not supervised.any():
            raise ValueError("Packed dataset contains no supervised target tokens")
        ds = object.__new__(cls)
        ds.tokens = token_blocks[supervised]
        ds.masks = mask_blocks[supervised]
        return ds

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, idx: int):
        block_tokens = self.tokens[idx]
        block_mask = self.masks[idx]
        x = block_tokens[:-1]
        y = block_tokens[1:]
        m = block_mask[1:]
        return x, y, m


# ── Convenience entry point ───────────────────────────────────────────
def build_dataset(
    seed_path: str | None = None,
    extra_dirs: Iterable[str] = (),
    *,
    duplication: int = 30,
    block_size: int = 512,
    seed: int = 1337,
    max_convs: int = 0,
    extra_dirs_full: Iterable[str] = (),
    val_fraction: float = 0.0,
    validation_dirs: Iterable[str] = (),
) -> tuple:
    """Load conversations, train (or load) the BPE, build a packed dataset.

    ``max_convs`` caps the number of conversations taken from ``extra_dirs``
    (the big HF shards — 0 disables the cap). Seed conversations are always
    kept in full and receive the ``duplication`` multiplier; HF conversations
    are included once so a 30k-conv dataset never balloons into an OOM.
    ``extra_dirs_full`` holds precious, already-capped dirs (e.g. grounded
    web-search data) that bypass ``max_convs`` entirely.
    """

    if not os.path.exists(config.BPE_PATH):
        raise FileNotFoundError(
            f"BPE not found at {config.BPE_PATH}. "
            "Train it first with: python bpe_tokenizer.py "
            "--data data/chat_train.txt --out bpe.json"
        )
    tokenizer = BPETokenizer()
    tokenizer.load(config.BPE_PATH)

    # Resolve configuration at call time (tests and reproducible launchers
    # may override paths after this module is imported).
    seed_path = seed_path or config.SEED_CONVERSATIONS_PATH
    # Seed conversations (small, hand-authored — define Jin's identity).
    seed_paths = [seed_path]
    seed_dir = Path(seed_path).parent
    if seed_dir.exists():
        for extra in sorted(seed_dir.glob("*.jsonl")):
            if extra.resolve() != Path(seed_path).resolve():
                seed_paths.append(str(extra))
    seed_convs = list(iter_conversations(seed_paths))

    # HF conversations (big shards) — budget-capped, balanced across files.
    hf_paths: list[str] = []
    for d in extra_dirs:
        d = str(d or "").strip()
        if d and os.path.isdir(d):
            hf_paths.extend(str(p) for p in sorted(Path(d).glob("*.jsonl")))
    hf_convs: list[list[dict]] = []
    if hf_paths:
        hf_budget = max(0, max_convs - len(seed_convs)) if max_convs > 0 else None
        if hf_budget is None or hf_budget > 0:
            per_file = max(1, hf_budget // len(hf_paths)) if hf_budget is not None else None
            hf_convs = list(iter_conversations(
                hf_paths, max_per_file=per_file, max_total=hf_budget,
            ))

    # Grounded web-search data (precious — bypasses the cap entirely).
    full_paths: list[str] = []
    for d in extra_dirs_full:
        d = str(d or "").strip()
        if d and os.path.isdir(d):
            full_paths.extend(str(p) for p in sorted(Path(d).glob("*.jsonl")))
    full_convs = list(iter_conversations(full_paths)) if full_paths else []

    # Unique conversations first; duplicate seeds only AFTER a disjoint split.
    # Random block splitting leaks duplicate dialogues into validation.
    from build_instruct_dataset import conversation_fingerprint
    from data_quality import check_contamination, load_probe_texts
    from chat_template import format_training_document
    unique = {}
    seed_keys = set()
    for conv in seed_convs:
        seed_keys.add(conversation_fingerprint(conv))
    for conv in seed_convs + hf_convs + full_convs:
        unique.setdefault(conversation_fingerprint(conv), conv)
    clean_docs, contamination = check_contamination(
        [format_training_document(c) for c in unique.values()])
    clean_set = set(clean_docs)
    unique = {key: conv for key, conv in unique.items() if format_training_document(conv) in clean_set}
    validation_paths = [str(p) for directory in validation_dirs
                        for p in sorted(Path(directory).glob("*.jsonl"))]
    if validation_dirs:
        if not validation_paths:
            raise ValueError("Explicit SFT validation directory has no JSONL files")
        validation_unique = {conversation_fingerprint(c): c for c in iter_conversations(validation_paths)}
        overlap = set(unique) & set(validation_unique)
        if overlap:
            raise ValueError("Explicit SFT train/validation conversation overlap")
        clean_val, val_contamination = check_contamination(
            [format_training_document(c) for c in validation_unique.values()])
        clean_val_set = set(clean_val)
        validation_unique = {key: conv for key, conv in validation_unique.items()
                             if format_training_document(conv) in clean_val_set}
        if not unique or not validation_unique:
            raise ValueError("Explicit SFT split is empty after filtering")
        train_convs = [conv for key, conv in unique.items()
                       for _ in range(max(1, duplication) if key in seed_keys else 1)]
        train_ds = PackedChatDataset.from_conversations(train_convs, tokenizer, block_size, seed=seed)
        val_ds = PackedChatDataset.from_conversations(list(validation_unique.values()), tokenizer, block_size, seed=seed)
        train_ds.split_manifest = {"method":"explicit_conversation_disjoint_validation",
            "train_conversation_hashes":sorted(unique),"val_conversation_hashes":sorted(validation_unique),
            "contamination_removed":len(contamination)+len(val_contamination)}
        print(f"[chat-dataset] Explicit split: {len(unique)} train / {len(validation_unique)} val conversations")
        return train_ds, val_ds, tokenizer
    if val_fraction:
        if not 0 < val_fraction < 0.5 or len(unique) < 2:
            raise ValueError("Need at least two unique conversations and val fraction in (0,.5)")
        keys = sorted(unique)
        random.Random(seed).shuffle(keys)
        n_val = max(1, int(len(keys) * val_fraction))
        val_keys, train_keys = keys[:n_val], keys[n_val:]
        train_convs = [unique[key] for key in train_keys for _ in range(max(1, duplication) if key in seed_keys else 1)]
        val_convs = [unique[key] for key in val_keys]
        train_ds = PackedChatDataset.from_conversations(train_convs, tokenizer, block_size, seed=seed)
        val_ds = PackedChatDataset.from_conversations(val_convs, tokenizer, block_size, seed=seed)
        train_ds.split_manifest = {"method":"unique_conversation_before_duplication",
            "train_conversation_hashes":train_keys,"val_conversation_hashes":val_keys,
            "contamination_removed":len(contamination)}
        print(f"[chat-dataset] Disjoint conversations: {len(train_keys)} train / {len(val_keys)} val; "
              f"blocks {len(train_ds)} / {len(val_ds)}")
        return train_ds, val_ds, tokenizer
    conversations = [conv for key, conv in unique.items()
                     for _ in range(max(1, duplication) if key in seed_keys else 1)]

    print(
        f"[chat-dataset] {len(conversations):,} conversations "
        f"({len(seed_convs):,} seed x{max(1, duplication)} + {len(hf_convs):,} hf "
        f"+ {len(full_convs):,} grounded) "
        f"from {len(seed_paths) + len(hf_paths) + len(full_paths)} file(s)."
    )
    ds = PackedChatDataset.from_conversations(
        conversations,
        tokenizer,
        block_size=block_size,
        seed=seed,
    )
    print(
        f"[chat-dataset] packed into {len(ds):,} blocks of {block_size} tokens; "
        f"trainable tokens = {int(ds.masks.sum()):,} / {ds.masks.numel():,}"
        f" ({100 * float(ds.masks.mean()):.1f}%)"
    )
    return ds, tokenizer


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the masked chat dataset.")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--duplication", type=int, default=30)
    ap.add_argument("--max-convs", type=int, default=30_000,
                    help="Cap HF conversations (0 = no cap).")
    args = ap.parse_args()

    hf_dir = os.path.join(config.ROOT_DIR, "data", "seed", "hf")
    extra_dirs = (hf_dir,) if os.path.isdir(hf_dir) else ()
    ds, tok = build_dataset(
        duplication=args.duplication,
        block_size=args.block_size,
        max_convs=args.max_convs,
        extra_dirs=extra_dirs,
    )
    print("Total blocks:", len(ds))
    print("First block trainable token ratio:", float(ds.masks[0].mean()))
