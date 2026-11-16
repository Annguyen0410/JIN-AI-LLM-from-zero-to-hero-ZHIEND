"""
Legacy char-level tokenizer — kept for loading old checkpoints.

New training uses ``bpe_tokenizer.BPETokenizer`` instead.
"""

from __future__ import annotations

import json
import os

import config


class CharTokenizer:
    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.vocab_size: int = 0

    def build(self, text: str):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def load(self, path: str = config.VOCAB_PATH) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data.get("char_to_id", {})
        self.id_to_char = data.get("id_to_char", {v: k for k, v in self.char_to_id.items()})
        self.vocab_size = len(self.char_to_id)

    def encode(self, text: str, return_stats: bool = False) -> list[int] | tuple[list[int], int]:
        ids = [self.char_to_id.get(ch, 0) for ch in text]
        if return_stats:
            return ids, 0
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char.get(i, "") for i in ids)

    def to_state_dict(self) -> dict:
        return {"char_to_id": self.char_to_id}

    def load_state_dict(self, state: dict) -> None:
        self.char_to_id = state.get("char_to_id", {})
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)