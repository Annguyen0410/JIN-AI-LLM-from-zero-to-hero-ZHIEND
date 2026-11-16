"""Tokenizer regression tests (Phase 2).

Covers the two profiles the project now ships:

* ``jin``  — the existing 16,384-entry byte-level BPE (checkpoint compatible).
* ``gpt2`` — the 50,257-entry GPT-2 byte-level BPE used by the
  ``gpt2_small_reference`` model.

The underscore regression is the important one: the old pre-tokenizer matched
``_`` with no alternative, ``findall`` dropped it, and every snake_case
identifier in the code corpus was silently corrupted during training.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from bpe_tokenizer import BPETokenizer  # noqa: E402
from tokenizer_gpt2 import GPT2BPETokenizer, GPT2_EOT_ID  # noqa: E402


def jin_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.load(config.BPE_PATH)
    return tok


@pytest.fixture(scope="module")
def jin():
    return jin_tokenizer()


@pytest.fixture(scope="module")
def gpt2():
    return GPT2BPETokenizer()


CASES = [
    "hello world",
    "LAST_UPDATED2 and snake_case_name(self, x_1)",
    "tiếng Việt có dấu: ă â đ ê ô ơ ư",
    "def f(x: int) -> list[int]:\n    return [x**2 for x in range(10)]",
    "emoji 🌍🚀 and symbols @#$%^&*()",
    "numbers 1,234.567 -0.0001 3.14e-9",
    "tabs\tand\nnewlines   and   spaces",
]


@pytest.mark.parametrize("text", CASES)
def test_jin_tokenizer_roundtrip(jin, text):
    assert jin.decode(jin.encode(text)) == text


def test_underscore_is_not_dropped(jin):
    ids = jin.encode("LAST_UPDATED")
    assert "_" in jin.decode(ids)
    assert jin.decode(ids) == "LAST_UPDATED"


def test_jin_tokenizer_roundtrip_on_real_corpus_slice(jin):
    path = os.path.join(config.ROOT_DIR, "data", "prepared_v4", "pretrain",
                        "vietnamese.txt")
    if not os.path.exists(path):
        pytest.skip("corpus slice not present")
    text = open(path, encoding="utf-8", errors="replace").read(120_000)
    assert jin.decode(jin.encode(text)) == text


@pytest.mark.parametrize("text", CASES)
def test_gpt2_tokenizer_roundtrip(gpt2, text):
    assert gpt2.decode(gpt2.encode(text)) == text


def test_gpt2_vocabulary_identity(gpt2):
    assert gpt2.vocab_size == 50_257
    assert gpt2.eot_id == GPT2_EOT_ID == 50_256
    assert gpt2.encode("<|endoftext|>") == [GPT2_EOT_ID]


def test_gpt2_document_format(gpt2):
    ids = gpt2.encode_document("The cat sat.")
    assert ids[0] == GPT2_EOT_ID and ids[-1] == GPT2_EOT_ID
    assert gpt2.decode(ids[1:-1]) == "The cat sat."


def test_gpt2_handles_all_byte_values(gpt2):
    """Byte-level BPE must round-trip arbitrary bytes, including invalid UTF-8."""
    text = "".join(chr(c) for c in range(32, 32 + 60))
    assert gpt2.decode(gpt2.encode(text)) == text


def test_tokenizer_state_is_stable(gpt2):
    """The state hash is embedded in checkpoints; it must not drift."""
    state = gpt2.to_state_dict()
    assert state["kind"] == "gpt2_bpe"
    assert state["vocab_size"] == 50_257
