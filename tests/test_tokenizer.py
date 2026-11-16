"""
BPE tokenizer tests: lossless round-trips (Vietnamese, emoji, code, math),
special-token atomicity, and state-dict round-trip (checkpoint embedding).

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from bpe_tokenizer import BPETokenizer, DEFAULT_SPECIAL_TOKENS


def make_tokenizer(vocab_size: int = 512) -> BPETokenizer:
    corpus = (
        "Xin chào thế giới! Hello world. "
        "def foo(x):\n    return x + 1  # comment\n"
        "The quick brown fox jumps over the lazy dog. "
        "1234567890 +=*/%<>[]{} ấ ă đ ê ô ơ ư. 👋🎉✨"
    ) * 20
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size)
    return tok


class TestRoundTrip(unittest.TestCase):
    def test_vietnamese_emoji_code(self):
        tok = make_tokenizer()
        samples = [
            "Xin chào, tôi là Jin. Rất vui được gặp bạn!",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "Emoji mixed: hello 👋 world 🎉 test ✨",
            "Math & symbols: 2 + 2 = 4, 15% of 80, a < b > c, {x: [1, 2, 3]}",
            "Vietnamese diacritics: ấ ắ ằ ẵ ặ ă â đ ê ô ơ ư ứ ừ",
        ]
        for text in samples:
            ids = tok.encode(text, allow_special=True)
            self.assertEqual(tok.decode(ids), text, f"round-trip failed for: {text!r}")

    def test_special_tokens_atomic(self):
        tok = make_tokenizer()
        for special in DEFAULT_SPECIAL_TOKENS:
            ids = tok.encode(special, allow_special=True)
            self.assertEqual(ids, [tok.special_to_id[special]], f"{special} must be one token")
            self.assertEqual(tok.decode(ids), special)

    def test_special_disabled_encodes_bytes(self):
        tok = make_tokenizer()
        # With allow_special=False the raw bytes are used (not the token id).
        for special in DEFAULT_SPECIAL_TOKENS:
            ids = tok.encode(special, allow_special=False)
            self.assertNotEqual(ids, [tok.special_to_id[special]])

    def test_state_dict_roundtrip(self):
        tok = make_tokenizer()
        state = tok.to_state_dict()
        tok2 = BPETokenizer()
        tok2.load_state_dict(state)
        self.assertEqual(tok2.vocab_size, tok.vocab_size)
        self.assertEqual(tok2.special_to_id, tok.special_to_id)
        text = "Kiểm tra tokenizer sau khi lưu/đọc state dict."
        self.assertEqual(tok2.decode(tok2.encode(text, allow_special=True)), text)


if __name__ == "__main__":
    unittest.main()