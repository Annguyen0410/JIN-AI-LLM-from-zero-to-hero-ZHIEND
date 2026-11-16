"""
Chat template tests: assistant-only loss masking, conversation formatting,
and stop-token handling.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from bpe_tokenizer import BPETokenizer
from chat_dataset import build_masked_example
from chat_template import (
    DEFAULT_SYSTEM_PROMPT,
    END_TAG,
    EOT_TAG,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
    STOP_TOKEN_STRINGS,
    format_conversation,
    format_training_document,
)


def make_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train("Hello world! Xin chào. Reply: yes. " * 50, vocab_size=512)
    return tok


class TestMasking(unittest.TestCase):
    def test_only_assistant_tokens_are_masked(self):
        tok = make_tokenizer()
        messages = [
            {"role": ROLE_SYSTEM, "content": "You are Jin."},
            {"role": ROLE_USER, "content": "What is 2+2?"},
            {"role": ROLE_ASSISTANT, "content": "It is 4."},
        ]
        ids, mask = build_masked_example(messages, tok)
        self.assertEqual(len(ids), len(mask))

        text = tok.decode(ids)
        user_body = tok.encode("What is 2+2?", allow_special=False)
        asst_body = tok.encode("It is 4.", allow_special=False)
        masked_text = tok.decode([i for i, m in zip(ids, mask) if m])
        # Assistant body must be fully masked...
        self.assertIn(tok.decode(asst_body), masked_text)
        # ...and the user body must not be.
        self.assertNotIn(tok.decode(user_body), masked_text)
        # No EOT inside the masked span (EOT is a boundary, never trained on).
        eot_ids = tok.encode(EOT_TAG, allow_special=True)
        masked_ids = [i for i, m in zip(ids, mask) if m]
        self.assertNotIn(eot_ids[0], masked_ids)

    def test_no_assistant_turn_means_no_mask(self):
        tok = make_tokenizer()
        messages = [
            {"role": ROLE_SYSTEM, "content": "System note."},
            {"role": ROLE_USER, "content": "Just a question?"},
        ]
        ids, mask = build_masked_example(messages, tok)
        self.assertEqual(sum(mask), 0)


class TestFormatting(unittest.TestCase):
    def test_generation_prompt_ends_with_assistant_tag(self):
        text = format_conversation(
            [{"role": ROLE_USER, "content": "Hi"}],
            add_generation_prompt=True, default_system=None,
        )
        self.assertTrue(text.endswith("<|assistant|>\n"))

    def test_training_document_ends_with_eot(self):
        text = format_training_document(
            [{"role": ROLE_USER, "content": "Hi"},
             {"role": ROLE_ASSISTANT, "content": "Hello!"}],
            default_system=None,
        )
        self.assertTrue(text.rstrip().endswith(EOT_TAG))

    def test_default_system_injected_when_absent(self):
        text = format_conversation([{"role": ROLE_USER, "content": "Hi"}])
        self.assertIn("<|system|>", text)
        self.assertIn(DEFAULT_SYSTEM_PROMPT[:20], text)

    def test_stop_tokens_are_defined(self):
        self.assertIn(END_TAG, STOP_TOKEN_STRINGS)
        self.assertIn(EOT_TAG, STOP_TOKEN_STRINGS)


if __name__ == "__main__":
    unittest.main()