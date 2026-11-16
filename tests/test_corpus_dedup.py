"""Corpus dedup: kill the canned-answer repetition that caused the echo.

The May corpus had 30,020 assistant turns but only 755 distinct answers, so the
model learned a fixed answer set and parroted it. These tests pin the two guards
that fix it: whole-conversation duplicates and over-repeated answer sequences.
"""

from __future__ import annotations

import unittest

import build_instruct_dataset as bid
from chat_template import ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER


def conv(user: str, assistant: str) -> list[dict]:
    return [
        {"role": ROLE_SYSTEM, "content": "You are Jin."},
        {"role": ROLE_USER, "content": user},
        {"role": ROLE_ASSISTANT, "content": assistant},
    ]


class TestDedup(unittest.TestCase):
    def test_exact_duplicates_are_dropped(self):
        data = [conv("Hi", "Hello!")] * 40 + [conv("Bye", "Goodbye!")]
        kept, stats = bid.dedup_conversations(data)
        self.assertEqual(len(kept), 2)
        self.assertEqual(stats["dropped_duplicates"], 39)

    def test_duplicates_that_differ_only_in_whitespace_and_case_are_dropped(self):
        data = [conv("Hi", "Hello  there!"), conv("hi", "hello there!")]
        kept, stats = bid.dedup_conversations(data)
        self.assertEqual(len(kept), 1)
        self.assertEqual(stats["dropped_duplicates"], 1)

    def test_repeated_answer_sequences_are_capped(self):
        # Different questions, identical answer set — the "2 + 2 = 4." pattern.
        data = [conv(f"What is {n} + {n}?", "2 + 2 = 4.") for n in range(10)]
        kept, stats = bid.dedup_conversations(data, max_identical_answers=2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(stats["dropped_repeated_answers"], 8)

    def test_answer_cap_can_be_disabled(self):
        data = [conv(f"Q{n}", "Same answer.") for n in range(10)]
        kept, _ = bid.dedup_conversations(data, max_identical_answers=0)
        self.assertEqual(len(kept), 10)

    def test_distinct_conversations_survive(self):
        data = [conv(f"Q{n}", f"A{n}") for n in range(50)]
        kept, stats = bid.dedup_conversations(data)
        self.assertEqual(len(kept), 50)
        self.assertEqual(stats["dropped_duplicates"], 0)
        self.assertEqual(stats["dropped_repeated_answers"], 0)

    def test_fingerprint_ignores_key_order_of_roles(self):
        a = conv("Hi", "Hello")
        b = conv("Hi", "Hello")
        self.assertEqual(bid.conversation_fingerprint(a), bid.conversation_fingerprint(b))


class TestAnswerStats(unittest.TestCase):
    def test_reports_the_echo_metric(self):
        data = [conv("Q1", "2 + 2 = 4."), conv("Q2", "2 + 2 = 4."), conv("Q3", "Unique.")]
        stats = bid.answer_stats(data)
        self.assertEqual(stats["turns"], 3)
        self.assertEqual(stats["unique_answers"], 2)
        self.assertAlmostEqual(stats["repeat_ratio"], 1 / 3, places=4)
        self.assertEqual(stats["top_answers"][0], ("2 + 2 = 4.", 2))

    def test_empty_corpus_does_not_divide_by_zero(self):
        stats = bid.answer_stats([])
        self.assertEqual(stats["turns"], 0)
        self.assertEqual(stats["repeat_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
