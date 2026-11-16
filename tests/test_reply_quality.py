"""Anti-echo decoding tests: n-gram ban, loop breaker, reply trimming.

The SFT 4,000-step checkpoint answered a greeting with
``"I'm Jin, a small language model — a small language model — ..."`` and ran to
the token cap. A token-level repetition penalty cannot stop that (every token is
individually legal, only the n-gram is not), so these tests pin the behaviour of
the three layers that now do: ban n-grams, stop on a looping tail, trim the
text. Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import torch

from model import (
    JinAI,
    repeated_repeat_count,
    repeated_tail_period,
)
from reply_quality import looks_like_echo, stop_reason_label, trim_repeated_tail

try:                      # ``python -m unittest discover -s tests``
    from test_model_shapes import VOCAB, make_model
except ModuleNotFoundError:   # ``python -m unittest tests.test_reply_quality``
    from tests.test_model_shapes import VOCAB, make_model

# Verbatim echo from the 4,000-step SFT sample in train_log.txt.
ECHO = ("Hello! I'm Jin, a small language model — a small language model — "
        "a small language model —")


class TestRepeatedTailPeriod(unittest.TestCase):
    def test_detects_echo(self):
        chunk = [1, 2, 3, 4]
        self.assertEqual(repeated_tail_period(chunk * 3), 4)

    def test_smallest_period_wins(self):
        self.assertEqual(repeated_tail_period([7] * 12), 1)

    def test_short_chunks_need_more_repeats(self):
        # "!!" x3 is style, not collapse: period 1 needs 12 copies.
        self.assertIsNone(repeated_tail_period([9, 9, 9]))
        self.assertEqual(repeated_tail_period([9] * 12), 1)
        # A 4-token chunk only needs 3 copies.
        self.assertEqual(repeated_tail_period([5, 6, 7, 8] * 3), 4)

    def test_clean_tail_is_not_a_loop(self):
        self.assertIsNone(repeated_tail_period([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_repeat_count_scales_with_period(self):
        self.assertEqual(repeated_repeat_count(1), 12)
        self.assertEqual(repeated_repeat_count(2), 6)
        self.assertEqual(repeated_repeat_count(4), 3)


class TestTrimRepeatedTail(unittest.TestCase):
    def test_trims_echo_to_one_copy(self):
        trimmed = trim_repeated_tail(ECHO)
        self.assertEqual(trimmed, "Hello! I'm Jin, a small language model")
        self.assertFalse(looks_like_echo(trimmed))

    def test_untouched_when_no_loop(self):
        text = "The capital of France is Paris, on the Seine."
        self.assertEqual(trim_repeated_tail(text), text)

    def test_stop_reason_labels(self):
        self.assertIn("repeat", stop_reason_label("repeat"))
        self.assertIn("budget", stop_reason_label("length"))
        self.assertIn("stop token", stop_reason_label("eos"))


class TestNoRepeatNgram(unittest.TestCase):
    """The ban is pure logits surgery — testable without a model."""

    def _logits(self, vocab: int = 8) -> torch.Tensor:
        return torch.zeros(1, vocab)

    def test_bans_the_completing_token(self):
        logits = self._logits()
        idx = torch.tensor([[1, 2, 3, 1, 2]])       # prefix [1, 2] seen before
        out = JinAI._apply_no_repeat_ngram(logits, idx, 3)
        self.assertTrue(torch.isinf(out[0, 3]) and out[0, 3] < 0)
        self.assertTrue(torch.isfinite(out[0, 4]).all())   # elsewhere untouched

    def test_keep_ids_is_never_banned(self):
        logits = self._logits()
        idx = torch.tensor([[1, 2, 3, 1, 2]])
        out = JinAI._apply_no_repeat_ngram(logits, idx, 3, keep_ids={3})
        self.assertTrue(torch.isfinite(out[0, 3]))

    def test_off_by_default(self):
        logits = self._logits()
        idx = torch.tensor([[1, 2, 3, 1, 2]])
        out = JinAI._apply_no_repeat_ngram(logits, idx, 0)
        self.assertTrue(torch.isfinite(out).all())

    def test_short_sequence_is_a_noop(self):
        logits = self._logits()
        idx = torch.tensor([[1, 2]])
        out = JinAI._apply_no_repeat_ngram(logits, idx, 3)
        self.assertTrue(torch.isfinite(out).all())


class TestGenerationWithAntiEcho(unittest.TestCase):
    def _prompt(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randint(0, VOCAB, (1, 12))

    def test_ngram_ban_prevents_any_repeated_3gram(self):
        """Property: with the ban on, no 3-gram may appear twice."""
        model = make_model()
        model.eval()
        with torch.no_grad():
            out = model.generate(
                self._prompt(), max_new_tokens=48, temperature=0.0,
                repetition_penalty=1.0, no_repeat_ngram_size=3,
            )[0].tolist()
        grams = [tuple(out[i:i + 3]) for i in range(len(out) - 2)]
        self.assertEqual(len(grams), len(set(grams)),
                         "a 3-gram repeated despite no_repeat_ngram_size=3")

    def test_speculative_path_obeys_the_same_ban(self):
        model = make_model()
        model.eval()
        with torch.no_grad():
            out = model.generate(
                self._prompt(), max_new_tokens=32, temperature=0.0,
                repetition_penalty=1.0, no_repeat_ngram_size=3,
                prompt_lookup_k=4,
            )[0].tolist()
        grams = [tuple(out[i:i + 3]) for i in range(len(out) - 2)]
        self.assertEqual(len(grams), len(set(grams)))

    def test_stop_on_repeat_reports_a_looping_tail(self):
        model = make_model()
        model.eval()
        stats: dict = {}
        with torch.no_grad():
            out = model.generate(
                self._prompt(), max_new_tokens=96, temperature=0.0,
                repetition_penalty=1.0, stop_on_repeat=True, stats=stats,
            )
        self.assertIn(stats["stop_reason"], {"length", "eos", "repeat"})
        self.assertEqual(stats["generated"], out.size(1) - 12)
        if stats["stop_reason"] == "repeat":
            tail = out[0].tolist()[12:]
            self.assertIsNotNone(repeated_tail_period(tail))

    def test_stream_reports_stats_and_matches_generate(self):
        model = make_model()
        model.eval()
        idx = self._prompt()
        stats: dict = {}
        with torch.no_grad():
            streamed = list(model.stream_generate(
                idx.clone(), max_new_tokens=16, temperature=0.0,
                repetition_penalty=1.0, no_repeat_ngram_size=3,
                stop_on_repeat=True, stats=stats,
            ))
            plain = model.generate(
                idx.clone(), max_new_tokens=16, temperature=0.0,
                repetition_penalty=1.0, no_repeat_ngram_size=3,
                stop_on_repeat=True, use_kv_cache=False,
            )
        self.assertEqual(streamed, plain[0].tolist()[idx.size(1):])
        self.assertEqual(stats["generated"], len(streamed))


if __name__ == "__main__":
    unittest.main()
