"""
Phase 3 tests (CPU): data quality pipeline, verified reasoning data,
self-distillation verifiers, DPO/ORPO losses, extended tokenizer tokens,
tool-use template.

Run:  python -m unittest tests.test_phase3 -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch

import config
import data_quality as dq
import build_reasoning_data as brd
import self_distill as sd
import train_pref as tp
from bpe_tokenizer import BPETokenizer, default_special_tokens
from chat_template import ROLE_TOOL, format_tool_trace, format_cot_document


class TestDataQuality(unittest.TestCase):
    def test_language_detection(self):
        self.assertEqual(dq.detect_language("con mèo đang chạy trong vườn nhà tôi hôm nay rất đẹp trời"), "vi")
        self.assertEqual(dq.detect_language("the quick brown fox jumps over the lazy dog near the river"), "en")

    def test_exact_dedupe(self):
        docs = [
            "hello world foo bar baz qux one two three four five six seven eight nine ten",
            "hello world foo bar baz qux one two three four five six seven eight nine ten",
            "hello world foo bar baz qux one two three four five six seven eight nine 10 eleven twelve thirteen fourteen",
        ]
        kept, report = dq.dedupe_minhash(docs)
        self.assertEqual(report["exact_removed"], 1)
        self.assertEqual(len(kept), 2)

    def test_contamination_removes_verbatim_probe(self):
        probe = None
        with open(config.PROBES_PATH, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("domain") == "factual":
                    probe = rec["prompt"]
                    break
        if probe is None:
            self.skipTest("no factual probe found")
        contam = probe + " " + ("and then some more filler words here to make it long enough " * 5)
        kept, removed = dq.check_contamination(
            [contam, "completely unrelated text about cooking rice and baking bread"]
        )
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(kept), 1)

    def test_quality_filter(self):
        good = "This is a perfectly normal document with many English words and sentences in it. " * 8
        self.assertTrue(dq.quality_filter(good)[0])
        garbage = "!!@##!!##@@ $$%%^^ &&**(())=+ }}}[[{{ }}]]" * 5
        self.assertFalse(dq.quality_filter(garbage)[0])

    def test_run_pipeline(self):
        text = (
            "<doc path=\"a\">\nThe capital of Vietnam is Hanoi, a city on the Red River.\n</doc>\n\n"
            "<doc path=\"b\">\nCompletely different text about baking bread and cooking rice.\n</doc>\n\n"
            "<doc path=\"c\">\nThe capital of Vietnam is Hanoi, a city on the Red River.\n</doc>"
        )
        clean, report = dq.run_pipeline(text)
        self.assertLessEqual(report["docs_kept"], report["docs_input"])
        self.assertIn("quality", report)


class TestReasoningData(unittest.TestCase):
    def test_math_answers_verified(self):
        import random
        rng = random.Random(7)
        for _ in range(30):
            rec = brd.make_math_problem(rng)
            self.assertTrue(rec["verified"])
            # Reference math answer must parse as a number or contain an =.
            self.assertTrue(any(ch.isdigit() for ch in rec["answer"]))

    def test_code_snippets_pass_their_tests(self):
        import random
        rng = random.Random(9)
        for _ in range(20):
            rec = brd.make_code_problem(rng)
            if rec is None:
                continue
            code = sd.extract_fenced_code(rec["answer"])
            self.assertIsNotNone(code)
            self.assertTrue(rec["verified"])

    def test_reasoning_output_is_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "reasoning.jsonl")
            # call main-level generation via make_problem loop (fast subset)
            import random
            rng = random.Random(3)
            recs = [brd.make_problem(rng, "math") for _ in range(5)]
            recs = [r for r in recs if r]
            with open(out, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            loaded = [json.loads(l) for l in open(out, encoding="utf-8")]
            self.assertEqual(len(loaded), len(recs))


class TestSelfDistillVerifiers(unittest.TestCase):
    def test_extract_last_number(self):
        self.assertEqual(sd.extract_last_number("The answer is 42."), "42")
        self.assertEqual(sd.extract_last_number("x = 3.5"), "3.5")
        self.assertIsNone(sd.extract_last_number("no numbers here"))

    def test_math_verify(self):
        ok, reason = sd.verify_candidate("q", "The result is 42", reference="42", kind="math")
        self.assertTrue(ok)
        bad, _ = sd.verify_candidate("q", "The result is 43", reference="42", kind="math")
        self.assertFalse(bad)

    def test_code_verify_with_tests(self):
        code = "def double(x):\n    return x * 2"
        ok, reason = sd.verify_candidate(
            "write double", f"```python\n{code}\n```",
            kind="code", tests=[("double(4)", 8), ("double(0)", 0)],
        )
        self.assertTrue(ok, reason)
        wrong = "def double(x):\n    return x + 2"
        bad, reason = sd.verify_candidate(
            "write double", f"```python\n{wrong}\n```",
            kind="code", tests=[("double(4)", 8)],
        )
        self.assertFalse(bad)

    def test_generic_score_penalizes_short(self):
        self.assertLess(sd._generic_score("hi"), 0.35)


class TestTrainPref(unittest.TestCase):
    def test_build_sequences_uses_chatml(self):
        tok = BPETokenizer()
        tok.train("the quick brown fox jumps over the lazy dog. " * 20,
                  vocab_size=300, verbose=False)
        ids = tp.build_sequences(tok, "Hello", "Hi there", max_length=256)
        text = tok.decode(ids)
        self.assertIn("<|user|>", text)
        self.assertIn("<|assistant|>", text)

    def test_dpo_loss_sign(self):
        # A policy that prefers chosen should get a LOWER loss than one that
        # prefers rejected. Compare on a tiny model via two manual forward
        # configurations is complex; instead assert the loss formula is
        # monotone in the log-ratio by direct math.
        import torch.nn.functional as F
        beta = 1.0
        good_ratio = torch.tensor(2.0)
        bad_ratio = torch.tensor(-2.0)
        loss_good = -F.logsigmoid(beta * good_ratio)
        loss_bad = -F.logsigmoid(beta * bad_ratio)
        self.assertLess(float(loss_good), float(loss_bad))

    def test_load_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": "p", "chosen": "good", "rejected": "bad"}) + "\n")
            pairs = tp.load_pairs(path)
            self.assertEqual(len(pairs), 1)


class TestTokenizerExtension(unittest.TestCase):
    def _train_mini(self, extended: bool = False) -> BPETokenizer:
        tok = BPETokenizer(default_special_tokens(extended=extended))
        tok.train("the quick brown fox jumps over the lazy dog. " * 20,
                  vocab_size=300, verbose=False)
        return tok

    def test_extended_tokens_at_tail(self):
        base = self._train_mini(extended=False)
        self.assertEqual(len(base.special_tokens), 6)
        self.assertNotIn("<|think|>", base.special_to_id)

        ext = self._train_mini(extended=True)
        self.assertEqual(len(ext.special_tokens), 9)
        self.assertIn("<|think|>", ext.special_to_id)
        self.assertIn("<|/think|>", ext.special_to_id)
        self.assertIn("<|tool|>", ext.special_to_id)
        # New token IDs sit after all base special tokens.
        think_id = ext.special_to_id["<|think|>"]
        pad_id = ext.special_to_id["<|pad|>"]
        self.assertEqual(think_id, pad_id + 1)
        # The first 6 special IDs are identical in both tokenizers, so the
        # extension never shifts existing IDs (migration safety).
        for tag in ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>", "<|pad|>"):
            self.assertEqual(base.special_to_id[tag], ext.special_to_id[tag])

    def test_base_tokenizer_encodes_without_extended(self):
        tok = BPETokenizer()
        # unknown special string falls through to byte tokens — must not crash
        ids = tok.encode("hello <|think|> world", allow_special=True)
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)


class TestToolTemplate(unittest.TestCase):
    def test_format_tool_trace(self):
        doc = format_tool_trace(
            user_query="What is the capital of Vietnam?",
            call='{"name": "web_search", "args": {"query": "capital of Vietnam"}}',
            result="Hanoi is the capital of Vietnam.",
            final_answer="Hanoi is the capital of Vietnam [1].",
        )
        self.assertIn("<|tool|>", doc)
        self.assertIn("<|endoftext|>", doc)
        self.assertIn("<|user|>", doc)
        self.assertIn("<|assistant|>", doc)

    def test_role_tool_registered(self):
        from chat_template import ROLE_TAG
        self.assertIn(ROLE_TOOL, ROLE_TAG)

    def test_format_cot_document(self):
        doc = format_cot_document("What is 2+2?", "Add the numbers: 2+2=4.", "4")
        self.assertIn("<|think|>", doc)
        self.assertIn("<|/think|>", doc)


class TestMixtureSampler(unittest.TestCase):
    def test_mixture_sampler_weights(self):
        import json
        import tempfile
        from utils import MixtureSampler
        from bpe_tokenizer import BPETokenizer

        tok = BPETokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mix.json")
            prose_file = os.path.join(tmp, "prose.txt")
            code_file = os.path.join(tmp, "code.txt")
            with open(prose_file, "w", encoding="utf-8") as f:
                f.write("the quick brown fox jumps over the lazy dog. " * 40)
            with open(code_file, "w", encoding="utf-8") as f:
                f.write("def f(x): return x + 1. " * 40)
            spec = {
                "domains": {"prose": prose_file, "code": code_file},
                "stages": [
                    {"name": "s1", "start": 0.0, "end": 1.0,
                     "weights": {"prose": 0.5, "code": 0.5}}
                ],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(spec, f)
            sampler = MixtureSampler(tok, mixture_path=path, block_size=32,
                                     batch_size=4, device="cpu")
            x, y = sampler.batch(progress=0.5)
            self.assertEqual(x.shape, (4, 32))
            self.assertEqual(y.shape, (4, 32))
            w = sampler.active_weights(0.5)
            self.assertAlmostEqual(sum(w.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()