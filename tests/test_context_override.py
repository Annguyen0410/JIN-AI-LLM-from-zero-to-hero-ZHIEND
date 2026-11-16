"""``--ctx N``: serve a longer context than the checkpoint was trained on.

RoPE rescaling at load time (NTK-aware) buys *usable* context without a
long-context fine-tune — the ``long_context`` probe in bench.py decides whether
it is worth keeping. These tests pin the config plumbing and the ceiling.
"""

from __future__ import annotations

import unittest

import torch

import config
from model import JinAI


def make_model(**kw) -> JinAI:
    defaults = dict(
        vocab_size=256, embed_dim=64, block_size=64, n_layers=2, n_heads=4,
        n_kv_heads=2, dropout=0.0, modern=True, qk_norm=True,
    )
    defaults.update(kw)
    return JinAI(**defaults)


class TestContextOverride(unittest.TestCase):
    def test_rejects_a_shrinking_window(self):
        with self.assertRaises(ValueError):
            config.context_override(512, base_ctx=1024)

    def test_factor_is_target_over_base(self):
        override = config.context_override(2048, base_ctx=1024)
        scaling = override["rope_scaling"]
        self.assertEqual(scaling["target_ctx"], 2048)
        self.assertAlmostEqual(scaling["factor"], 2.0)
        self.assertEqual(scaling["type"], "ntk")

    def test_base_defaults_to_the_trained_block_size(self):
        override = config.context_override(config.BLOCK_SIZE * 2)
        self.assertAlmostEqual(override["rope_scaling"]["factor"], 2.0)

    def test_yarn_mode_is_passed_through(self):
        override = config.context_override(4096, base_ctx=1024, mode="yarn")
        self.assertEqual(override["rope_scaling"]["type"], "yarn")
        self.assertAlmostEqual(override["rope_scaling"]["factor"], 4.0)

    def test_override_raises_the_model_ceiling(self):
        """A 4x window: forward passes beyond block_size must still work."""
        model = make_model(**config.context_override(256, base_ctx=64))
        model.eval()
        self.assertEqual(model.max_positions, 256)
        idx = torch.randint(0, 256, (1, 200))
        with torch.no_grad():
            logits, _ = model(idx)
            self.assertEqual(logits.shape[:2], (1, 200))
            out = model.generate(idx, max_new_tokens=4, temperature=0.0,
                                 repetition_penalty=1.0)
        self.assertEqual(out.size(1), 204)

    def test_untouched_model_keeps_its_ceiling(self):
        model = make_model()
        self.assertEqual(model.max_positions, 64)


if __name__ == "__main__":
    unittest.main()
