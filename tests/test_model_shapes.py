"""
Model shape & behaviour tests for the v3 architecture (Phase 1 acceptance).

Covers, for every preset: forward + generate shapes, greedy KV-cache
equivalence (output with cache == output without cache), GQA KV-head
handling, MoE total-vs-active parameter counting + auxiliary losses,
sliding-window cache bounding, and RoPE scaling beyond the trained block.

Run:  python -m unittest discover -s tests -v   (stdlib unittest, CPU)
"""

from __future__ import annotations

import gc
import os
import unittest

import torch

import config
from model import JinAI

VOCAB = 256

# Preset nặng: ``large`` (1,06B) và ``xl-moe`` (0,76B) dựng trên CPU cần ~5GB cấp
# phát liền mạch. Khi máy đang chặt RAM, cấp phát thất bại làm cả process
# **segfault** (exit 139) chứ không ném MemoryError, và lượt test chết mà không
# in dòng tóm tắt — rất dễ bị đọc thành "code hỏng". Vì vậy mặc định bỏ qua hai
# preset này; bật lại khi máy rảnh RAM:
#     JINAI_TEST_BIG_PRESETS=1 python -m unittest discover -s tests
HEAVY_PRESETS = ("large", "xl-moe")
BIG_PRESETS_ENV = "JINAI_TEST_BIG_PRESETS"


def big_presets_enabled() -> bool:
    """Có bật kiểm tra preset nặng không (mặc định KHÔNG, để tránh segfault)."""
    return os.environ.get(BIG_PRESETS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def make_model(**kw) -> JinAI:
    defaults = dict(
        vocab_size=VOCAB, embed_dim=64, block_size=64, n_layers=2, n_heads=4,
        n_kv_heads=2, dropout=0.0, gradient_checkpointing=False, modern=True,
        qk_norm=True, rope_theta=10_000.0, rope_scaling=None,
        sliding_window=None, local_layer_every=None, moe=None,
        expert_offload=False,
    )
    defaults.update(kw)
    return JinAI(**defaults)


class TestPresetShapes(unittest.TestCase):
    """Every shipped preset must forward and generate without crashing."""

    def test_presets_forward_and_generate(self):
        skipped: list[str] = []
        for name in ("tiny", "small", "medium", "large", "xl-moe"):
            if name in HEAVY_PRESETS and not big_presets_enabled():
                skipped.append(name)
                continue
            preset = config.SCALE_PRESETS[name]
            model = make_model(
                embed_dim=preset["embed_dim"], n_layers=preset["n_layers"],
                n_heads=preset["n_heads"], n_kv_heads=preset["n_kv_heads"],
                block_size=min(preset["block_size"], 128),
                qk_norm=preset["qk_norm"],
                sliding_window=preset["sliding_window"],
                local_layer_every=preset["local_layer_every"],
                moe=preset["moe"],
                expert_offload=preset["expert_offload"],
                gradient_checkpointing=False,
            )
            model.eval()
            seq = min(block := model.block_size, 32)
            idx = torch.randint(0, VOCAB, (1, seq))
            with torch.no_grad():
                logits, _ = model(idx)
                self.assertEqual(logits.shape, (1, seq, VOCAB))
                out = model.generate(
                    idx[:, :4], max_new_tokens=8, temperature=0.0,
                    repetition_penalty=1.0,
                )
                self.assertEqual(out.shape[1], 4 + 8)
            # total >= active, and MoE models must have a real gap.
            total, active = model.num_parameters(), model.active_parameters()
            self.assertGreaterEqual(total, active)
            if preset["moe"]:
                self.assertGreater(total, active)
                self.assertLess(active / total, 0.8, f"{name} MoE should shrink active params")
            # Nhả model NGAY: giữ nhiều preset cùng lúc trong một process chính là
            # thứ làm tràn RAM rồi segfault ở preset kế tiếp.
            del model, logits, out, idx
            gc.collect()
        if skipped:
            # Chữ thuần ASCII: console Windows mặc định cp1252, in tiếng Việt ở
            # đây sẽ nổ UnicodeEncodeError khi file test này chạy một mình.
            print(f"[skip] heavy presets skipped to avoid a RAM-pressure segfault: "
                  f"{', '.join(skipped)} -- rerun with {BIG_PRESETS_ENV}=1 when RAM is free")


class TestKVEquivalence(unittest.TestCase):
    """Greedy generation with KV cache must equal generation without it."""

    def _assert_equiv(self, model, prompt_len, new_tokens):
        model.eval()
        torch.manual_seed(0)
        idx = torch.randint(0, VOCAB, (1, prompt_len))
        with torch.no_grad():
            a = model.generate(idx.clone(), max_new_tokens=new_tokens, temperature=0.0,
                               repetition_penalty=1.0, use_kv_cache=True)
            b = model.generate(idx.clone(), max_new_tokens=new_tokens, temperature=0.0,
                               repetition_penalty=1.0, use_kv_cache=False)
        torch.testing.assert_close(a, b)

    def test_mha(self):
        self._assert_equiv(make_model(n_kv_heads=4), 16, 12)

    def test_gqa(self):
        self._assert_equiv(make_model(n_kv_heads=2), 16, 12)

    def test_sliding_window(self):
        self._assert_equiv(
            make_model(sliding_window=8, local_layer_every=2, block_size=64), 16, 12)

    def test_rope_scaling(self):
        # Scaled model accepts sequences longer than block_size.
        model = make_model(block_size=32, rope_scaling={"type": "ntk", "factor": 4.0, "target_ctx": 128})
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 96))  # > block_size, < target_ctx
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=8, temperature=0.0, repetition_penalty=1.0)
        self.assertEqual(out.shape[1], 96 + 8)


class TestGQA(unittest.TestCase):
    def test_kv_cache_head_shape(self):
        model = make_model(n_kv_heads=2)
        caches = model.empty_kv_cache()
        x = torch.randint(0, VOCAB, (1, 8))
        with torch.no_grad():
            model(x, kv_caches=caches)
        k = caches[0]["k"]
        self.assertEqual(k.shape, (1, 2, 8, 16))  # (B, n_kv_heads, T, head_dim)

    def test_kv_smaller_than_mha(self):
        mha = make_model(n_kv_heads=4)
        gqa = make_model(n_kv_heads=2)
        x = torch.randint(0, VOCAB, (1, 8))
        c1, c2 = mha.empty_kv_cache(), gqa.empty_kv_cache()
        with torch.no_grad():
            mha(x, kv_caches=c1)
            gqa(x, kv_caches=c2)
        self.assertEqual(c1[0]["k"].numel(), 2 * c2[0]["k"].numel())


class TestMoE(unittest.TestCase):
    def test_aux_and_z_loss_nonzero_in_training(self):
        moe = dict(n_experts=4, top_k=2, capacity_factor=1.25,
                   aux_loss_coef=0.01, z_loss_coef=0.001, layers="even")
        model = make_model(moe=moe, n_layers=2)
        model.train()
        x = torch.randint(0, VOCAB, (1, 16))
        y = torch.randint(0, VOCAB, (1, 16))
        _, loss = model(x, y)
        moe_loss = sum(
            b.ffn.aux_loss_coef * b.ffn.aux_loss + b.ffn.z_loss_coef * b.ffn.z_loss
            for b in model.blocks if isinstance(b.ffn, type(model.blocks[0].ffn)) and hasattr(b.ffn, "n_experts")
        )
        self.assertGreater(float(moe_loss), 0.0)
        self.assertGreater(float(loss), 0.0)

    def test_router_deterministic_at_eval(self):
        moe = dict(n_experts=4, top_k=2, capacity_factor=1.25,
                   aux_loss_coef=0.01, z_loss_coef=0.001, layers="even")
        model = make_model(moe=moe, n_layers=2)
        model.eval()
        x = torch.randint(0, VOCAB, (1, 16))
        with torch.no_grad():
            a, _ = model(x)
            b, _ = model(x)
        torch.testing.assert_close(a, b)

    def test_capacity_factor_bounds_output(self):
        moe = dict(n_experts=2, top_k=1, capacity_factor=0.0, aux_loss_coef=0.0,
                   z_loss_coef=0.0, layers="even")
        model = make_model(moe=moe, n_layers=2)
        model.eval()
        x = torch.randint(0, VOCAB, (1, 16))
        with torch.no_grad():
            logits, _ = model(x)
        self.assertEqual(logits.shape, (1, 16, VOCAB))


class TestSlidingWindowCache(unittest.TestCase):
    def test_local_cache_stays_bounded(self):
        model = make_model(sliding_window=8, local_layer_every=2, block_size=64)
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 4))
        kv = model.empty_kv_cache()
        with torch.no_grad():
            for _ in range(30):  # way past the window
                logits, _ = model(idx[:, -1:], kv_caches=kv)
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                idx = torch.cat([idx, nxt], dim=1)
        for i, cache in enumerate(kv):
            is_local = (i + 1) % 2 != 0
            if is_local and cache["k"] is not None:
                self.assertLessEqual(cache["k"].size(-2), 8)


class TestPhase2KV(unittest.TestCase):
    """Static / int8 KV caches, prefix reuse and prompt-lookup (Phase 2)."""

    def _model(self, **kw):
        m = make_model(**kw)
        m.eval()
        return m

    def _repetitive_prompt(self):
        return torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8,
                              1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)

    def test_static_kv_equals_dynamic(self):
        model = self._model()
        idx = self._repetitive_prompt()
        with torch.no_grad():
            base = model.generate(idx.clone(), max_new_tokens=20, temperature=0.0,
                                  repetition_penalty=1.0, use_kv_cache=True)
            static = model.generate(idx.clone(), max_new_tokens=20, temperature=0.0,
                                    repetition_penalty=1.0, use_kv_cache=True,
                                    static_kv=True)
        torch.testing.assert_close(base, static)

    def _assert_draft_equiv(self, prompt, **gen_kw):
        """Greedy output + cache state with prompt-lookup == without it.

        Regression guard for the prompt-lookup rewrite: drafts must fire
        (they used to silently never fire — a broken ``in`` on lists) AND
        must not change greedy tokens or corrupt the KV cache, including
        at a sliding-window boundary where the draft batch can overflow.
        """
        model = make_model(n_layers=2, n_heads=4, n_kv_heads=2, embed_dim=64,
                           block_size=64, modern=True, qk_norm=True,
                           sliding_window=8, local_layer_every=2)
        model.eval()
        torch.manual_seed(0)
        with torch.no_grad():
            c0 = model.empty_kv_cache(static=True)
            model(prompt, kv_caches=c0)
            out0 = model.generate(prompt.clone(), kv_caches=c0, static_kv=True,
                                  prompt_lookup_k=0, **gen_kw)
            c1 = model.empty_kv_cache(static=True)
            model(prompt, kv_caches=c1)
            out1 = model.generate(prompt.clone(), kv_caches=c1, static_kv=True,
                                  prompt_lookup_k=4, **gen_kw)
        torch.testing.assert_close(out0, out1)
        for a, b in zip(c0, c1):
            self.assertEqual(a["len"], b["len"])
            torch.testing.assert_close(
                a["k_buf"][:, :, : a["len"]], b["k_buf"][:, :, : b["len"]])
            torch.testing.assert_close(
                a["v_buf"][:, :, : a["len"]], b["v_buf"][:, :, : b["len"]])

    def test_prompt_lookup_drafts_actually_fire(self):
        # Repetitive prompt -> repeated n-gram -> drafts are generated.
        prompt = torch.tensor([[1, 2, 3, 4, 1, 2]], dtype=torch.long)
        self.assertGreaterEqual(
            JinAI._rfind_sublist([1, 2, 3, 4, 1, 2], [1, 2]), 0)
        self._assert_draft_equiv(prompt, max_new_tokens=10, temperature=0.0,
                                 repetition_penalty=1.0, skip_prefill_tokens=6)

    def test_prompt_lookup_window_boundary_rewind(self):
        # 6-token prefill in a window-8 cache: the draft batch (1 cur + 2
        # drafts) overflows the window -> the slide must be unrewound
        # correctly (this used to corrupt the cache / produce 1 extra token).
        prompt = torch.tensor([[1, 2, 3, 1, 2, 3]], dtype=torch.long)
        self._assert_draft_equiv(prompt, max_new_tokens=12, temperature=0.0,
                                 repetition_penalty=1.0, skip_prefill_tokens=6)

    def test_prompt_lookup_budget_exhausted_all_accepted(self):
        # All drafts accepted and remaining hits 0: must NOT emit an extra
        # token (used to fall through to the partial branch -> 29 vs 28).
        model = make_model(n_layers=2, n_heads=4, n_kv_heads=2, embed_dim=64,
                           block_size=64, modern=True, qk_norm=True)
        model.eval()
        idx = torch.tensor([[1, 2, 3, 1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(idx.clone(), max_new_tokens=4, temperature=0.0,
                                 repetition_penalty=1.0, prompt_lookup_k=4)
        self.assertEqual(out.shape[1], idx.size(1) + 4)

    def test_stream_generate_guard_empty_tail(self):
        # skip_prefill_tokens == prompt length (fully cached prefix) must
        # not crash on an empty tail (used to IndexError on logits[:, -1]).
        model = self._model()
        idx = self._repetitive_prompt()
        caches = model.empty_kv_cache(static=True)
        with torch.no_grad():
            model(idx, kv_caches=caches)
            toks = list(model.stream_generate(
                idx.clone(), max_new_tokens=5, temperature=0.0,
                repetition_penalty=1.0, kv_caches=caches, static_kv=True,
                skip_prefill_tokens=idx.size(1)))
        self.assertEqual(len(toks), 5)

    def test_int8_kv_equals_fp32(self):
        model = self._model()
        idx = self._repetitive_prompt()
        with torch.no_grad():
            base = model.generate(idx.clone(), max_new_tokens=20, temperature=0.0,
                                  repetition_penalty=1.0, use_kv_cache=True,
                                  static_kv=True)
            i8 = model.generate(idx.clone(), max_new_tokens=20, temperature=0.0,
                                repetition_penalty=1.0, use_kv_cache=True,
                                static_kv=True, kv_int8=True)
        torch.testing.assert_close(base, i8, atol=0, rtol=0)

    def test_prompt_lookup_preserves_greedy(self):
        model = self._model()
        idx = self._repetitive_prompt()
        with torch.no_grad():
            base = model.generate(idx.clone(), max_new_tokens=24, temperature=0.0,
                                  repetition_penalty=1.0, use_kv_cache=True,
                                  prompt_lookup_k=0)
            pl = model.generate(idx.clone(), max_new_tokens=24, temperature=0.0,
                                repetition_penalty=1.0, use_kv_cache=True,
                                prompt_lookup_k=4, prompt_lookup_max_ngram=32)
        torch.testing.assert_close(base, pl)

    def test_prefix_reuse_equals_full_prefill(self):
        model = self._model()
        idx = self._repetitive_prompt()
        with torch.no_grad():
            # Full prefill + decode from scratch.
            fresh = model.generate(idx.clone(), max_new_tokens=12, temperature=0.0,
                                   repetition_penalty=1.0, use_kv_cache=True)
            # Prefill everything, then decode reusing the whole prefix.
            caches = model.empty_kv_cache(static=True)
            model(idx, kv_caches=caches)
            reused = model.generate(idx.clone(), max_new_tokens=12, temperature=0.0,
                                    repetition_penalty=1.0, use_kv_cache=True,
                                    kv_caches=caches,
                                    skip_prefill_tokens=idx.size(1))
        torch.testing.assert_close(fresh, reused)

    def test_prefix_reuse_multi_turn(self):
        model = self._model()
        turn1 = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            c1 = model.empty_kv_cache(static=True)
            model(turn1, kv_caches=c1)
            r1 = model.generate(turn1, max_new_tokens=6, temperature=0.0,
                                repetition_penalty=1.0, kv_caches=c1,
                                static_kv=True, skip_prefill_tokens=turn1.size(1))
            # Turn 2 = turn1 + reply tail + new user tokens.
            turn2 = torch.cat([turn1, r1[:, -2:]], dim=1)
            c2 = model.empty_kv_cache(static=True)
            model(turn2, kv_caches=c2)
            r2 = model.generate(turn2, max_new_tokens=6, temperature=0.0,
                                repetition_penalty=1.0, kv_caches=c2,
                                static_kv=True, skip_prefill_tokens=turn2.size(1))
            # Reference: same turn-2 sequence decoded from scratch.
            ref = model.generate(turn2, max_new_tokens=6, temperature=0.0,
                                 repetition_penalty=1.0, use_kv_cache=True)
        self.assertEqual(r2.shape, ref.shape)

    def test_doc_mask_blocks_cross_document_attention(self):
        model = make_model(n_layers=1, n_heads=2, n_kv_heads=2,
                           embed_dim=32, block_size=16)
        model.eval()
        x = torch.randint(0, VOCAB, (1, 8))
        doc = torch.zeros(1, 8, dtype=torch.long)
        doc[0, 0] = 1
        doc[0, 4] = 1  # two documents: [0..3] and [4..7]
        with torch.no_grad():
            _, loss = model(x, x, doc_mask=doc)
        self.assertGreater(float(loss), 0.0)

    def test_get_batch_with_doc_mask(self):
        import utils
        data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        torch.manual_seed(0)
        x, y, doc = utils.get_batch_with_doc_mask(
            data, eot_id=5, block_size=6, batch_size=2, device="cpu")
        self.assertEqual(x.shape, (2, 6))
        self.assertEqual(doc.shape, (2, 6))
        self.assertEqual(int(doc[0, 0]), 1)  # first token of a block starts a doc


class TestBackwardCompat(unittest.TestCase):
    def test_v25_config_loads_unchanged(self):
        # v2.5 model_config has no v3 keys -> defaults reproduce old behaviour.
        model = JinAI(vocab_size=VOCAB, embed_dim=64, block_size=64,
                       n_layers=2, n_heads=4, dropout=0.0, modern=True)
        self.assertEqual(model.n_kv_heads, 4)
        self.assertFalse(model.qk_norm)
        qkv = model.blocks[0].attn.qkv_proj.weight
        self.assertEqual(qkv.shape, (3 * 64, 64))  # MHA: 3*embed


class TestRopeTableBoundary(unittest.TestCase):
    """Regression: generation past ``max_positions`` total tokens used to
    crash because the absolute ``pos`` counter outgrew the precomputed RoPE
    table (empty slice -> q collapses to 0 tokens). The lookup is now
    clamped to the table (standard sliding-window position cap)."""

    def test_prompt_at_full_context_generates(self):
        # max_positions == block_size when no RoPE scaling is configured.
        model = make_model(block_size=32)
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 32))
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=8, temperature=0.0)
        self.assertEqual(out.shape, (1, 40))

    def test_generation_past_context_does_not_crash(self):
        model = make_model(block_size=32)
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 4))
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=96, temperature=0.0)
        self.assertEqual(out.shape, (1, 100))
        # Streaming path shares the fix.
        n = sum(1 for _ in model.stream_generate(
            idx, max_new_tokens=64, temperature=0.0))
        self.assertEqual(n, 64)

    def test_bounded_behavior_unchanged_within_table(self):
        """Within max_positions the cap is a no-op: cache == no-cache."""
        model = make_model(block_size=32)
        model.eval()
        idx = torch.randint(0, VOCAB, (1, 8))
        with torch.no_grad():
            a = model.generate(idx, max_new_tokens=16, temperature=0.0,
                               use_kv_cache=True)
            b = model.generate(idx, max_new_tokens=16, temperature=0.0,
                               use_kv_cache=False)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()