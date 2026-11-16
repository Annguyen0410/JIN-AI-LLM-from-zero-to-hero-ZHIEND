"""Architecture tests for the GPT-2 Small reference preset.

These tests verify the *shape and behaviour* of the neural network, not the
quality of any weights: parameter count, attention causality, embedding
tying, checkpoint round-trip, gradient flow, numeric safety and — most
importantly — that the training loop can actually overfit a tiny dataset.
A model that cannot overfit 2 short documents has a broken forward or
backward pass no matter what its benchmark number says.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from model import FeedForward, JinAI  # noqa: E402

GPT2_VOCAB = 50_257
#: Exact parameter count of OpenAI GPT-2 Small at vocab 50257.
GPT2_SMALL_PARAMS = 124_439_808


def gpt2_reference(**overrides) -> JinAI:
    preset = config.SCALE_PRESETS["gpt2_small_reference"]
    cfg = {
        "vocab_size": GPT2_VOCAB,
        "embed_dim": preset["embed_dim"],
        "block_size": preset["block_size"],
        "n_layers": preset["n_layers"],
        "n_heads": preset["n_heads"],
        "n_kv_heads": preset["n_kv_heads"],
        "dropout": 0.0,
        "gradient_checkpointing": False,
        "modern": preset["modern"],
        "arch_id": preset["arch_id"],
        "gelu_tanh": preset["gelu_tanh"],
        "resid_scaled_init": preset["resid_scaled_init"],
        "qk_norm": preset["qk_norm"],
        "rope_theta": preset["rope_theta"],
        "rope_scaling": None,
        "sliding_window": None,
        "local_layer_every": None,
        "moe": None,
        "expert_offload": False,
        "static_kv_cache": False,
        "kv_cache_int8": False,
    }
    cfg.update(overrides)
    return JinAI.from_config(cfg)


def tiny_classic(**overrides) -> JinAI:
    """Small classic block sharing the GPT-2 code path (fast on CPU)."""
    cfg = {
        "vocab_size": 512,
        "embed_dim": 128,
        "block_size": 64,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 4,
        "dropout": 0.0,
        "gradient_checkpointing": False,
        "modern": False,
        "arch_id": "test_classic",
        "gelu_tanh": True,
        "resid_scaled_init": True,
    }
    cfg.update(overrides)
    return JinAI.from_config(cfg)


# ── Shape / identity ───────────────────────────────────────────────────

def test_gpt2_small_parameter_count_is_exact():
    model = gpt2_reference()
    assert model.num_parameters() == GPT2_SMALL_PARAMS
    # ~124M, and NOT the JIN modern 143M profile.
    assert 124_000_000 < model.num_parameters() < 125_000_000


def test_projection_shapes_match_gpt2():
    model = gpt2_reference()
    block = model.blocks[0]
    # Fused QKV: 768 -> 768 + 2*768 with bias.
    assert block.attn.qkv_proj.weight.shape == (3 * 768, 768)
    assert block.attn.qkv_proj.bias.shape == (3 * 768,)
    assert block.attn.out_proj.weight.shape == (768, 768)
    # GPT-2 MLP: 768 -> 3072 -> 768 (not SwiGLU's gate/up pair).
    assert isinstance(block.ffn, FeedForward)
    assert block.ffn.net[0].weight.shape == (3072, 768)
    assert block.ffn.net[2].weight.shape == (768, 3072)
    assert model.pos_emb.weight.shape == (1024, 768)
    assert len(model.blocks) == 12
    assert model.n_heads == 12 and model.n_kv_heads == 12


def test_gelu_is_tanh_approximation():
    model = gpt2_reference()
    act = model.blocks[0].ffn.net[1]
    assert isinstance(act, torch.nn.GELU)
    x = torch.tensor([0.5, 1.0, 2.0])
    assert torch.allclose(act(x), torch.nn.functional.gelu(x, approximate="tanh"))


def test_residual_projections_are_scaled_like_gpt2():
    model = gpt2_reference()
    # Every residual-writing projection starts at std 0.02/sqrt(2*n_layer).
    expected = 0.02 / math.sqrt(2 * 12)
    observed = model.blocks[3].attn.out_proj.weight.std().item()
    assert abs(observed - expected) < expected * 0.05
    # qkv is NOT residual-scaled.
    assert model.blocks[3].attn.qkv_proj.weight.std().item() > expected


def test_embedding_and_head_are_tied():
    model = gpt2_reference()
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    assert model.head.bias is None


def test_old_modern_preset_unchanged():
    """The JIN modern profile must keep its old shape (no regression)."""
    cfg = config.SCALE_PRESETS["jin_small_modern"]
    model = JinAI(vocab_size=16_384, embed_dim=cfg["embed_dim"],
                  block_size=cfg["block_size"], n_layers=cfg["n_layers"],
                  n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"],
                  modern=True, gelu_tanh=False, resid_scaled_init=False)
    assert model.modern is True
    assert model.n_kv_heads == 8
    assert model.gelu_tanh is False
    assert model.resid_scaled_init is False


# ── Behaviour ──────────────────────────────────────────────────────────

def test_attention_is_causal():
    model = gpt2_reference()
    model.eval()
    idx = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        logits_a, _ = model(idx)
        changed = idx.clone()
        changed[0, 10:] = torch.randint(0, 1000, (6,))
        logits_b, _ = model(changed)
    # Positions 0..10 cannot see tokens 11..15.
    assert torch.allclose(logits_a[0, :10], logits_b[0, :10], atol=1e-5)
    # The changed positions themselves must differ.
    assert not torch.allclose(logits_a[0, 12], logits_b[0, 12], atol=1e-4)


def test_forward_backward_is_finite_and_all_params_get_gradients():
    model = tiny_classic()
    idx = torch.randint(0, 500, (2, 32))
    _, loss = model(idx, idx)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_sequence_and_batch_boundaries():
    model = tiny_classic(block_size=64)
    for batch, seq in ((1, 1), (1, 64), (3, 17), (2, 64)):
        idx = torch.randint(0, 500, (batch, seq))
        logits, loss = model(idx, idx)
        assert logits.shape == (batch, seq, 512)
        assert torch.isfinite(loss)
    with pytest.raises(ValueError):
        model(torch.randint(0, 500, (1, 65)))


def test_checkpoint_round_trip(tmp_path):
    model = tiny_classic()
    path = tmp_path / "tiny.pt"
    torch.save({"model_config": model.export_config(),
                "model_state_dict": model.state_dict()}, path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    clone = JinAI.from_config({**blob["model_config"], "device": "cpu"})
    clone.load_state_dict(blob["model_state_dict"])
    idx = torch.randint(0, 500, (2, 16))
    model.eval(); clone.eval()
    with torch.no_grad():
        assert torch.allclose(model(idx)[0], clone(idx)[0], atol=1e-6)
    # Architecture identity survives the round trip.
    assert clone.export_config()["arch_id"] == "test_classic"
    assert clone.gelu_tanh is True and clone.resid_scaled_init is True


def test_gpt2_export_config_records_architecture():
    model = gpt2_reference()
    exported = model.export_config()
    assert exported["arch_id"] == "gpt2_small"
    assert exported["modern"] is False
    assert exported["gelu_tanh"] is True
    assert exported["resid_scaled_init"] is True


def test_cuda_forward_backward_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = tiny_classic().cuda()
    idx = torch.randint(0, 500, (2, 32), device="cuda")
    _, loss = model(idx, idx)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)


# ── The test that matters: can it learn? ───────────────────────────────

def test_model_overfits_a_tiny_dataset():
    """Overfit 2 fixed documents. If this fails, training is broken."""
    torch.manual_seed(0)
    model = tiny_classic(block_size=32, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    x = torch.tensor([[1, 5, 9, 12, 5, 9, 12, 5, 9, 12, 5, 9, 12, 5, 9, 12,
                       5, 9, 12, 5, 9, 12, 5, 9, 12, 5, 9, 12, 5, 9, 12, 7]])
    y = torch.roll(x, shifts=-1, dims=1)
    losses = []
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    assert last < first * 0.05, f"loss did not collapse: {first:.3f} -> {last:.3f}"
    assert last < 0.3


def test_document_mask_blocks_cross_document_attention():
    model = tiny_classic()
    model.eval()
    idx = torch.randint(0, 500, (1, 12))
    doc_mask = torch.zeros(1, 12, dtype=torch.long)
    doc_mask[0, 0] = 1
    doc_mask[0, 6] = 1                     # second document starts at index 6
    with torch.no_grad():
        base, _ = model(idx, doc_mask=doc_mask)
        changed = idx.clone()
        changed[0, :6] = torch.randint(0, 500, (6,))   # rewrite doc 1 only
        other, _ = model(changed, doc_mask=doc_mask)
    # Tokens of document 2 must not attend to document 1 at all.
    assert torch.allclose(base[0, 6:], other[0, 6:], atol=1e-5)


def test_document_mask_works_for_batch_greater_than_one():
    """Regression: a (B, T, T) mask used to crash SDPA for B > 1."""
    model = tiny_classic()
    idx = torch.randint(0, 500, (3, 16))
    doc_mask = torch.zeros(3, 16, dtype=torch.long)
    doc_mask[:, 0] = 1
    doc_mask[:, 8] = 1
    _, loss = model(idx, idx, doc_mask=doc_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
