"""Exact-resume test (Phase 4.3).

Two paths are compared on a fixed small model and a fixed token stream:

    A: train N steps continuously
    B: train K steps, save, rebuild everything from the checkpoint, train N-K

The weights after B must equal the weights after A. Loading a checkpoint is not
an exact-resume test; matching the *continuation* is.

Everything that can influence a step must therefore be restored: model,
optimizer moments, Torch CPU RNG, CUDA RNG, NumPy global RNG, Python ``random``
and the NumPy Generator that picks sample offsets in ``ShardBatcher``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_gpt2  # noqa: E402
from model import JinAI  # noqa: E402

VOCAB = 64
SEQ = 16
BATCH = 2
EOT = 63


def build_model() -> JinAI:
    torch.manual_seed(0)
    return JinAI(vocab_size=VOCAB, embed_dim=32, block_size=SEQ, n_layers=2,
                 n_heads=2, n_kv_heads=2, dropout=0.0, modern=False,
                 gelu_tanh=True, resid_scaled_init=True, arch_id="resume_test")


def write_shard(path: str, tokens: int = 4000) -> None:
    rng = np.random.default_rng(1234)
    data = rng.integers(0, VOCAB - 1, size=tokens, dtype=np.uint16)
    data[::97] = EOT                       # sprinkle document boundaries
    np.asarray(data, dtype=np.uint16).tofile(path)


def make_batcher(path: str, seed: int = 7) -> train_gpt2.ShardBatcher:
    return train_gpt2.ShardBatcher(path, SEQ, BATCH, "cpu", seed, EOT)


def train_steps(model, optimizer, batcher, meta, start: int, end: int,
                total_steps: int) -> list[float]:
    """The same step body train_gpt2.main uses, minus AMP and logging."""
    losses = []
    model.train()
    for step in range(start + 1, end + 1):
        lr = train_gpt2.lr_at(step, total_steps, 6e-4, 2, 0.1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        x, y, m = batcher.batch()
        _, loss = model(x, y, doc_mask=m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def state_of(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_exact_resume_matches_continuous_training(tmp_path):
    total_steps, split = 6, 3
    train_bin = str(tmp_path / "train.bin")
    write_shard(train_bin)
    dataset_manifest = {"train": {"sha256": "train-hash"}, "val": {"sha256": "val-hash"}}
    meta = {
        "architecture_id": "resume_test",
        "training_args": {"lr": 6e-4},
        "dataset_manifest": dataset_manifest,
        "tokenizer": {"kind": "gpt2_bpe", "vocab_size": VOCAB},
        "hardware": {"device": "cpu", "torch": str(torch.__version__)},
        "schedule": {"kind": "cosine_with_warmup", "total_steps": total_steps,
                     "warmup": 2, "peak_lr": 6e-4, "min_lr_ratio": 0.1},
        "effective_batch_tokens": SEQ * BATCH,
        "revision": "test",
    }

    def fresh():
        np.random.seed(123)
        model = build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))
        batcher = make_batcher(train_bin)
        return model, optimizer, batcher

    # ── path A: continuous ────────────────────────────────────────────
    model_a, opt_a, bat_a = fresh()
    losses_a = train_steps(model_a, opt_a, bat_a, meta, 0, total_steps, total_steps)
    weights_a = state_of(model_a)

    # ── path B: stop, save, rebuild, resume ───────────────────────────
    model_b, opt_b, bat_b = fresh()
    losses_b1 = train_steps(model_b, opt_b, bat_b, meta, 0, split, total_steps)
    ckpt = str(tmp_path / "ckpt.pt")
    train_gpt2.save_state(ckpt, model_b, opt_b, None, split, SEQ * BATCH * split,
                          best_val=float("inf"), history=[], meta=meta, batcher=bat_b)

    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    model_r, opt_r, bat_r = fresh()          # different RNG state on purpose
    train_gpt2.verify_resume_compatibility(blob, model_r, type("T", (), {
        "vocab_size": VOCAB, "kind": "gpt2_bpe"})(), dataset_manifest, total_steps, False)
    train_gpt2.restore_training_state(blob, model_r, opt_r, None, bat_r, "cpu")
    start = int(blob["iteration"])
    losses_b2 = train_steps(model_r, opt_r, bat_r, meta, start, total_steps, total_steps)
    weights_b = state_of(model_r)

    # Loss sequence must match exactly, not approximately.
    assert losses_a[:split] == pytest.approx(losses_b1, rel=0, abs=1e-9)
    assert losses_a[split:] == pytest.approx(losses_b2, rel=0, abs=1e-9)

    # The continuation must reproduce the continuous run's weights.
    for name, tensor in weights_a.items():
        assert torch.allclose(tensor, weights_b[name], rtol=0, atol=1e-6), name

    # Optimizer moments must also match (a fresh AdamW would differ here).
    sd_a = opt_a.state_dict()["state"]
    sd_b = opt_r.state_dict()["state"]
    assert sd_a.keys() == sd_b.keys()
    for idx in sd_a:
        for key in sd_a[idx]:
            assert torch.allclose(sd_a[idx][key], sd_b[idx][key], rtol=0, atol=1e-6)

    # Batcher position is restored, so the next batch is the same one.
    assert bat_a.tokens_served == bat_r.tokens_served


def test_resume_refuses_changed_corpus(tmp_path):
    meta = {
        "architecture_id": "resume_test",
        "training_args": {}, "dataset_manifest": {"train": {"sha256": "a"},
                                                  "val": {"sha256": "b"}},
        "tokenizer": {"kind": "gpt2_bpe", "vocab_size": VOCAB},
        "hardware": {}, "schedule": {"total_steps": 4},
        "effective_batch_tokens": SEQ * BATCH, "revision": "test",
    }
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = str(tmp_path / "c.pt")
    train_gpt2.save_state(ckpt, model, optimizer, None, 1, 10, 1.0, [], meta)
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    tok = type("T", (), {"vocab_size": VOCAB, "kind": "gpt2_bpe"})()

    changed = {"train": {"sha256": "DIFFERENT"}, "val": {"sha256": "b"}}
    with pytest.raises(SystemExit):
        train_gpt2.verify_resume_compatibility(blob, model, tok, changed, 4, False)

    # Different budget without --extend-budget is refused too.
    with pytest.raises(SystemExit):
        train_gpt2.verify_resume_compatibility(
            blob, model, tok, meta["dataset_manifest"], 8, False)

    # ...and allowed with the explicit flag.
    train_gpt2.verify_resume_compatibility(
        blob, model, tok, meta["dataset_manifest"], 8, True, log_fn=lambda *_: None)


def test_validation_is_deterministic_and_fp32():
    """Same seed -> same windows -> identical loss, computed in fp32."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "val.bin")
        write_shard(path, 2000)
        model = build_model().eval()

        def run():
            batcher = make_batcher(path, seed=99)
            return train_gpt2.evaluate(model, batcher, iters=2, amp_dtype=torch.float16,
                                       use_amp=False)

        first, second = run(), run()
        assert first["loss"] == second["loss"]
        assert first["tokens_scored"] == second["tokens_scored"] == 2 * BATCH * SEQ
        assert first["loss_dtype"] == "fp32_cross_entropy"
