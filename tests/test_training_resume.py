"""Resume / manifest tests for ``train_gpt2.py``.

The empirical resume check is recorded in ``docs/GPT2_CLASS_RELEASE_REPORT.md``
(a real 220-step run continued at step 221 with its optimizer, tokens_seen and
RNG state intact). These tests pin the two properties that make that work:

1. the LR schedule is a function of the ABSOLUTE step, so a resumed run
   continues the same curve instead of restarting warmup/decay;
2. the checkpoint payload carries everything a restart needs — step,
   tokens_seen, best loss, optimizer state, RNG state, config, dataset hash.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_gpt2  # noqa: E402
from model import JinAI  # noqa: E402


def test_lr_schedule_is_absolute_not_relative():
    peak, warmup, total, floor = 6e-4, 100, 1000, 0.1
    first = [train_gpt2.lr_at(s, total, peak, warmup, floor) for s in range(1, 1001)]
    # Warmup rises, then cosine decays: never increases after warmup.
    assert first[0] < first[50] < first[warmup]
    assert first[warmup] >= first[500] >= first[999]
    # Asking for step 500 directly must give exactly the value the long run
    # would have used — this is what makes a resumed run reproducible.
    assert train_gpt2.lr_at(500, total, peak, warmup, floor) == first[499]
    assert train_gpt2.lr_at(1000, total, peak, warmup, floor) == pytest.approx(
        peak * floor, rel=1e-6)


def test_save_state_round_trip_carries_resume_state(tmp_path):
    torch.manual_seed(0)
    model = JinAI(vocab_size=64, embed_dim=32, block_size=16, n_layers=1,
                  n_heads=2, n_kv_heads=2, dropout=0.0, modern=False,
                  gelu_tanh=True, resid_scaled_init=True, arch_id="test")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # One real step: AdamW creates its state on first step, and a checkpoint
    # saved before any step would carry an empty optimizer (that is exactly the
    # case the resume path must not pretend to restore).
    idx = torch.randint(0, 64, (2, 8))
    _, loss = model(idx, idx)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    path = str(tmp_path / "ckpt.pt")
    meta = {
        "architecture_id": "test",
        "training_args": {"lr": 1e-3},
        "dataset_manifest": {"train": {"sha256": "abc", "tokens": 10},
                             "val": {"sha256": "def", "tokens": 2}},
        "tokenizer": {"kind": "gpt2_bpe", "vocab_size": 50257},
        "hardware": {"device": "cpu", "torch": str(torch.__version__)},
    }
    train_gpt2.save_state(path, model, optimizer, None, step=7, tokens_seen=1234,
                          best_val=2.5, history=[{"step": 7}], meta=meta)

    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert blob["iteration"] == 7
    assert blob["tokens_seen"] == 1234
    assert blob["best_val_loss"] == 2.5
    assert blob["architecture_id"] == "test"
    assert blob["optimizer_state_dict"]["state"], "optimizer state must be saved"
    assert blob["rng_state"] is not None
    assert blob["model_config"]["gelu_tanh"] is True
    assert blob["dataset_manifest"]["train"]["sha256"] == "abc"
    assert blob["history"][-1]["step"] == 7

    manifest = json.load(open(path + ".manifest.json", encoding="utf-8"))
    assert manifest["iteration"] == 7
    assert manifest["tokens_seen"] == 1234
    assert len(manifest["sha256"]) == 64
    assert manifest["dataset_sha256"] == "abc"


def test_shard_batcher_marks_document_boundaries():
    import numpy as np

    eot = 50256
    tokens = np.array([eot, 10, 11, eot, 12, 13, 14, eot, 15] * 8, dtype=np.uint16)
    batcher = train_gpt2.ShardBatcher.__new__(train_gpt2.ShardBatcher)
    batcher.data = tokens
    batcher.seq_len = 4
    batcher.batch_size = 3
    batcher.device = "cpu"
    batcher.eot_id = eot
    batcher.rng = np.random.default_rng(0)
    batcher.tokens_served = 0

    x, y, doc_mask = batcher.batch()
    assert x.shape == (3, 4) and y.shape == (3, 4)
    # First column of every row is a document start.
    assert torch.all(doc_mask[:, 0] == 1)
    # Every position whose predecessor is EOT is flagged.
    expected = (x[:, :-1] == eot).long()
    assert torch.equal(doc_mask[:, 1:], expected)
