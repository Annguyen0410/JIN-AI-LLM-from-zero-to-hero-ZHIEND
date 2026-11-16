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


# ── Phase 15F: full experiment-configuration fingerprint ─────────────────

class StubTokenizer:
    kind = "gpt2_bpe"

    def __init__(self, vocab_size: int = VOCAB, salt: str = ""):
        self.vocab_size = vocab_size
        self.salt = salt

    def to_state_dict(self):
        return {"kind": self.kind, "vocab_size": self.vocab_size, "salt": self.salt}


def make_args(**overrides):
    base = dict(preset="gpt2_small_reference", seq_len=SEQ, batch_size=BATCH,
                grad_accum=1, warmup=2, lr=6e-4, min_lr_ratio=0.1, seed=1337,
                dropout=0.0, no_amp=True)
    base.update(overrides)
    return type("Args", (), base)()


def make_blob(model, tokenizer, manifest, args, steps):
    fp = train_gpt2.experiment_fingerprint(model, tokenizer, manifest, args, steps)
    return {"experiment_config": fp,
            "architecture_id": model.arch_id,
            "dataset_manifest": manifest,
            "tokenizer": {"kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size},
            "schedule": {"total_steps": steps}}, fp


MANIFEST = {"train": {"sha256": "train-hash"}, "val": {"sha256": "val-hash"}}


@pytest.mark.parametrize("override", [
    {"lr": 3e-4},            # learning rate
    {"warmup": 5},           # warmup
    {"seq_len": 32},         # sequence length (also changes effective tokens)
    {"batch_size": 4},       # batch size
    {"grad_accum": 2},       # gradient accumulation
    {"dropout": 0.1},        # dropout
    {"seed": 7},             # sampling seed
    {"preset": "tiny"},      # model preset
    {"min_lr_ratio": 0.0},   # schedule floor
])
def test_resume_refuses_any_changed_hyperparameter(override):
    model = build_model()
    tok = StubTokenizer()
    args = make_args()
    blob, old_fp = make_blob(model, tok, MANIFEST, args, 4)
    changed = make_args(**override)
    new_fp = train_gpt2.experiment_fingerprint(model, tok, MANIFEST, changed, 4)
    with pytest.raises(SystemExit):
        train_gpt2.verify_resume_compatibility(
            blob, model, tok, MANIFEST, 4, False, log_fn=lambda *_: None,
            current_fingerprint=new_fp)


def test_resume_refuses_tokenizer_state_change_with_same_vocab():
    model = build_model()
    tok_old = StubTokenizer()
    args = make_args()
    blob, _ = make_blob(model, tok_old, MANIFEST, args, 4)
    tok_new = StubTokenizer(salt="rebuilt")          # same vocab, different state
    assert tok_new.vocab_size == tok_old.vocab_size
    new_fp = train_gpt2.experiment_fingerprint(model, tok_new, MANIFEST, args, 4)
    with pytest.raises(SystemExit):
        train_gpt2.verify_resume_compatibility(
            blob, model, tok_new, MANIFEST, 4, False, log_fn=lambda *_: None,
            current_fingerprint=new_fp)


def test_resume_refuses_sampler_version_change():
    model = build_model()
    tok = StubTokenizer()
    args = make_args()
    blob, old_fp = make_blob(model, tok, MANIFEST, args, 4)
    mutated = dict(old_fp)
    mutated["data"] = dict(old_fp["data"], sampler="numpy-pcg64.v999")
    with pytest.raises(SystemExit):
        train_gpt2.verify_resume_compatibility(
            blob, model, tok, MANIFEST, 4, False, log_fn=lambda *_: None,
            current_fingerprint=mutated)


def test_extend_budget_changes_only_the_step_budget():
    model = build_model()
    tok = StubTokenizer()
    args = make_args()
    blob, _ = make_blob(model, tok, MANIFEST, args, 4)
    longer = train_gpt2.experiment_fingerprint(model, tok, MANIFEST, args, 8)

    with pytest.raises(SystemExit):
        train_gpt2.verify_resume_compatibility(
            blob, model, tok, MANIFEST, 8, False, log_fn=lambda *_: None,
            current_fingerprint=longer)

    logs = []
    train_gpt2.verify_resume_compatibility(
        blob, model, tok, MANIFEST, 8, True, log_fn=logs.append,
        current_fingerprint=longer)
    assert any("NOT the original trajectory" in m for m in logs)

    # --extend-budget must not license a hyperparameter change.
    tampered = train_gpt2.experiment_fingerprint(
        model, tok, MANIFEST, make_args(lr=1e-3), 8)
    with pytest.raises(SystemExit):
        train_gpt2.verify_resume_compatibility(
            blob, model, tok, MANIFEST, 8, True, log_fn=lambda *_: None,
            current_fingerprint=tampered)


def test_fingerprint_is_deterministic_and_covers_key_fields():
    model = build_model()
    tok = StubTokenizer()
    args = make_args()
    fp1 = train_gpt2.experiment_fingerprint(model, tok, MANIFEST, args, 4)
    fp2 = train_gpt2.experiment_fingerprint(model, tok, MANIFEST, args, 4)
    assert fp1 == fp2
    flat = {}
    train_gpt2._flatten("", fp1, flat)
    for key in ("schedule.peak_lr", "schedule.warmup", "data.seq_len",
                "data.effective_batch_tokens", "data.sampler", "data.framing",
                "optimizer.weight_decay", "optimizer.grad_clip", "dropout",
                "seed", "tokenizer.state_hash", "corpus.train_sha256",
                "architecture.vocab_size", "precision.amp_dtype"):
        assert key in flat, key
