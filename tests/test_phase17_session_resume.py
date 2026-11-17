"""Phase 17 §2.3 — a session stop must not move the training schedule.

Kaggle sessions are bounded by wall-clock time, not by the experiment budget. A
session stop saves state and exits; the next session resumes the same curve with
the same global total_steps, warmup and corpus fingerprint.
"""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_gpt2                     # noqa: E402
from model import JinAI               # noqa: E402

VOCAB = 64
SEQ = 16
BATCH = 2
EOT = 63
TOTAL_STEPS = 4


def build_model() -> JinAI:
    torch.manual_seed(0)
    return JinAI(vocab_size=VOCAB, embed_dim=32, block_size=SEQ, n_layers=2,
                 n_heads=2, n_kv_heads=2, dropout=0.0, modern=False,
                 gelu_tanh=True, resid_scaled_init=True, arch_id="session_test")


def write_shard(path: str, tokens: int = 4000) -> None:
    rng = np.random.default_rng(7)
    data = rng.integers(0, VOCAB - 1, size=tokens, dtype=np.uint16)
    data[::97] = EOT
    data.tofile(path)


def manifest(tmp_path) -> dict:
    return {
        "train": {"path": str(tmp_path / "train.bin"), "sha256": "trainhash",
                  "tokens": 4000},
        "val": {"path": str(tmp_path / "val.bin"), "sha256": "valhash",
                "tokens": 4000},
    }


def fake_args(**over):
    base = dict(preset="gpt2_small_reference", seq_len=SEQ, batch_size=BATCH,
                grad_accum=1, lr=6e-4, warmup=1, min_lr_ratio=0.1, seed=3,
                dropout=0.0, no_amp=True, session_max_minutes=0.0,
                init_from=None, resume_from=None, resume=False)
    base.update(over)
    return types.SimpleNamespace(**base)


class FakeTokenizer:
    kind = "gpt2_bpe"
    vocab_size = VOCAB

    def to_state_dict(self) -> dict:
        return {"kind": self.kind, "vocab_size": self.vocab_size}


def optimizer_for(model) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))


def train_steps(model, optimizer, batcher, start: int, end: int,
                total_steps: int = TOTAL_STEPS) -> None:
    model.train()
    for step in range(start + 1, end + 1):
        lr = train_gpt2.lr_at(step, total_steps, 6e-4, 1, 0.1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        x, y, m = batcher.batch()
        _, loss = model(x, y, doc_mask=m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def test_session_limit_is_not_part_of_the_experiment_fingerprint(tmp_path):
    write_shard(str(tmp_path / "train.bin"))
    write_shard(str(tmp_path / "val.bin"))
    model = build_model()
    tok = FakeTokenizer()
    man = manifest(tmp_path)
    fp_a = train_gpt2.experiment_fingerprint(model, tok, man,
                                             fake_args(session_max_minutes=0.0),
                                             TOTAL_STEPS)
    fp_b = train_gpt2.experiment_fingerprint(model, tok, man,
                                             fake_args(session_max_minutes=90.0),
                                             TOTAL_STEPS)
    assert fp_a == fp_b
    assert not train_gpt2.compare_fingerprints(fp_a, fp_b)


def test_resume_from_wins_over_resume(tmp_path):
    """`--resume --resume-from X` must restore X, not <out-dir>/latest.pt."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    default = str(out_dir / "latest.pt")
    chosen = str(tmp_path / "attached" / "latest.pt")
    os.makedirs(os.path.dirname(chosen))
    torch.save({"iteration": 5}, chosen)
    assert train_gpt2.resolve_restore_path(True, chosen, default) == chosen
    assert train_gpt2.resolve_restore_path(False, chosen, default) == chosen



def test_missing_resume_checkpoint_is_fatal(tmp_path):
    """A resume with nothing to restore must stop, never train from scratch."""
    with pytest.raises(SystemExit):
        train_gpt2.resolve_restore_path(True, None, str(tmp_path / "latest.pt"))
    with pytest.raises(SystemExit):
        train_gpt2.resolve_restore_path(True, str(tmp_path / "missing.pt"),
                                        str(tmp_path / "latest.pt"))


def test_no_resume_means_fresh_run(tmp_path):
    assert train_gpt2.resolve_restore_path(False, None,
                                           str(tmp_path / "latest.pt")) is None


def test_session_expired_helper():
    assert train_gpt2.session_expired(0.0, 0.0, now=10_000.0) is False
    assert train_gpt2.session_expired(0.0, 90.0, now=60.0) is False
    assert train_gpt2.session_expired(0.0, 1.0, now=61.0) is True


def test_session_stop_preserves_state_and_resumes_identically(tmp_path):
    """Continuous 4 steps == 2 steps + save + resume 2 steps."""
    write_shard(str(tmp_path / "train.bin"))
    write_shard(str(tmp_path / "val.bin"))
    man = manifest(tmp_path)
    meta = {"architecture_id": "session_test", "training_args": {},
            "dataset_manifest": man,
            "tokenizer": {"kind": "gpt2_bpe", "vocab_size": VOCAB},
            "schedule": {"total_steps": TOTAL_STEPS, "warmup": 1, "peak_lr": 6e-4},
            "hardware": {"device": "cpu"}, "init_mode": "continued_pretrain",
            "parent_sha256": "abc", "experiment_config_hash": "fp"}

    # A: continuous
    model_a = build_model()
    opt_a = optimizer_for(model_a)
    batcher_a = train_gpt2.ShardBatcher(str(tmp_path / "train.bin"), SEQ, BATCH,
                                        "cpu", 3, EOT)
    train_steps(model_a, opt_a, batcher_a, 0, TOTAL_STEPS)

    # B: two steps, save (as a session stop would), rebuild, restore, two more
    model_b = build_model()
    opt_b = optimizer_for(model_b)
    batcher_b = train_gpt2.ShardBatcher(str(tmp_path / "train.bin"), SEQ, BATCH,
                                        "cpu", 3, EOT)
    train_steps(model_b, opt_b, batcher_b, 0, 2)
    ckpt = str(tmp_path / "latest.pt")
    train_gpt2.save_state(ckpt, model_b, opt_b, None, 2, 2 * BATCH * SEQ, 99.0,
                          [], meta, batcher=batcher_b)
    saved = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    assert saved["iteration"] == 2
    assert saved["schedule"]["total_steps"] == TOTAL_STEPS   # budget unchanged

    model_c = build_model()
    opt_c = optimizer_for(model_c)
    batcher_c = train_gpt2.ShardBatcher(str(tmp_path / "train.bin"), SEQ, BATCH,
                                        "cpu", 3, EOT)
    train_gpt2.restore_training_state(saved, model_c, opt_c, None, batcher_c, "cpu")
    train_steps(model_c, opt_c, batcher_c, int(saved["iteration"]), TOTAL_STEPS)

    for (ka, va), (kc, vc) in zip(model_a.state_dict().items(),
                                  model_c.state_dict().items()):
        assert torch.equal(va, vc), ka


def test_saved_manifest_hash_matches_the_checkpoint(tmp_path):
    write_shard(str(tmp_path / "train.bin"))
    write_shard(str(tmp_path / "val.bin"))
    model = build_model()
    opt = optimizer_for(model)
    man = manifest(tmp_path)
    meta = {"architecture_id": "session_test", "training_args": {},
            "dataset_manifest": man,
            "tokenizer": {"kind": "gpt2_bpe", "vocab_size": VOCAB},
            "schedule": {"total_steps": TOTAL_STEPS}, "hardware": {"device": "cpu"},
            "init_mode": "continued_pretrain", "parent_sha256": "abc",
            "parent_checkpoint": "phase14.pt"}
    path = str(tmp_path / "latest.pt")
    train_gpt2.save_state(path, model, opt, None, 1, BATCH * SEQ, 1.0, [], meta)

    saved_manifest = json.load(open(path + ".manifest.json", encoding="utf-8"))
    assert saved_manifest["sha256"] == train_gpt2.sha256_file(path)
    assert saved_manifest["init_mode"] == "continued_pretrain"
    assert saved_manifest["parent_sha256"] == "abc"
    assert saved_manifest["additional_tokens_this_run"] == BATCH * SEQ


def test_resume_refuses_a_changed_corpus(tmp_path):
    write_shard(str(tmp_path / "train.bin"))
    write_shard(str(tmp_path / "val.bin"))
    model = build_model()
    tok = FakeTokenizer()
    man = manifest(tmp_path)
    args = fake_args()
    fp = train_gpt2.experiment_fingerprint(model, tok, man, args, TOTAL_STEPS)
    blob = {"dataset_manifest": man, "architecture_id": "session_test",
            "tokenizer": {"kind": "gpt2_bpe", "vocab_size": VOCAB},
            "schedule": {"total_steps": TOTAL_STEPS}, "experiment_config": fp}
    train_gpt2.verify_resume_compatibility(blob, model, tok, man, TOTAL_STEPS,
                                           False, log_fn=lambda *_: None,
                                           current_fingerprint=fp)

    changed = json.loads(json.dumps(man))
    changed["train"]["sha256"] = "different"
    with pytest.raises(SystemExit):
        train_gpt2.verify_resume_compatibility(blob, model, tok, changed,
                                               TOTAL_STEPS, False,
                                               log_fn=lambda *_: None,
                                               current_fingerprint=fp)
