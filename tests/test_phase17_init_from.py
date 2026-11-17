"""Phase 17 §2.3 — continued pretraining must start from the parent's weights.

`--init-from` is weight transfer, not a resume: a new optimizer, a new schedule
and a new data state. These tests pin the compatibility checks that make the
difference observable.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_gpt2                     # noqa: E402
from model import JinAI               # noqa: E402

VOCAB = 64
BLOCK = 16


def make_model(seed: int = 0, block: int = BLOCK) -> JinAI:
    torch.manual_seed(seed)
    return JinAI(vocab_size=VOCAB, embed_dim=32, block_size=block, n_layers=2,
                 n_heads=2, n_kv_heads=2, dropout=0.0, modern=False,
                 gelu_tanh=True, resid_scaled_init=True, arch_id="init_test")


class FakeTokenizer:
    kind = "gpt2_bpe"
    vocab_size = VOCAB


def parent_blob(model: JinAI, *, arch_id: str = "init_test", vocab: int = VOCAB,
                kind: str = "gpt2_bpe", drop: str | None = None,
                iteration: int = 640, tokens_seen: int = 47_185_920) -> dict:
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if drop:
        sd.pop(drop)
    return {
        "architecture_id": arch_id,
        "model_config": model.export_config(),
        "tokenizer": {"kind": kind, "vocab_size": vocab},
        "iteration": iteration,
        "tokens_seen": tokens_seen,
        "model_state_dict": sd,
    }


def save(blob: dict, path) -> str:
    torch.save(blob, path)
    return str(path)


def test_init_from_loads_the_exact_parent_weights(tmp_path):
    parent = make_model(seed=1)
    blob = parent_blob(parent)
    path = save(blob, tmp_path / "parent.pt")

    child = make_model(seed=99)                     # deliberately different
    before = child.state_dict()["tok_emb.weight"].clone()
    lineage = train_gpt2.init_from_checkpoint(child, FakeTokenizer(), path,
                                              log_fn=lambda *_: None)
    after = child.state_dict()

    assert not torch.equal(before, after["tok_emb.weight"])
    for key, tensor in blob["model_state_dict"].items():
        assert torch.equal(after[key], tensor), key
    assert lineage["init_mode"] == "continued_pretrain"


def test_init_from_records_the_parent_sha256(tmp_path):
    blob = parent_blob(make_model())
    path = save(blob, tmp_path / "parent.pt")
    lineage = train_gpt2.init_from_checkpoint(make_model(), FakeTokenizer(), path,
                                              log_fn=lambda *_: None)
    import hashlib
    expected = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert lineage["parent_sha256"] == expected
    assert lineage["parent_iteration"] == 640
    assert lineage["parent_tokens_seen"] == 47_185_920


def test_missing_parent_checkpoint_refuses_to_start(tmp_path):
    with pytest.raises(SystemExit):
        train_gpt2.init_from_checkpoint(make_model(), FakeTokenizer(),
                                        str(tmp_path / "nope.pt"),
                                        log_fn=lambda *_: None)


def test_architecture_mismatch_is_rejected(tmp_path):
    path = save(parent_blob(make_model(), arch_id="some_other_arch"), tmp_path / "p.pt")
    with pytest.raises(SystemExit):
        train_gpt2.init_from_checkpoint(make_model(), FakeTokenizer(), path,
                                        log_fn=lambda *_: None)


def test_tokenizer_mismatch_is_rejected(tmp_path):
    path = save(parent_blob(make_model(), vocab=VOCAB + 1), tmp_path / "p.pt")
    with pytest.raises(SystemExit):
        train_gpt2.init_from_checkpoint(make_model(), FakeTokenizer(), path,
                                        log_fn=lambda *_: None)


def test_positional_embedding_mismatch_is_rejected(tmp_path):
    longer = make_model(block=BLOCK * 2)
    path = save(parent_blob(longer), tmp_path / "p.pt")
    with pytest.raises(SystemExit) as exc:
        train_gpt2.init_from_checkpoint(make_model(), FakeTokenizer(), path,
                                        log_fn=lambda *_: None)
    # Rejected either by the config check or by the positional-embedding shape
    # check; both are hard failures with the offending field named.
    message = str(exc.value).lower()
    assert "positional" in message or "block_size" in message


def test_missing_weights_are_not_silently_ignored(tmp_path):
    path = save(parent_blob(make_model(), drop="blocks.0.attn.out_proj.weight"),
                tmp_path / "p.pt")
    with pytest.raises(SystemExit):
        train_gpt2.init_from_checkpoint(make_model(), FakeTokenizer(), path,
                                        log_fn=lambda *_: None)


def test_head_stays_tied_after_init(tmp_path):
    parent = make_model()
    blob = parent_blob(parent)
    # make the duplicate head identical, as the real Phase 14 checkpoint has it
    blob["model_state_dict"]["head.weight"] = \
        blob["model_state_dict"]["tok_emb.weight"].clone()
    path = save(blob, tmp_path / "p.pt")

    child = make_model(seed=5)
    train_gpt2.init_from_checkpoint(child, FakeTokenizer(), path,
                                    log_fn=lambda *_: None)
    assert child.head.weight.data_ptr() == child.tok_emb.weight.data_ptr()


def test_continue_pretrain_without_init_from_is_refused():
    with pytest.raises(SystemExit):
        train_gpt2.validate_continued_args(None, False, None, True)


def test_init_from_cannot_be_combined_with_resume():
    with pytest.raises(SystemExit):
        train_gpt2.validate_continued_args("ckpt.pt", True, None, True)
    with pytest.raises(SystemExit):
        train_gpt2.validate_continued_args("ckpt.pt", False, "other.pt", False)


def test_plain_training_is_allowed_without_init_from():
    train_gpt2.validate_continued_args(None, False, None, False)


def test_init_from_does_not_restore_optimizer_state(tmp_path):
    """A continuation is a new experiment: the optimizer starts empty."""
    parent = make_model()
    blob = parent_blob(parent)
    blob["optimizer_state_dict"] = {"fake": "parent optimizer"}
    path = save(blob, tmp_path / "p.pt")
    lineage = train_gpt2.init_from_checkpoint(make_model(), FakeTokenizer(), path,
                                              log_fn=lambda *_: None)
    # The lineage contract exposes only model provenance; nothing from the parent
    # optimizer is carried over, and the caller builds a fresh AdamW.
    assert "optimizer_state_dict" not in lineage
    assert lineage["init_mode"] == "continued_pretrain"
