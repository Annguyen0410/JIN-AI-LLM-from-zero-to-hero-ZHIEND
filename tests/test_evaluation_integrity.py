"""Evaluation integrity tests (Phase 15D).

The Phase 14 evaluator padded short documents with EOT and then scored the
padded tensor, so artificial tokens entered the metric. These tests pin the new
contract:

* every scored token is a real token (``padding_tokens == 0``);
* adding a document that is too short to form a window cannot change the loss;
* documents are scored independently, so their losses combine additively;
* stratum sampling is deterministic, source-aware and capped per stratum.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import eval_phase15  # noqa: E402


VOCAB = 64


class StubModel(torch.nn.Module):
    """Returns uniform logits: cross-entropy is exactly ln(VOCAB) per token.

    With uniform logits the loss is a pure token-count question, which is what
    makes padding contamination visible: any padding in the denominator would
    still give ln(V) per token but the token COUNT would be wrong.
    """

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, doc_mask=None):
        b, t = x.shape
        logits = torch.zeros(b, t, VOCAB)
        return logits, None


class StubTokenizer:
    kind = "stub"
    vocab_size = VOCAB
    eot_id = VOCAB - 1

    def encode(self, text: str) -> list[int]:
        # one token per word, ids kept inside the vocabulary
        return [abs(hash(w)) % (VOCAB - 2) + 1 for w in text.split()]


def doc(doc_id: str, words: int, source="src_a", language="en", domain="knowledge"):
    return {"document_id": doc_id, "source_id": source, "language": language,
            "domain": domain, "text": " ".join(f"w{i}" for i in range(words))}


def test_loss_is_exactly_ln_vocab_per_real_token():
    metrics = eval_phase15.score_documents(StubModel(), StubTokenizer(),
                                           [doc("d1", 40)], seq_len=8)
    assert metrics["loss"] == pytest.approx(math.log(VOCAB), abs=1e-3)   # rounded to 4dp
    assert metrics["padding_tokens"] == 0
    # framed = [EOT] + 40 tokens + [EOT] = 42 tokens, 41 possible targets;
    # stride = seq_len = 8, so windows = 41 // 8 = 5 and 40 targets are scored
    assert metrics["windows"] == 5
    assert metrics["tokens_scored"] == 40


def test_adding_a_too_short_document_cannot_change_the_loss():
    model, tok = StubModel(), StubTokenizer()
    base = eval_phase15.score_documents(model, tok, [doc("d1", 40)], seq_len=8)
    with_short = eval_phase15.score_documents(
        model, tok, [doc("d1", 40), doc("short", 3)], seq_len=8)
    assert with_short["loss"] == base["loss"]
    assert with_short["tokens_scored"] == base["tokens_scored"]
    assert with_short["padding_tokens"] == 0


def test_documents_are_scored_independently():
    model, tok = StubModel(), StubTokenizer()
    a = eval_phase15.score_documents(model, tok, [doc("a", 30)], seq_len=8)
    b = eval_phase15.score_documents(model, tok, [doc("b", 50)], seq_len=8)
    both = eval_phase15.score_documents(model, tok, [doc("a", 30), doc("b", 50)],
                                        seq_len=8)
    assert both["tokens_scored"] == a["tokens_scored"] + b["tokens_scored"]
    assert both["nll_sum"] == pytest.approx(a["nll_sum"] + b["nll_sum"], abs=0.05)


def test_boundary_tokens_are_framed_once_per_document():
    """A document is framed with its own EOT at both ends and split into
    non-overlapping complete windows; nothing is invented or dropped."""
    tok = StubTokenizer()
    framed = [tok.eot_id] + [5] * 20 + [tok.eot_id]
    windows = eval_phase15.document_windows(framed, 8)
    assert len(windows) == 2, "22 tokens -> two complete windows of 9"
    for i, window in enumerate(windows):
        assert window == framed[i * 9:(i + 1) * 9]
    assert windows[0][0] == tok.eot_id, "the document starts with its boundary token"
    assert all(w == framed[i * 9:(i + 1) * 9] for i, w in enumerate(windows))


def write_val(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_stratified_sampling_is_deterministic_and_covers_every_source(tmp_path):
    rows = []
    for source in ("wiki", "cosmo", "prepared"):
        for language in ("en", "vi"):
            for i in range(40):
                rows.append(doc(f"{source}-{language}-{i}", 30, source=source,
                                language=language))
    path = str(tmp_path / "val.jsonl")
    write_val(path, rows)

    first = eval_phase15.load_documents(path, 5, seed=15)
    second = eval_phase15.load_documents(path, 5, seed=15)
    assert {k: [d["document_id"] for d in v] for k, v in first.items()} == \
           {k: [d["document_id"] for d in v] for k, v in second.items()}

    assert len(first) == 6, "every (source, language, domain) stratum is sampled"
    for key, docs in first.items():
        assert len(docs) == 5, f"cap not respected for {key}"
        assert len({d["document_id"] for d in docs}) == 5

    for source in ("wiki", "cosmo", "prepared"):
        assert any(k[0] == source for k in first), f"source {source} not sampled"


def test_sampling_does_not_take_file_order(tmp_path):
    rows = [doc(f"doc-{i}", 30, source="a" if i < 100 else "b") for i in range(200)]
    path = str(tmp_path / "val.jsonl")
    write_val(path, rows)
    chosen = eval_phase15.load_documents(path, 10, seed=7)
    ids = [d["document_id"] for docs in chosen.values() for d in docs]
    assert ids != [f"doc-{i}" for i in range(len(ids))], "sampling followed file order"


def test_different_seeds_pick_different_documents(tmp_path):
    rows = [doc(f"d{i}", 30) for i in range(100)]
    path = str(tmp_path / "val.jsonl")
    write_val(path, rows)
    a = eval_phase15.load_documents(path, 10, seed=1)
    b = eval_phase15.load_documents(path, 10, seed=2)
    ids_a = {d["document_id"] for docs in a.values() for d in docs}
    ids_b = {d["document_id"] for docs in b.values() for d in docs}
    assert ids_a != ids_b
