"""Phase 16 §2 — regression tests for the coverage-aware evaluator.

The model used here is a stub with uniform logits, so the exact loss is ln(vocab)
per target token and any denominator error shows up immediately.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_phase16 import (PROTOCOL_VERSION, document_windows,  # noqa: E402
                         evaluate, length_bucket)

VOCAB = 32
EOT = 0


class StubTokenizer:
    """One token per character; deterministic and inspectable."""

    eot_id = EOT
    vocab_size = VOCAB

    def encode(self, text: str) -> list[int]:
        return [(ord(ch) % (VOCAB - 1)) + 1 for ch in text]


class UniformModel(torch.nn.Module):
    """Uniform logits → loss = ln(vocab) on every real target, no matter what."""

    block_size = 1024

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, doc_mask=None):
        logits = torch.zeros(*x.shape, VOCAB, device=x.device)
        return logits, None


def doc(text: str, did: str = "d1", source: str = "wiki"):
    return {"text": text, "document_id": did, "source_id": source,
            "language": "en", "domain": "knowledge"}


def run(docs, seq_len=8, per_stratum=10):
    chosen = {("wiki", "en", "knowledge"): docs}
    return evaluate(UniformModel(), StubTokenizer(), chosen, seq_len, "cpu")


def test_short_document_is_scored_not_skipped():
    """A document shorter than one window must contribute tokens."""
    out = run([doc("abc", "short")])
    agg = out["aggregate"]
    assert agg["documents_scored"] == 1
    assert agg["tokens_scored"] == 4          # 3 body tokens + terminal boundary
    assert agg["token_coverage"] > 0


def test_document_exactly_one_window():
    out = run([doc("abcdefg", "exact")], seq_len=8)   # framed = 9 = seq_len+1
    agg = out["aggregate"]
    assert agg["tokens_scored"] == 8
    assert agg["documents_scored"] == 1


def test_document_just_longer_than_one_window():
    out = run([doc("abcdefghij", "longer")], seq_len=8)   # framed = 12
    agg = out["aggregate"]
    # 8 targets from the full window + 3 from the 4-token remainder
    assert agg["tokens_scored"] == 11
    assert agg["eligible_tokens"] == 11


def test_short_final_remainder_is_scored():
    out = run([doc("abcdefghijk", "tail")], seq_len=8)     # framed = 13
    agg = out["aggregate"]
    assert agg["tokens_scored"] == 12       # 8 + 4 from the 5-token remainder


def test_mixed_lengths_cover_every_token():
    docs = [doc("ab", "d1"), doc("abcdefghij", "d2"), doc("abcdefgh", "d3")]
    out = run(docs, seq_len=8)
    agg = out["aggregate"]
    assert agg["documents_selected"] == 3
    assert agg["documents_scored"] == 3
    assert agg["eligible_tokens"] == agg["tokens_scored"]
    assert agg["token_coverage"] == 1.0


def test_no_padding_tokens_enter_the_denominator():
    """Loss must be exactly ln(vocab) for any document, which padding would break."""
    for text in ("ab", "abcdef", "abcdefghijklmnop"):
        out = run([doc(text)], seq_len=8)
        for stratum in out["strata"].values():
            assert abs(stratum["loss"] - math.log(VOCAB)) < 1e-4
        assert out["aggregate"]["padding_tokens"] == 0


def test_windows_never_cross_documents():
    """Two documents scored together must equal the sum of scoring them apart."""
    a = doc("abcdefghij", "a")
    b = doc("klmnop", "b")
    together = run([a, b], seq_len=8)["aggregate"]
    only_a = run([a], seq_len=8)["aggregate"]
    only_b = run([b], seq_len=8)["aggregate"]
    assert together["tokens_scored"] == only_a["tokens_scored"] + only_b["tokens_scored"]
    # ln(vocab) everywhere, so a leaked context would have to shift this
    assert abs(together["token_weighted_loss"] - math.log(VOCAB)) < 1e-3


def test_identical_results_on_repeated_evaluation():
    docs = [doc("abcdefghij", "a"), doc("xyz", "b")]
    first = run(docs, seq_len=8)
    second = run(docs, seq_len=8)
    assert first["aggregate"] == second["aggregate"]
    assert first["strata"] == second["strata"]
    assert first["protocol"] == PROTOCOL_VERSION


def test_empty_document_is_reported_not_counted():
    out = run([doc("", "empty")])
    agg = out["aggregate"]
    assert agg["documents_scored"] == 0
    assert agg["documents_unscored"] == 1
    assert out["unscored_documents"][0]["reason"] == "empty_document"


def test_windows_helper_returns_tail_and_dropped_count():
    windows, dropped = document_windows(list(range(13)), seq_len=8)
    assert [len(w) for w in windows] == [9, 5]   # 8 + 4 targets = 12
    assert dropped == 0
    windows2, dropped2 = document_windows(list(range(10)), seq_len=8)
    assert [len(w) for w in windows2] == [9, 2]  # 8 + 1 targets = 9
    assert dropped2 == 0
    # every token after the first is a target exactly once, at any length
    for n in range(2, 40):
        ws, dr = document_windows(list(range(n)), seq_len=8)
        assert sum(len(w) - 1 for w in ws) == n - 1
        assert dr == 0
    assert document_windows([1], seq_len=8) == ([], 0)


def test_length_buckets():
    assert length_bucket(0) == "0-128"
    assert length_bucket(200) == "128-256"
    assert length_bucket(5000) == "2048+"
