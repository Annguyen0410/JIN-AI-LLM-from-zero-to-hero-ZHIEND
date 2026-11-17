"""Phase 17 §8 — system numbers must mean what their labels say."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import phase17_system_benchmark as bench  # noqa: E402


def test_corpus_is_larger_and_harder_than_phase16():
    docs, sources = bench.build_corpus()
    assert len(docs) >= 80
    assert len(sources) == len(docs)
    assert any(d["id"].startswith("filler_") for d in docs)      # distractors
    assert all(d["license"] == "project_authored" for d in docs)


def test_queries_cover_every_required_track():
    kinds = {q["kind"] for q in bench.build_queries()}
    assert {"direct", "paraphrase", "vietnamese", "unanswerable",
            "multi_doc"} <= kinds


def test_retrieval_metrics_are_bounded_and_consistent():
    report = bench.run_retrieval_benchmark()
    assert report["corpus_documents"] >= 80
    assert report["answerable_queries"] >= 60
    for key, value in report["recall_at"].items():
        assert 0.0 <= value <= 1.0
        assert value >= report["recall_at"]["1"] - 1e-9, "recall@k must be monotone"
    assert 0.0 <= report["mrr"] <= 1.0
    assert 0.0 <= report["evidence_selection_accuracy"] <= 1.0
    assert 0.0 <= report["evidence_support_rate"] <= 1.0
    assert report["credited_to_model"] == 0


def test_unanswerable_questions_are_abstained():
    report = bench.run_retrieval_benchmark()
    assert report["unanswerable_queries"] == 6
    assert report["abstention_accuracy"] == 1.0


def test_evidence_support_is_stricter_than_citation_syntax():
    """Support must be at most selection: a picked passage can still lack the answer."""
    report = bench.run_retrieval_benchmark()
    assert report["evidence_support_rate"] <= report["evidence_selection_accuracy"] + 1e-9
    assert report["citation_syntax_checks_are_not_support"] is True


def test_router_and_arithmetic_are_reported_separately():
    section = bench.run_router_section()
    assert "router_accuracy" in section and "arithmetic_accuracy" in section


def test_written_report_has_no_model_credit(tmp_path):
    out = tmp_path / "sys.json"
    argv = sys.argv
    sys.argv = ["phase17_system_benchmark", "--out", str(out)]
    try:
        bench.main()
    finally:
        sys.argv = argv
    import json
    report = json.load(open(out, encoding="utf-8"))
    assert report["protocol"] == "phase17.system.v1"
    assert report["retrieval"]["credited_to_model"] == 0
    assert report["router"]["credited_to_model"] == 0
    assert "not measured here" in report["attribution"]["model_in_loop"]
    assert report["limitations"]
