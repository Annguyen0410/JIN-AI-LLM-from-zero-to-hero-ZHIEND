"""Phase 17 §8.1 — the router must pick the tool, not just compute correctly.

Routing accuracy and arithmetic accuracy are asserted separately: a router that
sends everything to the calculator would score 100 % on arithmetic and 0 % on
routing, and the tests must be able to tell those apart.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import phase17_intent_router as router  # noqa: E402


@pytest.mark.parametrize("prompt,expected", [
    ("2 + 3", 5),
    ("what is 12 * 7?", 84),
    ("calculate (12 + 8) * 3", 60),
    ("tính 12 + 5", 17),
    ("2 ** 10", 1024),
    ("2^8", 256),
    ("17 % 5", 2),
    ("100 // 9", 11),
    ("-4 + 10", 6),
    ("3.5 * 2", 7),
])
def test_calculator_routing_and_value(prompt, expected):
    route = router.route(prompt)
    assert route.intent == router.INTENT_CALCULATOR
    assert route.status == router.STATUS_OK
    assert float(route.value) == expected
    assert route.executor == "calculator"


@pytest.mark.parametrize("prompt,expected", [
    ("15% of 240", 36.0),
    ("20% off 250", 200.0),
    ("30% của 400", 120.0),
])
def test_percentage_routing(prompt, expected):
    route = router.route(prompt)
    assert route.intent in (router.INTENT_PERCENT, router.INTENT_CALCULATOR)
    assert route.status == router.STATUS_OK
    assert float(route.value) == expected


@pytest.mark.parametrize("prompt,expected", [
    ("100 c to f", 212.0),
    ("32 f to c", 0.0),
    ("5 km to miles", 5 * 1000 / 1609.344),
    ("2 kg to lb", 2 / 0.45359237),
    ("1 hour to minutes", 60.0),
])
def test_unit_conversion_routing(prompt, expected):
    route = router.route(prompt)
    assert route.intent == router.INTENT_CONVERT
    assert route.status == router.STATUS_OK
    assert float(route.value) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("prompt", router.NEGATIVE_CASES)
def test_non_arithmetic_requests_are_not_computed(prompt):
    route = router.route(prompt)
    assert route.intent == router.INTENT_NONE
    assert route.status == router.STATUS_NO_TOOL
    assert route.value is None


def test_dates_and_phones_are_named_as_such():
    assert "date" in router.route("12/03/2024").explanation
    assert "phone" in router.route("090-123-4567").explanation


def test_no_eval_is_used():
    source = open(os.path.join(ROOT, "scripts", "phase17_intent_router.py"),
                  encoding="utf-8").read()
    assert "eval(" not in source.replace("evaluate_expression(", "")
    assert "exec(" not in source


@pytest.mark.parametrize("expression", ["__import__('os')", "sqrt(9)", "x + 1",
                                        "2 ** 99999", "3 + * 4"])
def test_unsupported_expressions_are_refused_not_guessed(expression):
    route = router.route(expression)
    assert route.status in (router.STATUS_UNSUPPORTED, router.STATUS_ERROR,
                           router.STATUS_NO_TOOL)
    assert route.value is None


def test_division_by_zero_is_an_error_not_a_crash():
    route = router.route("1/0")
    assert route.status in (router.STATUS_ERROR, router.STATUS_UNSUPPORTED)
    assert "zero" in route.explanation.lower()


def test_bare_percentage_is_ambiguous():
    route = router.route("50 + 10%")
    assert route.status == router.STATUS_AMBIGUOUS
    assert route.value is None


def test_trace_preserves_the_original_request():
    original = "  What is (12 + 8) * 3?  "
    route = router.route(original)
    assert route.request == original
    trace = route.to_dict()
    for key in ("request", "intent", "tool", "args", "status", "executor"):
        assert key in trace


def test_router_benchmark_separates_the_two_accuracies():
    report = router.run_router_benchmark()
    assert report["supported_cases"] >= 200
    assert report["router_accuracy"] == 1.0
    assert report["arithmetic_accuracy"] == 1.0
    assert report["negative_rejection_rate"] == 1.0
    assert report["credited_to_model"] == 0


def test_retrieval_routing_requires_sources_above_the_coverage_threshold():
    sources = [{"id": "doc1", "title": "Doc",
                "body": "Kestrel Bridge in Vallis is 742 metres long."}]
    hit = router.route("how long is the Kestrel Bridge?", sources=sources)
    assert hit.intent == router.INTENT_RETRIEVAL
    assert hit.status == router.STATUS_OK
    assert hit.evidence and hit.executor == "retrieval"

    miss = router.route("what is the population of the moon?", sources=sources)
    assert miss.status == router.STATUS_UNSUPPORTED
    assert "abstain" in miss.explanation
