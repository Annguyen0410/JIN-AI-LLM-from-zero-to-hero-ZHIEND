"""Phase 16 §3 — tests for the deterministic task benchmark and its sandbox.

These tests exist to prove the benchmark can fail, can pass, and cannot be gamed
by empty output or by valid-looking-but-wrong code.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase16_task_benchmark import (CODE_PROBLEMS, TASKS, check_answer,  # noqa: E402
                                    extract_code, run_code_tests)
from phase16_gold import GOLD, GOLD_CODE                            # noqa: E402


def test_gold_answers_pass_every_non_code_task():
    failures = [(t["id"], check_answer(GOLD[t["id"]], t["check"])[1])
                for t in TASKS if not check_answer(GOLD[t["id"]], t["check"])[0]]
    assert failures == []


def test_gold_code_passes_every_problem():
    for func, _prompt, tests in CODE_PROBLEMS:
        res = run_code_tests(GOLD_CODE[func], func, tests)
        assert res["ran"] and res["passed"] == res["total"], (func, res)


def test_empty_output_never_passes():
    for task in TASKS:
        ok, failed = check_answer("", task["check"])
        assert not ok and "empty_output" in failed


def test_wrong_number_fails():
    ok, failed = check_answer("The answer is 94.", {"number": 95})
    assert not ok and any(f.startswith("wrong_number") for f in failed)


def test_abstention_requires_no_fabricated_number():
    ok, _ = check_answer("There is not enough information to answer.", {"abstain": True})
    assert ok
    ok2, failed2 = check_answer("It is about 12 grams.", {"abstain": True})
    assert not ok2 and "did_not_abstain" in failed2
    ok3, failed3 = check_answer("Not enough information, maybe 12 grams.",
                                {"abstain": True})
    assert not ok3 and "fabricated_number" in failed3


def test_json_checks_require_real_json():
    ok, _ = check_answer('{"name": "Lan", "age": 30}', {"json_valid": True,
                                                       "required_keys": ["name", "age"]})
    assert ok
    ok2, failed2 = check_answer("name: Lan, age: 30", {"json_valid": True})
    assert not ok2 and "invalid_json" in failed2
    ok3, failed3 = check_answer('{"city": "Paris"}', {"json_valid": True,
                                                     "exact_keys": ["city", "country"]})
    assert not ok3


def test_word_limit_is_enforced():
    ok, _ = check_answer("one two three", {"max_words": 3})
    assert ok
    ok2, failed2 = check_answer("one two three four", {"max_words": 3})
    assert not ok2 and any(f.startswith("too_many_words") for f in failed2)


def test_syntax_only_is_not_success():
    """Valid Python that is wrong must fail; the harness must not credit syntax."""
    wrong = "def sum_list(nums):\n    return 0"
    tests = [[([1, 2, 3],), 6]]
    res = run_code_tests(wrong, "sum_list", tests)
    assert res["syntax_ok"] is True
    assert res["ran"] is True
    assert res["passed"] == 0, "a parseable but wrong function must not pass"


def test_code_that_crashes_is_reported_not_passed():
    res = run_code_tests("def sum_list(nums):\n    return nums.undefined_attr",
                         "sum_list", [[([1],), 1]])
    assert res["passed"] == 0 and res["reason"] in ("runtime_error", "test_failure")


def test_missing_function_is_reported():
    res = run_code_tests("x = 1", "sum_list", [[([1],), 1]])
    assert res["passed"] == 0 and res["reason"] == "function_not_defined"


def test_infinite_loop_times_out():
    res = run_code_tests("def sum_list(nums):\n    while True:\n        pass",
                         "sum_list", [[([1],), 1]], timeout=1.0)
    assert res["passed"] == 0 and res["reason"] == "timeout"


def test_code_sandbox_has_no_network_module():
    res = run_code_tests(
        "import urllib.request\ndef sum_list(nums):\n"
        "    return urllib.request.urlopen('http://example.com').status",
        "sum_list", [[([1],), 1]])
    assert res["passed"] == 0


def test_fenced_code_is_extracted():
    text = "Here you go:\n```python\ndef sum_list(nums):\n    return sum(nums)\n```\nDone."
    assert "def sum_list" in extract_code(text)
    res = run_code_tests(text, "sum_list", [[([1, 2],), 3]])
    assert res["ran"] and res["passed"] == res["total"]


def test_benchmark_has_the_required_categories_and_size():
    categories = {t["category"] for t in TASKS}
    assert {"instruction", "qa", "math", "vi", "extract", "summarize",
            "multiturn", "tool"} <= categories
    assert len(CODE_PROBLEMS) >= 30
    assert len(TASKS) + len(CODE_PROBLEMS) >= 60
