"""No new dependencies: compile source, run unittest, capture honest results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import unittest
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from storage import configure_runtime, output_path
configure_runtime(ROOT)


# Test classes remain at their original paths; grouping does not break discovery.
GROUPS = {
    "pretrain": {
        "test_corpus_dedup", "test_optim_muon.TestMuon",
        "test_phase3.TestDataQuality", "test_phase3.TestMixtureSampler",
    },
    "sft": {
        "test_chat_template", "test_phase3.TestReasoningData",
        "test_phase3.TestSelfDistillVerifiers", "test_phase3.TestToolTemplate",
        "test_system_upgrade.DatasetTests",
    },
    "preference": {"test_phase3.TestTrainPref"},
    "shared": set(),  # Every test not explicitly assigned above.
}


def test_group(test_id: str) -> str:
    # Both unittest discovery (test_x) and module invocation (tests.test_x).
    name = test_id.removeprefix("tests.")
    for group, prefixes in GROUPS.items():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            return group
    return "shared"


def iter_tests(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from iter_tests(test)
        else:
            yield test


def select_suite(suite, groups):
    selected = set(groups)
    return unittest.TestSuite(test for test in iter_tests(suite)
                              if "all" in selected or test_group(test.id()) in selected
                              or isinstance(test, unittest.loader._FailedTest))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--syntax-only",action="store_true")
    parser.add_argument("--report",default="bench_results/upgrade_tests.json",help="Test report output path")
    parser.add_argument("--skip-preset-shapes",action="store_true",help="Explicitly skip allocation-heavy preset shape test on RAM-constrained hosts")
    parser.add_argument("--group", action="append", choices=("all", *GROUPS),
                        help="Run a test category; repeat to combine. Default: all")
    parser.add_argument("--list-groups", action="store_true", help="List categories without importing/running tests")
    args=parser.parse_args()
    if args.list_groups:
        print(json.dumps({**{k: sorted(v) for k,v in GROUPS.items()},
                          "shared_note": "All remaining infrastructure/model/tokenizer/data/workflow tests"}, indent=2))
        return
    groups=args.group or ["all"]
    files=list(ROOT.glob("*.py"))+list((ROOT/"scripts").glob("*.py"))+list((ROOT/"tests").glob("*.py"))
    for path in files:
        compile(path.read_text(encoding="utf-8-sig"),str(path),"exec")
    print(f"Syntax checked: {len(files)} Python files")
    if args.syntax_only:return
    start=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    if args.skip_preset_shapes:
        from test_model_shapes import TestPresetShapes
        TestPresetShapes.test_presets_forward_and_generate = unittest.skip(
            "Explicit --skip-preset-shapes: local host allocation pressure; not verified in this run")(
                TestPresetShapes.test_presets_forward_and_generate)
    suite=select_suite(suite, groups)
    group_counts = {name: sum(test_group(test.id()) == name for test in iter_tests(suite)) for name in GROUPS}
    print("Test groups: " + ", ".join(groups))
    print("Selected counts: " + json.dumps(group_counts))
    if not suite.countTestCases():
        raise SystemExit("No tests selected; check group mappings")
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    report={"groups":groups,"group_counts":group_counts,"kind":"software regression, not learned-model benchmark","tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
            "skips":[{"test":str(t),"reason":why} for t,why in result.skipped],
            "elapsed_s":round(time.monotonic()-start,3),"syntax_files":len(files),
            "typecheck":"NOT RUN: no static checker configured/installed; syntax plus executable tests",
            "passed":result.wasSuccessful()}
    report_path=Path(output_path(args.report,root=ROOT))
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__=="__main__":main()
