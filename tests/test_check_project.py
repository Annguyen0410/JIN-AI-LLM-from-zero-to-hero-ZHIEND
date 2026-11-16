from __future__ import annotations
import unittest
import scripts.check_project as check_project
from scripts.check_project import GROUPS, iter_tests, select_suite

# NOTE: ``test_group`` is imported through the module on purpose. Importing the
# function by name made pytest collect it as a test and fail on its ``test_id``
# parameter ("fixture 'test_id' not found"), which is a real defect in this repo
# and not a reason to skip the file.
# (also deliberately not rebound to a module-level test_* name)


def named_case(name):
    case = unittest.FunctionTestCase(lambda: None)
    case.id = lambda: name
    return case


class TestGroups(unittest.TestCase):
    def test_classifies_mixed_phase3_by_class(self):
        for name, expected in (
            ("test_phase3.TestMixtureSampler.test_x", "pretrain"),
            ("tests.test_phase3.TestReasoningData.test_x", "sft"),
            ("test_phase3.TestTrainPref.test_x", "preference"),
            ("test_phase3.TestTokenizerExtension.test_x", "shared"),
            ("test_storage.StorageTests.test_x", "shared"),
        ):
            self.assertEqual(check_project.test_group(name), expected)

    def test_partition_no_loss_or_duplicate(self):
        cases = [named_case(name) for name in (
            "test_corpus_dedup.Case.test_x", "test_chat_template.Case.test_x",
            "test_phase3.TestTrainPref.test_x", "test_memory.Case.test_x")]
        suite = unittest.TestSuite([unittest.TestSuite(cases[:2]), unittest.TestSuite(cases[2:])])
        selected = [t.id() for group in GROUPS for t in iter_tests(select_suite(suite, [group]))]
        self.assertCountEqual(selected, [t.id() for t in cases])
        self.assertEqual(select_suite(suite, ["all"]).countTestCases(), 4)
        self.assertEqual(select_suite(suite, ["sft", "sft"]).countTestCases(), 1)
        self.assertEqual(select_suite(suite, ["pretrain", "sft"]).countTestCases(), 2)

    def test_discovery_errors_never_filtered_out(self):
        loader = unittest.TestLoader()
        broken = loader.loadTestsFromName("missing_jin_module_for_test")
        self.assertEqual(select_suite(broken, ["preference"]).countTestCases(), 1)


if __name__ == "__main__":
    unittest.main()
