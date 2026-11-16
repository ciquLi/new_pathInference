"""Ablation controls must isolate a single numerical intervention."""
import unittest

from pathInference.research.ablation import paired_assignment, recursion_plan


class RecursionAblationTests(unittest.TestCase):
    def test_no_recursion_preserves_firsthops_even_with_zero_observed_tails(self):
        def never_called(prefix):
            self.fail("No-recursion must not expand any firsthop")
        result = recursion_plan("1", ["2", "3"], lambda p, t: dict(valid_tail_count=0),
                                never_called, "none", 5)
        self.assertEqual(result["prefixes"], [("1", "2"), ("1", "3")])

    def test_all_low_recurses_despite_adequate_tails_and_filters_children(self):
        def predict(prefix):
            return dict(candidates=[dict(asn=n, admitted=ok) for n, ok in
                                    [("1", True), ("9", False), ("4", True), ("4", True),
                                     ("5", True), ("6", True), ("7", True), ("8", True), ("10", True)]])
        result = recursion_plan("1", ["2"], lambda p, t: dict(valid_tail_count=300),
                                predict, "all", 0.5)
        self.assertEqual(result["prefixes"], [("1", "2")] + [("1", "2", str(n)) for n in range(4, 9)])
        self.assertNotIn(("1", "2", "10"), result["prefixes"])

    def test_all_high_is_identical_to_production_mandatory_policy(self):
        from pathInference.pats.policy import mandatory_plan
        capacity = lambda p, t: dict(valid_tail_count=0)
        predict = lambda p: dict(candidates=[dict(asn="4", admitted=True)])
        self.assertEqual(recursion_plan("1", ["2"], capacity, predict, "all", 2),
                         mandatory_plan("1", ["2"], capacity, predict))

    def test_shared_pairs_cannot_change_split_when_the_library_changes(self):
        original = dict(train_pairs={"1->2"}, validation_pairs={"1->3"})
        self.assertEqual("train", paired_assignment("1->2", original, 7, 0.99))
        self.assertEqual("validation", paired_assignment("1->3", original, 7, 0.01))
        first = paired_assignment("1->4", original, 7, 0.1)
        expanded = dict(train_pairs={"1->2", "1->100"}, validation_pairs={"1->3", "1->101"})
        self.assertEqual(first, paired_assignment("1->4", expanded, 7, 0.1))


if __name__ == "__main__":
    unittest.main()
