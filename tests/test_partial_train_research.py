"""Admission must be label independent, bounded, and preserve the frozen tree."""
import unittest

from pathInference.research.partial_train import choose_paths


class PartialTrainAdmission(unittest.TestCase):
    def test_budget_destination_uniqueness_legality_and_new_endpoint_priority(self):
        augmented = [("1", "2", str(d)) for d in range(10, 26)]
        original = [
            ("1", "2", "30"), ("1", "2", "40", "30"),
            ("1", "3", "31"), ("1", "2", "32"), ("1", "3", "33"),
            ("1", "2", "34"), ("1", "2", "10"), ("1", "9", "35"),
            ("1", "2", "1", "36"), ("1", "2", "037"),
        ]
        chosen, counts = choose_paths(original, augmented, [("1", "2"), ("1", "3")],
                                      lambda a, b: "unknown")
        self.assertEqual(len(chosen), 4)
        self.assertEqual(len({p[-1] for p in chosen}), 4)
        self.assertTrue(all(p[-1] not in {a[-1] for a in augmented} for p in chosen))
        self.assertEqual({p[1] for p in chosen}, {"2", "3"})
        self.assertEqual(counts["outside_frozen_tree"], 1)
        self.assertEqual(counts["invalid"], 2)
        self.assertNotIn(("1", "2", "40", "30"), chosen)
        reversed_chosen, _ = choose_paths(reversed(original), augmented, [("1", "2"), ("1", "3")],
                                          lambda a, b: "unknown")
        self.assertEqual(chosen, reversed_chosen)

    def test_full_twohop_prefix_is_required_and_known_conflicts_are_rejected(self):
        augmented = [("1", "2", str(d)) for d in range(10, 18)]
        chosen, counts = choose_paths(
            [("1", "2", "3", "30"), ("1", "2", "4", "31"), ("1", "2", "3", "32")],
            augmented, [("1", "2", "3")],
            lambda a, b: "p2c" if (a, b) == ("1", "2") else "c2p")
        self.assertEqual(chosen, [])
        self.assertEqual(counts["known_commercial_conflict"], 2)
        self.assertEqual(counts["outside_frozen_tree"], 1)


if __name__ == "__main__":
    unittest.main()
