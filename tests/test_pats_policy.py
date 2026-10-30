import unittest

from pathveil.pats.policy import POLICY, RECURSION_TOPK, mandatory_plan, select_root
from pathveil.pats.workflow import pats_training_tasks


class PatsPolicyTests(unittest.TestCase):
    def test_source_rho_selects_branch_without_neighbor_rho(self):
        original = {
            "method": "combined",
            "root": {
                "rankings": {"combined": ["10", "20", "30", "40", "50"]},
                "support_probabilities": {"10": 0.9, "20": 0.8, "30": 0.7, "40": 0.6},
                "provenance": {str(n): {} for n in (10, 20, 30, 40)},
            },
            "top3_union_sources": {"ranker_a": [10, 20, 30], "ranker_b": [20, 30, 40]},
            "selected_neighbors": [10, 20, 30, 40, 50],
        }
        self.assertEqual(select_root(original, 0.5)["selected_neighbors"], [10, 20, 30, 40, 50])
        high = select_root(original, 2.0)
        self.assertEqual(set(high["selected_neighbors"]), {10, 20, 30, 40})
        self.assertEqual(high["source_rho"], 2.0)

    def test_high_recursion_uses_fixed_five(self):
        def capacity(prefix, minimum):
            return 0

        def predict(prefix):
            return {"candidates": [{"asn": str(n), "admitted": True} for n in range(10, 20)]}

        result = mandatory_plan("1", ["2", "3"], capacity, predict)
        self.assertEqual(RECURSION_TOPK, 5)
        self.assertEqual(len(result["prefixes"]), 10)
        self.assertTrue(all(len(prefix) == 3 for prefix in result["prefixes"]))
        self.assertEqual(result["policy"]["next_hop_topk"], 5)
        self.assertTrue(POLICY["source_rho_only"])

    def test_batch_path_generation_leaves_only_training_tasks_for_scheduler(self):
        tasks = pats_training_tasks(["10", "20"])
        self.assertEqual([task.stage for task in tasks], ["train", "annotate", "evaluate", "rescore"] * 2)
        self.assertTrue(all(not task.stage.startswith("pg_") for task in tasks))


if __name__ == "__main__":
    unittest.main()
