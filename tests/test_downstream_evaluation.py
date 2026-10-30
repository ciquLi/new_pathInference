"""Independent scoring never changes the query denominator or trusts CSV scores."""
import csv
from pathlib import Path
import tempfile
import unittest

from pathveil.downstream.evaluation import reference_identity, score_source


class IndependentEvaluation(unittest.TestCase):
    def test_missing_queries_are_zero_and_ordered_exact_is_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth = root / "truth.csv"
            truth.write_text("10->11->20\n10->30\n10->11->20\n")
            prediction = root / "predictions.csv"
            prediction.write_text("pair,predicted_path\n10->20,10->12->20\n")
            identity = reference_identity(truth, "10")
            result = score_source(truth, prediction, "10", identity)
            self.assertEqual(result["query_count"], 2)
            self.assertEqual(result["missing_predictions"], 1)
            self.assertEqual(result["coverage"], 0.5)
            self.assertEqual(result["jaccard_accuracy"], 0.25)
            self.assertEqual(result["ordered_exact_accuracy"], 0.0)

    def test_set_jaccard_can_be_perfect_when_route_order_is_wrong(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth = root / "truth.csv"
            truth.write_text("10->11->12->20\n")
            prediction = root / "prediction.csv"
            prediction.write_text("pair,predicted_path\n10->20,10->12->11->20\n")
            result = score_source(truth, prediction, "10")
            self.assertEqual(result["jaccard_accuracy"], 1.0)
            self.assertEqual(result["ordered_exact_accuracy"], 0.0)

    def test_duplicate_unregistered_and_fabricated_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth = root / "truth.csv"
            truth.write_text("10->20\n")
            prediction = root / "prediction.csv"
            for content in [
                "pair,predicted_path\n10->20,10->20\n10->20,10->20\n",
                "pair,predicted_path\n10->30,10->30\n",
                "pair,predicted_path,jaccard\n10->20,10->20,0.2\n",
            ]:
                prediction.write_text(content)
                with self.assertRaises(ValueError):
                    score_source(truth, prediction, "10")


if __name__ == "__main__":
    unittest.main()
