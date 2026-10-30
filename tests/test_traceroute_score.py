import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathveil.traceroute.score import (
    read_truth,
    pair_inventory_sha256,
    score_predictions,
    FROZEN_TRUTH_IDENTITIES,
)
from pathveil.utils import sha256


class TraceScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.truth = self.root / "truth.csv"
        self.truth.write_text("1->2\n1->3->4\n1->5\n")
        truth, _ = read_truth(self.truth)
        self.identity = dict(
            pairs=3,
            unique_paths=3,
            sha256=sha256(self.truth),
            pair_inventory_sha256=pair_inventory_sha256(truth),
        )
        self.pred = self.root / "pred.jsonl"

    def predictions(self, rows):
        self.pred.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def score(self):
        return score_predictions(self.truth, self.pred, self.root / "score", identity=self.identity)

    def test_missing_pairs_remain_zero_in_full_denominator(self):
        self.predictions(
            [
                dict(
                    source_asn="1",
                    destination_asn="2",
                    predicted_path=["1", "2"],
                    status="anything",
                ),
                dict(
                    source_asn="1",
                    destination_asn="4",
                    predicted_path=["1", "4"],
                    status="predicted",
                ),
            ]
        )
        result = self.score()["metrics"]
        self.assertEqual(result["all_truth_pairs"], 3)
        self.assertEqual(result["missing_prediction_rows"], 1)
        self.assertAlmostEqual(result["pair_jaccard_accuracy"], (1 + 2 / 3) / 3)
        self.assertAlmostEqual(result["pair_jaccard_accuracy_rounded3"], (1 + 0.667) / 3)
        self.assertEqual(result["pairs_matching_any_truth"], 1)

    def test_wrong_endpoints_do_not_gain_overlap_credit(self):
        self.predictions(
            [
                dict(
                    source_asn="1",
                    destination_asn="2",
                    predicted_path=["2", "1"],
                    status="predicted",
                )
            ]
        )
        scores = self.score()["metrics"]
        self.assertEqual(scores["wrong_endpoint_prediction_pairs"], 1)
        self.assertEqual(scores["pair_jaccard_accuracy"], 0)

    def test_duplicate_predictions_and_changed_truth_are_rejected(self):
        row = dict(source_asn="1", destination_asn="2", predicted_path=["1", "2"])
        self.predictions([row, row])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.score()
        self.predictions([row])
        self.truth.write_text("1->2\n1->4\n1->5\n")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            self.score()

    def test_canonical_inventory_is_fixed_at_1890_pairs(self):
        self.assertEqual(FROZEN_TRUTH_IDENTITIES["20261015"]["pairs"], 1890)
        self.assertEqual(FROZEN_TRUTH_IDENTITIES["20261015"]["unique_paths"], 2198)
        self.predictions([])
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            score_predictions(self.truth, self.pred, self.root / "canonical")

    def test_same_as_query_keeps_its_single_as_reference(self):
        self.truth.write_text("1\n")
        truth, rows = read_truth(self.truth)
        self.assertEqual(rows, 1)
        self.assertEqual(truth, {("1", "1"): {("1",)}})


if __name__ == "__main__":
    unittest.main()
