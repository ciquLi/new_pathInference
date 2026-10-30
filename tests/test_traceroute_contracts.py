"""Trace labels stay outside training and target-only prediction contracts."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathveil.traceroute.common import load_contract, stable_bgp_split, path_signature
from pathveil.traceroute.predict_web import read_targets
from pathveil.utils import sha256, write_json


class TraceContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_target_inventory_rejects_labels_and_duplicate_source_ip(self):
        path = self.root / "targets.csv"
        path.write_text("target_ip,destination_asn,as_path\n1.2.3.4,2,1->2\n")
        with self.assertRaisesRegex(ValueError, "path labels"):
            read_targets(path, "1")
        path.write_text("target_ip,destination_asn\n1.2.3.4,2\n1.2.3.4,3\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_targets(path, "1")
        path.write_text("target_ip,destination_asn\n1.2.3.4,\n1.2.3.5,2\n")
        targets = read_targets(path, "1")
        self.assertIsNone(targets[0]["destination_asn"])
        self.assertEqual(targets[1]["destination_asn"], "2")

    def test_bgp_validation_split_keeps_all_rows_of_a_pair_together(self):
        path = self.root / "generated.csv"
        rows = [
            ["candidate", "history", "pairs", "scores", pair]
            for pair in ["1->2", "1->2", "1->3", "1->4", "1->5"]
        ]
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        result = stable_bgp_split(path, self.root / "split", validation_fraction=0.5, seed=7)
        train, validation = set(result["train_pairs"]), set(result["validation_pairs"])
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, {"1->2", "1->3", "1->4", "1->5"})
        self.assertEqual(sum(result["row_counts"].values()), 5)
        self.assertIn("no traceroute", result["validation_role"])

    def release_fixture(self):
        pool = self.root / "pool.csv"
        pool.write_text("1->2->3\n1->4\n")
        graph = self.root / "neighbors.csv"
        graph.write_text("center_as,neighbor_as\n1,2\n")
        anchors = self.root / "anchors.csv"
        anchors.write_text("first_hop,second_hop\n2,3\n")
        data = self.root / "data.json"
        write_json(data, {"date": "20261015"})
        release = self.root / "RELEASE.json"
        value = {
            "date": "20261015",
            "sources": ["1"],
            "formal_reference_labels_read": False,
            "training_and_PG_pool": "same_R_s_hash_augmentation_only",
            "files_by_source": {"1": {"path": str(pool), "sha256": sha256(pool), "rows": 2}},
            "selected_neighbors_csv": str(graph),
            "recursive_anchors_csv": str(anchors),
            "outputs_sha256": {str(graph): sha256(graph), str(anchors): sha256(anchors)},
        }
        write_json(release, value)
        return data, release, pool

    def test_release_contract_binds_full_route_set_and_graph(self):
        data, release, pool = self.release_fixture()
        contract = load_contract(data, release, "1")
        self.assertEqual(contract["path_set_sha256"], path_signature([["1", "2", "3"], ["1", "4"]]))
        self.assertEqual(contract["pool_mode"], "augmentation_only")
        pool.write_text("1->9->3\n1->4\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_contract(data, release, "1")

    def test_release_date_and_reference_access_are_rejected(self):
        data, release, pool = self.release_fixture()
        write_json(data, {"date": "20260301"})
        with self.assertRaisesRegex(ValueError, "date"):
            load_contract(data, release, "1")
        write_json(data, {"date": "20261015"})
        value = json.loads(release.read_text())
        value["formal_reference_labels_read"] = True
        write_json(release, value)
        with self.assertRaisesRegex(ValueError, "no-formal-reference"):
            load_contract(data, release, "1")


if __name__ == "__main__":
    unittest.main()
