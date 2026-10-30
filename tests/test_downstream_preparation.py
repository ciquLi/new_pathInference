"""Source derivation is proved by parent observations, never CSV filenames."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pathveil.downstream import common
from pathveil.downstream.runner import audit_derived_source, prepare_downstream
from pathveil.utils import read_json, sha256, write_json


class DownstreamPreparation(unittest.TestCase):
    def test_nonempty_source_csv_can_be_forward_derived_without_source_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            source = train / "10.csv"
            source.write_text("10->20\n10->20->30\n10->40\n")
            merged = train / "merged_paths.txt"
            merged.write_text("1->10->20->30\n2->10->40\n")
            (train / "expand_paths.txt").write_text(source.read_text())
            (root / "split_log.txt").write_text("Source-origin holdout\n")
            data = dict(date="20260501", merged_train=str(merged), train_dir=str(train))
            audit = audit_derived_source(source, "10", data, ["10", "99"], root / "audit.json")
            result = read_json(audit)
            self.assertEqual(result["train_merged_source_origin_rows"], 0)
            self.assertEqual(result["source_csv_rows"], 3)
            self.assertFalse(result["test_paths_read"])
            self.assertEqual(result["decision"], "allow_explicit_audited_derived_source_csv")

    def test_actual_heldout_origin_in_canonical_stream_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            train.mkdir()
            source = train / "10.csv"
            source.write_text("10->20\n")
            merged = train / "merged_paths.txt"
            merged.write_text("10->20\n1->10->20\n")
            (train / "expand_paths.txt").write_text(source.read_text())
            (root / "split_log.txt").write_text("split\n")
            with self.assertRaises(ValueError):
                audit_derived_source(
                    source,
                    "10",
                    dict(date="20260501", merged_train=str(merged), train_dir=str(train)),
                    ["10"],
                    root / "audit.json",
                )

    def test_prepare_copies_same_pool_for_both_consumers_without_launching(self):
        self._assert_fresh_prepare("20260501")

    def test_fresh_march_does_not_force_the_historical_regression_graph(self):
        self._assert_fresh_prepare("20260301")

    def _assert_fresh_prepare(self, date):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = root / "runs"
            train, test, graph = root / "train", root / "test", root / "graph"
            for folder in (train, test, graph):
                folder.mkdir()
            (train / "merged_paths.txt").write_text("1->2\n")
            (test / "10.csv").write_text("10->20\n")
            (graph / "edge_frequency.json").write_text("{}")
            data = dict(
                date=date,
                sources=["10"],
                common9=["10"],
                public_files={},
                graph_dir=str(graph),
                test_dir=str(test),
                train_dir=str(train),
                merged_train=str(train / "merged_paths.txt"),
            )
            data_file = root / "data.json"
            write_json(data_file, data)
            pool = root / "pool.csv"
            pool.write_text("10->20\n")
            pool_json = root / "pool.json"
            write_json(pool_json, {"10-20": [["10", "20"]]})
            neighbors, anchors = root / "neighbors.csv", root / "anchors.csv"
            neighbors.write_text("source,neighbor\n10,20\n")
            anchors.write_text("source,neighbor,anchor\n")
            release = dict(
                status="PASS",
                date=date,
                sources=["10"],
                files_by_source={
                    "10": dict(
                        path=str(pool),
                        sha256=sha256(pool),
                        json_path=str(pool_json),
                        json_sha256=sha256(pool_json),
                        rows=1,
                    )
                },
                selected_neighbors_csv=str(neighbors),
                recursive_anchors_csv=str(anchors),
                outputs_sha256={},
            )
            release_file = root / "release.json"
            write_json(release_file, release)
            with patch.object(common, "RUNS", runs):
                path = prepare_downstream(runs / "trial", data_file, release_file)
            prepared = read_json(path)
            admitted = path.parent / "path_augmentation_output/train/10_augmented_paths.csv"
            self.assertEqual(sha256(admitted), sha256(pool))
            self.assertEqual(prepared["fixed_test_pairs"], 1)
            self.assertTrue(prepared["no_model_training_started"])
            self.assertEqual(prepared["train_extra_data_mode"], "augmentation_only")
            self.assertEqual(prepared["pg_local_knowledge_mode"], "augmentation_only")
            self.assertEqual(prepared["dispatch_replay"], {})
            self.assertFalse((path.parent / "COMPLETED.json").exists())


if __name__ == "__main__":
    unittest.main()
