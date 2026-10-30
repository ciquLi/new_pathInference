"""Exercise the exFAT-compatible preparation contract with isolated fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathveil.data.prepare import prepare_date, resolve_read_path


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo, self.files, self.run = [self.root / name for name in ["repo", "files", "run"]]
        self.source = self.files / "data/aspath/path_by_vantage/20260301"
        self.graph = self.source / "RL_Graph"
        self.graph.mkdir(parents=True)
        for name, value in {
            "edge_frequency.json": '"{\\"1-2\\":1}"',
            "AS_Feature.json": "legacy AS feature bytes",
            "edge_feature.json": "legacy edge feature bytes",
            "ASEmbedding_collectedPath_256.txt": "legacy combined embedding bytes",
            "20260301_246_embeddings_collectedPath.emb": "1 1\n1 .25\n",
            "vantage_dst.json": "large read-only cache stand-in",
        }.items():
            (self.graph / name).write_text(value)
        (self.source / "AS_frequency.json").write_text('"{\\"1\\":1}"')
        for split in ["train", "test"]:
            path = self.source / "path_without_prefix" / split
            path.mkdir(parents=True)
            (path / ("1.csv" if split == "train" else "9.csv")).write_text(
                "1->2\n" if split == "train" else "9->2->3\n"
            )
            for name in [
                "merged_paths.txt",
                "expand_paths.txt",
                "pathWithFreq.json",
                "pathWithKey.pkl",
            ]:
                (path / name).write_text("source aggregate bytes\n")
        public = self.repo / "data"
        public.mkdir(parents=True)
        for name, value in {
            "asns.jsonl": '{"asn":"1","date":"2026-03-01"}\n',
            "links.jsonl": '{"asn0":{"asn":"1"},"asn1":{"asn":"2"},"date":"2026-03-01"}\n',
            "asrel.txt": "1|2|-1\n",
            "pdb.json": '{"net":{"data":[{}]}}',
        }.items():
            (public / name).write_text(value)
        raw = [
            {
                "id": identifier,
                "type": "file",
                "candidates": [str(public / name)],
                "snapshot_date": "20260301",
            }
            for identifier, name in [
                ("asrank_asns", "asns.jsonl"),
                ("asrank_asnLinks", "links.jsonl"),
                ("relationships", "asrel.txt"),
                ("peeringdb", "pdb.json"),
            ]
        ]
        config = {
            "schema_version": 1,
            "date": "20260301",
            "routing_date": "20260301",
            "mode": "bgp",
            "auxiliary_dates": {
                "asrank": "20260301",
                "relationships": "20260301",
                "peeringdb": "20260301",
            },
            "files_root": str(self.files),
            "test_sources": ["9"],
            "common9": [],
            "raw_inputs": raw,
            "derived_inputs": [],
        }
        registry = self.repo / "configs/dates"
        registry.mkdir(parents=True)
        (registry / "20260301.json").write_text(json.dumps(config))

    def prepare(self):
        return prepare_date("20260301", self.run, project_root=self.repo, files_root=self.files)

    def test_external_inputs_stay_external_and_mutable_features_are_copied(self):
        before = {str(path): path.read_bytes() for path in self.files.rglob("*") if path.is_file()}
        manifest = self.prepare()
        self.assertEqual(
            manifest["merged_train"],
            str(self.source / "path_without_prefix/train/merged_paths.txt"),
        )
        local_graph = Path(manifest["graph_dir"])
        self.assertEqual(
            (local_graph / "AS_Feature.json").read_bytes(),
            (self.graph / "AS_Feature.json").read_bytes(),
        )
        self.assertFalse(any(path.is_symlink() for path in self.run.rglob("*")))
        self.assertFalse((local_graph / "vantage_dst.json").exists())
        self.assertEqual(
            resolve_read_path(manifest, local_graph / "vantage_dst.json"),
            self.graph / "vantage_dst.json",
        )
        self.assertEqual(
            resolve_read_path(manifest, local_graph / "AS_Feature.json"),
            local_graph / "AS_Feature.json",
        )
        self.assertEqual(
            before,
            {str(path): path.read_bytes() for path in self.files.rglob("*") if path.is_file()},
        )
        self.assertEqual(self.prepare(), manifest)

    def test_resume_rejects_changed_feature_or_registry(self):
        manifest = self.prepare()
        (Path(manifest["graph_dir"]) / "AS_Feature.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "derived output changed"):
            self.prepare()

    def test_missing_test_aggregates_rebuilt_without_copying_csvs(self):
        original = self.source / "path_without_prefix/test/merged_paths.txt"
        original.unlink()
        manifest = self.prepare()
        self.assertIn("test_aggregates", manifest["repairs"])
        replacement = resolve_read_path(manifest, original)
        self.assertEqual(replacement.read_text(), "9->2->3\n")
        self.assertFalse(original.exists())
        self.assertFalse((replacement.parent / "9.csv").exists())

    def test_copy_budget_failure_leaves_external_data_unchanged(self):
        before = (self.graph / "AS_Feature.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "budget"):
            prepare_date(
                "20260301",
                self.run,
                project_root=self.repo,
                files_root=self.files,
                copy_budget_bytes=1,
            )
        self.assertEqual((self.graph / "AS_Feature.json").read_bytes(), before)
        self.assertFalse((self.run / "data/data_manifest.json").exists())

    def trace_fixture(self):
        registry = self.repo / "configs/dates/20260301.json"
        config = json.loads(registry.read_text())
        config.update(date="20261015", routing_date="20261015", mode="traceroute")
        trace_source = self.source.with_name("20261015")
        self.source.rename(trace_source)
        old_embedding = trace_source / "RL_Graph/20260301_246_embeddings_collectedPath.emb"
        old_embedding.rename(old_embedding.with_name("20261015_246_embeddings_collectedPath.emb"))
        test_dir = trace_source / "path_without_prefix/test"
        for path in test_dir.iterdir():
            path.unlink()
        holdout_path = trace_source / "path_without_prefix/source_holdout.json"
        holdout_path.write_text(
            json.dumps(
                {
                    "requested_sources": ["9"],
                    "observed_source_files": [],
                    "absent_sources": ["9"],
                    "training_source_paths": 0,
                }
            )
        )
        truth = self.repo / "data/traceroute/20261015/measurement/as_paths_path_inference.csv"
        truth.parent.mkdir(parents=True)
        truth.write_text("9->8->7\n")
        config["raw_inputs"].append(
            {
                "id": "trace_truth",
                "type": "file",
                "candidates": [str(truth)],
                "snapshot_date": "20261015",
            }
        )
        (registry.parent / "20261015.json").write_text(json.dumps(config))
        return trace_source, holdout_path, truth

    def test_trace_accepts_registered_absent_bgp_source_and_independent_measurements(self):
        source, holdout, truth = self.trace_fixture()
        manifest = prepare_date("20261015", self.run, project_root=self.repo, files_root=self.files)
        self.assertEqual(manifest["sources"], ["9"])
        self.assertEqual(list(Path(manifest["test_dir"]).glob("*.csv")), [])
        self.assertNotIn("test_aggregates", manifest["repairs"])
        self.assertEqual(
            manifest["source_holdout"]["reference_kind"], "independent_traceroute_measurements"
        )
        self.assertFalse(
            (
                Path(manifest["data_root"])
                / "data/aspath/path_by_vantage/20261015/path_without_prefix/test"
            ).exists()
        )
        self.assertEqual(truth.read_text(), "9->8->7\n")
        holdout.write_text(json.dumps({"requested_sources": [], "training_source_paths": 0}))
        with self.assertRaisesRegex(ValueError, "holdout manifest changed"):
            prepare_date("20261015", self.run, project_root=self.repo, files_root=self.files)

    def test_trace_requires_positive_source_holdout_registration(self):
        _, holdout, _ = self.trace_fixture()
        holdout.write_text(json.dumps({"requested_sources": [], "training_source_paths": 0}))
        with self.assertRaisesRegex(ValueError, "source holdout is incomplete"):
            prepare_date("20261015", self.run, project_root=self.repo, files_root=self.files)

    def test_bgp_still_rejects_a_missing_formal_source_csv(self):
        (self.source / "path_without_prefix/test/9.csv").unlink()
        with self.assertRaisesRegex(
            ValueError, "Required processed routing input missing or empty"
        ):
            self.prepare()

    def test_bgp_still_rejects_an_empty_formal_source_csv(self):
        (self.source / "path_without_prefix/test/9.csv").write_text("")
        with self.assertRaisesRegex(
            ValueError, "Required processed routing input missing or empty"
        ):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
