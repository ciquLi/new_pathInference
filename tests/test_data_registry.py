import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathveil.data.registry import load_registry, select_asset
from pathveil.data.preflight import check_date, classify_empty_csv, inspect_asset, inspect_jsonl
from pathveil.data.prepare import (
    prepare_date,
    rebuild_test_aggregates,
    read_double_json,
    rebuild_as_features,
)


class RegistryTests(unittest.TestCase):
    def extracted_input_fixture(self, root):
        """A dated processed corpus with its original RIB archive offline."""
        root = Path(root)
        repo, files = root / "repo", root / "files"
        source = files / "data/aspath/path_by_vantage/20260301"
        graph = source / "RL_Graph"
        graph.mkdir(parents=True)
        for name in [
            "edge_frequency.json",
            "AS_Feature.json",
            "edge_feature.json",
            "ASEmbedding_collectedPath_256.txt",
            "20260301_246_embeddings_collectedPath.emb",
        ]:
            (graph / name).write_text("processed graph fixture\n")
        (source / "AS_frequency.json").write_text("processed frequency fixture\n")
        for split, asn in [("train", "1"), ("test", "9")]:
            directory = source / "path_without_prefix" / split
            directory.mkdir(parents=True)
            (directory / (asn + ".csv")).write_text(asn + "->2->3\n")
            for name in [
                "merged_paths.txt",
                "expand_paths.txt",
                "pathWithFreq.json",
                "pathWithKey.pkl",
            ]:
                (directory / name).write_text("processed aggregate fixture\n")
        public = repo / "data"
        public.mkdir(parents=True)
        raw = []
        for identifier, name, value, kind in [
            ("asrank_asns", "asns.jsonl", {"asn": "1", "date": "2026-03-01"}, "asns"),
            (
                "asrank_asnLinks",
                "links.jsonl",
                {"asn0": {"asn": "1"}, "asn1": {"asn": "2"}, "date": "2026-03-01"},
                "asnLinks",
            ),
            ("relationships", "asrel.txt", "1|2|-1\n", None),
            ("peeringdb", "pdb.json", {"net": {"data": [{}]}}, None),
        ]:
            path = public / name
            path.write_text(value if isinstance(value, str) else json.dumps(value) + "\n")
            asset = {
                "id": identifier,
                "type": "file",
                "candidates": [str(path)],
                "required": True,
                "snapshot_date": "20260301",
            }
            if kind:
                asset["download"] = {"kind": kind, "date": "20260301", "expected_count": 1}
            raw.append(asset)
        for collector in ["riperis", "routeviews"]:
            raw.append(
                {
                    "id": "bgp_" + collector,
                    "type": "directory",
                    "candidates": [str(root / "unmounted_archive/20260301" / collector)],
                    "required": False,
                    "purpose": "archive_only",
                    "snapshot_date": "20260301",
                }
            )
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
            "files_root": str(files),
            "test_sources": ["9"],
            "common9": [],
            "raw_inputs": raw,
            "derived_inputs": [
                {
                    "id": "graph",
                    "type": "file",
                    "candidates": [str(graph / "edge_frequency.json")],
                }
            ],
        }
        registry = repo / "configs/dates/20260301.json"
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps(config))
        return repo, files, source

    def test_registered_ribs_are_explicit_optional_archives(self):
        for date in ["20260201", "20260301", "20260401", "20260501", "20260601", "20261015"]:
            config = load_registry(date)
            for asset in config["raw_inputs"]:
                if not asset["id"].startswith("bgp_"):
                    self.assertTrue(asset["required"])
                    continue
                self.assertFalse(asset["required"])
                self.assertEqual(asset["purpose"], "archive_only")
                self.assertEqual(
                    asset["candidates"][0],
                    "/mnt/56DAEA7BDAEA56B1/pathInference/bgp/"
                    + config["routing_date"]
                    + "/"
                    + asset["id"][4:],
                )
                self.assertTrue(any("/data/raw_data/bgp/" in name for name in asset["candidates"]))

    def test_offline_optional_rib_archive_does_not_block_processed_input_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, files, _ = self.extracted_input_fixture(directory)
            report = check_date("20260301", project_root=repo, files_root=files, deep=True)
            self.assertTrue(report["raw_ready"])
            self.assertTrue(report["ready"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["required_downloads"], [])
            self.assertEqual(
                report["optional_inputs_unavailable"], ["bgp_riperis", "bgp_routeviews"]
            )
            for asset in report["raw_inputs"]:
                if asset["id"].startswith("bgp_"):
                    self.assertFalse(asset["required"])
                    self.assertTrue(asset["optional"])
                    self.assertEqual(asset["purpose"], "archive_only")
                    self.assertEqual(asset["status"], "optional_missing_or_empty")
                    self.assertFalse(asset["ready"])
            manifest = prepare_date(
                "20260301", Path(directory) / "run", project_root=repo, files_root=files, deep=True
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse(any(key.startswith("bgp_") for key in manifest["public_files"]))

    def test_missing_required_asrank_still_blocks_with_optional_ribs_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, files, _ = self.extracted_input_fixture(directory)
            (repo / "data/links.jsonl").unlink()
            report = check_date("20260301", project_root=repo, files_root=files, deep=True)
            self.assertFalse(report["raw_ready"])
            self.assertFalse(report["ready"])
            self.assertEqual(
                report["errors"], ["Missing or invalid required input: asrank_asnLinks"]
            )
            self.assertEqual(report["required_downloads"][0]["kind"], "asnLinks")
            with self.assertRaisesRegex(ValueError, "asrank_asnLinks"):
                prepare_date(
                    "20260301", Path(directory) / "run", project_root=repo, files_root=files
                )

    def test_missing_consumed_graph_still_fails_preparation_with_ribs_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, files, source = self.extracted_input_fixture(directory)
            (source / "RL_Graph/edge_frequency.json").unlink()
            report = check_date("20260301", project_root=repo, files_root=files, deep=True)
            self.assertTrue(report["raw_ready"])
            self.assertFalse(report["derived_available"])
            with self.assertRaisesRegex(ValueError, "edge_frequency.json"):
                prepare_date(
                    "20260301", Path(directory) / "run", project_root=repo, files_root=files
                )

    def test_missing_consumed_training_corpus_still_fails_with_ribs_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, files, source = self.extracted_input_fixture(directory)
            (source / "path_without_prefix/train/merged_paths.txt").unlink()
            with self.assertRaisesRegex(ValueError, "merged_paths.txt"):
                prepare_date(
                    "20260301", Path(directory) / "run", project_root=repo, files_root=files
                )

    def test_resume_rejects_changed_consumed_training_corpus_with_ribs_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, files, source = self.extracted_input_fixture(directory)
            run = Path(directory) / "run"
            prepare_date("20260301", run, project_root=repo, files_root=files)
            (source / "path_without_prefix/train/merged_paths.txt").write_text("1->2->4\n")
            with self.assertRaisesRegex(ValueError, "corpus inputs changed"):
                prepare_date("20260301", run, project_root=repo, files_root=files)

    def test_present_optional_archive_is_never_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "rib.bz2").write_bytes(b"archived RIB fixture")
            asset = {
                "id": "bgp_riperis",
                "type": "directory",
                "candidates": [str(path)],
                "required": False,
                "purpose": "archive_only",
            }
            with patch(
                "pathveil.data.preflight.sha256_file", side_effect=AssertionError("RIB hash")
            ):
                report = inspect_asset(asset, deep=True)
            self.assertEqual(report["status"], "optional_present")
            self.assertEqual(report["validation"], "archive_presence_only")
            self.assertFalse(report["used_by_experiment"])
            self.assertNotIn("sha256", report)
            self.assertNotIn("file_count", report)

    def test_auxiliary_dates_and_frozen_common_cohort(self):
        trace = load_registry("20261015")
        self.assertEqual(trace["routing_date"], "20261015")
        self.assertEqual(
            trace["auxiliary_dates"],
            {"asrank": "20261001", "relationships": "20261001", "peeringdb": "20261014"},
        )
        may, march = load_registry("20260501"), load_registry("20260301")
        self.assertEqual(set(may["common9"]), set(march["common9"]))
        self.assertEqual(set(may["test_sources"]) - set(march["test_sources"]), {"16276"})
        self.assertEqual(set(march["test_sources"]) - set(may["test_sources"]), {"42473"})
        self.assertEqual(len(may["common9"]), 9)

    def test_missing_date_never_falls_back(self):
        with self.assertRaises(ValueError):
            load_registry("0501")
        with self.assertRaises(ValueError):
            load_registry("20260701")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "20260401links.jsonl").write_text("available")
            (root / "20260501links.jsonl").write_text("")
            asset = {"type": "file", "candidates": [str(root / "20260501links.jsonl")]}
            self.assertIsNone(select_asset(asset))

    def test_empty_ipv6_named_csv_needs_positive_parse_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route-views6.csv"
            path.touch()
            self.assertEqual(classify_empty_csv(path)["status"], "empty_unverified")
            source = Path(directory) / "rib.bz2"
            source.write_bytes(b"raw")
            path.with_suffix(".csv.parse.json").write_text(
                json.dumps(
                    {
                        "status": "no_usable_paths",
                        "paths": 0,
                        "parse_errors": 0,
                        "records": 12,
                        "source": str(source),
                        "source_size": source.stat().st_size,
                        "source_mtime_ns": source.stat().st_mtime_ns,
                    }
                )
            )
            self.assertEqual(classify_empty_csv(path)["status"], "valid_empty_after_filtering")
            source.write_bytes(b"new raw RIB")
            self.assertEqual(
                classify_empty_csv(path)["status"], "empty_parse_report_requires_review"
            )

    def test_test_rebuild_does_not_write_through_source_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("source remains untouched\n")
            target = root / "test"
            target.mkdir()
            (target / "123.csv").write_text("123->2->3\n123->2->4\n")
            (target / "merged_paths.txt").symlink_to(source)
            result = rebuild_test_aggregates(target)
            self.assertEqual(source.read_text(), "source remains untouched\n")
            self.assertFalse((target / "merged_paths.txt").is_symlink())
            self.assertEqual(
                set((target / "expand_paths.txt").read_text().splitlines()),
                {"123->2", "123->2->3", "123->2->4", "2->3", "2->4"},
            )
            self.assertEqual(result["unique_expanded_paths"], 5)

    def test_feature_rebuild_uses_registered_month_and_keeps_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "old.json"
            features.write_text(json.dumps(json.dumps({"1": [0] * 10, "2": [0] * 10})))
            edges = root / "edges.json"
            edges.write_text(json.dumps(json.dumps({"1-2": 7})))
            rank = root / "asns.jsonl"
            row = {
                "asn": "1",
                "date": "2026-05-01",
                "rank": 42,
                "asnDegree": dict(total=5, transit=2, provider=1, peer=2, customer=2),
                "country": {"iso": "CN"},
                "organization": {"orgId": "org"},
                "longitude": 1.234,
                "latitude": 4.567,
            }
            rank.write_text(json.dumps(row) + "\n")
            old_edges = edges.read_bytes()
            result = rebuild_as_features(rank, features, edges, root, "20260501")
            self.assertEqual(result["default_feature_nodes"], 1)
            self.assertEqual(
                read_double_json(root / "AS_Feature.json")["1"],
                [42, 5, 2, 1, 2, 2, 1, 1, 1.23, 4.57],
            )
            self.assertEqual(edges.read_bytes(), old_edges)
            with self.assertRaisesRegex(ValueError, "date"):
                rebuild_as_features(rank, features, edges, root, "20260601")


if __name__ == "__main__":
    unittest.main()
