"""Date isolation, frozen numerics, branch admission and evaluation boundaries."""
from contextlib import redirect_stdout
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pathveil.experiment import assets, augmentation, report, runner, train, utilities, worker
from pathveil.downstream.common import StageGuard
from pathveil.experiment.compile_queries import LabelBlindStageGuard
from pathveil.experiment.contracts import (
    freeze,
    normalize_pool,
    quality_selection,
    require,
    signature,
)
from pathveil.experiment.schedule import downstream_tasks
from pathveil.utils import read_json, sha256, write_json
from pathveil.prefix_tree.provenance import symbol_sha256


class FakeTopology:
    def __init__(self, shapes=None):
        self.shapes = shapes or {}

    def source_features(self, source):
        b1, second = self.shapes[str(source)]
        return dict(public_b1=b1, public_l2_distinct_asns=second)

    def neighbors(self, source):
        return ["410", "420"]

    def public_role(self, a, b):
        return "unknown"


class FrozenNumerics(unittest.TestCase):
    def test_extracted_numerical_symbols_keep_their_verified_ast(self):
        root = Path(__file__).resolve().parents[1] / "src/pathveil/prefix_tree"
        provenance = read_json(root / "COMPUTATION_PROVENANCE.json")["modules"]
        for name, record in provenance.items():
            module = ast.parse((root / (name + ".py")).read_text())
            definitions = {
                n.name: n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            for symbol, expected in record["numerical_symbols_ast_sha256"].items():
                with self.subTest(module=name, symbol=symbol):
                    actual = symbol_sha256(definitions[symbol])
                    self.assertEqual(actual, expected)
            assignments = {
                target.id: n
                for n in module.body
                if isinstance(n, ast.Assign)
                for target in n.targets
                if isinstance(target, ast.Name)
            }
            for constant, expected in record["constant_assignments_ast_sha256"].items():
                self.assertEqual(symbol_sha256(assignments[constant]), expected, (name, constant))

    def test_pool_normalization_retains_order_and_maximum_duplicate_score(self):
        paths, values, _ = normalize_pool(
            "400",
            [("400", "410", "500"), ("400", "420", "600"), ("400", "410", "500")],
            [0.2, 0.8, 0.6],
        )
        self.assertEqual(paths, [("400", "410", "500"), ("400", "420", "600")])
        np.testing.assert_array_equal(values, [0.6, 0.8])
        for scores in ([float("nan")], [-0.1], [1.01]):
            with self.assertRaises(ValueError):
                normalize_pool("400", [("400", "410", "500")], scores)

    def test_inclusive_gate_and_quantity_floor_have_no_upper_cap(self):
        paths = [("400", "410", str(i)) for i in range(500, 510)]
        selected, rule = quality_selection(paths, [1.0] * 10, 0.75, 2)
        self.assertEqual(len(selected), 10)
        self.assertFalse(rule["threshold_relaxed"])
        selected, rule = quality_selection(paths[:3], [1, 0.75, 0.4], 0.9, 2)
        self.assertEqual(selected, paths[:2])
        self.assertTrue(rule["threshold_relaxed"])


class RhoPartition(unittest.TestCase):
    def test_threshold_equal_one_and_unknown_are_explicit(self):
        topo = FakeTopology({"400": (2, 4), "401": (2, 5), "402": (0, 0)})
        features, neighbors, origins = runner.structure(topo, ["400", "401", "402"])
        self.assertEqual(features["400"]["rho"], 1)
        self.assertEqual(features["401"]["rho"], 1.25)
        self.assertIsNone(features["402"]["rho"])
        self.assertEqual(neighbors["400"], [])
        self.assertEqual(neighbors["401"], ["410", "420"])
        self.assertEqual(set(origins), {"400", "410", "420"})

    def test_plan_only_cannot_start_upstream_training(self):
        args = SimpleNamespace(
            method_config=None, check_only=False, plan_only=True, deep_check=False
        )
        registry = {"date": "20300101"}
        with patch.object(runner, "load_configuration", return_value={}), patch.object(
            runner, "plan", return_value={"status": "INPUTS_READY"}
        ), patch.object(
            assets, "obtain_upstream", side_effect=AssertionError("unexpected fitting")
        ), redirect_stdout(
            io.StringIO()
        ):
            self.assertEqual(runner.execute(args, registry), 0)


class AdmissionFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.protocol = dict(
            date="20300101",
            public_features={"400": {"rho": 2}},
            neighbors={"400": ["410", "420"]},
            method={
                "augmentation": dict(
                    neighbor_threshold=0.9, source_threshold=0.75, low_minimum_paths=2
                )
            },
        )
        write_json(self.run / "PROTOCOL.json", self.protocol)

    def pool(self, source, rows):
        directory = self.run / "score_pools" / ("as" + source)
        file = directory / "SCORED_PATHS.json"
        freeze(file, rows)
        freeze(
            directory / "SCORED.json",
            dict(
                date="20300101",
                protocol_sha256=sha256(self.run / "PROTOCOL.json"),
                outputs_sha256={str(file): sha256(file)},
            ),
        )

    def row(self, path, score):
        return dict(path=list(path), prefix=list(path[:2]), raw_score=score, relative_score=score)

    def test_neighbor_first_keeps_threshold_ties_unknowns_and_rejects_source_loops(self):
        self.pool("410", [self.row(("410", "500", "600"), 0.9), self.row(("410", "400", "601"), 1)])
        self.pool(
            "420", [self.row(("420", "510", "602"), 0.89), self.row(("420", "511", "603"), 0.95)]
        )
        result = augmentation.build_library(self.run, "400", FakeTopology())
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["filters"], {"loop_including_source": 1})
        self.assertIsNone(result["selection"]["quantity_cap"])
        self.assertFalse(result["selection"]["source_rescoring"])
        self.assertEqual(result["date"], "20300101")
        records = read_json(self.run / "augmentation/as400/PATH_RECORDS.json")
        self.assertTrue(all(len(r["prefix"]) == 3 for r in records))
        self.assertEqual(result["path_set_sha256"], signature(r["path"] for r in records))

    def test_low_rho_uses_its_own_model_pool(self):
        self.protocol["public_features"]["400"]["rho"] = 0.5
        write_json(self.run / "PROTOCOL.json", self.protocol)
        self.pool(
            "400",
            [
                self.row(("400", "410", "600"), 1),
                self.row(("400", "420", "601"), 0.75),
                self.row(("400", "430", "602"), 0.5),
            ],
        )
        result = augmentation.build_library(self.run, "400", FakeTopology())
        self.assertEqual(result["strategy"], "direct_model_frontier")
        self.assertEqual(result["rows"], 2)

    def test_unknown_structure_is_not_silently_treated_as_low(self):
        self.protocol["public_features"]["400"]["rho"] = None
        write_json(self.run / "PROTOCOL.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "Unknown public structure"):
            augmentation.build_library(self.run, "400", FakeTopology())

    def test_another_date_score_pool_is_rejected(self):
        self.pool("410", [self.row(("410", "500", "600"), 1)])
        marker = self.run / "score_pools/as410/SCORED.json"
        record = read_json(marker)
        record["date"] = "20260201"
        write_json(marker, record)
        with self.assertRaisesRegex(ValueError, "date/method"):
            augmentation.build_library(self.run, "400", FakeTopology())


class SourceView(unittest.TestCase):
    def test_upstream_cache_rejects_another_date_before_model_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "upstream/IDENTITY.json", {"date": "20260201", "sources": ["400"]})
            write_json(root / "upstream/path_model/MODEL.json", {"date": "20260201"})
            write_json(root / "upstream/NEIGHBOR_BUNDLE.json", {"date": "20260201"})
            with self.assertRaisesRegex(ValueError, "Cross-date upstream cache"):
                assets.verify_upstream(root, {"date": "20260301", "sources": ["400"]}, {})

    def test_upstream_cache_rejects_another_heldout_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "upstream/IDENTITY.json", {"date": "20260201", "sources": ["401"]})
            write_json(root / "upstream/path_model/MODEL.json", {"date": "20260201"})
            write_json(root / "upstream/NEIGHBOR_BUNDLE.json", {"date": "20260201"})
            with self.assertRaisesRegex(ValueError, "another explicitly held-out cohort"):
                assets.verify_upstream(root, {"date": "20260201", "sources": ["400"]}, {})

    def fixture(self, root, canonical="10->42->43\n"):
        original = root / "original"
        train_dir = original / "paths/train"
        test_dir = original / "paths/test"
        train_dir.mkdir(parents=True)
        test_dir.mkdir()
        (train_dir / "42.csv").write_text("")
        (train_dir / "merged_paths.txt").write_text(canonical)
        (train_dir / "expand_paths.txt").write_text("42->43\n")
        (train_dir.parent / "split_log.txt").write_text("fixture")
        reference = test_dir / "42.csv"
        reference.write_text("42->45\n")
        return dict(
            date="20300101",
            sources=["42"],
            source_data_root=str(original),
            train_dir=str(train_dir),
            test_dir=str(test_dir),
            merged_train=str(train_dir / "merged_paths.txt"),
            corpus_input_sha256={str(reference): sha256(reference)},
            read_path_map={},
            path_prefix_map={},
        )

    def test_two_AS_restore_uses_independent_parents_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            manifest = assets.downstream_view(data, root / "run")
            projected = Path(read_json(manifest)["train_dir"]) / "42.csv"
            self.assertEqual(projected.read_text(), "42->43\n")
            self.assertEqual((Path(data["train_dir"]) / "42.csv").read_text(), "")
            self.assertEqual(assets.downstream_view(data, root / "run"), manifest)
            (Path(data["train_dir"]) / "merged_paths.txt").write_text("10->42->44\n")
            with self.assertRaisesRegex(ValueError, "Frozen admission input changed"):
                assets.downstream_view(data, root / "run")

    def test_formal_origin_cannot_supply_training_derivatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root, "42->43\n")
            with self.assertRaisesRegex(ValueError, "Formal source-origin"):
                assets.downstream_view(data, root / "run")

    def test_missing_source_csv_restores_multihop_only_from_training_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root, "10->42->43->44\n")
            train_dir = Path(data["train_dir"])
            (train_dir / "42.csv").unlink()
            (train_dir / "expand_paths.txt").write_text("42->43\n42->43->44\n")
            reference = Path(data["test_dir"]) / "42.csv"
            original_reference = reference.read_bytes()
            manifest = assets.downstream_view(data, root / "run")
            result = read_json(manifest)
            projected = Path(result["train_dir"]) / "42.csv"
            self.assertEqual(projected.read_text(), "42->43\n42->43->44\n")
            self.assertNotIn("42->45", projected.read_text())
            self.assertFalse((train_dir / "42.csv").exists())
            self.assertEqual(reference.read_bytes(), original_reference)
            receipt = read_json(manifest.parent / "VIEW_COMPLETED.json")
            difference = receipt["differences"]["42"]
            self.assertFalse(difference["original_csv_present"])
            self.assertEqual(difference["restored_two_AS"], 1)
            self.assertEqual(difference["restored_multihop_paths"], 1)
            self.assertEqual(difference["restored_by_AS_count"], {"2": 1, "3": 1})

    def test_existing_source_csv_can_be_completed_from_proven_multihop_fragments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root, "10->42->43->44\n")
            train_dir = Path(data["train_dir"])
            (train_dir / "42.csv").write_text("42->43\n")
            (train_dir / "expand_paths.txt").write_text("42->43\n42->43->44\n")
            manifest = assets.downstream_view(data, root / "run")
            self.assertEqual(
                (Path(read_json(manifest)["train_dir"]) / "42.csv").read_text(),
                "42->43\n42->43->44\n",
            )
            self.assertEqual((train_dir / "42.csv").read_text(), "42->43\n")

    def test_multihop_expansion_without_independent_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            (Path(data["train_dir"]) / "expand_paths.txt").write_text("42->43\n42->43->44\n")
            with self.assertRaisesRegex(ValueError, "lacks exact parent provenance"):
                assets.downstream_view(data, root / "run")

    def test_source_csv_without_independent_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.fixture(root)
            (Path(data["train_dir"]) / "42.csv").write_text("42->43->44\n")
            with self.assertRaisesRegex(ValueError, "unsupported derivatives"):
                assets.downstream_view(data, root / "run")


class SelectionAndLabels(unittest.TestCase):
    def test_one_normal_history_checkpoint_for_both_rho_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "METHOD_FROZEN.json", {"training_objective": "normal_history_huber"})
            info = {"root": root, "source": "400", "identity": {"high_rho": False}}
            verified = dict(
                validation={"criterion_value": 0.8},
                model="p0",
                model_sha256="p0",
                training_manifest="p0",
                training_manifest_sha256="p0",
            )
            with patch.object(train, "verify_training", return_value=verified):
                selected = train.selection_value(info)
            self.assertEqual(selected["chosen_variant"], "p0")
            self.assertEqual(set(selected["candidates"]), {"p0"})
            self.assertFalse(selected["final_test_used_for_selection"])
            for high in [True, False]:
                info["identity"]["high_rho"] = high
                self.assertEqual(train.configured_variants(info), ["p0"])

    def test_unregistered_loss_cannot_change_training_implicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "METHOD_FROZEN.json", {"training_objective": "ce"})
            with self.assertRaisesRegex(ValueError, "Only the frozen normal-history Huber"):
                train.configured_variants({"root": root})

    def test_test_compiler_cannot_open_reference_even_if_legacy_guard_relaxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "private_route_labels.csv"
            reference.write_text("42->43\n")
            guard = LabelBlindStageGuard(
                root, {"test_identity": {"42": {"reference_file": str(reference)}}}
            )
            guard.allow_raw_test = True
            with self.assertRaises(PermissionError):
                guard.check("open", (str(reference), "r", 0))
            with self.assertRaises(PermissionError):
                guard.allow_evaluation(reference, sha256(reference))


class SchedulingAndPublishing(unittest.TestCase):
    def test_PG_receipts_stay_inside_their_source_write_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            child = run / "T-PG-B/pg_inputs/as400"
            guard = StageGuard(child)
            for stage in ("pg_train", "pg_test"):
                file = worker.receipt_path(run, stage, "400")
                guard.check("os.mkdir", (str(file.parent), 0o755, -1))
                self.assertTrue(child in file.parents)

    def test_per_source_stages_are_serial_but_sources_independent(self):
        tasks = downstream_tasks(["400", "401"])
        by = {t.key: t for t in tasks}
        self.assertEqual(len(tasks), 12)
        self.assertEqual(by["pg_train:400"].dependencies, ())
        self.assertEqual(by["pg_train:401"].dependencies, ())
        self.assertEqual(by["pg_test:400"].dependencies, ("pg_train:400",))
        self.assertEqual(by["train:400"].dependencies, ("pg_test:400",))

    def test_partial_results_do_not_count_unfinished_AS_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "PROTOCOL.json",
                dict(
                    date="20300101",
                    sources=["400", "401"],
                    public_features={"400": {"rho": 2}, "401": {"rho": 0.1}},
                ),
            )

            def row(source, count, score):
                file = root / (source + ".csv")
                file.write_text(source)
                return dict(
                    source_asn=source,
                    rho=2,
                    admitted_paths=1000,
                    query_count=count,
                    jaccard_accuracy=score,
                    total_jaccard_reward=count * score,
                    ordered_exact_accuracy=0.2,
                    coverage=1,
                    prediction_file=str(file),
                    prediction_sha256=sha256(file),
                )

            directory = root / "T-PG-B/independent_evaluation"
            write_json(directory / "source_400.json", row("400", 10, 0.8))
            value = report.publish(root)
            self.assertEqual(value["status"], "RUNNING")
            self.assertEqual(value["macro_jaccard"], 0.8)
            write_json(directory / "source_401.json", row("401", 30, 0.9))
            value = report.publish(root)
            self.assertEqual(value["status"], "COMPLETED")
            self.assertAlmostEqual(value["macro_jaccard"], 0.85)
            self.assertAlmostEqual(value["query_weighted_jaccard"], 0.875)
            (root / "400.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "prediction bytes changed"):
                report.publish(root)


if __name__ == "__main__":
    unittest.main()
