"""Prediction-first neighbor evaluation on tiny original-origin fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pathveil.upstream.contracts import UpstreamError, read, sha, write
from pathveil.upstream import heldout_evaluation as evaluation


class FixtureRuntime:
    def __init__(self, root):
        self.folds = {"10": 0}
        self.config = {"root_selection": {"max_k": 32}}
        self.calls = []
        models, hashes = {}, {}
        for family in evaluation._ROOT_FAMILIES:
            path = root / (family + ".txt")
            path.write_text("fixture full model " + family)
            models[family] = {"full": str(path)}
            hashes[str(path)] = sha(path)
        self.bundle = {
            "date": "20260201",
            "model_fit_date": "20260201",
            "formal_labels_read": False,
            "validation_labels_read": False,
            "test_labels_read": False,
            "training_sources": ["10"],
            "source_to_fold": {"10": 0},
            "models": models,
            "model_files_sha256": hashes,
        }
        self.results = {
            20: {
                "selected_neighbors": [101, 999],
                "root": {
                    "candidates": [101, 102, 999],
                    "rankings": {"frozen": [999, 101, 102]},
                },
            },
            30: {"selected_neighbors": [], "root": {"candidates": []}},
        }

    def _head_key(self, source):
        return "oof" + str(self.folds[str(source)]) if str(source) in self.folds else "full"

    def infer_root(self, source):
        self.calls.append(source)
        result = deepcopy(self.results[source])
        result.setdefault("original_source_model_head", "full")
        result["source_labels_excluded_from_predictive_heads"] = True
        return result


class HeldoutEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.canonical = self.root / "original_merged.txt"
        self.canonical.write_text(
            "10->20->888\n"
            "20->101->900\n"
            "20->101->900\n"
            "20->101->777->900\n"
            "20->102->900\n"
            "20->101->901\n"
            "20->103->902\n"
            "20->104->903\n"
            "20->20->900\n"
            "20->bad\n"
            "20\n"
            "30->201->950\n"
            "30->202->950\n"
            "99->101->903\n"
        )
        self.data = {
            "date": "20260201",
            "sources": ["99"],
            "merged_train": str(self.canonical),
        }
        self.split = {
            "split": {"train": [10], "validation": [20], "test": [30], "formal_test": [99]},
            "source_availability": {"canonical_sha256": sha(self.canonical)},
        }
        self.runtime = FixtureRuntime(self.root)
        self.out = self.root / "evaluation"

    def evaluate(self):
        return evaluation.evaluate_heldout_neighbors(self.data, self.split, self.runtime, self.out)

    def test_all_predictions_are_frozen_before_original_reference_extraction(self):
        original = evaluation._extract_heldout

        def check_order(*args):
            self.assertEqual(self.runtime.calls, [20, 30])
            marker = read(self.out / "PREDICTIONS_COMPLETED.json")
            self.assertTrue(marker["frozen_before_truth_read"])
            self.assertEqual(marker["prediction_sha256"], sha(self.out / "PREDICTIONS.json"))
            return original(*args)

        with mock.patch.object(evaluation, "_extract_heldout", side_effect=check_order):
            summary = read(self.evaluate())
        self.assertTrue(summary["all_predictions_frozen_before_truth"])
        self.assertEqual(set(summary["per_source"]), {"20", "30"})
        self.assertFalse(summary["formal_cohort_included_in_neighbor_evaluation"])
        self.assertFalse(summary["test_metrics_may_select_checkpoint"])

    def test_exact_full_path_dedup_destination_balance_and_unconfirmed_semantics(self):
        summary = read(self.evaluate())
        source = read(summary["per_source"]["20"])
        self.assertEqual(
            source["truth_provenance"]["parser_counts"],
            {"paths": 6, "duplicates": 1, "excluded": 3},
        )
        self.assertEqual(source["truth_provenance"]["extracted_original_source_rows"], 10)
        self.assertEqual(source["counts"]["destinations"], 4)
        metrics = source["metrics"]
        self.assertAlmostEqual(metrics["selected_neighbor_observed_support"], 0.5)
        self.assertAlmostEqual(metrics["all_firsthop_recall"], 0.25)
        self.assertAlmostEqual(metrics["destination_balanced_top3_recall"], 1 / 3)
        self.assertAlmostEqual(metrics["destination_balanced_top10_recall"], 0.25)
        self.assertAlmostEqual(metrics["destination_balanced_top3_mass_recall"], 5 / 11)
        self.assertAlmostEqual(metrics["destination_balanced_selected_mass"], 5 / 12)
        self.assertAlmostEqual(metrics["candidate_observed_neighbor_coverage"], 0.5)
        self.assertAlmostEqual(metrics["candidate_destination_balanced_mass"], 0.5)
        self.assertEqual(
            source["selected_neighbor_evidence"], {"101": "observed", "999": "unconfirmed"}
        )
        self.assertEqual(source["unconfirmed_selected_neighbors"], ["999"])
        self.assertNotIn("888", source["destination_balanced_firsthop_weights"])
        self.assertFalse(source["unobserved_selected_neighbors_are_false_edges"])
        self.assertFalse(list(self.out.glob("heldout_origins_*")))

    def test_empty_selection_is_defined_as_abstention_not_false_support(self):
        summary = read(self.evaluate())
        source = read(summary["per_source"]["30"])
        self.assertIsNone(source["metrics"]["selected_neighbor_observed_support"])
        self.assertEqual(source["metrics"]["all_firsthop_recall"], 0)
        self.assertEqual(
            summary["test"]["macro_source_denominators"]["selected_neighbor_observed_support"], 0
        )
        self.assertIsNone(summary["test"]["macro"]["selected_neighbor_observed_support"])
        self.assertEqual(summary["status"], "PASS")

    def test_candidate_coverage_may_be_unavailable(self):
        self.runtime.results[20]["root"] = {}
        source = read(read(self.evaluate())["per_source"]["20"])
        self.assertIsNone(source["metrics"]["candidate_observed_neighbor_coverage"])
        self.assertIsNone(source["metrics"]["candidate_destination_balanced_mass"])

    def test_runtime_cannot_open_canonical_truth_during_inference(self):
        def leaking(_source):
            return self.canonical.read_text()

        self.runtime.infer_root = leaking
        with self.assertRaisesRegex(UpstreamError, "before all predictions are frozen"):
            self.evaluate()
        self.assertFalse((self.out / "PREDICTIONS.json").exists())

    def test_heldout_oof_prediction_and_overlapping_roles_are_rejected(self):
        self.runtime.results[20]["original_source_model_head"] = "oof0"
        with self.assertRaisesRegex(UpstreamError, "full model"):
            self.evaluate()
        self.runtime.results[20].pop("original_source_model_head")
        self.split["split"]["train"].append(20)
        with self.assertRaisesRegex(UpstreamError, "Source roles overlap"):
            self.evaluate()

    def test_formal_sources_and_fold_contamination_are_rejected(self):
        self.split["split"]["validation"].append(99)
        with self.assertRaisesRegex(UpstreamError, "Source roles overlap"):
            self.evaluate()
        self.split["split"]["validation"].remove(99)
        self.runtime.folds["20"] = 1
        with self.assertRaisesRegex(UpstreamError, "Runtime folds"):
            self.evaluate()

    def test_missing_reference_fails_after_prediction_freeze(self):
        self.canonical.write_text("20->101->900\n")
        self.split["source_availability"]["canonical_sha256"] = sha(self.canonical)
        with self.assertRaisesRegex(UpstreamError, "no original reference: 30"):
            self.evaluate()
        self.assertTrue((self.out / "PREDICTIONS_COMPLETED.json").exists())
        self.assertFalse((self.out / "AUDIT_COMPLETED.json").exists())

    def test_reference_with_only_invalid_paths_fails_explicitly(self):
        self.canonical.write_text("20->101->900\n30->30->950\n30->bad\n")
        self.split["source_availability"]["canonical_sha256"] = sha(self.canonical)
        with self.assertRaisesRegex(UpstreamError, "no valid original reference: 30"):
            self.evaluate()
        self.assertFalse((self.out / "AUDIT_COMPLETED.json").exists())

    def test_mutating_configuration_during_prediction_is_rejected(self):
        original = self.runtime.infer_root

        def mutate(source):
            result = original(source)
            self.runtime.config["root_selection"]["max_k"] += 1
            return result

        self.runtime.infer_root = mutate
        with self.assertRaisesRegex(UpstreamError, "configuration changed during prediction"):
            self.evaluate()
        self.assertFalse((self.out / "PREDICTIONS.json").exists())

    def test_validated_cache_reuses_frozen_predictions_without_new_inference(self):
        result = self.evaluate()
        self.runtime.infer_root = mock.Mock(side_effect=AssertionError("No new prediction"))
        self.assertEqual(self.evaluate(), result)
        self.runtime.infer_root.assert_not_called()
        marker = read(self.out / "AUDIT_COMPLETED.json")
        self.assertEqual(marker["summary_sha256"], sha(result))

    def test_prediction_or_summary_tampering_does_not_silently_regenerate(self):
        self.evaluate()
        prediction_path = self.out / "PREDICTIONS.json"
        previous = prediction_path.read_text()
        value = json.loads(previous)
        value["predictions"]["20"]["selected_neighbors"] = ["102"]
        write(prediction_path, value)
        with self.assertRaisesRegex(UpstreamError, "Frozen held-out predictions changed"):
            self.evaluate()
        prediction_path.write_text(previous)
        (self.out / "summary.json").write_text("{}")
        with self.assertRaisesRegex(UpstreamError, "Frozen input changed"):
            self.evaluate()

    def test_canonical_split_and_model_mutations_invalidate_cache(self):
        self.evaluate()
        original = self.canonical.read_text()
        self.canonical.write_text(original + "20->105->960\n")
        with self.assertRaisesRegex(UpstreamError, "canonical original stream changed"):
            self.evaluate()
        self.canonical.write_text(original)
        self.split["evaluation_note"] = "new split identity"
        with self.assertRaisesRegex(UpstreamError, "prediction cache identity changed"):
            self.evaluate()
        self.split.pop("evaluation_note")
        filename = next(iter(self.runtime.bundle["model_files_sha256"]))
        Path(filename).write_text("changed model")
        with self.assertRaisesRegex(UpstreamError, "Frozen input changed"):
            self.evaluate()

    def test_model_date_and_unregistered_canonical_hash_are_rejected(self):
        self.runtime.bundle["model_fit_date"] = "20260301"
        with self.assertRaisesRegex(UpstreamError, "model date differs"):
            self.evaluate()
        self.runtime.bundle["model_fit_date"] = "20260201"
        self.split["source_availability"].clear()
        with self.assertRaisesRegex(UpstreamError, "canonical SHA256 before prediction"):
            self.evaluate()

    def test_full_source_protocol_canonical_registration_and_conflicts(self):
        self.split["canonical_evidence_input"] = {
            "path": str(self.canonical),
            "sha256": sha(self.canonical),
        }
        self.split.pop("source_availability")
        self.assertEqual(read(self.evaluate())["status"], "PASS")
        self.split["source_availability"] = {"canonical_sha256": "0" * 64}
        with self.assertRaisesRegex(UpstreamError, "hashes registered.*disagree"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
