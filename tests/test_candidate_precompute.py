"""Parallel PG publication preserves frozen inputs, dispatch and resume proofs."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import precompute_downstream_candidates as candidates
import run_registered_experiment as entry
from pathveil.downstream import runner
from pathveil.utils import read_json, write_json, sha256


class CandidatePrecompute(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "runs/20260301/unified255/fixture"
        self.run.mkdir(parents=True)
        self.prepared = dict(
            date="20260301",
            sources=["1", "2"],
            prepared_files={},
            admitted_pool_manifest_sha256="pool",
            derived_source_audits={},
            configuration=dict(seed=1, hash_seed=1),
        )
        for source in self.prepared["sources"]:
            for suffix in (".csv", ".json"):
                p = (
                    self.run
                    / "path_augmentation_output/train"
                    / (source + "_augmented_paths" + suffix)
                )
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(source + "->9\n" if suffix == ".csv" else "{}")
                self.prepared["prepared_files"][str(p)] = sha256(p)
        for name in (
            "reviewed_selected_neighbors.csv",
            "recursive_secondhop_anchors.csv",
        ):
            p = self.run / "neighbor_selection" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("source,neighbor\n1,9\n")
            self.prepared["prepared_files"][str(p)] = sha256(p)
        self.manifest = self.run / "data.json"
        write_json(
            self.manifest,
            dict(
                source_data_root="/original",
                graph_dir="/graph",
                public_files=dict(asrank_links="/asrank"),
            ),
        )
        self.prepared["data_manifest"] = str(self.manifest)
        write_json(self.run / "PREPARED.json", self.prepared)

    def produce(self, source="1", stage="test_inputs"):
        subrun = candidates.isolated_run(self.run, source, self.prepared)
        output, marker = candidates.stage_paths(subrun, source, stage)
        output.parent.mkdir(parents=True)
        output.write_text("unchanged PG candidate bytes\n")
        receipt = dict(
            source=source,
            mode=stage,
            output=str(output),
            sha256=sha256(output),
            admitted_pool_manifest_sha256="pool",
            dispatch_version=1,
            date="20260301",
            graph_delta_edges=[["1", "9", 1]],
            direct_pairs=["1->9"],
            graph_sha256="graph",
        )
        write_json(marker, receipt)
        # The original PG's extra-training report uses its meta directory.
        report = subrun / (
            "pathgen_results/PG-test-result.csv"
            if stage == "test_inputs"
            else "pathgen_meta_results/PG-train-result.csv"
        )
        report.parent.mkdir(parents=True)
        report.write_text("vantage,coverage,time_cost\n" + source + ",1.0,123\n")
        write_json(
            subrun / "control/consumer_audit" / (stage + "_" + source + "_local.json"),
            dict(same_admitted_pool_verified=True),
        )
        return subrun, output, receipt

    def test_as_workers_have_distinct_caches_and_identical_frozen_inputs(self):
        first = candidates.isolated_run(self.run, "1", self.prepared)
        second = candidates.isolated_run(self.run, "2", self.prepared)
        self.assertNotEqual(first / "runtime_cache", second / "runtime_cache")
        for subrun, source in ((first, "1"), (second, "2")):
            snapshot = read_json(subrun / "PREPARED.json")
            self.assertEqual(snapshot["data_manifest"], self.prepared["data_manifest"])
            self.assertEqual(
                snapshot["parent_prepared_sha256"], sha256(self.run / "PREPARED.json")
            )
            for suffix in (".csv", ".json"):
                p = (
                    subrun
                    / "path_augmentation_output/train"
                    / (source + "_augmented_paths" + suffix)
                )
                self.assertEqual(
                    sha256(p),
                    self.prepared["prepared_files"][
                        str(self.run / p.relative_to(subrun))
                    ],
                )

    def test_changed_isolated_input_is_rejected(self):
        subrun = candidates.isolated_run(self.run, "1", self.prepared)
        (subrun / "path_augmentation_output/train/1_augmented_paths.csv").write_text(
            "different"
        )
        with self.assertRaisesRegex(ValueError, "input changed"):
            candidates.isolated_run(self.run, "1", self.prepared)

    def test_command_matches_frozen_runner_contract(self):
        class StopAfterCandidates(Exception):
            pass

        captured = []

        def execute(run, name, command, *args):
            if name.startswith("gmpi_"):
                raise StopAfterCandidates()
            captured.append((name, command))

        with patch.object(
            runner, "prepare_downstream", return_value=self.run / "PREPARED.json"
        ), patch.object(runner, "_execute", side_effect=execute):
            with self.assertRaises(StopAfterCandidates):
                runner.run_downstream(
                    self.run, self.manifest, self.run / "release.json", device="cuda"
                )
        self.assertEqual(
            captured,
            [
                (stage + "_1", candidates.command(self.run, "1", stage, self.prepared))
                for stage in candidates.STAGES
            ],
        )

    def test_publish_preserves_dispatch_and_is_accepted_by_original_cache(self):
        subrun, output, receipt = self.produce()
        command = candidates.command(subrun, "1", "test_inputs", self.prepared)
        candidates.publish(self.run, subrun, "1", "test_inputs", self.prepared, command)
        target, marker = candidates.stage_paths(self.run, "1", "test_inputs")
        self.assertFalse(output.exists())
        self.assertEqual(sha256(target), receipt["sha256"])
        published = read_json(marker)
        self.assertEqual(published["graph_delta_edges"], receipt["graph_delta_edges"])
        self.assertEqual(published["direct_pairs"], receipt["direct_pairs"])
        proof = read_json(published["parallel_producer"])
        self.assertEqual(proof["actual_command"], command)
        self.assertTrue(proof["consumer_audits_sha256"])
        self.assertTrue(candidates.cached(self.run, "1", "test_inputs", self.prepared))
        # Re-publication recovers the moved output and retains one report row.
        candidates.publish(self.run, subrun, "1", "test_inputs", self.prepared, command)
        with (self.run / "pathgen_results/PG-test-result.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_resume_recovers_crash_between_csv_move_and_receipt_publication(self):
        subrun, output, receipt = self.produce()
        target, marker = candidates.stage_paths(self.run, "1", "test_inputs")
        target.parent.mkdir(parents=True)
        output.replace(target)
        candidates.publish(
            self.run,
            subrun,
            "1",
            "test_inputs",
            self.prepared,
            candidates.command(subrun, "1", "test_inputs", self.prepared),
        )
        self.assertEqual(read_json(marker)["sha256"], receipt["sha256"])

    def test_extra_training_report_uses_original_meta_directory(self):
        subrun, _, _ = self.produce(stage="training_inputs")
        candidates.publish(
            self.run,
            subrun,
            "1",
            "training_inputs",
            self.prepared,
            candidates.command(subrun, "1", "training_inputs", self.prepared),
        )
        self.assertTrue(
            (self.run / "pathgen_meta_results/PG-train-result.csv").is_file()
        )
        self.assertFalse((self.run / "pathgen_results/PG-train-result.csv").exists())
        self.assertTrue(
            candidates.cached(self.run, "1", "training_inputs", self.prepared)
        )

    def test_corrupt_completed_output_cannot_be_adopted(self):
        subrun, output, _ = self.produce()
        output.write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            candidates.publish(self.run, subrun, "1", "test_inputs", self.prepared, [])
        self.assertFalse(
            candidates.stage_paths(self.run, "1", "test_inputs")[1].exists()
        )

    def test_report_merge_rejects_conflicting_row(self):
        subrun, _, _ = self.produce()
        target = self.run / "pathgen_results/PG-test-result.csv"
        target.parent.mkdir(parents=True)
        target.write_text("vantage,coverage,time_cost\n1,0.0,123\n")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            candidates.merge_report(
                subrun / "pathgen_results/PG-test-result.csv", target, "1"
            )

    def test_cpu_environment_uses_original_data_and_one_native_thread_per_worker(self):
        env = candidates.environment(self.prepared)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["ASINFER_FILES_ROOT"], "/original")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")

    def test_entrypoint_only_precomputes_explicit_full_run(self):
        args = ["20260301", "--run-dir", str(self.run)]
        self.assertIsNone(entry.precompute_route(args, self.root))
        write_json(
            self.run / "control/pg_precompute/REQUEST.json", dict(date="20260301")
        )
        self.assertEqual(entry.precompute_route(args, self.root), self.run)
        for option in (
            "--upstream-only",
            "--check-only",
            "--prepare-only",
            "--migration-replay",
            "--help",
            "--unknown",
        ):
            self.assertIsNone(entry.precompute_route(args + [option], self.root))
        with self.assertRaisesRegex(ValueError, "frozen registered method"):
            entry.precompute_route(
                args + ["--method-config", "alternate.json"], self.root
            )
        write_json(
            self.run / "control/pg_precompute/REQUEST.json", dict(date="20260201")
        )
        with self.assertRaisesRegex(ValueError, "another date"):
            entry.precompute_route(args, self.root)

    def configure_cpu_first(self):
        write_json(
            self.run / "RUN_IDENTITY.json", dict(date="20260301", code_sha256={})
        )
        write_json(
            self.run / "control/pg_precompute/REQUEST.json",
            dict(date="20260301", prepared_sha256=sha256(self.run / "PREPARED.json")),
        )

    def test_gpu_admission_occurs_only_after_all_candidates_are_frozen(self):
        for source in self.prepared["sources"]:
            for stage in candidates.STAGES:
                subrun, _, _ = self.produce(source, stage)
                candidates.publish(
                    self.run,
                    subrun,
                    source,
                    stage,
                    self.prepared,
                    candidates.command(subrun, source, stage, self.prepared),
                )
        self.configure_cpu_first()

        def admit(*args, **kwargs):
            proof = read_json(self.run / "control/pg_precompute/COMPLETED.json")
            self.assertEqual(proof["status"], "PASS")
            self.assertEqual(proof["sources"], ["1", "2"])
            self.assertEqual(len(proof["outputs_sha256"]), 10)
            candidates.common.verify_files(proof["outputs_sha256"])
            return True

        with patch.object(candidates, "PROJECT", self.root), patch.object(
            candidates.common, "validate_run", return_value=self.run
        ), patch.object(
            candidates.common,
            "verify_runtime_authorization",
            return_value=self.prepared,
        ), patch.object(
            candidates, "gpu_idle", side_effect=admit
        ) as gpu:
            candidates.prepare_all(self.run)
        gpu.assert_called_once()

    def test_existing_model_cannot_be_relabelled_as_cpu_first(self):
        self.configure_cpu_first()
        write_json(
            self.run / "control/gmpi/as1/seed1/training_manifest.json",
            dict(model_sha256="existing"),
        )
        with patch.object(candidates, "PROJECT", self.root), patch.object(
            candidates.common, "validate_run", return_value=self.run
        ), patch.object(
            candidates.common,
            "verify_runtime_authorization",
            return_value=self.prepared,
        ), patch.object(
            candidates, "gpu_idle"
        ) as gpu:
            with self.assertRaisesRegex(ValueError, "precede model fitting"):
                candidates.prepare_all(self.run)
        gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
