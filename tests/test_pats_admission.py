import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from pathveil.experiment.contracts import AdmissionStopped, UnknownSourceScale
from pathveil.experiment.contracts import quality_selection as strict_quality_selection
from pathveil.pats.admission import quality_selection
from pathveil.pats import runner
from pathveil.pats.policy import runtime_policy
from pathveil.utils import sha256, write_json


class PatsAdmissionTests(unittest.TestCase):
    def test_short_pool_keeps_all_legal_paths_and_reports_shortfall(self):
        paths = [("1", "2", str(i)) for i in range(3, 6)]
        chosen, rule = quality_selection(paths, [0.8, 0.4, 0.0], 0.75, 1000)
        self.assertEqual(chosen, paths)
        self.assertEqual(rule["minimum_shortfall"], 997)
        self.assertEqual(rule["original_quality_paths"], 1)
        self.assertEqual(rule["insufficient_legal_pool_policy"], "retain_all")
        self.assertEqual(rule["effective_threshold"], 0)

    def test_available_pool_preserves_existing_gate_exactly(self):
        paths = [("1", "2", str(i)) for i in range(3, 7)]
        for minimum in (None, 2, 4):
            self.assertEqual(quality_selection(paths, [0.8, 0.7, 0.4, 0.2], 0.75, minimum),
                             strict_quality_selection(paths, [0.8, 0.7, 0.4, 0.2], 0.75, minimum))
        with self.assertRaises(AdmissionStopped):
            quality_selection(paths, [0.8, 0.7, 0.4, 0.0], 0.75, 4)

    def test_empty_or_invalid_scores_do_not_create_a_library(self):
        with self.assertRaises(UnknownSourceScale):
            quality_selection([], [], 0.75, 1000)
        for scores in ([0.0], [float("nan")], [-1.0]):
            with self.assertRaises(ValueError):
                quality_selection([("1", "2", "3")], scores, 0.75, 1000)

    def test_explicit_frozen_resume_reuses_inputs_and_checks_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/20300101/pats_rho_v1/existing"
            code = root / "source.py"
            code.write_text("version = 1\n")
            write_json(root / "method.json", {})
            write_json(run / "PROTOCOL.json", dict(date="20300101", sources=["1"],
                       method={"selection_policy": runtime_policy({})},
                       code_sha256={str(code): sha256(code)}))
            args = SimpleNamespace(date="20300101", check_only=False, plan_only=False,
                                   method_config=None, run_dir=run, fresh_run=False,
                                   prepare_only=False, upstream_only=False,
                                   augmentation_only=False, device="cpu")
            registry = {"test_sources": ["1"], "workflow_config": "method.json"}
            with patch.object(runner, "project_root", return_value=root), patch.object(
                runner, "prepare", side_effect=AssertionError("must reuse frozen inputs")
            ), patch.object(runner, "run") as execute:
                self.assertEqual(runner.execute(args, registry), 0)
                execute.assert_called_once_with(run, "cpu")
                code.write_text("version = 2\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    runner.execute(args, registry)


if __name__ == "__main__":
    unittest.main()
