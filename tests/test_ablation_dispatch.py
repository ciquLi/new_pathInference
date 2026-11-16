"""The evaluation guard must be opened only after the checkpoint gate."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathInference.utils import read_json, sha256, write_json
from pathInference.experiment import worker

_file = Path(__file__).resolve().parents[1] / "research/generator_recursion_v1/dispatch.py"
_spec = importlib.util.spec_from_file_location("ablation_dispatch", _file)
dispatch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dispatch)


class AblationDispatchTests(unittest.TestCase):
    def test_gmpi_child_does_not_publish_outer_research_receipt(self):
        for stage in dispatch.GMPI_STAGES:
            key = ("20260201", "no_recursion", "19529", stage)
            command = dispatch.stage_command(key)
            self.assertIn("gmpi_stage(*sys.argv[1:])", command[4])
            self.assertEqual(list(key), command[5:])

    def test_completed_numerical_work_is_registered_without_execution(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "CONFIG.json"
            write_json(config, {})
            write_json(root / "PROTOCOL.json", {})
            child = root / "T-PG-B/pg_inputs/as19529"
            child.mkdir(parents=True)
            write_json(child / "PREPARED.json", {})
            output = root / "selected_model.pt"
            output.write_bytes(b"frozen checkpoint")
            key = ("20260201", "no_recursion", "19529", "train")
            receipt = worker.receipt_path(root, "train", "19529")
            write_json(receipt, dict(status="PASS", stage="train", source="19529",
                                    protocol_sha256=sha256(root / "PROTOCOL.json"),
                                    prepared_sha256=sha256(child / "PREPARED.json"),
                                    outputs_sha256={str(output): sha256(output)}))
            with patch.object(dispatch.experiment, "CONFIG", config), patch.object(
                    dispatch.experiment, "run_dir", return_value=root), patch.object(
                    worker, "execute", side_effect=AssertionError("must not retrain")):
                self.assertTrue(dispatch.recover_gmpi_task(key))
                dispatch.verify_task(dispatch.task_path(*key))
                self.assertFalse(dispatch.recover_gmpi_task(key))
                dispatch.task_path(*key).unlink()
                output.write_bytes(b"changed checkpoint")
                with self.assertRaises(ValueError):
                    dispatch.recover_gmpi_task(key)
                self.assertFalse(dispatch.task_path(*key).exists())

    def test_gmpi_guard_matches_the_root_opened_by_evaluation_adapter(self):
        run = Path("/example/run")
        for stage in ("train", "annotate", "evaluate", "rescore"):
            env = dispatch.stage_environment({"NUMEXPR_NUM_THREADS": "1"}, run, stage)
            self.assertEqual(str(run / "T-PG-B"), env["PATHVEIL_STAGE_GUARD_ROOT"])
            self.assertEqual("1", env["NUMEXPR_NUM_THREADS"])

    def test_candidate_guard_retains_run_boundary_for_source_publication(self):
        run = Path("/example/run")
        for stage in ("prepare", "pg_train", "pg_test", "metrics"):
            self.assertEqual(str(run), dispatch.stage_environment({}, run, stage)["PATHVEIL_STAGE_GUARD_ROOT"])


if __name__ == "__main__":
    unittest.main()
