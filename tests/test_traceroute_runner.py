"""Orchestration tests use tiny workers; they never train or predict routes."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathveil.traceroute.runner import run_traceroute
from pathveil.utils import sha256, write_json


class TraceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        pool = self.root / "pool.csv"
        pool.write_text("1->2\n1->3\n")
        self.data, self.release = self.root / "data.json", self.root / "RELEASE.json"
        write_json(self.data, {"date": "20261015"})
        neighbors, anchors = self.root / "neighbors.csv", self.root / "anchors.csv"
        neighbors.write_text("center_as,neighbor_as\n1,2\n")
        anchors.write_text("first_hop,second_hop\n2,3\n")
        write_json(
            self.release,
            {
                "date": "20261015",
                "sources": ["1"],
                "formal_reference_labels_read": False,
                "training_and_PG_pool": "same_R_s_hash_augmentation_only",
                "files_by_source": {"1": {"path": str(pool), "sha256": sha256(pool), "rows": 2}},
                "selected_neighbors_csv": str(neighbors),
                "recursive_anchors_csv": str(anchors),
                "outputs_sha256": {
                    str(neighbors): sha256(neighbors),
                    str(anchors): sha256(anchors),
                },
            },
        )
        targets = self.root / "data/traceroute/20261015/inputs/web_targets.csv"
        targets.parent.mkdir(parents=True)
        targets.write_text("target_ip,destination_asn\n1.2.3.4,2\n")
        # There is deliberately no measurement file in this fixture.
        self.calls = []

    def fake_worker(self, command, **kwargs):
        module = command[2]
        output = Path(command[command.index("--outdir") + 1])
        output.mkdir(parents=True)
        self.calls.append(module)
        if module.endswith("generate_training"):
            csv = output / "generated.csv"
            csv.write_text("generated BGP row fixture\n")
            write_json(output / "TRAINING_INPUTS.json", {"generated_csv": str(csv)})
        elif module.endswith("train_bgp"):
            (output / "model.pkl").write_bytes(b"fixed model fixture")
            write_json(output / "training_manifest.json", {})
            write_json(output / "training_history.json", [])
        elif module.endswith("predict_web"):
            (output / "pair_predictions.jsonl").write_text("{}\n")
            (output / "target_predictions.jsonl").write_text("{}\n")
            write_json(output / "prediction_manifest.json", {})
        else:
            write_json(output / "summary.json", {})
            for name in ["per_pair.csv", "per_path.csv", "gmpi_result.txt"]:
                (output / name).write_text("evaluation fixture\n")
        return SimpleNamespace(returncode=0)

    def run_fixture(self):
        return run_traceroute(self.root / "run", self.data, self.release, "cpu", source="1")

    def test_stage_resume_and_changed_checkpoint_detection(self):
        with patch.dict("os.environ", {"PATHVEIL_PROJECT_ROOT": str(self.root)}), patch(
            "pathveil.traceroute.runner.subprocess.run", side_effect=self.fake_worker
        ):
            result = self.run_fixture()
            self.assertEqual(len(self.calls), 4)
            self.assertFalse(result["measurement_labels_used_for_training_or_selection"])
            self.assertEqual(self.run_fixture(), result)
            self.assertEqual(len(self.calls), 4)
            Path(result["model"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "output changed"):
                self.run_fixture()

    def test_failed_worker_keeps_failure_status_and_can_retry(self):
        with patch.dict("os.environ", {"PATHVEIL_PROJECT_ROOT": str(self.root)}), patch(
            "pathveil.traceroute.runner.subprocess.run", return_value=SimpleNamespace(returncode=7)
        ):
            with self.assertRaisesRegex(RuntimeError, "stage failed"):
                self.run_fixture()
        state = json.loads((self.root / "run/traceroute/control/status.json").read_text())
        self.assertEqual(state["status"], "failed")
        with patch.dict("os.environ", {"PATHVEIL_PROJECT_ROOT": str(self.root)}), patch(
            "pathveil.traceroute.runner.subprocess.run", side_effect=self.fake_worker
        ):
            self.assertEqual(self.run_fixture()["status"], "complete")
        logs = list((self.root / "run/traceroute/control/attempts/training_inputs").glob("*.log"))
        self.assertEqual(len(logs), 2)


if __name__ == "__main__":
    unittest.main()
