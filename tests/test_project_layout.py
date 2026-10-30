"""The production workspace must not accumulate another historical pipeline."""

from pathlib import Path
import unittest


class ProductionLayout(unittest.TestCase):
    def test_old_workflows_and_loose_root_scripts_are_outside_the_project(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "agent_review",
            "analysis",
            "artifacts",
            "code_backup",
            "experiments",
            "pipeline",
            "result",
            "result---",
            "run_gateway_pipeline.sh",
        ):
            self.assertFalse((root / name).exists(), name)
        self.assertEqual(list(root.glob("*.py")), [])
        self.assertEqual(list(root.glob("*.sh")), [])
        self.assertTrue((root / "scripts/run_experiment.sh").is_file())


if __name__ == "__main__":
    unittest.main()
