"""A freeze must detect code drift and damaged historical artifacts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "verify_frozen_release",
    Path(__file__).resolve().parents[1] / "scripts/verify_frozen_release.py",
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class FrozenReleaseTests(unittest.TestCase):
    def test_code_drift_and_new_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "method.py"
            code.write_text("original")
            manifest = dict(release_tag="frozen", source_files_sha256={
                "method.py": release.digest(code),
            })
            with patch.object(release, "source_files", return_value={"method.py"}):
                self.assertEqual("PASS", release.verify(root, manifest)["status"])
                code.write_text("changed numerical method")
                self.assertEqual("FAIL", release.verify(root, manifest)["status"])
            with patch.object(release, "source_files", return_value={"new_method.py"}):
                self.assertIn("unfrozen source: new_method.py",
                              release.verify(root, manifest)["errors"])

    def test_missing_or_damaged_asset_fails_without_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "checkpoint"
            asset.write_bytes(b"frozen model")
            manifest = dict(release_tag="frozen", source_files_sha256={},
                            baseline_assets_sha256={"checkpoint": release.digest(asset)})
            with patch.object(release, "source_files", return_value=set()):
                self.assertEqual("PASS", release.verify(root, manifest, True)["status"])
                asset.write_bytes(b"damaged model")
                self.assertEqual(["changed: checkpoint"],
                                 release.verify(root, manifest, True)["errors"])
                self.assertEqual(b"damaged model", asset.read_bytes())
                asset.unlink()
                self.assertEqual(["missing: checkpoint"],
                                 release.verify(root, manifest, True)["errors"])
                self.assertFalse(asset.exists())


if __name__ == "__main__":
    unittest.main()
