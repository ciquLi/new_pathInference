"""Restore a known CSV omission without weakening the frozen provenance gates."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import repair_downstream_inputs as repair
import run_registered_experiment as entry
from pathveil.data.prepare import resolve_read_path
from pathveil.downstream.runner import audit_derived_source


class DerivedInputRepair(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.merged = self.root / "merged.txt"
        self.expand = self.root / "expand.txt"
        self.log = self.root / "split_log.txt"
        self.merged.write_text("9->1->2->3\n8->1->7\n9->4->5->6\n")
        self.expand.write_text("1->2\n1->2->3\n1->7\n4->5\n4->5->6\n")
        self.log.write_text("fixture\n")
        self.originals = {"1": {"1->2->3"}, "4": {"4->5->6"}}

    def derive(self):
        return repair.derive(self.merged, self.expand, self.originals, {"1", "4"})

    def test_restores_only_omitted_two_AS_paths(self):
        full, counts = self.derive()
        self.assertEqual(full["1"], {"1->2", "1->2->3", "1->7"})
        self.assertEqual(counts["1"]["restored_two_AS_rows"], 2)
        self.assertEqual(counts["4"]["restored_two_AS_rows"], 1)

    def test_missing_longer_path_is_rejected(self):
        self.originals["1"] = {"1->2"}
        with self.assertRaisesRegex(ValueError, "longer paths"):
            self.derive()

    def test_unproven_original_path_is_rejected(self):
        self.originals["1"].add("1->999")
        with self.assertRaisesRegex(ValueError, "without independent provenance"):
            self.derive()

    def test_expand_must_equal_independent_derivation(self):
        self.expand.write_text("1->2->3\n4->5\n4->5->6\n")
        with self.assertRaisesRegex(ValueError, "exact independent derivation"):
            self.derive()

    def test_formal_origin_rows_are_rejected(self):
        with self.merged.open("a") as stream:
            stream.write("1->8->9\n")
        with self.assertRaisesRegex(ValueError, "Formal-origin"):
            self.derive()

    def test_both_original_frozen_gates_accept_reconstructed_CSV(self):
        full, _ = self.derive()
        original = self.root / "old.csv"
        original.write_text("1->2->3\n")
        data = dict(
            date="20260201",
            train_dir=str(self.root),
            merged_train=str(self.merged),
            expanded_train=str(self.expand),
            split_log=str(self.log),
        )
        with self.assertRaisesRegex(ValueError, "exact independent forward derivation"):
            audit_derived_source(
                original, "1", data, ["1", "4"], self.root / "old_audit.json"
            )
        projected = self.root / "1.csv"
        projected.write_text("".join(path + "\n" for path in sorted(full["1"])))
        audit_path = audit_derived_source(
            projected, "1", data, ["1", "4"], self.root / "audit.json"
        )
        audit = json.loads(audit_path.read_text())
        self.assertEqual(audit["train_expand_minus_source_csv_count"], 0)
        from pathveil.downstream.train_evaluate_gmpi import audit_source_training_file

        result = audit_source_training_file(projected, "1", audit_path)
        self.assertEqual(result["derived_external_rows"], 3)
        self.assertEqual(original.read_text(), "1->2->3\n")

    def test_manifest_changes_only_downstream_view_and_preserves_read_targets(self):
        old = self.root / "original"
        local = self.root / "local"
        projected = (
            local / "data/aspath/path_by_vantage/20260201/path_without_prefix/train"
        )
        projected.mkdir(parents=True)
        csv = projected / "1.csv"
        csv.write_text("1->2\n")
        aggregate = (
            old
            / "data/aspath/path_by_vantage/20260201/path_without_prefix/test/merged_paths.txt"
        )
        data = dict(
            date="20260201",
            source_data_root=str(old),
            train_dir=str(old / "train"),
            test_dir=str(old / "test"),
            graph_dir="/unchanged/graph",
            read_path_map={str(aggregate): "/prepared/test_merged.txt"},
            path_prefix_map={},
            public_files={"asrel": "/unchanged/asrel"},
        )
        before = copy.deepcopy(data)
        result = repair.make_manifest(data, local, projected, old / "train")
        self.assertEqual(data, before)
        self.assertEqual(result["test_dir"], before["test_dir"])
        self.assertEqual(result["graph_dir"], before["graph_dir"])
        self.assertEqual(resolve_read_path(result, csv), csv)
        self.assertEqual(
            str(resolve_read_path(result, local / aggregate.relative_to(old))),
            "/prepared/test_merged.txt",
        )

    def test_formal_entry_uses_repair_only_for_its_exact_normal_run(self):
        run = self.root / "runs/20260201/unified255/fixture"
        receipt = run / "control/derived_csv_repair/REPAIR.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps(dict(date="20260201", status="PASS")))
        arguments = ["20260201", "--run-dir", str(run)]
        self.assertEqual(entry.repaired_route(arguments, self.root), (run, "cuda"))
        for mode in (
            "--upstream-only",
            "--check-only",
            "--prepare-only",
            "--migration-replay",
            "--unknown",
        ):
            self.assertIsNone(entry.repaired_route(arguments + [mode], self.root))
        with self.assertRaisesRegex(ValueError, "another date"):
            entry.repaired_route(["20260401", "--run-dir", str(run)], self.root)

    def test_evaluation_mirrors_are_byte_identical_and_never_overwrite_changes(self):
        original = self.root / "test"
        original.mkdir()
        source = original / "1.csv"
        source.write_bytes(b"1->2->3\n")
        data = dict(
            date="20260201",
            sources=["1"],
            test_dir=str(original),
            corpus_input_sha256={str(source): repair.sha256(source)},
        )
        run = self.root / "run"
        outputs = repair.mirror_test_inputs(run, data)
        copied = Path(next(iter(outputs)))
        self.assertEqual(copied.read_bytes(), source.read_bytes())
        self.assertEqual(repair.mirror_test_inputs(run, data), outputs)
        copied.write_text("1->999\n")
        with self.assertRaisesRegex(ValueError, "differs from frozen input"):
            repair.mirror_test_inputs(run, data)


if __name__ == "__main__":
    unittest.main()
