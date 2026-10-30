"""Date-local origin availability preserves frozen source roles and OOF folds."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from pathveil.upstream.contracts import UpstreamError
from pathveil.upstream.evidence import select_dated_sources, source_fold_map


class DatedSourceSplit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.canonical = self.root / "merged_paths.txt"
        self.train = list(range(10, 170))
        self.split = {
            "date": "20260301",
            "split": {
                "train": self.train,
                "validation": [9000],
                "formal_test": [9001],
                "old_test": [9002],
            },
            "excluded_task_origins": self.train + [9000, 9001, 9002, 9003],
            "source_to_fold": {str(source): index % 5 for index, source in enumerate(self.train)},
        }
        self.data = {"date": "20260201", "merged_train": str(self.canonical)}

    def write_origins(self, missing=()):
        self.canonical.write_text(
            "".join(f"{source}->2000\n" for source in self.train if source not in missing)
            + "2001->2002\n"
        )

    def select(self, manifest=None, requested=(9001,), allow_date_template=True):
        return select_dated_sources(
            self.data,
            manifest if manifest is not None else self.split,
            requested,
            allow_date_template=allow_date_template,
        )

    def test_missing_origins_keep_157_or_156_sources_without_reassigning_folds(self):
        for missing in ((10, 11, 12), (10, 11, 12, 13)):
            with self.subTest(missing=missing):
                self.write_origins(missing)
                result = self.select()
                available = set(self.train) - set(missing)
                self.assertEqual(set(result["split"]["train"]), available)
                self.assertEqual(len(available), 160 - len(missing))
                self.assertEqual(
                    result["source_to_fold"],
                    {str(s): self.split["source_to_fold"][str(s)] for s in sorted(available)},
                )
                receipt = result["source_availability"]
                self.assertEqual(receipt["unavailable_training_sources"], list(missing))
                self.assertEqual(receipt["requested_training_source_count"], 160)
                self.assertEqual(receipt["available_training_source_count"], len(available))
                self.assertFalse(receipt["held_out_labels_read"])
                self.assertFalse(receipt["replacement_sources"])
                self.assertTrue(set(missing) <= set(result["excluded_task_origins"]))
                self.assertEqual(result["split"]["validation"], [9000])
                self.assertEqual(result["split"]["old_test"], [9002])
                self.assertIn(9003, result["excluded_task_origins"])

    def test_all_available_keeps_original_training_membership_and_folds(self):
        self.write_origins()
        self.data["date"] = "20260301"
        result = self.select(allow_date_template=False)
        self.assertEqual(result["split"], self.split["split"])
        self.assertEqual(result["source_to_fold"], self.split["source_to_fold"])
        self.assertEqual(result["source_availability"]["unavailable_training_sources"], [])

    def test_formal_cannot_take_a_registered_train_or_validation_identity(self):
        self.write_origins(missing=(10,))
        for requested in ((10,), (9000,), (9002,)):
            with self.subTest(requested=requested), self.assertRaises(UpstreamError):
                self.select(requested=requested)

    def test_formal_change_keeps_old_formal_origin_masked(self):
        self.write_origins()
        result = self.select(requested=(9010,))
        self.assertEqual(result["split"]["formal_test"], [9010])
        self.assertTrue({9001, 9010} <= set(result["excluded_task_origins"]))

    def test_empty_fold_and_incomplete_fold_keys_fail(self):
        self.write_origins(missing=tuple(s for s in self.train if s % 5 == 4))
        with self.assertRaisesRegex(UpstreamError, "five nonempty"):
            self.select()
        self.write_origins()
        invalid = copy.deepcopy(self.split)
        del invalid["source_to_fold"]["10"]
        with self.assertRaisesRegex(UpstreamError, "fold keys"):
            self.select(invalid)

    def test_resume_is_identical_but_changed_canonical_is_rejected(self):
        self.write_origins(missing=(10, 11, 12))
        first = self.select()
        self.assertEqual(first, self.select(first, allow_date_template=False))
        self.assertEqual(
            first["source_availability"]["canonical_sha256"],
            hashlib.sha256(self.canonical.read_bytes()).hexdigest(),
        )
        with self.canonical.open("a") as stream:
            stream.write("2001->2003\n")
        with self.assertRaisesRegex(UpstreamError, "different canonical"):
            self.select(first, allow_date_template=False)

    def test_per_as_csv_filename_or_derived_suffix_cannot_create_an_origin(self):
        self.write_origins(missing=(10,))
        (self.root / "10.csv").write_text("10->3000\n")
        with self.canonical.open("a") as stream:
            stream.write("3001->10->3000\n0->10\n-1->10\n4294967296->10\n10\n10->\n")
        result = self.select()
        self.assertEqual(result["source_availability"]["unavailable_training_sources"], [10])
        self.assertEqual(result["source_availability"]["malformed_origin_rows"], 5)

    def test_missing_or_empty_canonical_stream_is_not_accepted(self):
        for exists in (False, True):
            if exists:
                self.canonical.write_bytes(b"")
            with self.subTest(exists=exists), self.assertRaisesRegex(
                UpstreamError, "missing or empty"
            ):
                self.select()

    def test_explicit_other_date_split_is_not_relabelled(self):
        self.write_origins()
        with self.assertRaisesRegex(UpstreamError, "split date differs"):
            self.select(allow_date_template=False)

    def test_fold_fallback_is_explicit_and_five_folds_are_required(self):
        manifest = {"split": self.split["split"]}
        expected = self.split["source_to_fold"]
        self.assertEqual(source_fold_map(manifest, self.train, lambda sources: expected), expected)
        with self.assertRaisesRegex(UpstreamError, "no registered fold"):
            source_fold_map(manifest, self.train)


if __name__ == "__main__":
    unittest.main()
