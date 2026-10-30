"""All-source roles use original unique paths and preserve an auditable split."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pathveil.upstream.contracts import UpstreamError
from pathveil.upstream.source_split import (
    inventory_sources,
    prepare_source_split,
    split_sources,
)


class SourceInventory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pathveil_source_protocol_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.canonical = self.root / "merged_paths.txt"

    def write(self, rows):
        self.canonical.write_bytes(b"".join(rows))
        return inventory_sources(self.canonical)

    def population(self, count=404):
        rows = [
            f"{source}->{destination}\n".encode()
            for source in range(100, 100 + count)
            for destination in (900001, 900002)
        ]
        # A qualifying formal source is excluded; one exactly-at-threshold
        # background source remains available as independent evidence.
        rows += [b"9000->900001\n", b"9000->900002\n", b"9500->900001\n"]
        return self.write(rows)

    def test_unique_valid_paths_match_existing_reference_rules(self):
        rows = [
            b"10->20->30\n",
            b"010->020->030\n",  # Same integer tuple, distinct raw bytes.
            b"10->30\n",
            b"10->20->10\n",  # Repeated ASN invalidates the whole path.
            b"10\n",  # Valid source identity but invalid complete path.
            b"10->4294967296\n",
            b"10->\xff\n",
            b"0->20\n",
            b"4294967296->20\n",
            b"malformed\n",
        ]
        inventory = self.write(rows)
        source = inventory["sources"]["10"]
        self.assertEqual(source["raw_rows"], 7)
        self.assertEqual(source["valid_rows"], 3)
        self.assertEqual(source["unique_valid_paths"], 2)
        self.assertEqual(source["duplicate_valid_rows"], 1)
        self.assertEqual(source["invalid_rows"], 4)
        self.assertEqual(inventory["totals"]["malformed_source_rows"], 3)
        self.assertEqual(source["raw_sha256"], hashlib.sha256(b"".join(rows[:7])).hexdigest())
        self.assertEqual(
            source["reference_unique_sha256"],
            hashlib.sha256(b"10->20->30\n10->30\n").hexdigest(),
        )
        self.assertFalse(inventory["canonical_is_unique_for_valid_paths"])

    def test_source_is_original_first_asn_not_a_filename_or_terminal_asn(self):
        (self.root / "10.csv").write_text("10->50\n" * 10001)
        inventory = self.write([b"20->10->50\n", b"20->60->10\n"])
        self.assertEqual(list(inventory["sources"]), ["20"])
        self.assertFalse(inventory["expanded_CSVs_read"])
        self.assertEqual(inventory["sources"]["20"]["unique_valid_paths"], 2)

    def test_ten_thousand_boundary_is_exclusive_and_duplicates_do_not_qualify(self):
        rows = [f"10->{100000 + i}\n".encode() for i in range(10000)]
        rows += [b"10->100000\n"] * 20
        rows += [f"11->{200000 + i}\n".encode() for i in range(10001)]
        inventory = self.write(rows)
        self.assertEqual(inventory["sources"]["10"]["unique_valid_paths"], 10000)
        self.assertGreater(inventory["sources"]["10"]["raw_rows"], 10000)
        self.assertEqual(inventory["sources"]["11"]["unique_valid_paths"], 10001)
        eligible = {
            int(source)
            for source, row in inventory["sources"].items()
            if row["unique_valid_paths"] > 10000
        }
        self.assertEqual(eligible, {11})

    def test_interleaved_exact_fallback_matches_grouped_counts_and_reference_hashes(self):
        groups = {
            source: [f"{source}->{dst}\n".encode() for dst in (2, 10, 2)]
            for source in (100, 101, 102)
        }
        grouped = self.write([raw for rows in groups.values() for raw in rows])
        interleaved_rows = [groups[source][i] for i in range(3) for source in groups]
        # A malformed source cannot disappear or be double-counted in fallback.
        interleaved_rows.append(b"invalid\n")
        interleaved = self.write(interleaved_rows)
        self.assertFalse(interleaved["source_grouped"])
        self.assertEqual(interleaved["scan_passes"], 2)
        self.assertEqual(interleaved["sources"], grouped["sources"])
        self.assertEqual(interleaved["totals"]["malformed_source_rows"], 1)
        for name in ("valid_rows", "unique_valid_paths", "duplicate_valid_rows"):
            self.assertEqual(interleaved["totals"][name], grouped["totals"][name])

    def test_every_eligible_source_is_assigned_with_no_160_cap(self):
        inventory = self.population()
        split = split_sources(inventory, "20260201", [9000], min_paths=1)
        roles = split["split"]
        self.assertEqual(
            {k: len(roles[k]) for k in ("train", "validation", "test")},
            {"train": 269, "validation": 67, "test": 68},
        )
        assigned = set(roles["train"] + roles["validation"] + roles["test"])
        self.assertEqual(assigned, set(range(100, 504)))
        self.assertNotIn(9000, assigned)
        self.assertEqual(len(split["excluded_task_origins"]), 405)
        self.assertEqual(split["remaining_background"]["unique_valid_paths"], 1)
        self.assertEqual(split["remaining_background"]["source_count"], 1)
        self.assertEqual(set(split["source_to_fold"].values()), set(range(5)))
        self.assertFalse(split["eligibility"]["legacy_160_template_used"])

    def test_role_order_and_per_source_folds_are_stable_across_dates_and_subsets(self):
        large = self.population()
        first = split_sources(large, "20260201", [9000], min_paths=1)
        smaller = dict(
            large,
            sources={k: v for k, v in large["sources"].items() if k not in {"100", "101", "102"}},
        )
        second = split_sources(smaller, "20260401", [9000], min_paths=1)
        common_train = set(first["source_to_fold"]) & set(second["source_to_fold"])
        self.assertGreater(len(common_train), 200)
        for source in common_train:
            self.assertEqual(first["source_to_fold"][source], second["source_to_fold"][source])
        self.assertEqual(first, split_sources(large, "20260201", [9000], min_paths=1))

    def test_freeze_resume_detects_input_configuration_or_output_tampering(self):
        self.population()
        data = dict(date="20260201", merged_train=str(self.canonical), sources=[9000])
        out = self.root / "frozen"
        path = prepare_source_split(data, out, min_paths=1)
        first = path.read_bytes()
        self.assertEqual(path, prepare_source_split(data, out, min_paths=1))
        self.assertEqual(first, path.read_bytes())
        with self.assertRaisesRegex(UpstreamError, "configuration changed"):
            prepare_source_split(data, out, min_paths=2)
        (out / "inventory.json").write_text("{}")
        with self.assertRaisesRegex(UpstreamError, "output changed"):
            prepare_source_split(data, out, min_paths=1)

    def test_changed_original_input_cannot_reuse_frozen_roles(self):
        self.population()
        data = dict(date="20260201", merged_train=str(self.canonical), sources=[9000])
        out = self.root / "frozen"
        prepare_source_split(data, out, min_paths=1)
        with self.canonical.open("ab") as stream:
            stream.write(b"9500->999999\n")
        with self.assertRaisesRegex(UpstreamError, "Canonical stream changed"):
            prepare_source_split(data, out, min_paths=1)

    def test_no_background_is_a_clear_failure(self):
        inventory = self.population()
        del inventory["sources"]["9500"]
        with self.assertRaisesRegex(UpstreamError, "No independent original background"):
            split_sources(inventory, "20260201", [9000], min_paths=1)


if __name__ == "__main__":
    unittest.main()
