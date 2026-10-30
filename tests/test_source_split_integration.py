"""The full-source split reaches the existing source-isolation contract intact."""

import json
from pathlib import Path
import tempfile
import unittest

from pathveil.upstream.contracts import read
from pathveil.upstream.evidence import prepare_evidence, select_dated_sources, source_fold_map
from pathveil.upstream.method import MethodLibrary
from pathveil.upstream.source_split import prepare_source_split


class FullSourceIntegration(unittest.TestCase):
    def test_generated_roles_remain_disjoint_and_all_task_origins_leave_background(self):
        with tempfile.TemporaryDirectory(prefix="pathveil_full_source_integration_") as temporary:
            root = Path(temporary)
            canonical = root / "merged_paths.txt"
            rows = []
            for source in range(100, 200):
                rows.extend(
                    [
                        f"{source}->901001->903001\n",
                        f"{source}->901002->903002\n",
                        f"{source}->901001->903001\n",
                    ]
                )
            rows.extend(["9000->901001->903001\n", "9000->901002->903002\n"])
            rows.append("9500->901001->903001\n")
            canonical.write_text("".join(rows))
            data = dict(date="20260201", merged_train=str(canonical), sources=["9000"])
            path = prepare_source_split(data, root / "source_protocol", min_paths=1)
            frozen = read(path)
            selected = select_dated_sources(data, path, ["9000"], allow_date_template=False)
            self.assertEqual(selected["split"], frozen["split"])
            self.assertEqual(selected["source_to_fold"], frozen["source_to_fold"])
            self.assertEqual(set(selected["split"]), {"train", "validation", "test", "formal_test"})
            self.assertEqual(selected["source_availability"]["unavailable_training_sources"], [])
            self.assertEqual(
                source_fold_map(selected, selected["split"]["train"]),
                frozen["source_to_fold"],
            )
            resumed = select_dated_sources(data, selected, ["9000"], allow_date_template=False)
            self.assertEqual(selected, resumed)
            prepared = prepare_evidence(data, selected, root / "evidence", MethodLibrary().core())
            references = read(prepared["training_references"])
            self.assertEqual(set(references), set(map(str, selected["split"]["train"])))
            self.assertFalse(
                set(references)
                & set(
                    map(str, selected["split"]["validation"] + selected["split"]["test"] + [9000])
                )
            )
            audit = read(root / "evidence/evidence_audit.json")
            self.assertEqual(audit["source_path_counts"], {"9500": 1})
            self.assertEqual(audit["total_paths"], 1)
            self.assertFalse(any(audit["withheld_start_counts"].values()))
            self.assertFalse(prepared["formal_labels_read"])
            self.assertFalse(prepared["validation_labels_read"])


if __name__ == "__main__":
    unittest.main()
