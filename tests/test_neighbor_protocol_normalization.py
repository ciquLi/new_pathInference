"""Tiny serialized model receipts exercise audited metadata-only normalization."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import normalize_neighbor_protocol as normalizer


class NeighborProtocolNormalization(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pathveil_protocol_normalize_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "runs/20260201/full_sources"
        self.run.mkdir(parents=True)
        source = self.root / "src/pathveil/__init__.py"
        source.parent.mkdir(parents=True)
        source.write_text("# frozen fixture source\n")
        self.method = normalizer.read(normalizer.PROJECT / "configs/method_full_sources.json")
        method_path = self.root / "configs/method_full_sources.json"
        self.write(method_path, self.method)
        registry = self.root / "configs/dates/20260201.json"
        self.write(
            registry, dict(date="20260201", method_config="configs/method_full_sources.json")
        )
        self.train = set(range(100, 130))
        folds = {
            str(s): int(
                hashlib.sha256(f"20261021:neighbor_oof_stable_source_v1:{s}".encode()).hexdigest(),
                16,
            )
            % 5
            for s in self.train
        }
        self.assertEqual(set(folds.values()), set(range(5)))
        self.roles = dict(
            train=sorted(self.train), validation=[200, 201], test=[300, 301], formal_test=[9000]
        )
        assigned = set(s for sources in self.roles.values() for s in sources)
        self.canonical = self.root / "inputs/merged_paths.txt"
        self.canonical.parent.mkdir()
        self.canonical.write_text(
            "".join(f"{s}->800001->800002\n" for s in sorted(assigned)) + "9500->800001->800002\n"
        )
        canonical_sha = normalizer.sha(self.canonical)
        split = dict(
            date="20260201",
            protocol=normalizer.SELECTION_POLICY,
            split=self.roles,
            source_to_fold=folds,
            excluded_task_origins=sorted(assigned),
            source_availability=dict(canonical_sha256=canonical_sha),
            eligibility=dict(
                operator=">",
                threshold=10000,
                path_count_field="unique_valid_paths",
                all_eligible_sources_assigned=True,
                eligible_sources=sorted(assigned - {9000}),
            ),
            fold_assignment=dict(
                policy="fixed_ASN_SHA256_modulo_5_v1",
                salt="20261021:neighbor_oof_stable_source_v1:",
            ),
        )
        self.split = split
        data = dict(date="20260201", sources=["9000"], merged_train=str(self.canonical))
        self.write(self.run / "data/data_manifest.json", data)
        self.write(self.run / "method_config.json", self.method)
        self.write(self.run / "upstream/split.json", split)
        self.write(
            self.run / "RUN_IDENTITY.json",
            dict(
                date="20260201",
                method_config=self.method,
                registry_sha256=normalizer.sha(registry),
                code_sha256=normalizer._code_identity(self.root),
            ),
        )
        code = normalizer.MethodLibrary(self.root).receipt()
        self.write(
            self.run / "upstream/IDENTITY.json",
            dict(method_config=self.method, split=split, code=code, canonical_sha256=canonical_sha),
        )
        computational = copy.deepcopy(code)
        actual_sources = normalizer.PROJECT / "src/pathveil/upstream"
        computational["orchestration_sources_sha256"] = {
            str(actual_sources / name): normalizer.sha(actual_sources / name)
            for name in ["neighbor_training.py", "neighbors.py", "evidence.py"]
        }
        references, raw_hashes = {}, {}
        for source_as in self.train:
            path = (
                self.run
                / "upstream/neighbor_models/evidence/training_origins"
                / (str(source_as) + ".csv")
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{source_as}->800001->800002\n")
            raw_hashes[str(source_as)] = normalizer.sha(path)
            references[str(source_as)] = dict(
                source=source_as, path=str(path), sha256=normalizer.sha(path)
            )
        refs_path = self.run / "upstream/neighbor_models/evidence/training_references.json"
        self.write(refs_path, references)
        inputs = {str(self.canonical): canonical_sha, str(refs_path): normalizer.sha(refs_path)}
        for name in ["evidence_audit.json", "safe_index_audit.json"]:
            path = refs_path.parent / name
            self.write(
                path,
                dict(
                    canonical_sha256=canonical_sha,
                    source_path_counts={"9500": 1},
                    withheld_start_counts={},
                    withheld_source_sha256=raw_hashes,
                ),
            )
            inputs[str(path)] = normalizer.sha(path)
        resources = normalizer.runtime_resources(self.method)
        binding = normalizer._binding(
            dict(
                date="20260201",
                split=self.roles,
                excluded_task_origins=sorted(assigned),
                source_to_fold=folds,
                input_hashes={str(self.canonical): canonical_sha},
                config=self.method,
                code=computational,
                runtime_resources=resources,
            )
        )
        bundle = dict(
            status="PASS",
            date="20260201",
            model_fit_date="20260201",
            protocol=normalizer.SOURCE_PROTOCOL,
            formal_labels_read=False,
            validation_labels_read=False,
            training_sources=list(map(str, sorted(self.train))),
            source_to_fold=folds,
            excluded_task_origins=sorted(assigned),
            computational_receipt=computational,
            binding=binding,
            runtime_resources=resources,
            training_references=str(refs_path),
            inputs_sha256=inputs,
            model_files_sha256={},
            models={},
            distribution={},
        )
        folder = self.run / "upstream/neighbor_models/models"
        for family in normalizer.FAMILIES:
            bundle["models"][family] = {}
            for key in sorted(normalizer.HEADS):
                path = folder / family / (key + ".txt")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("unchanged fixture model asset " + family + "/" + key)
                digest = normalizer.sha(path)
                bundle["models"][family][key] = str(path)
                bundle["model_files_sha256"][str(path)] = digest
                allowed = (
                    self.train
                    if key == "full"
                    else {s for s in self.train if folds[str(s)] != int(key[3:])}
                )
                # Some model families legitimately lack labels for a source.
                declared = sorted(allowed)[1:] if family == "role" else sorted(allowed)
                self.write(
                    path.with_suffix(".json"),
                    dict(
                        binding=dict(
                            binding=binding,
                            key=key,
                            sources=declared,
                            runtime_resources=dict(cpu_threads=resources["cpu_threads"]),
                        ),
                        sha256=digest,
                        formal_labels_read=False,
                    ),
                )
        for key in sorted(normalizer.HEADS):
            paths = {}
            hashes = {}
            for name, suffix in [("branch", ".txt"), ("length", ".pkl"), ("coverage", ".pkl")]:
                path = folder / "distribution" / key / (name + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("unchanged fixture " + key + "/" + name)
                paths[name] = str(path)
                hashes[str(path)] = normalizer.sha(path)
            bundle["distribution"][key] = paths
            bundle["model_files_sha256"].update(hashes)
            allowed = (
                self.train
                if key == "full"
                else {s for s in self.train if folds[str(s)] != int(key[3:])}
            )
            self.write(
                folder / "distribution" / key / "HEAD_COMPLETED.json",
                dict(
                    binding=dict(
                        binding=binding,
                        key=key,
                        sources=sorted(allowed),
                        runtime_resources=dict(cpu_threads=resources["cpu_threads"]),
                    ),
                    files_sha256=hashes,
                    formal_labels_read=False,
                    validation_labels_read=False,
                ),
            )
        self.bundle_path = self.run / "upstream/neighbor_models/neighbor_bundle.json"
        self.write(self.bundle_path, bundle)
        self.original = self.bundle_path.read_bytes()
        self.model_hashes = dict(bundle["model_files_sha256"])

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        normalizer.write_json(path, value)

    def normalize(self):
        return normalizer.normalize_run(self.run, project_root=self.root)

    def test_only_descriptive_fields_change_and_restart_is_idempotent(self):
        receipt_path = self.normalize()
        normalized = normalizer.read(self.bundle_path)
        original = json.loads(self.original)
        self.assertEqual(normalized["protocol"], normalizer.MODEL_PROTOCOL)
        self.assertEqual(normalized["source_selection_protocol"], normalizer.SOURCE_PROTOCOL)
        for name, value in original.items():
            if name != "protocol":
                self.assertEqual(normalized[name], value)
        self.assertEqual(
            set(normalized) - set(original),
            {"source_selection_protocol", "protocol_metadata_migration"},
        )
        receipt = normalizer.read(receipt_path)
        self.assertTrue(receipt["audit"]["original_validator_passed"])
        self.assertEqual(len(receipt["audit"]["head_source_audits"]), 48)
        self.assertEqual(Path(receipt["original_bundle_path"]).read_bytes(), self.original)
        self.assertEqual(self.normalize(), receipt_path)
        self.assertEqual({p: normalizer.sha(p) for p in self.model_hashes}, self.model_hashes)

    def test_lock_requires_original_training_process_to_release_run(self):
        with normalizer._run_lock(self.run):
            with self.assertRaisesRegex(RuntimeError, "still running"):
                self.normalize()
        self.assertEqual(self.bundle_path.read_bytes(), self.original)

    def test_unknown_protocol_or_source_change_is_rejected_without_publication(self):
        bundle = json.loads(self.original)
        bundle["protocol"] = "unknown"
        self.write(self.bundle_path, bundle)
        with self.assertRaisesRegex(RuntimeError, "Unknown protocol"):
            self.normalize()
        self.bundle_path.write_bytes(self.original)
        (self.root / "src/pathveil/__init__.py").write_text("# changed source")
        with self.assertRaisesRegex(RuntimeError, "source code differs"):
            self.normalize()
        self.assertEqual(self.bundle_path.read_bytes(), self.original)
        self.assertFalse((self.run / "control/protocol_metadata_migration/migration.json").exists())

    def test_heldout_or_own_fold_head_membership_is_rejected(self):
        marker = self.run / "upstream/neighbor_models/models/role/oof0.json"
        receipt = normalizer.read(marker)
        receipt["binding"]["sources"].append(200)
        self.write(marker, receipt)
        with self.assertRaisesRegex(RuntimeError, "held-out or own-fold"):
            self.normalize()
        self.assertEqual(self.bundle_path.read_bytes(), self.original)

    def test_changed_weights_are_rejected_by_the_unmodified_original_gate(self):
        filename = next(iter(self.model_hashes))
        Path(filename).write_text("changed weights")
        with self.assertRaisesRegex(RuntimeError, "Frozen input changed"):
            self.normalize()
        self.assertEqual(self.bundle_path.read_bytes(), self.original)
        self.assertFalse((self.run / "control/protocol_metadata_migration/migration.json").exists())

    def test_raw_reference_bytes_and_completed_migration_evidence_cannot_drift(self):
        receipt = self.normalize()
        marker = normalizer.read(receipt)
        original_path = Path(marker["original_bundle_path"])
        original_path.write_bytes(self.original + b" ")
        with self.assertRaisesRegex(RuntimeError, "Original bundle evidence changed"):
            self.normalize()


if __name__ == "__main__":
    unittest.main()
