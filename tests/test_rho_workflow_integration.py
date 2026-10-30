"""Opt-in engineering smoke test; its synthetic references are not research results."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from pathveil.experiment import preparation
from pathveil.experiment.contracts import freeze, path_bytes, signature
from pathveil.prefix_tree.public_prefix_tree import PublicPrefixTopology
from pathveil.utils import project_root, read_json, sha256, write_json


@unittest.skipUnless(
    os.environ.get("PATHVEIL_RUN_RHO_SMOKE") == "1", "explicit engineering smoke run only"
)
class PortableEndToEnd(unittest.TestCase):
    def test_real_environment_bindings_with_small_synthetic_library_and_references(self):
        root = project_root()
        current = root / "runs/20260201/rho_prefix_v1/current.json"
        base = Path(os.environ.get("PATHVEIL_RHO_SMOKE_BASE") or read_json(current)["run_dir"])
        upstream = read_json(base / "UPSTREAM_BOUND.json")
        protocol = copy.deepcopy(read_json(base / "PROTOCOL.json"))
        source = "19529"
        original = read_json(
            root
            / "runs/20260201/neighbor_first_augmentation_v1/augmentation/as19529/q0.9/ASSEMBLED.json"
        )
        selected = []
        destinations = set()
        with Path(original["paths_file"]).open() as stream:
            for row in csv.reader(stream):
                path = tuple(row[0].split("->"))
                if len(path) >= 4 and path[-1] not in destinations:
                    selected.append(path)
                    destinations.add(path[-1])
                if len(selected) == 12:
                    break
        self.assertEqual(len(selected), 12)
        verification = root / "runs/verification"
        verification.mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="rho_prefix_smoke_", dir=verification))
        protocol.update(
            sources=[source],
            public_features={source: protocol["public_features"][source]},
            score_origins=[],
            engineering_fixture=True,
            synthetic_references_are_not_heldout_performance=True,
            code_sha256=preparation.code_identity(),
        )
        protocol["method"]["gmpi"].update(epochs=1, patience=1)
        protocol["method"].pop("loss_candidates", None)
        protocol["method"]["training_objective"] = "normal_history_huber"
        freeze(run / "PROTOCOL.json", protocol)
        freeze(run / "UPSTREAM_BOUND.json", upstream)
        data = copy.deepcopy(read_json(base / "data/downstream_view/data_manifest.json"))
        files = run / "data/fixture_files"
        train = files / "data/aspath/path_by_vantage/20260201/path_without_prefix/train"
        train.mkdir(parents=True)
        # This source has no independently derived measured training paths.
        original_source = Path(data["train_dir"]) / (source + ".csv")
        self.assertEqual(original_source.stat().st_size, 0)
        shutil.copyfile(original_source, train / original_source.name)
        reference = train.parent / "test" / (source + ".csv")
        reference.parent.mkdir()
        csv_bytes, json_bytes = path_bytes(selected)
        reference.write_bytes(csv_bytes)
        endpoints = run / "data/fixture_endpoints.json"
        freeze(endpoints, {source: sorted(destinations, key=int)})
        data.update(
            sources=[source],
            common9=[source],
            source_data_root=str(files),
            train_dir=str(train),
            test_dir=str(reference.parent),
            expanded_train=str(Path(data["expanded_train"])),
            path_prefix_map={
                **data.get("path_prefix_map", {}),
                str(files): data["legacy_source_root"],
            },
            corpus_input_sha256={
                data["merged_train"]: sha256(data["merged_train"]),
                str(reference): sha256(reference),
            },
            engineering_fixture=True,
        )
        data.setdefault("read_path_map", {})[
            str(Path(data["graph_dir"]) / "vantage_dst_test.json")
        ] = str(endpoints)
        manifest = run / "data/data_manifest.json"
        freeze(manifest, data)
        directory = run / "augmentation" / ("as" + source)
        directory.mkdir(parents=True)
        (directory / "paths.csv").write_bytes(csv_bytes)
        (directory / "paths.json").write_bytes(json_bytes)
        freeze(
            directory / "ADMISSION.json",
            dict(
                date=protocol["date"],
                source=source,
                protocol_sha256=sha256(run / "PROTOCOL.json"),
                rows=len(selected),
                destinations=len(destinations),
                rho=protocol["public_features"][source]["rho"],
                strategy="engineering_fixture",
                path=str(directory / "paths.csv"),
                json_path=str(directory / "paths.json"),
                sha256=sha256(directory / "paths.csv"),
                json_sha256=sha256(directory / "paths.json"),
                path_set_sha256=signature(selected),
            ),
        )
        topology = PublicPrefixTopology.from_manifest(manifest)
        preparation.prepare_group(run, manifest, topology)
        env = dict(
            os.environ,
            PATHVEIL_PROJECT_ROOT=str(root),
            PYTHONPATH=str(root / "src"),
            PYTHONHASHSEED="20261015",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        completed = []
        for stage in ("pg_train", "pg_test", "train", "annotate", "evaluate", "rescore"):
            log = run / (stage + ".log")
            with log.open("wb") as stream:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "pathveil.experiment.worker",
                        "--run",
                        str(run),
                        "--stage",
                        stage,
                        "--source",
                        source,
                        "--device",
                        "cpu",
                        "--cpu-threads",
                        "2",
                    ],
                    cwd=root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                )
            self.assertEqual(result.returncode, 0, "Engineering stage failed; see " + str(log))
            completed.append(stage)
        group = run / "T-PG-B"
        selection = read_json(group / "gmpi/selections/as19529/SELECTION_FROZEN.json")
        self.assertEqual(set(selection["candidates"]), {"p0"})
        self.assertFalse(selection["final_test_used_for_selection"])
        summary = read_json(group / "evaluations/as19529/evaluation_summary.json")
        self.assertEqual(summary["query_count"], 12)
        self.assertEqual(summary["cached_label_mismatch_count"], 0)
        self.assertFalse(summary["original_test_used_for_selection"])
        zero_input = (
            group
            / "pg_inputs/as19529/model_input_data_PG/test/meta-devoid/19529_generated_data.csv"
        )
        with zero_input.open() as stream:
            self.assertTrue(
                all(all(float(v) == 0 for v in row[3].split("|")) for row in csv.reader(stream))
            )
        receipt = dict(
            status="PASS",
            scope="engineering smoke only; synthetic references and 12-path library",
            run=str(run),
            stages=completed,
            selected_model_count=1,
            cold_objectives_tested=["p0"],
            query_count=12,
            heldout_performance_claim=False,
            code_sha256=protocol["code_sha256"],
        )
        write_json(verification / "RHO_SMOKE_COMPLETED.json", receipt)


if __name__ == "__main__":
    unittest.main()
