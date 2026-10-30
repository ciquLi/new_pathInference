"""One frozen255 scorer; rho selects direct or neighbor-first admission."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import pickle

import numpy as np

from pathveil.utils import read_json, sha256
from pathveil.prefix_tree.public_prefix_tree import commercial_phases
from .contracts import (
    freeze,
    freeze_bytes,
    forbid_formal_references,
    normalize_pool,
    path_bytes,
    quality_selection,
    require,
    signature,
    verify_files,
)


class ScoreRuntime:
    """An independent numerical runtime for one CPU resource worker."""

    def __init__(self, run):
        import lightgbm as lgb
        from pathveil.upstream.destination import DestinationFeatureStore
        from pathveil.upstream.neighbors import NeighborRuntime
        from pathveil.upstream.method import MethodLibrary

        self.run = Path(run)
        self.protocol = read_json(self.run / "PROTOCOL.json")
        self.upstream = read_json(self.run / "UPSTREAM_BOUND.json")
        self.data_file = self.run / "data/data_manifest.json"
        self.data = read_json(self.data_file)
        self.config = self.protocol["upstream_config"]
        verify_files(
            {
                self.upstream["model"]: self.upstream["model_sha256"],
                self.upstream["bundle"]: self.upstream["bundle_sha256"],
            }
        )
        bundle = read_json(self.upstream["bundle"])
        self.runtime = NeighborRuntime(
            bundle, self.data, self.config, method=MethodLibrary(), threads=1
        )
        self.store = DestinationFeatureStore(bundle, self.runtime.public)
        self.model = lgb.Booster(model_file=self.upstream["model"])

    def score(self, source):
        from pathveil.upstream.path_pipeline import construct_case

        source = str(source)
        output = self.run / "score_pools" / ("as" + source)
        marker = output / "SCORED.json"
        if marker.exists():
            value = read_json(marker)
            require(
                value["protocol_sha256"] == sha256(self.run / "PROTOCOL.json"),
                "Score pool protocol differs",
            )
            verify_files(value["outputs_sha256"])
            verify_files(value["inputs_sha256"])
            return value
        with forbid_formal_references():
            # An existing CASE is a numerical cache, never an implementation dependency.
            cached = Path(self.upstream["run"]) / "upstream/inference_cases" / source / "CASE.json"
            if cached.exists():
                case = read_json(cached)
                require(
                    case["date"] == self.data["date"]
                    and case["source"] == source
                    and not case["formal_labels_read"],
                    "Cross-date/source CASE cache rejected",
                )
                verify_files(case["files_sha256"])
                for key in (
                    "computational_sha256",
                    "model_files_sha256",
                    "donor_inputs_sha256",
                    "public_files_sha256",
                ):
                    verify_files(case["identity"][key])
                require(
                    case["identity"]["method_config"] == self.config
                    and case["identity"]["evidence_sha256"] == self.runtime.evidence_sha256,
                    "Cached CASE evidence or method differs",
                )
            else:
                case = construct_case(
                    source,
                    self.runtime,
                    self.config,
                    self.store,
                    self.run / "construction" / ("as" + source),
                )
                cached = Path(case["features"]).parent / "CASE.json"
            with Path(case["candidates"]).open("rb") as stream:
                candidates = pickle.load(stream)["candidates"]
            with np.load(case["features"], allow_pickle=False) as archive:
                matrix = archive["X"]
            require(
                self.model.feature_name() == case["feature_names"]
                and matrix.shape == (len(candidates), 255),
                "Frozen255 path/feature alignment differs",
            )
            raw = self.model.predict(matrix, num_threads=1) if len(matrix) else np.empty(0)
            paths, values, diagnostics = normalize_pool(
                source, [c["path"] for c in candidates], raw
            )
            best = {}
            for i, candidate in enumerate(candidates):
                path = tuple(map(str, candidate["path"]))
                if path not in best or raw[i] > raw[best[path]]:
                    best[path] = i
            maximum = float(values.max())
            records = [
                dict(
                    path=list(path),
                    prefix=list(map(str, candidates[best[path]]["prefix"])),
                    raw_score=float(value),
                    relative_score=float(value) / maximum,
                )
                for path, value in zip(paths, values)
            ]
            freeze(output / "SCORED_PATHS.json", records)
            value = dict(
                status="PASS",
                date=self.data["date"],
                source=source,
                paths=len(records),
                source_max=maximum,
                diagnostics=diagnostics,
                feature_schema=case["feature_names"],
                protocol_sha256=sha256(self.run / "PROTOCOL.json"),
                model_sha256=self.upstream["model_sha256"],
                inputs_sha256={str(cached): sha256(cached), **case["files_sha256"]},
                outputs_sha256={
                    str(output / "SCORED_PATHS.json"): sha256(output / "SCORED_PATHS.json")
                },
                final_reference_labels_read=False,
            )
            freeze(marker, value)
            return value


def build_library(run, source, topology):
    run, source = Path(run), str(source)
    protocol = read_json(run / "PROTOCOL.json")
    shape = protocol["public_features"][source]
    policy = protocol["method"]["augmentation"]
    require(shape["rho"] is not None, "Unknown public structure has no default admission mode")
    inputs, paths, filters, per_neighbor = {}, {}, Counter(), {}

    def load(origin):
        marker = run / "score_pools" / ("as" + origin) / "SCORED.json"
        receipt = read_json(marker)
        require(
            receipt["date"] == protocol["date"]
            and receipt["protocol_sha256"] == sha256(run / "PROTOCOL.json"),
            "Candidate pool date/method differs",
        )
        verify_files(receipt["outputs_sha256"])
        inputs[str(marker)] = sha256(marker)
        inputs.update(receipt["outputs_sha256"])
        return read_json(marker.with_name("SCORED_PATHS.json"))

    with forbid_formal_references():
        if shape["rho"] > 1:
            neighbors = sorted(map(str, topology.neighbors(source)), key=int)
            require(
                neighbors == protocol["neighbors"][source], "Source public neighbor set changed"
            )
            threshold = policy["neighbor_threshold"]
            for neighbor in neighbors:
                counts = Counter()
                for row in load(neighbor):
                    if row["relative_score"] < threshold:
                        continue
                    counts["neighbor_selected"] += 1
                    path, prefix = (source, *row["path"]), (source, *row["prefix"])
                    require(
                        path[1] == neighbor and prefix == path[: len(prefix)] and len(prefix) <= 4,
                        "Neighbor prefix splice differs",
                    )
                    reason = None
                    if len(path) != len(set(path)):
                        reason = "loop_including_source"
                    elif len(path) > 20:
                        reason = "full_path_longer_than20"
                    elif (
                        commercial_phases(
                            [topology.public_role(a, b) for a, b in zip(path, path[1:])]
                        )["status"]
                        == "incompatible"
                    ):
                        reason = "known_commercial_incompatible"
                    if reason:
                        filters[reason] += 1
                        counts[reason] += 1
                        continue
                    paths[path] = dict(
                        path=list(path),
                        prefix=list(prefix),
                        source_neighbor=neighbor,
                        raw_score=row["raw_score"],
                        relative_score=row["relative_score"],
                    )
                    counts["retained"] += 1
                per_neighbor[neighbor] = dict(counts)
            selection = dict(
                threshold=threshold,
                normalization="independent complete legal pool maximum per neighbor",
                source_rescoring=False,
                quantity_floor=None,
                quantity_cap=None,
            )
            strategy = "neighbor_first_then_prepend_source"
        else:
            pool = load(source)
            chosen, selection = quality_selection(
                [tuple(r["path"]) for r in pool],
                [r["raw_score"] for r in pool],
                policy["source_threshold"],
                policy["low_minimum_paths"],
            )
            chosen = set(chosen)
            paths = {tuple(r["path"]): r for r in pool if tuple(r["path"]) in chosen}
            strategy = "direct_model_frontier"
        require(paths, "No admissible paths; keep the source unresolved rather than dropping it")
        output = run / "augmentation" / ("as" + source)
        output.mkdir(parents=True, exist_ok=True)
        csv_data, json_data = path_bytes(sorted(paths))
        freeze_bytes(output / "paths.csv", csv_data)
        freeze_bytes(output / "paths.json", json_data)
        freeze(output / "PATH_RECORDS.json", [paths[p] for p in sorted(paths)])
        item = dict(
            path=str(output / "paths.csv"),
            json_path=str(output / "paths.json"),
            rows=len(paths),
            sha256=sha256(output / "paths.csv"),
            json_sha256=sha256(output / "paths.json"),
            path_set_sha256=signature(paths),
        )
        record = dict(
            status="FROZEN",
            date=protocol["date"],
            source=source,
            strategy=strategy,
            rho=shape["rho"],
            public_features=shape,
            selection=selection,
            destinations=len({p[-1] for p in paths}),
            per_neighbor=per_neighbor,
            filters=dict(filters),
            protocol_sha256=sha256(run / "PROTOCOL.json"),
            inputs_sha256=inputs,
            final_labels_read=False,
            **item,
        )
        freeze(output / "ADMISSION.json", record)
        return record
