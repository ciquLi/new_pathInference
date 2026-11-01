"""Migration checks over canonical data and models, independent of old code."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np

from .contracts import read, require, write
from .destination import DestinationFeatureStore
from .implementation.final_features import ActualQueryStore, FinalPathFeatures
from .implementation.frontier_adapter import SafeFrontierAdapter
from .implementation.frontier_policy import FrontierPolicy
from .implementation.prior_spec import prediction_spec
from .method import MethodLibrary
from .neighbors import NeighborRuntime
from .path_pipeline import CachedPathFeatures


def verify_feature_migration(
    data_manifest, output, project_root=None, sources=("58057", "16552"), sample_rows=25
):
    """Regenerate 222/33 columns and context plans against the frozen fixtures."""
    library = MethodLibrary(project_root)
    project = library.project
    data = read(data_manifest) if isinstance(data_manifest, (str, Path)) else data_manifest
    require(
        data["date"] == "20260301",
        "March migration fixtures cannot validate another month's features",
    )
    bundle = read(project / "models/unified255/20260301/MANIFEST.json")
    config = read(project / "configs/method_unified255.json")
    fixtures = read(project / "data/upstream/20260301/REGRESSION.json")
    runtime = NeighborRuntime(bundle, data, config, library)
    destination = DestinationFeatureStore(bundle, runtime.public)
    records = []
    for source in map(str, sources):
        s = int(source)
        fixture = fixtures["cases"][source]
        with Path(fixture["candidates"]).open("rb") as handle:
            metadata = pickle.load(handle)
        with np.load(fixture["base222"], allow_pickle=False) as archive:
            original = archive["X"]
        with np.load(fixture["extra33"], allow_pickle=False) as archive:
            original33 = archive["X_extra"]
        ids = np.unique(
            np.linspace(
                0, len(metadata["candidates"]) - 1, min(sample_rows, len(metadata["candidates"]))
            ).astype(int)
        )
        selected = [dict(metadata["candidates"][int(i)], row_index=j) for j, i in enumerate(ids)]
        prediction = runtime.infer_root(s)
        require(
            prediction == read(fixture["root_prediction"]),
            "Root inference migration differs: " + source,
        )
        queries = ActualQueryStore(runtime)
        for neighbor in prediction["selected_neighbors"]:
            queries.query_frontier(s, int(neighbor), [s, int(neighbor)])
        adapter = SafeFrontierAdapter(
            s,
            prediction,
            runtime.safe["index"],
            runtime.public.relations,
            queries,
            FrontierPolicy(**config["frontier"]),
        )
        plan = adapter.plan()
        require(
            json.loads(json.dumps(plan)) == read(fixture["frontier_plan"]),
            "Actual frontier plan migration differs: " + source,
        )
        _, _, priors, _ = prediction_spec(s, prediction, runtime.public)
        for n in map(str, prediction["selected_neighbors"]):
            for rank, candidate in enumerate(queries.queries[(s, int(n))]["candidates"], 1):
                priors["recursive"].setdefault(n, {})[str(candidate["asn"])] = dict(
                    center_evidence={source: dict(pred_rank=rank)}
                )
        base = CachedPathFeatures(
            runtime.safe["index"],
            runtime.safe["origin_support"],
            runtime.public.relations,
            runtime.public,
            priors,
        )
        builder = FinalPathFeatures(base, plan, runtime, queries)
        values = [builder.features(candidate) for candidate in selected]
        matrix = np.asarray(
            [[row[name] for name in builder.feature_names] for row in values], np.float32
        )
        extra = destination.append(s, selected)
        require(
            np.array_equal(matrix, original[ids]), "222-feature byte equivalence failed: " + source
        )
        require(
            np.array_equal(extra, original33[ids]), "33-feature byte equivalence failed: " + source
        )
        records.append(
            dict(
                source=source,
                rows=len(ids),
                root_equal=True,
                frontier_plan_equal=True,
                base222_byte_equal=True,
                destination33_byte_equal=True,
            )
        )
        base._destination_aggregate.cache_clear()
        queries.clear()
    receipt = dict(
        status="PASS",
        date="20260301",
        checks=records,
        old_experiment_directory_reads=False,
        code=library.receipt(),
        scope="Sampled regenerated features; full-score and release equality are separate replay checks",
    )
    write(output, receipt)
    return receipt
