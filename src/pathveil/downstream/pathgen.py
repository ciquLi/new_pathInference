"""Isolated candidate generation with one admitted library for both consumers."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import runpy
import sys

from pathveil.core import activate
from pathveil.downstream import common as c


def signature(paths):
    return hashlib.sha256(
        ("\n".join(sorted("->".join(map(str, path)) for path in paths)) + "\n").encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--stage", choices=("training_inputs", "test_inputs"), required=True)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    options, _ = parser.parse_known_args()
    run = c.validate_run(options.experiment_dir)
    prepared = c.verify_runtime_authorization(run)
    core = activate(data_manifest=prepared["data_manifest"])
    os.chdir(c.PROJECT)
    cache = run / "runtime_cache"
    cache.mkdir(exist_ok=True)
    for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "TORCH_HOME"):
        path = run / "runtime" / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    guard = c.StageGuard(run, allow_raw_test=options.stage == "test_inputs")
    sys.addaudithook(guard.check)
    import getData
    import HighFrequencyVertexBasedPathGen as hf
    from pathveil.downstream.train_evaluate_gmpi import graph_edges, graph_sha256

    def check_pool(source, paths, consumer, mode):
        expected = c.csv_paths(
            run / "path_augmentation_output/train" / (str(source) + "_augmented_paths.csv"),
            str(source),
        )
        actual, wanted = signature(paths), signature(expected)
        c.require(
            mode == "augmentation_only" and actual == wanted,
            consumer + " escaped admitted route library",
        )
        c.json_write(
            run
            / "control/consumer_audit"
            / (options.stage + "_" + str(source) + "_" + consumer + ".json"),
            dict(
                source=str(source),
                consumer=consumer,
                paths=len(paths),
                mode=mode,
                path_set_sha256=actual,
                same_admitted_pool_verified=True,
            ),
        )

    original_truth, original_local = (
        getData.load_train_extra_truth_paths,
        hf.load_local_knowledge_paths,
    )

    def truth(env, source):
        result = original_truth(env, source)
        check_pool(
            source,
            [path for rows in result.values() for path in rows],
            "truth",
            env.train_extra_data_mode,
        )
        return result

    def local(env, source):
        result = original_local(env, source)
        check_pool(source, result, "local", env.pg_local_knowledge_mode)
        return result

    getData.load_train_extra_truth_paths, hf.load_local_knowledge_paths = truth, local
    original_multi, original_task = (
        getData.multi_main_sparse_intensive,
        getData.pathGeneration_PerformanceAnalysis,
    )

    def isolated_multi(env, *args, **kwargs):
        if not hasattr(env, "_pathveil_base_graph"):
            env._pathveil_base_graph = env.G.copy()
            env._pathveil_base_edges = graph_edges(env.G)
        previous = env.curDate_baseDir
        env.curDate_baseDir = str(cache) + "/"
        try:
            return original_multi(env, *args, **kwargs)
        finally:
            env.curDate_baseDir = previous

    getData.multi_main_sparse_intensive = isolated_multi

    def task(parameters):
        source, destinations, env = parameters
        source = str(source)
        replay = prepared.get("dispatch_replay", {}).get(source)
        old = None
        if options.stage == "test_inputs" and replay:
            old = c.read(replay)
            c.require(
                graph_sha256(env._pathveil_base_edges) == old["base_graph_sha256"],
                "Regression base topology differs",
            )
            env.G = env._pathveil_base_graph.copy()
            for left, right, weight in old["graph_delta_edges"]:
                c.require(
                    env._add_graph_edge(left, right, weight), "Repeated regression graph edge"
                )
            env.rebuild_networkit_graph()
            c.require(
                graph_sha256(graph_edges(env.G)) == old["graph_sha256"],
                "Regression topology differs",
            )
            env.augment_graph_for_external_nodes = lambda nodes: 0
        result = original_task(parameters)
        extra = getattr(env, "generation_data_type", "") == "train_extra_data"
        suffix = "_train_extra_data_generated_data.csv" if extra else "_generated_data.csv"
        output = Path(env.analyze_data_dir) / env.dataType / env.dataAttr / (source + suffix)
        c.require(output.is_file(), "PG output missing")
        record = dict(
            source=source,
            mode=options.stage,
            output=str(output),
            sha256=c.sha(output),
            admitted_pool_manifest_sha256=prepared["admitted_pool_manifest_sha256"],
        )
        if options.stage == "test_inputs":
            base, current = env._pathveil_base_edges, graph_edges(env.G)
            delta = [
                [left, right, weight]
                for (left, right), weight in sorted(current.items())
                if (left, right) not in base
            ]
            direct = sorted(
                source + "->" + str(dest)
                for dest in set(destinations)
                if env.G.has_edge(source, str(dest))
            )
            if old:
                c.require(
                    delta == old["graph_delta_edges"] and direct == old["direct_pairs"],
                    "Regression graph/direct dispatch changed",
                )
            record.update(
                dispatch_version=1,
                date=prepared["date"],
                base_graph_sha256=graph_sha256(base),
                graph_sha256=graph_sha256(current),
                base_graph_edge_count=len(base),
                graph_edge_count=len(current),
                graph_delta_edges=delta,
                direct_pairs=direct,
                direct_pair_count=len(direct),
                topology_scope="source-specific actual generation topology",
                original_dispatch_replayed=bool(old),
                pg_local_knowledge_mode=env.pg_local_knowledge_mode,
                train_extra_data_mode=env.train_extra_data_mode,
            )
        c.json_write(
            run / "control/pathgen_completed" / (options.stage + "_" + source + ".json"), record
        )
        return result

    getData.pathGeneration_PerformanceAnalysis = task
    arguments = sys.argv[1:]
    position = arguments.index("--stage")
    del arguments[position : position + 2]
    if options.stage == "training_inputs":
        arguments.append("--extra_training_only")
    else:
        original_destinations = getData.get_meta_vas_dests
        getData.get_meta_vas_dests = lambda *args, **kwargs: (
            {},
            original_destinations(*args, **kwargs)[1],
        )
        getData.main_extra_training = lambda *args, **kwargs: None
    sys.argv = [str(core / "mainPathGen.py")] + arguments
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
