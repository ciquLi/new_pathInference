"""Build an explicitly registered BGP cohort without mutating extracted corpora.

Graph, expanded knowledge and collected-path vectors are rebuilt from the
remaining original training sources. Held-out paths supply evaluation and
endpoint inventories only. Each costly phase has a verified reuse receipt.
"""
from __future__ import annotations

from collections import Counter
import gc
import json
from pathlib import Path
import shutil

from pathveil.data.prepare import (
    DEFAULT_FEATURE, rebuild_as_embedding, rebuild_as_features,
    rebuild_test_aggregates, write_double_json,
)
from pathveil.data.registry import select_asset
from pathveil.utils import read_json, sha256, utc_now, write_json


def _verify(files):
    for name, digest in files.items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise ValueError("Cohort artifact changed: " + name)


def _phase(root, name, inputs, builder, progress):
    receipt = root / (name + ".json")
    if receipt.exists():
        value = read_json(receipt)
        if value["inputs_sha256"] != inputs:
            raise ValueError("Cohort inputs changed for " + name)
        _verify(value["outputs_sha256"])
        progress("cohort_" + name + "_reused")
        return value
    progress("cohort_" + name)
    outputs, detail = builder()
    value = dict(status="COMPLETED", inputs_sha256=inputs,
                 outputs_sha256={str(p): sha256(p) for p in outputs},
                 details=detail, updated_utc=utc_now())
    write_json(receipt, value)
    gc.collect()
    return value


def build_training_graph(merged, graph, as_frequency, asrank, snapshot_date, formal):
    """Use original TRAIN routes only; do not read held-out route sequences."""
    edges, nodes, frequency, endpoints = Counter(), {}, Counter(), {}
    count = 0
    with Path(merged).open() as stream:
        for line in stream:
            path = line.strip().split("->")
            if path[0] in formal:
                raise ValueError("Held-out original source entered training graph")
            endpoints.setdefault(path[0], set()).add(path[-1])
            for node in path:
                nodes.setdefault(node, DEFAULT_FEATURE[:])
                frequency[node] += 1
            edges.update(a + "-" + b for a, b in zip(path, path[1:]))
            count += 1
    graph = Path(graph)
    graph.mkdir(parents=True, exist_ok=True)
    write_double_json(graph / "edge_frequency.json", edges)
    write_double_json(graph / "AS_Feature.json", nodes)
    write_double_json(as_frequency, frequency)
    write_double_json(graph / "vantage_dst.json",
                      {s: sorted(d, key=int) for s, d in endpoints.items()})
    detail = rebuild_as_features(asrank, graph / "AS_Feature.json",
                                 graph / "edge_frequency.json", graph, snapshot_date)
    detail.update(original_training_rows=count, heldout_sources_excluded=sorted(formal, key=int))
    return detail


def ensure_cohort(registry, progress=print):
    """Materialize the registry's source holdout using original full-path CSVs."""
    policy = registry.get("rebuild_bgp_holdout")
    if not policy:
        return None
    date = registry["date"]
    project = Path(registry["project_root"])
    local = Path(registry["routing_files_root"])
    if project / "data" not in local.parents:
        raise ValueError("Rebuilt cohort must be in project data/")
    external = Path(registry["files_root"])
    if local == external or external in local.parents:
        raise ValueError("Cannot rebuild inside external corpus")
    original = external / "data/aspath/path_by_vantage" / date / "path_without_prefix"
    source_files = sorted(original.glob("*.csv"), key=lambda p: int(p.stem))
    formal = set(map(str, registry["test_sources"]))
    if not formal <= {p.stem for p in source_files if p.stat().st_size}:
        raise ValueError("Registered formal source has no original BGP observations")
    root = local.parent
    root.mkdir(parents=True, exist_ok=True)
    target = local / "data/aspath/path_by_vantage" / date
    paths, graph = target / "path_without_prefix", target / "RL_Graph"
    train, test = paths / "train", paths / "test"
    train.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)
    inputs = {str(p): sha256(p) for p in source_files}
    inputs[str(Path(__file__).resolve())] = sha256(__file__)
    from pathveil.data import prepare as preparation
    inputs[str(Path(preparation.__file__).resolve())] = sha256(preparation.__file__)
    inputs[str(Path(registry["registry_path"]).resolve())] = sha256(registry["registry_path"])

    def split():
        outputs, endpoints, rows = [], {}, {}
        for file in source_files:
            destination = (test if file.stem in formal else train) / file.name
            shutil.copyfile(file, destination)
            outputs.append(destination)
            if file.stem in formal:
                destinations, count = set(), 0
                with file.open() as stream:
                    for line in stream:
                        nodes = line.strip().split("->")
                        if len(nodes) < 2 or nodes[0] != file.stem:
                            raise ValueError("Original source CSV has a malformed source: " + str(file))
                        destinations.add(nodes[-1])
                        count += 1
                endpoints[file.stem] = sorted(destinations, key=int)
                rows[file.stem] = count
        graph.mkdir(parents=True, exist_ok=True)
        write_double_json(graph / "vantage_dst_test.json", endpoints)
        write_json(paths / "source_holdout.json", dict(
            requested_sources=sorted(formal, key=int), training_source_paths=0,
            observed_source_files=[str(original / (s + ".csv")) for s in sorted(formal, key=int)],
            absent_sources=[], reference_kind="original_BGP_source_holdout"))
        (paths / "split_log.txt").write_text(
            "meta_test_as_num:10\nglobal_ratio:1.000000\nmeta_train_path_num:0.000000\n"
            + "holdout_sources:" + ",".join(sorted(formal, key=int)) + "\nmeta_files:\n"
            + "".join(str(original / (s + ".csv")) + "\n" for s in sorted(formal, key=int)))
        outputs += [graph / "vantage_dst_test.json", paths / "source_holdout.json", paths / "split_log.txt"]
        return outputs, dict(formal_rows=rows, original_source_count=len(source_files),
                             formal_originals_in_training=False)

    split_receipt = _phase(root, "SPLIT", inputs, split, progress)
    phase_inputs = {str(root / "SPLIT.json"): sha256(root / "SPLIT.json")}
    for name, folder in (("TRAIN_AGGREGATES", train), ("TEST_AGGREGATES", test)):
        def aggregates(folder=folder):
            detail = rebuild_test_aggregates(folder)
            return [folder / n for n in detail["files"]], detail
        _phase(root, name, phase_inputs, aggregates, progress)
    public = {a["id"]: select_asset(a) for a in registry["raw_inputs"]}
    asrank = public["asrank_asns"]
    graph_inputs = {str(train / "merged_paths.txt"): sha256(train / "merged_paths.txt"),
                    str(asrank): sha256(asrank), str(Path(__file__).resolve()): sha256(__file__)}

    def make_graph():
        detail = build_training_graph(train / "merged_paths.txt", graph,
            target / "AS_frequency.json", asrank, registry["auxiliary_dates"]["asrank"], formal)
        files = [graph / n for n in ("edge_frequency.json", "AS_Feature.json", "edge_feature.json", "vantage_dst.json")]
        return files + [target / "AS_frequency.json"], detail
    _phase(root, "GRAPH", graph_inputs, make_graph, progress)
    embedding = graph / (date + "_246_embeddings_collectedPath.emb")
    vector_inputs = {str(train / "expand_paths.txt"): sha256(train / "expand_paths.txt"),
                     str(graph / "AS_Feature.json"): sha256(graph / "AS_Feature.json"),
                     str(Path(__file__).resolve()): sha256(__file__)}

    def vectors():
        from gensim.models import Word2Vec
        from gensim.models.callbacks import CallbackAny2Vec

        class Routes:
            def __iter__(self):
                with (train / "expand_paths.txt").open() as stream:
                    for line in stream:
                        yield line.strip().split("->")

        class Epochs(CallbackAny2Vec):
            count = 0
            def on_epoch_end(self, model):
                self.count += 1
                progress("cohort_embedding_epoch_" + str(self.count) + "_of_5")

        # The legacy collectedPath Node2Vec.fit is Word2Vec over expanded TRAIN
        # paths, sg=1, dimension=246, window=10, min_count=1, batch_words=4.
        model = Word2Vec(Routes(), vector_size=246, window=10, min_count=1,
                         batch_words=4, workers=2, sg=1, seed=20261015,
                         epochs=5, callbacks=[Epochs()])
        model.wv.save_word2vec_format(str(embedding))
        del model
        detail = rebuild_as_embedding(embedding, graph / "AS_Feature.json",
                                       graph / "ASEmbedding_collectedPath_256.txt")
        detail.update(training_only=True, seed=20261015, workers=2, epochs=5)
        return [embedding, graph / "ASEmbedding_collectedPath_256.txt"], detail
    _phase(root, "EMBEDDING", vector_inputs, vectors, progress)
    outputs = {str(root / (n + ".json")): sha256(root / (n + ".json"))
               for n in ("SPLIT", "TRAIN_AGGREGATES", "TEST_AGGREGATES", "GRAPH", "EMBEDDING")}
    write_json(root / "COHORT.json", dict(status="COMPLETED", date=date,
        formal_sources=sorted(formal, key=int), formal_rows=split_receipt["details"]["formal_rows"],
        inputs_sha256=inputs, phase_receipts_sha256=outputs,
        graph_and_embeddings_from_training_only=True, external_files_modified=False))
    return local
