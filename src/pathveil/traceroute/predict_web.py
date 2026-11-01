#!/usr/bin/env python3
"""Predict unique AS-pair queries from a target-only CSV; never read traces."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

from .common import (
    assert_source_held_out,
    digest,
    json_write,
    project_imports,
    vocab_digest,
    load_contract,
    install_label_guard,
    audit_pool_consumers,
)


def read_targets(path, default_source):
    targets, seen = [], set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"target_ip", "destination_asn"} <= set(reader.fieldnames or []):
            raise ValueError("targets.csv requires target_ip,destination_asn columns")
        forbidden = {"hops", "as_segments", "as_path", "truth", "traceroute_path"} & set(
            reader.fieldnames or []
        )
        if forbidden:
            raise ValueError("Target inventory must not contain path labels: " + str(forbidden))
        for row in reader:
            source = str(row.get("source_asn") or default_source)
            target_ip = row["target_ip"]
            if (source, target_ip) in seen:
                raise ValueError(
                    "Duplicate source/target-IP in inventory: " + str((source, target_ip))
                )
            seen.add((source, target_ip))
            destination = row["destination_asn"].strip()
            if destination and (
                not destination.isdigit() or not 0 < int(destination) <= 4294967295
            ):
                raise ValueError("destination_asn must be single ASN or empty")
            targets.append(
                {
                    "target_ip": target_ip,
                    "source_asn": source,
                    "destination_asn": destination or None,
                    "domain_count": int(row.get("domain_count") or 0),
                    "mapping_status": row.get("mapping_status")
                    or ("single_origin" if destination else "unresolved"),
                }
            )
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--support-data-dir", required=True, type=Path)
    parser.add_argument("--selected-neighbors-csv", required=True, type=Path)
    parser.add_argument("--recursive-anchors-csv", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--date", default="20261015")
    parser.add_argument("--source-asn", default="63835")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--prefix-tree-dir", type=Path)
    parser.add_argument("--destination-batch-size", type=int, default=256)
    parser.add_argument("--dgi-threshold", type=float, default=4.0)
    parser.add_argument(
        "--local-knowledge-mode", default="augmentation_only", choices=["augmentation_only"]
    )
    args = parser.parse_args()
    for name in (
        "targets",
        "support_data_dir",
        "selected_neighbors_csv",
        "recursive_anchors_csv",
        "model",
        "outdir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    targets = read_targets(args.targets, args.source_asn)
    if any(row["source_asn"] != args.source_asn for row in targets):
        raise ValueError(
            "One source-specific model per invocation; inventory contains another source"
        )
    if args.destination_batch_size < 1:
        parser.error("destination-batch-size must be positive")
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    contract = load_contract(args.data_manifest, args.release, args.source_asn)
    if (
        str(args.selected_neighbors_csv) != contract["selected_neighbors_csv"]
        or str(args.recursive_anchors_csv) != contract["recursive_anchors_csv"]
    ):
        raise ValueError("Prediction topology differs from the released graph")
    project_imports(args.data_manifest)
    install_label_guard(args.outdir)
    import torch
    from pathveil.downstream.device_runtime import configure_device

    device, _ = configure_device(torch, args.device)
    args.device = str(device)
    import environment
    import getData
    import myDataset
    import HighFrequencyVertexBasedPathGen as generator

    if contract["date"] != args.date:
        raise ValueError("Trace routing date and release differ")
    env = environment.Env(args.date, args.date)
    audit = assert_source_held_out(env, args.source_asn)
    env.support_data_dir = str(args.support_data_dir) + "/"
    env.graph_selected_neighbors_csv = str(args.selected_neighbors_csv)
    env.graph_recursive_anchors_csv = str(args.recursive_anchors_csv)
    env.augment_graph_from_pipeline_priors()
    env.k = args.k
    env.observed_path_num = args.k
    env.as_type = "devoid"
    env.dataType = "test"
    env.pathGenAlgorithm = "PG"
    env.length_border = args.dgi_threshold
    env.pg_local_knowledge_mode = args.local_knowledge_mode
    env.prefix_level = False
    env.train_extra_data_mode = "augmentation_only"
    pool_audit = audit_pool_consumers(env, args.source_asn, contract)
    env.existPath = env.getExistPath("train")
    if args.prefix_tree_dir is not None:
        from pathveil.prefix_tree.traceroute import install_generator
        install_generator(args.prefix_tree_dir, args.data_manifest, contract, args.outdir)
    model = torch.load(str(args.model), map_location=args.device)
    metadata = getattr(model, "_pathveil_metadata", None)
    if (
        not metadata
        or metadata.get("admitted_path_set_sha256") != contract["path_set_sha256"]
        or metadata.get("release_sha256") != contract["release_sha256"]
        or metadata.get("data_manifest_sha256") != contract["data_manifest_sha256"]
    ):
        raise ValueError("Checkpoint admitted pool or release hash mismatch")
    if (
        metadata["date"] != args.date
        or metadata["source_asn"] != args.source_asn
        or metadata["vocab_sha256"] != vocab_digest(env)
    ):
        raise ValueError("Checkpoint date/source/vocabulary does not match inference environment")
    if metadata["k"] != args.k:
        raise ValueError("Checkpoint candidate-count configuration mismatch")
    history_argument_order = metadata.get("history_argument_order")
    input_version = getattr(model, "input_semantics_version", None)
    if (
        input_version != 2
        or metadata.get("input_semantics_version") != 2
        or history_argument_order != "corrected"
    ):
        raise ValueError(
            "Canonical prediction requires input semantics v2 and corrected history arguments"
        )
    history_path_index, history_hop_index = 4, 5
    model.to(args.device).eval()
    tensorizer = myDataset.IterableDatasetShuffle(
        "", env, metadata["max_length"], args.k, 1, 0, False
    )
    destinations = sorted(
        {row["destination_asn"] for row in targets if row["destination_asn"]}, key=int
    )
    nearest_asrank = env._nearest_asrank_link_file()
    config = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "targets_sha256": digest(args.targets),
        "model_sha256": digest(args.model),
        "model_metadata": metadata,
        "source_holdout_audit": audit,
        "history_argument_order": history_argument_order,
        "input_semantics_version": input_version,
        "selected_neighbors_sha256": digest(args.selected_neighbors_csv),
        "recursive_anchors_sha256": digest(args.recursive_anchors_csv),
        "asrank_link_file_used_for_external_nodes": str(nearest_asrank) if nearest_asrank else None,
        "target_count": len(targets),
        "unique_destination_as_count": len(destinations),
        "prediction_granularity": "source AS / destination AS; endpoints within a destination AS share one prediction",
        "traceroute_input": False,
        "pool_contract": contract,
        "pool_consumer_audit": pool_audit,
        "status": "running",
    }
    json_write(args.outdir / "prediction_manifest.json", config)
    results = {}

    def base_result(dest):
        return {
            "source_asn": args.source_asn,
            "destination_asn": dest,
            "predicted_path": [],
            "candidate_count": 0,
            "observed_path_count": 0,
            "discarded_unknown_vocab_candidates": 0,
            "discarded_long_candidates": 0,
            "discarded_invalid_endpoint_candidates": 0,
        }

    with (args.outdir / "pair_predictions.jsonl").open("x") as out:
        for start in range(0, len(destinations), args.destination_batch_size):
            batch = destinations[start : start + args.destination_batch_size]
            # Existing PG can extend topology for external destination nodes.
            env.augment_graph_for_external_nodes(batch)
            pending = []
            for dest in batch:
                result = base_result(dest)
                if args.source_asn not in env.stoi or dest not in env.stoi:
                    result["status"] = "unsupported_vocabulary"
                elif dest == args.source_asn:
                    result.update(
                        status="same_as", predicted_path=[args.source_asn], candidate_count=1
                    )
                elif env.G.has_edge(args.source_asn, dest):
                    result.update(
                        status="direct_graph_edge",
                        predicted_path=[args.source_asn, dest],
                        candidate_count=1,
                    )
                else:
                    pending.append(dest)
                results[dest] = result
            generated = (
                generator.generate_kpath_nosave(env, args.k, args.source_asn, pending)
                if pending
                else {}
            )
            for dest in pending:
                result = results[dest]
                raw_candidates = generated.get(args.source_asn + "-" + dest, [])[: args.k]
                candidates, seen = [], set()
                for candidate in raw_candidates:
                    path = [str(node) for node in candidate]
                    if len(path) > metadata["max_length"]:
                        result["discarded_long_candidates"] += 1
                        continue
                    if any(node not in env.stoi for node in path):
                        result["discarded_unknown_vocab_candidates"] += 1
                        continue
                    if len(path) < 2 or path[0] != args.source_asn or path[-1] != dest:
                        # The legacy generator can emit an internally valid
                        # walk for a requested destination after graph
                        # augmentation, but with stale endpoints.  Such a
                        # walk cannot be scored for this query; discard it
                        # and continue evaluating the remaining candidates.
                        result["discarded_invalid_endpoint_candidates"] += 1
                        continue
                    if tuple(path) not in seen:
                        candidates.append(path)
                        seen.add(tuple(path))
                result["candidate_count"] = len(candidates)
                if not candidates:
                    result["status"] = (
                        "unsupported_candidates" if raw_candidates else "no_candidate"
                    )
                    continue
                observed, _ = getData.getObservedPaths(env, [args.source_asn, dest])
                observed = [
                    path
                    for path in observed
                    if 2 <= len(path) <= metadata["max_length"]
                    and all(node in env.stoi for node in path)
                ][: args.k]
                result["observed_path_count"] = len(observed)
                indices = lambda path: "->".join(str(env.stoi[node]) for node in path)
                row = [
                    "|".join(indices(path) for path in candidates),
                    "|".join(indices(path) for path in observed),
                    "|".join(indices([path[0], path[-1]]) for path in observed),
                    "|".join("0" for _ in candidates),
                    args.source_asn + "->" + dest,
                ]
                tensors = tensorizer.convertToTensor(row)
                with torch.no_grad():
                    moved = [
                        torch.as_tensor(tensors[i]).unsqueeze(0).to(args.device) for i in range(7)
                    ]
                    scores = model(
                        kpathsEmbedding=moved[0],
                        k_lastSecHops=moved[1],
                        kcurASPair=moved[2],
                        hcurASPair=moved[3],
                        hispathEmbedding=moved[history_path_index],
                        his_lastSecHops=moved[history_hop_index],
                        hisASPairsEmbedding=moved[6],
                    ).reshape(-1)
                    # Padding rows are not candidate routes and must never be selected.
                    scores = scores[: len(candidates)]
                    if not bool(torch.isfinite(scores).all()):
                        result["status"] = "nonfinite_model_output"
                        continue
                    selected = int(torch.argmax(scores).item())
                result.update(
                    status="predicted",
                    predicted_path=candidates[selected],
                    selected_candidate_index=selected,
                    model_score=float(scores[selected].item()),
                )
            for dest in batch:
                out.write(json.dumps(results[dest], sort_keys=True) + "\n")
            out.flush()
            print(
                json.dumps(
                    {
                        "processed_destination_as": min(start + len(batch), len(destinations)),
                        "total_destination_as": len(destinations),
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    statuses = Counter()
    with (args.outdir / "target_predictions.jsonl").open("x") as out:
        for target in targets:
            result = results.get(
                target["destination_asn"],
                {
                    "predicted_path": [],
                    "status": "unresolved_destination_mapping",
                    "candidate_count": 0,
                },
            )
            merged = {**target, **result}
            statuses[merged["status"]] += 1
            out.write(json.dumps(merged, sort_keys=True) + "\n")
    config.update(
        status="complete",
        elapsed_seconds=time.time() - started,
        target_status_counts=dict(statuses),
        pair_predictions_sha256=digest(args.outdir / "pair_predictions.jsonl"),
        target_predictions_sha256=digest(args.outdir / "target_predictions.jsonl"),
    )
    json_write(args.outdir / "prediction_manifest.json", config)


if __name__ == "__main__":
    main()
