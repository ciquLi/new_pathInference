"""Explicit dated assets and independently derived downstream source views."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
import shutil
import sys

from pathveil.data.prepare import prepare_date
from pathveil.utils import project_root, read_json, sha256, write_json
from .contracts import freeze, freeze_bytes, require, verify_files


def prepare_data(date, run):
    data = prepare_date(date, run, deep=True)
    require(
        data["mode"] in {"bgp", "traceroute"} and data["date"] == data["routing_date"] == date,
        "The rho workflow requires an explicitly registered routing date",
    )
    registry = read_json(data["registry_path"])
    require(data["auxiliary_dates"] == registry["auxiliary_dates"], "Public snapshot registration differs")
    require(all(day <= date for day in data["auxiliary_dates"].values()), "Future public snapshot")
    return data


def downstream_view(data, run):
    """Rebuild source fragments proven by independent training parents and expansion."""
    run = Path(run)
    directory = run / "data/downstream_view"
    manifest = directory / "data_manifest.json"
    receipt = directory / "VIEW_COMPLETED.json"
    if receipt.exists():
        record = read_json(receipt)
        verify_files(record["inputs_sha256"])
        verify_files(record["outputs_sha256"])
        return manifest
    sources = set(map(str, data["sources"]))
    original_train = Path(data["train_dir"])
    merged = Path(data["merged_train"])
    expanded = original_train / "expand_paths.txt"
    inputs = {
        str(p): sha256(p) for p in [merged, expanded, original_train.parent / "split_log.txt"]
    }
    originals, derived, observed = {}, {}, {}
    for source in sources:
        path = original_train / (source + ".csv")
        rows = []
        if path.exists():
            with path.open() as stream:
                rows = [line.strip() for line in stream if line.strip()]
        require(len(rows) == len(set(rows)), "Repeated source CSV path: " + source)
        originals[source], derived[source], observed[source] = set(rows), set(), set()
        if path.exists():
            inputs[str(path)] = sha256(path)
    with merged.open() as stream:
        for line in stream:
            nodes = line.strip().split("->")
            require(nodes[0] not in sources, "Formal source-origin paths appear in training")
            for start, source in enumerate(nodes[:-1]):
                if source not in sources or start == 0:
                    continue
                require(len(nodes) == len(set(nodes)), "Cyclic independent parent route")
                derived[source].update(
                    "->".join(nodes[start:stop]) for stop in range(start + 2, len(nodes) + 1)
                )
    with expanded.open() as stream:
        for line in stream:
            source = line.split("->", 1)[0].strip()
            if source in sources:
                observed[source].add(line.strip())
    local_root = directory / "files"
    projected = (
        local_root / "data/aspath/path_by_vantage" / data["date"] / "path_without_prefix/train"
    )
    projected.mkdir(parents=True, exist_ok=True)
    differences = {}
    for source in sorted(sources, key=int):
        require(
            derived[source] == observed[source],
            "Training expansion lacks exact parent provenance: " + source,
        )
        require(
            originals[source] <= derived[source],
            "Source CSV contains unsupported derivatives: " + source,
        )
        missing = derived[source] - originals[source]
        # A held-out source may have no per-source training CSV at all. Its
        # independently observed transit fragments can have any legal length;
        # the parent/expansion equality above proves their training provenance.
        restored_lengths = Counter(len(path.split("->")) for path in missing)
        freeze_bytes(
            projected / (source + ".csv"),
            "".join(p + "\n" for p in sorted(derived[source])).encode(),
        )
        differences[source] = dict(
            original=len(originals[source]),
            normalized=len(derived[source]),
            original_csv_present=(original_train / (source + ".csv")).is_file(),
            restored_two_AS=restored_lengths.get(2, 0),
            restored_multihop_paths=sum(
                count for length, count in restored_lengths.items() if length > 2
            ),
            restored_by_AS_count={
                str(length): count for length, count in sorted(restored_lengths.items())
            },
        )
    result = copy.deepcopy(data)
    previous_root = Path(data["source_data_root"])
    result.update(
        source_data_root=str(local_root),
        train_dir=str(projected),
        expanded_train=str(expanded),
        split_log=str(original_train.parent / "split_log.txt"),
        downstream_input_view="exact_independent_forward_derivatives_v2",
    )
    for key in ("read_path_map", "path_prefix_map"):
        for name, value in data.get(key, {}).items():
            try:
                relative = Path(name).relative_to(previous_root)
            except ValueError:
                continue
            result.setdefault(key, {})[str(local_root / relative)] = value
    result.setdefault("path_prefix_map", {})[str(local_root)] = str(previous_root)
    for path in projected.glob("*.csv"):
        result.setdefault("read_path_map", {})[str(path)] = str(path)
    mirror = projected.parent / "test"
    mirror.mkdir(parents=True, exist_ok=True)
    for source in sorted(sources, key=int) if data.get("mode", "bgp") == "bgp" else []:
        original = Path(data["test_dir"]) / (source + ".csv")
        wanted = data["corpus_input_sha256"][str(original)]
        require(sha256(original) == wanted, "Reference mirror input changed")
        inputs[str(original)] = wanted
        destination = mirror / original.name
        if not destination.exists():
            temporary = destination.with_suffix(".copying")
            shutil.copyfile(original, temporary)
            require(sha256(temporary) == wanted, "Reference mirror bytes differ")
            temporary.replace(destination)
        require(sha256(destination) == wanted, "Reference mirror changed")
    freeze(manifest, result)
    outputs = {
        str(p): sha256(p) for p in [manifest, *projected.glob("*.csv"), *mirror.glob("*.csv")]
    }
    freeze(
        receipt,
        dict(
            status="PASS",
            date=data["date"],
            differences=differences,
            inputs_sha256=inputs,
            outputs_sha256=outputs,
            original_files_modified=False,
            reference_mirrors_are_byte_identical=True,
            final_labels_used_for_training=False,
        ),
    )
    return manifest


def verify_upstream(cache, data, config):
    from pathveil.upstream.contracts import validate_model_scope
    from pathveil.upstream.method import MethodLibrary

    cache = Path(cache).resolve()
    identity = read_json(cache / "upstream/IDENTITY.json")
    model_file = cache / "upstream/path_model/MODEL.json"
    model = read_json(model_file)
    bundle = read_json(cache / "upstream/NEIGHBOR_BUNDLE.json")
    require(
        identity["date"] == model["date"] == bundle["date"] == data["date"],
        "Cross-date upstream cache rejected",
    )
    require(
        set(map(str, identity["sources"])) == set(map(str, data["sources"])),
        "Upstream cache has another explicitly held-out cohort",
    )
    require(identity["method_config"] == config, "Upstream cache has another method configuration")
    require(identity["code"] == MethodLibrary().receipt(), "Upstream numerical source differs")
    require(
        identity["canonical_sha256"] == sha256(data["merged_train"]),
        "Canonical training input differs",
    )
    require(
        identity["public_files_sha256"]
        == {str(p): sha256(p) for p in data["public_files"].values()},
        "Upstream public input identities differ",
    )
    validate_model_scope(bundle, data["date"], identity["split"]["split"])
    require(
        not model["formal_labels_read"] and not model["validation_labels_read"],
        "Path model consumed held-out labels",
    )
    require(
        set(map(str, model["training_sources"])) == set(map(str, bundle["training_sources"])),
        "Path/neighbor fit roles differ",
    )
    require(
        not set(map(str, data["sources"])) & set(map(str, model["training_sources"])),
        "Formal source fitted upstream",
    )
    require(len(model["path_feature_names"]) == 255, "Expected frozen255 feature schema")
    verify_files({model["path_model"]: model["model_sha256"], **bundle["model_files_sha256"]})
    return dict(
        run=str(cache),
        date=data["date"],
        model=str(model["path_model"]),
        model_sha256=model["model_sha256"],
        bundle=str(cache / "upstream/NEIGHBOR_BUNDLE.json"),
        bundle_sha256=sha256(cache / "upstream/NEIGHBOR_BUNDLE.json"),
        identity_sha256=sha256(cache / "upstream/IDENTITY.json"),
        independently_verified=True,
    )


def obtain_upstream(run, data, registry, config):
    root, run = project_root(), Path(run)
    receipt = run / "UPSTREAM_BOUND.json"
    if receipt.exists():
        record = read_json(receipt)
        require(verify_upstream(record["run"], data, config) == record, "Upstream binding changed")
        return record
    registered = registry.get("upstream_cache")
    candidate = (root / registered).resolve() if registered else None
    if (
        candidate
        and (candidate / "upstream/NEIGHBOR_BUNDLE.json").exists()
        and (candidate / "upstream/path_model/MODEL.json").exists()
    ):
        record = verify_upstream(candidate, data, config)
    else:
        from pathveil.cli import run_identity
        from pathveil.upstream.entry import run_upstream

        candidate = run / "upstream_training"
        candidate.mkdir(parents=True, exist_ok=True)
        freeze(candidate / "RUN_IDENTITY.json", run_identity(data["date"], registry, config))
        freeze(candidate / "method_config.json", config)
        freeze(candidate / "data/data_manifest.json", data)
        try:
            run_upstream(candidate / "data/data_manifest.json", candidate, config)
        except Exception as error:
            if (
                type(error).__name__ != "UpstreamError"
                or str(error) != "Only explicit date-local source-disjoint models are accepted"
            ):
                raise
            scripts = str(root / "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)
            from normalize_neighbor_protocol import normalize_run

            normalize_run(candidate, project_root=root)
            run_upstream(candidate / "data/data_manifest.json", candidate, config)
        record = verify_upstream(candidate, data, config)
    freeze(receipt, record)
    return record
