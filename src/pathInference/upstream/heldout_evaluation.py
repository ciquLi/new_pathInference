"""Freeze full-model neighbor predictions before opening held-out origin labels.

These finite-observation metrics describe support in the original BGP corpus.
An unobserved selected neighbor remains unconfirmed; it is not a false edge.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from .contracts import read, require, sha, verify, write
from .implementation.neighbor_features import read_reference


_ROOT_FAMILIES = (
    "root_pdb_rank",
    "root_struct_rank",
    "root_pdb_present",
    "root_struct_present",
)
_ROLES = ("train", "validation", "test", "formal_test")
_METRICS = (
    "selected_neighbor_observed_support",
    "all_firsthop_recall",
    "destination_balanced_top3_recall",
    "destination_balanced_top10_recall",
    "destination_balanced_top3_mass_recall",
    "destination_balanced_top10_mass_recall",
    "destination_balanced_selected_mass",
    "candidate_observed_neighbor_coverage",
    "candidate_destination_balanced_mass",
    "candidate_top3_coverage",
    "candidate_top10_coverage",
)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _sources(values, label):
    require(isinstance(values, (list, tuple)), "Missing source list: " + label)
    result = []
    for source in values:
        require(not isinstance(source, bool), "Invalid ASN in " + label)
        try:
            number = int(source)
        except (TypeError, ValueError):
            require(False, "Invalid ASN in " + label)
        require(
            str(source) == str(number) and 0 < number < 2**32,
            "Invalid ASN in " + label,
        )
        result.append(str(number))
    require(len(result) == len(set(result)), "Duplicate ASN in " + label)
    return result


def _audit_roles(data, split_info, runtime):
    require(
        str(split_info.get("date", data["date"])) == str(data["date"]),
        "Held-out split date differs from the experiment",
    )
    split = split_info.get("split", split_info)
    roles = {role: _sources(split.get(role), role) for role in _ROLES}
    require(all(roles[role] for role in _ROLES[:3]), "Train/validation/test must be nonempty")
    intersections = {}
    for index, first in enumerate(_ROLES):
        for second in _ROLES[index + 1 :]:
            overlap = sorted(set(roles[first]) & set(roles[second]), key=int)
            intersections[first + "__" + second] = overlap
            require(not overlap, "Source roles overlap: " + first + "/" + second)
    bundle = runtime.bundle
    train = set(roles["train"])
    require(
        set(map(str, bundle.get("training_sources", ()))) == train,
        "Neighbor model training sources differ from the frozen split",
    )
    folds = {str(source) for source in bundle.get("source_to_fold", {})}
    require(folds == train, "Neighbor fold keys differ from training sources")
    require(
        {str(source) for source in runtime.folds} == train,
        "Runtime folds differ from the frozen training sources",
    )
    formal = set(roles["formal_test"])
    if "sources" in data:
        registered = set(_sources(data["sources"], "registered formal sources"))
        require(registered == formal, "Registered formal cohort differs from formal_test")
    heldout = roles["validation"] + roles["test"]
    for source in heldout:
        require(runtime._head_key(int(source)) == "full", "Held-out inference must use full heads")
    require(
        bundle.get("date") == data["date"] and bundle.get("model_fit_date") == data["date"],
        "Held-out neighbor model date differs from the experiment",
    )
    require(
        bundle.get("formal_labels_read") is False
        and bundle.get("validation_labels_read") is False
        and bundle.get("test_labels_read", False) is False,
        "Model bundle declares held-out label access",
    )
    return roles, dict(
        status="PASS",
        source_roles=roles,
        role_counts={role: len(sources) for role, sources in roles.items()},
        pairwise_intersections=intersections,
        model_training_sources_equal_split=True,
        heldout_sources_absent_from_all_oof_folds=True,
        all_heldout_model_heads="full",
        formal_cohort_included_in_neighbor_evaluation=False,
    )


def _model_hashes(bundle):
    files = {
        str(Path(path).resolve()): expected
        for path, expected in bundle.get("model_files_sha256", {}).items()
    }
    require(files, "Neighbor bundle has no frozen model file hashes")
    verify(files)
    full = {}
    for family in _ROOT_FAMILIES:
        filename = bundle.get("models", {}).get(family, {}).get("full")
        require(filename is not None, "Missing full root model: " + family)
        path = str(Path(filename).resolve())
        require(path in files, "Full root model has no frozen hash: " + family)
        full[family] = dict(path=path, sha256=files[path])
    return files, full


class _TruthReadBarrier:
    """CPython open-event guard; this is an ordering audit, not OS isolation."""

    def __init__(self, canonical):
        self.canonical = canonical
        self.enabled = False

    def check(self, event, args):
        if not self.enabled or event != "open" or not args:
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1]
        reading = (
            "r" in mode or "+" in mode
            if isinstance(mode, str)
            else (int(args[2]) & os.O_ACCMODE) != os.O_WRONLY
        )
        if reading:
            require(
                Path(os.fsdecode(args[0])).resolve() != self.canonical,
                "Held-out canonical labels cannot be read before all predictions are frozen",
            )


def _freeze_predictions(data, split_info, runtime, out, roles, model_files, full_models):
    canonical = Path(data["merged_train"]).resolve()
    require(canonical.is_file(), "Canonical original stream is missing")
    availability = split_info.get("source_availability", {})
    original = split_info.get("canonical_evidence_input", {})
    registered = [
        value
        for value in (availability.get("canonical_sha256"), original.get("sha256"))
        if value is not None
    ]
    require(
        len(set(registered)) <= 1,
        "Canonical hashes registered by the source split disagree",
    )
    expected = registered[0] if registered else None
    for filename in (availability.get("canonical_path"), original.get("path")):
        if filename is not None:
            require(
                Path(filename).resolve() == canonical,
                "Canonical path registered by the source split differs from merged_train",
            )
    require(
        isinstance(expected, str)
        and len(expected) == 64
        and all(c in "0123456789abcdef" for c in expected),
        "Held-out evaluation needs the split's canonical SHA256 before prediction",
    )
    code = [
        Path(__file__),
        Path(__file__).with_name("neighbors.py"),
        Path(__file__).with_name("contracts.py"),
        Path(__file__).parent / "implementation/neighbor_features.py",
    ]
    binding = dict(
        schema_version=1,
        date=str(data["date"]),
        split_sha256=_digest(split_info),
        bundle_sha256=_digest(runtime.bundle),
        method_config_sha256=_digest(getattr(runtime, "config", {})),
        model_files_sha256=model_files,
        full_root_models=full_models,
        canonical_path=str(canonical),
        canonical_sha256=expected,
        code_sha256={str(path.resolve()): sha(path) for path in code},
    )
    predictions_path, marker_path = out / "PREDICTIONS.json", out / "PREDICTIONS_COMPLETED.json"
    barrier = _TruthReadBarrier(canonical)
    sys.addaudithook(barrier.check)
    barrier.enabled = True
    try:
        if predictions_path.exists():
            require(marker_path.is_file(), "Prediction freeze is incomplete; choose a fresh outdir")
            previous, marker = read(predictions_path), read(marker_path)
            require(previous["binding"] == binding, "Held-out prediction cache identity changed")
            require(
                marker.get("prediction_sha256") == sha(predictions_path)
                and marker.get("binding_sha256") == _digest(binding)
                and marker.get("status") == "PASS"
                and marker.get("frozen_before_truth_read") is True,
                "Frozen held-out predictions changed",
            )
            return previous, canonical
        require(not marker_path.exists(), "Prediction marker has no frozen predictions")
        predictions = {}
        for role in ("validation", "test"):
            for source in sorted(roles[role], key=int):
                prediction = runtime.infer_root(int(source))
                require(
                    prediction.get("original_source_model_head") == "full"
                    and prediction.get("source_labels_excluded_from_predictive_heads") is True,
                    "Held-out prediction does not attest full model and source-label exclusion",
                )
                selected = _sources(prediction.get("selected_neighbors"), "selected neighbors")
                root = prediction.get("root", {})
                candidates = (
                    _sources(root["candidates"], "root candidates")
                    if "candidates" in root
                    else None
                )
                if candidates is not None:
                    require(
                        set(selected) <= set(candidates), "Selected neighbor absent from candidates"
                    )
                predictions[source] = dict(
                    source=source,
                    role=role,
                    model_head="full",
                    selected_neighbors=selected,
                    candidates=candidates,
                    rankings={
                        str(name): _sources(order, "root ranking")
                        for name, order in root.get("rankings", {}).items()
                    },
                    inference_status=prediction.get("status", "predicted"),
                    evidence_sha256=prediction.get("evidence_sha256"),
                    source_labels_excluded_from_predictive_heads=True,
                )
        require(
            _digest(runtime.bundle) == binding["bundle_sha256"],
            "Model bundle changed during prediction",
        )
        require(
            _digest(getattr(runtime, "config", {})) == binding["method_config_sha256"],
            "Neighbor configuration changed during prediction",
        )
        verify(model_files)
        frozen = dict(
            binding=binding, predictions=predictions, all_predictions_frozen_before_truth=True
        )
        write(predictions_path, frozen)
        write(
            marker_path,
            dict(
                status="PASS",
                prediction_sha256=sha(predictions_path),
                binding_sha256=_digest(binding),
                frozen_before_truth_read=True,
            ),
        )
        return frozen, canonical
    finally:
        barrier.enabled = False


def _extract_heldout(canonical, sources, directory):
    """Read the original stream once; bound open writers, not source identities."""
    sources = set(map(int, sources))
    handles = OrderedDict()
    digest = hashlib.sha256()
    counts = {str(source): 0 for source in sources}
    total = malformed_origin = 0
    try:
        with canonical.open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                total += 1
                try:
                    source = int(raw.split(b"->", 1)[0])
                except ValueError:
                    malformed_origin += 1
                    continue
                if source not in sources:
                    continue
                if source in handles:
                    handle = handles.pop(source)
                else:
                    if len(handles) >= 64:
                        _, old = handles.popitem(last=False)
                        old.close()
                    handle = (directory / (str(source) + ".txt")).open("ab")
                handles[source] = handle
                handle.write(raw)
                counts[str(source)] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return dict(
        canonical_sha256=digest.hexdigest(),
        canonical_lines=total,
        malformed_origin_lines=malformed_origin,
        extracted_original_source_rows=counts,
        selection="actual first ASN in original merged_train; no suffix expansion or per-source CSV lookup",
    )


def _score_source(prediction, reference, canonical, extracted_rows):
    weights = {str(source): float(weight) for source, weight in reference["first"].items()}
    require(
        reference["counts"].get("paths", 0) > 0 and weights,
        "Held-out source has no valid original reference: " + prediction["source"],
    )
    observed, selected = set(weights), set(prediction["selected_neighbors"])
    order = sorted(weights, key=lambda source: (-round(weights[source], 12), int(source)))
    confirmed = selected & observed
    metrics = dict(
        selected_neighbor_observed_support=len(confirmed) / len(selected) if selected else None,
        all_firsthop_recall=len(confirmed) / len(observed),
        destination_balanced_selected_mass=math.fsum(
            weights.get(source, 0.0) for source in selected
        ),
    )
    for size in (3, 10):
        target = set(order[:size])
        covered = target & selected
        metrics["destination_balanced_top%d_recall" % size] = len(covered) / len(target)
        metrics["destination_balanced_top%d_mass_recall" % size] = math.fsum(
            weights[source] for source in covered
        ) / math.fsum(weights[source] for source in target)
    candidates = prediction["candidates"]
    candidate_set = set(candidates) if candidates is not None else None
    metrics.update(
        candidate_observed_neighbor_coverage=len(candidate_set & observed) / len(observed)
        if candidate_set is not None
        else None,
        candidate_destination_balanced_mass=math.fsum(
            weights.get(source, 0.0) for source in candidate_set
        )
        if candidate_set is not None
        else None,
    )
    for size in (3, 10):
        target = set(order[:size])
        metrics["candidate_top%d_coverage" % size] = (
            len(candidate_set & target) / len(target) if candidate_set is not None else None
        )
    return dict(
        status="PASS",
        source=prediction["source"],
        role=prediction["role"],
        metrics=metrics,
        counts=dict(
            selected_neighbors=len(selected),
            observed_firsthop_neighbors=len(observed),
            confirmed_selected_neighbors=len(confirmed),
            unconfirmed_selected_neighbors=len(selected - observed),
            candidates=len(candidate_set) if candidate_set is not None else None,
            reference_paths=reference["counts"]["paths"],
            destinations=reference["destinations"],
        ),
        selected_neighbor_evidence={
            source: "observed" if source in observed else "unconfirmed"
            for source in prediction["selected_neighbors"]
        },
        confirmed_selected_neighbors=sorted(confirmed, key=int),
        unconfirmed_selected_neighbors=sorted(selected - observed, key=int),
        destination_balanced_firsthop_weights=weights,
        destination_balanced_reference_order=order,
        truth_provenance=dict(
            canonical_path=str(canonical),
            actual_original_source=prediction["source"],
            canonical_source_bytes_sha256=reference["sha256"],
            extracted_original_source_rows=extracted_rows,
            parser_counts=reference["counts"],
            parser="implementation.neighbor_features.read_reference",
            deduplication="full valid loop-free AS path, not source/firsthop/destination pair",
            destination_weight="Each observed destination has equal mass; its distinct full paths share that mass equally",
        ),
        unobserved_selected_neighbors_are_false_edges=False,
    )


def _macro(sources, rows):
    values, denominators = {}, {}
    for name in _METRICS:
        available = [
            rows[source]["metrics"][name]
            for source in sources
            if rows[source]["metrics"][name] is not None
        ]
        denominators[name] = len(available)
        values[name] = math.fsum(available) / len(available) if available else None
    return dict(
        sources=sources,
        source_count=len(sources),
        macro=values,
        macro_source_denominators=denominators,
    )


def evaluate_heldout_neighbors(data, split_info, runtime, outdir):
    """Return summary.json after prediction-first validation/test evaluation.

    This function neither trains nor selects a checkpoint. PASS certifies the
    completed evaluation and provenance checks, never an accuracy threshold.
    """
    data, split_info = dict(data), dict(split_info)
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    roles, role_audit = _audit_roles(data, split_info, runtime)
    model_files, full_models = _model_hashes(runtime.bundle)
    frozen, canonical = _freeze_predictions(
        data, split_info, runtime, out, roles, model_files, full_models
    )
    expected_sha = frozen["binding"]["canonical_sha256"]
    prediction_sha = sha(out / "PREDICTIONS.json")
    summary_path, marker_path = out / "summary.json", out / "AUDIT_COMPLETED.json"
    if marker_path.exists():
        marker = read(marker_path)
        require(
            marker.get("binding_sha256") == _digest(frozen["binding"])
            and marker.get("prediction_sha256") == prediction_sha,
            "Held-out evaluation cache identity changed",
        )
        require(sha(canonical) == expected_sha, "Held-out canonical original stream changed")
        verify(marker["outputs_sha256"])
        return summary_path
    rows, outputs = {}, {}
    with tempfile.TemporaryDirectory(prefix="heldout_origins_", dir=str(out)) as temporary:
        directory = Path(temporary)
        extraction = _extract_heldout(canonical, frozen["predictions"], directory)
        require(
            extraction["canonical_sha256"] == expected_sha,
            "Held-out canonical original stream changed",
        )
        for source, prediction in frozen["predictions"].items():
            path = directory / (source + ".txt")
            require(path.is_file(), "Held-out source has no original reference: " + source)
            reference = read_reference(path, int(source))
            result = _score_source(
                prediction,
                reference,
                canonical,
                extraction["extracted_original_source_rows"][source],
            )
            result["prediction_provenance"] = dict(
                prediction_file=str(out / "PREDICTIONS.json"),
                prediction_sha256=prediction_sha,
                source_prediction_sha256=_digest(prediction),
                model_head="full",
                model_bundle_sha256=frozen["binding"]["bundle_sha256"],
                full_root_models=full_models,
                split_sha256=frozen["binding"]["split_sha256"],
                canonical_sha256=expected_sha,
            )
            path = out / ("source_" + source + ".json")
            write(path, result)
            outputs[str(path)] = sha(path)
            rows[source] = result
    verify(model_files)
    summary = dict(
        status="PASS",
        date=str(data["date"]),
        evaluation="prediction_first_full_head_original_origin_neighbor_evaluation_v1",
        pass_semantics="Evaluation and provenance checks completed; no metric threshold is implied",
        binding=frozen["binding"],
        role_audit=role_audit,
        validation=_macro(roles["validation"], rows),
        test=_macro(roles["test"], rows),
        per_source={source: str(out / ("source_" + source + ".json")) for source in rows},
        truth_provenance=extraction,
        prediction_file=str(out / "PREDICTIONS.json"),
        prediction_sha256=prediction_sha,
        all_predictions_frozen_before_truth=True,
        final_formal_route_labels_read=False,
        formal_cohort_included_in_neighbor_evaluation=False,
        checkpoint_selection_performed=False,
        test_metrics_may_select_checkpoint=False,
        finite_observation_support_only=True,
        metric_definitions=dict(
            selected_neighbor_observed_support="Observed selected first-hop neighbors / selected neighbors; null for no selected neighbors. Unobserved selections are unconfirmed, not false edges.",
            all_firsthop_recall="Selected observed first-hop neighbors / all distinct observed first-hop neighbors.",
            destination_balanced_topk_recall="Unweighted recall of the strict first min(k,N) reference neighbors ranked by destination-balanced mass; ties use rounded-12 mass then ascending ASN.",
            destination_balanced_topk_mass_recall="Selected reference Top-k mass / all reference Top-k mass.",
            destination_balanced_selected_mass="Sum of the selected neighbors' destination-balanced first-hop mass over all observed reference neighbors.",
            candidate_coverage="Coverage of the same observed neighbor sets/mass by the frozen candidate list; null when unavailable.",
            macro="Unweighted mean over sources with a defined metric; every metric reports its source denominator.",
        ),
    )
    write(summary_path, summary)
    for path in [summary_path, out / "PREDICTIONS.json", out / "PREDICTIONS_COMPLETED.json"]:
        outputs[str(path)] = sha(path)
    write(
        marker_path,
        dict(
            status="PASS",
            binding_sha256=_digest(frozen["binding"]),
            prediction_sha256=prediction_sha,
            canonical_sha256=expected_sha,
            summary_sha256=sha(summary_path),
            outputs_sha256=outputs,
        ),
    )
    return summary_path
