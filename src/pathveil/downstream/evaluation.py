"""Independent standard-library scoring on the complete frozen query universe."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path

from .common import pair_identity, require
from pathveil.utils import read_json, sha256, write_json


def canonical_pair(text, source=None):
    fields = text.replace("->", "-").split("-")
    require(
        len(fields) == 2
        and all(x.isascii() and x.isdecimal() and 0 < int(x) < 2**32 for x in fields),
        "Malformed prediction query: " + text,
    )
    require(source is None or fields[0] == str(source), "Prediction query has wrong source")
    return "->".join(fields)


def reference_identity(path, source):
    """Register only endpoints and a byte hash; retain no reference sequences."""
    pairs, observations = set(), 0
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                raise ValueError("Empty reference row")
            text = next(csv.reader([raw.decode().strip()], strict=True))[0]
            first, separator, tail = text.partition("->")
            require(
                separator and first == str(source),
                "Reference origin differs from registered source",
            )
            pairs.add(canonical_pair(first + "->" + tail.rsplit("->", 1)[-1], source))
            observations += 1
    require(pairs, "Empty fixed reference universe")
    return dict(
        pairs=len(pairs),
        reference_sha256=digest.hexdigest(),
        pair_inventory_sha256=pair_identity(pairs),
        reference_file=str(Path(path).resolve()),
        observations=observations,
        registration="Only endpoints and raw SHA256; reference sequences not retained",
    )


def read_references(path, source):
    truth, count = defaultdict(set), 0
    with Path(path).open(newline="") as stream:
        for number, row in enumerate(csv.reader(stream, strict=True), 1):
            require(row and row[0], "Empty reference at row " + str(number))
            route = tuple(row[0].strip().split("->"))
            require(
                len(route) >= 2
                and route[0] == str(source)
                and all(n.isascii() and n.isdecimal() and 0 < int(n) < 2**32 for n in route),
                "Invalid fixed reference route at row " + str(number),
            )
            truth[route[0] + "->" + route[-1]].add(route)
            count += 1
    require(truth, "Empty fixed reference")
    return dict(truth), count


def score_source(
    reference_file, prediction_file, source, expected_identity=None, reported_summary=None
):
    source = str(source)
    truth, observations = read_references(reference_file, source)
    identity = dict(
        pairs=len(truth),
        reference_sha256=sha256(reference_file),
        pair_inventory_sha256=pair_identity(truth),
    )
    if expected_identity:
        for key in ("pairs", "reference_sha256", "pair_inventory_sha256"):
            require(
                identity[key] == expected_identity[key], "Fixed reference identity changed: " + key
            )
    scores, rounded, exact = {}, {}, {}
    statuses = Counter()
    valid_count = invalid_count = rows = 0
    with Path(prediction_file).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            {"pair", "predicted_path"} <= set(reader.fieldnames or []),
            "Prediction CSV lacks required columns",
        )
        for row in reader:
            pair = canonical_pair(row["pair"], source)
            require(
                pair in truth and pair not in scores,
                "Unexpected or duplicated prediction query: " + pair,
            )
            text = row["predicted_path"].strip()
            route = tuple(text.split("->")) if text else ()
            valid = bool(route) and all(
                n.isascii() and n.isdecimal() and 0 < int(n) < 2**32 for n in route
            )
            valid = valid and len(route) == len(set(route)) and route[0] + "->" + route[-1] == pair
            predicted = frozenset(route)
            score = (
                max(
                    (
                        len(predicted & set(actual)) / len(predicted | set(actual))
                        for actual in truth[pair]
                    ),
                    default=0.0,
                )
                if valid
                else 0.0
            )
            score3 = round(score, 3)
            for key, expected in [
                ("jaccard", score),
                ("jaccard_rounded3", score3),
                ("endpoint_valid", int(valid)),
            ]:
                if row.get(key, "") != "":
                    require(
                        math.isclose(float(row[key]), expected, rel_tol=0, abs_tol=1e-9),
                        "Recorded prediction metric differs from independent scoring: "
                        + pair
                        + ":"
                        + key,
                    )
            scores[pair], rounded[pair], exact[pair] = (
                score,
                score3,
                int(valid and route in truth[pair]),
            )
            statuses[row.get("status") or "unspecified"] += 1
            valid_count += int(valid)
            invalid_count += int(bool(text) and not valid)
            rows += 1
    missing = set(truth) - set(scores)
    total = len(truth)
    result = dict(
        source_asn=source,
        query_count=total,
        reference_observations=observations,
        reference_unique_paths=sum(map(len, truth.values())),
        prediction_rows=rows,
        valid_predictions=valid_count,
        invalid_nonempty_predictions=invalid_count,
        missing_predictions=len(missing),
        missing_prediction_policy="zero; fixed denominator preserved",
        statuses=dict(statuses),
        jaccard_accuracy=math.fsum(scores.values()) / total,
        jaccard_accuracy_rounded3=math.fsum(rounded.values()) / total,
        total_jaccard_reward=math.fsum(scores.values()),
        total_jaccard_reward_rounded3=math.fsum(rounded.values()),
        ordered_exact_matches=sum(exact.values()),
        ordered_exact_accuracy=sum(exact.values()) / total,
        coverage=valid_count / total,
        fixed_identity=identity,
        reference_file=str(Path(reference_file).resolve()),
        prediction_file=str(Path(prediction_file).resolve()),
        prediction_sha256=sha256(prediction_file),
        stdlib_independent_rescore=True,
    )
    if reported_summary:
        reported = read_json(reported_summary)
        require(
            reported["query_count"] == total
            and reported.get("prediction_csv_sha256") == result["prediction_sha256"],
            "Worker summary denominator or prediction hash differs",
        )
        for key in ("jaccard_accuracy", "jaccard_accuracy_rounded3"):
            require(
                math.isclose(reported[key], result[key], rel_tol=0, abs_tol=1e-12),
                "Worker aggregate differs from independent rescore: " + key,
            )
    return result


def evaluate_run(run_dir):
    run = Path(run_dir).resolve()
    prepared = read_json(run / "PREPARED.json")
    output = run / "independent_evaluation"
    per_source = {}
    for source in prepared["sources"]:
        identity = prepared["test_identity"][source]
        worker = (
            run
            / "control/gmpi"
            / ("as" + source)
            / ("seed" + str(prepared["configuration"]["seed"]))
        )
        summary = worker / "evaluation_summary.json"
        require(
            summary.is_file() and (worker / "training_manifest.json").is_file(),
            "Source model evaluation is incomplete: " + source,
        )
        checkpoint = read_json(worker / "training_manifest.json")
        require(
            sha256(worker / "model.pkl") == checkpoint["model_sha256"], "Source checkpoint changed"
        )
        result = score_source(
            identity["reference_file"], worker / "pair_predictions.csv", source, identity, summary
        )
        write_json(output / ("source_" + source + ".json"), result)
        per_source[source] = result

    def aggregate(sources):
        sources = [s for s in sources if s in per_source]
        require(sources, "Cannot aggregate an empty cohort")
        rows = [per_source[s] for s in sources]
        count = sum(r["query_count"] for r in rows)
        return dict(
            sources=sources,
            query_count=count,
            jaccard_accuracy=math.fsum(r["total_jaccard_reward"] for r in rows) / count,
            jaccard_accuracy_rounded3=math.fsum(r["total_jaccard_reward_rounded3"] for r in rows)
            / count,
            ordered_exact_accuracy=sum(r["ordered_exact_matches"] for r in rows) / count,
            coverage=sum(r["valid_predictions"] for r in rows) / count,
            missing_predictions=sum(r["missing_predictions"] for r in rows),
        )

    result = dict(
        status="PASS",
        date=prepared["date"],
        per_source=per_source,
        original_cohort=aggregate(prepared["sources"]),
        common_cohort=aggregate(prepared.get("common9", prepared["sources"])),
        all_sources_included=True,
        stdlib_independent_rescore=True,
        fixed_full_pair_denominator=True,
        test_based_policy_selection=False,
        prepared_sha256=sha256(run / "PREPARED.json"),
    )
    write_json(output / "summary.json", result)
    write_json(
        output / "AUDIT_COMPLETED.json",
        dict(
            status="PASS",
            summary_sha256=sha256(output / "summary.json"),
            prediction_sha256={s: r["prediction_sha256"] for s, r in per_source.items()},
            fixed_test_pairs=result["original_cohort"]["query_count"],
        ),
    )
    return output / "summary.json"
