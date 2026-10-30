"""Compile endpoint-only test inputs through the unchanged observed-prefix PG.

Test endpoints are queries, never fabricated observed paths. Candidate/history
columns preserve the existing PG/model encoding; test labels are unavailable
zero placeholders. Only the separately frozen GMPI evaluation may read routes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

from pathveil.downstream import common
from pathveil.utils import read_json, sha256, write_json

VERSION = "label_blind_observed_prefix_test_compilation_v1"
LABEL_SEMANTICS = "unavailable_zero_placeholder"
REFERENCE_FIELDS = (
    "all_similarity",
    "total_exact_same",
    "exact_ratio",
    "two_hops_same",
    "two_hops_similarity",
    "avg_length",
    "sorted_total_result_by_length",
)
SUMMARY_FIELDS = (
    "vantage",
    "num_dests",
    "total_available_APs",
    "haspath",
    "coverage",
    "all_similarity",
    "num_vantages",
    "time_cost",
    "isGetData",
    "total_exact_same",
    "exact_ratio",
    "two_hops",
    "two_hops_same",
    "two_hops_similarity",
    "avg_length",
    "sorted_total_result_by_length",
    "k",
    "k2",
    "top_n",
    "top_n2",
    "neighborNum",
    "neighborNum2",
    "reference_metrics_computed",
    "label_semantics",
    "model_queries",
    "missing_candidate_queries",
    "same_as_queries",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_asn(value):
    node = str(value)
    require(
        not isinstance(value, bool)
        and node.isascii()
        and node.isdecimal()
        and 0 < int(node) < 2**32
        and node == str(int(node)),
        "Noncanonical query ASN",
    )
    return node


@dataclass(frozen=True)
class TestQuery:
    """Typed endpoints with deliberately no route or reference-label member."""

    source: str
    destination: str

    @classmethod
    def parse(cls, value, source):
        require(
            isinstance(value, str) and len(value.split("->")) == 2, "Expected one endpoint pair"
        )
        left, right = map(canonical_asn, value.split("->"))
        require(left == str(source), "Query belongs to another source")
        return cls(left, right)

    @property
    def pair(self):
        return self.source + "->" + self.destination


def load_fixed_inventory(prepared, source):
    source = canonical_asn(source)
    require(
        source in prepared["sources"] and prepared["date"] == prepared["configuration"]["date"],
        "Prepared source/date differs",
    )
    binding = prepared["fixed_test_pairs_inventory"][source]
    path = Path(binding["path"]).resolve()
    require(
        path.is_file() and sha256(path) == binding["sha256"], "Frozen endpoint inventory changed"
    )
    document = read_json(path)
    require(
        document.get("version")
        and document["date"] == prepared["date"]
        and document["source"] == source
        and isinstance(document["pairs"], list),
        "Endpoint inventory source/date/schema differs",
    )
    queries = [TestQuery.parse(pair, source) for pair in document["pairs"]]
    pairs = [q.pair for q in queries]
    require(
        len(pairs) == len(set(pairs))
        and pairs == sorted(pairs, key=lambda p: tuple(map(int, p.split("->")))),
        "Endpoint inventory must be unique and in canonical numeric order",
    )
    identity = prepared["test_identity"][source]
    digest = common.pair_identity(pairs)
    require(
        len(pairs) == document["pair_count"] == binding["pairs"] == identity["pairs"]
        and digest
        == document["pair_inventory_sha256"]
        == binding["pair_inventory_sha256"]
        == identity["pair_inventory_sha256"],
        "Complete registered endpoint denominator changed",
    )
    require(pairs, "An empty endpoint universe is not a valid registered source")
    return queries, dict(
        path=str(path), sha256=binding["sha256"], pairs=len(pairs), pair_inventory_sha256=digest
    )


class LabelBlindStageGuard(common.StageGuard):
    """An outer guard stays closed even when the legacy inner guard opens."""

    def __init__(self, run, prepared):
        super().__init__(run, allow_raw_test=False)
        self.explicit_references = {
            Path(v["reference_file"]).resolve()
            for v in prepared.get("test_identity", {}).values()
            if v.get("reference_file")
        }
        manifest_path = prepared.get("data_manifest")
        if manifest_path:
            from pathveil.data.prepare import resolve_read_path

            manifest = read_json(manifest_path)
            self.explicit_references |= {
                Path(resolve_read_path(manifest, p)).resolve() for p in self.explicit_references
            }

    def allow_evaluation(self, checkpoint, expected_sha):
        raise PermissionError("PG compilation cannot open final route labels")

    def check(self, event, values):
        if event == "open" and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).resolve()
            if path in self.explicit_references:
                self.denied.append(str(path))
                raise PermissionError(
                    "Final route labels remain closed during PG compilation: " + str(path)
                )
        # Do not allow an accidental caller to relax inherited path-pattern checks.
        self.allow_raw_test = False
        return super().check(event, values)


def _write_summary(path, row):
    """Dispatch counts only; every reference-dependent metric remains blank."""
    path = Path(path)
    previous = []
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            require(tuple(reader.fieldnames or ()) == SUMMARY_FIELDS, "PG summary schema differs")
            previous = [r for r in reader if r["vantage"] != row["vantage"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(previous)
        writer.writerow(row)
    temporary.replace(path)


def compile_test_source(env, source, destinations, queries, binding, run, prepared, gd, hf):
    """Swap the label serializer only; PG and history remain original callbacks."""
    started = time.monotonic()
    source = str(source)
    run = Path(run)
    require(
        str(env.parent) == str(env.curDate) == prepared["date"] and source in prepared["sources"],
        "PG environment date/source differs",
    )
    require(
        env.pathGenAlgorithm == "PG" and env.dataType == "test" and env.isGetData,
        "Label-blind serializer requires the original PG test stage",
    )
    require(
        not getattr(env, "prefix_level", False) and not getattr(env, "max_dests", 0),
        "Use complete AS endpoint queries without measurement-prefix or destination caps",
    )
    require(
        list(map(str, destinations)) == [q.destination for q in queries],
        "PG received a different or truncated endpoint inventory",
    )
    try:
        env.length_border = env.better_borderLen_dict[source]
    except (AttributeError, KeyError):
        pass
    average = hf.get_local_path_profile(env, source)["avg_length"]
    env.asn_collected_path_avglen[source] = average
    hf.configure_pg_strategy(env, average)
    # This callback is installed by the frozen observed-prefix adapter. It has
    # exactly the original (env,k,S,destinations) numerical API.
    proposals = hf.generate_kpath(env, env.k, source, list(destinations))
    output = Path(env.analyze_data_dir) / "test" / env.dataAttr / (source + "_generated_data.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp." + str(os.getpid()))
    status_path = run / "control/label_blind_test_inputs" / (source + "_queries.jsonl")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_temporary = status_path.with_name(status_path.name + ".tmp." + str(os.getpid()))
    model_rows = direct = missing = same = 0
    import json

    with temporary.open("w", newline="") as stream, status_temporary.open("w") as statuses:
        writer = csv.writer(stream)
        for query in queries:
            count = histories = 0
            if env.G.has_edge(source, query.destination):
                status = "direct_graph_edge_candidate"
                direct += 1
            elif source == query.destination:
                status = "same_as_query"
                same += 1
            else:
                candidates = gd._filter_model_paths(
                    env,
                    proposals.get(source + "-" + query.destination, ()),
                    (source, query.destination),
                    limit=env.k,
                )
                count = len(candidates)
                if not candidates:
                    status = "no_usable_candidate"
                    missing += 1
                else:
                    paths, pairs = gd.getObservedPaths(env, [source, query.destination])
                    histories = len(paths)
                    require(len(paths) == len(pairs), "History path/pair count differs")
                    encode_path = lambda p: "->".join(map(str, gd.path_represent_by_idx(env, p)))
                    encode_pair = lambda p: "->".join(map(str, gd.AP_represent_by_idx(env, p)))
                    writer.writerow(
                        [
                            "|".join(map(encode_path, candidates)),
                            "|".join(map(encode_path, paths)),
                            "|".join(map(encode_pair, pairs)),
                            "|".join("0.0" for _ in candidates),
                            query.pair,
                        ]
                    )
                    model_rows += 1
                    status = "model_candidates"
            statuses.write(
                json.dumps(
                    dict(
                        pair=query.pair,
                        status=status,
                        candidate_count=count,
                        history_count=histories,
                        reference_route_available=False,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(output)
    status_temporary.replace(status_path)
    summary = dict.fromkeys(SUMMARY_FIELDS)
    summary.update(
        vantage=source,
        num_dests=len(queries),
        total_available_APs=model_rows + direct,
        coverage=(model_rows + direct) / len(queries),
        num_vantages=1,
        time_cost=time.monotonic() - started,
        isGetData=1,
        two_hops=direct,
        k=env.k,
        k2=env.k2,
        top_n=env.top_n,
        top_n2=env.top_n2,
        neighborNum=env.neighbor_num_1,
        neighborNum2=env.neighbor_num_2,
        reference_metrics_computed=False,
        label_semantics=LABEL_SEMANTICS,
        model_queries=model_rows,
        missing_candidate_queries=missing,
        same_as_queries=same,
    )
    summary_path = Path(env.result_dir) / "PG-test-result.csv"
    _write_summary(summary_path, summary)
    return dict(
        version=VERSION,
        date=prepared["date"],
        source=source,
        stage="test_inputs",
        label_semantics=LABEL_SEMANTICS,
        labels_all_zero=True,
        reference_metrics_computed=False,
        reference_routes_read=False,
        fabricated_observed_routes=False,
        query_count=len(queries),
        full_query_inventory_file=binding["path"],
        full_query_inventory_sha256=binding["sha256"],
        pair_inventory_sha256=binding["pair_inventory_sha256"],
        output=str(output),
        sha256=sha256(output),
        model_rows=model_rows,
        direct_query_count=direct,
        missing_candidate_queries=missing,
        same_as_queries=same,
        dispatch_status_file=str(status_path),
        dispatch_status_sha256=sha256(status_path),
        pg_summary_file=str(summary_path),
        pg_summary_sha256=sha256(summary_path),
        generator=prepared.get("experimental_generator", {}).get(
            "name", "original_observed_prefix_PG"
        ),
        history="original_independent_training_index",
        admitted_library_path_set_sha256=prepared["admitted_library_path_set_sha256"][source],
    )


@contextmanager
def install_test_compiler(gd, hf, functions, env_class, prepared, run, inventories, records):
    """Keep the legacy setup/audits, replacing endpoint and test-label providers."""
    changes = []

    def replace(owner, name, value):
        changes.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def endpoints(vases, *ignored):
        require(set(map(str, vases)) <= set(inventories), "Unregistered endpoint source")
        return {}, {str(s): [q.destination for q in inventories[str(s)][0]] for s in vases}

    def compile_task(parameters):
        source, destinations, env = parameters
        source = str(source)
        queries, binding = inventories[source]
        record = compile_test_source(
            env, source, destinations, queries, binding, run, prepared, gd, hf
        )
        records[source] = record
        return record

    original_ensure = gd._ensure_result_csv

    def ensure(env, headers):
        if env.dataType != "test":
            return original_ensure(env, headers)
        path = Path(env.result_dir) / "PG-test-result.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as stream:
                csv.writer(stream).writerow(SUMMARY_FIELDS)

    original_source = env_class.get_exist_path_by_ASN

    def source_paths(env, source, data_type):
        if data_type == "test":
            raise PermissionError("No test truth loader during endpoint-only PG")
        return original_source(env, source, data_type)

    original_index = env_class.getExistPath

    def index(env, data_type):
        if data_type != "train":
            raise PermissionError("Only the registered training history index may load")
        return original_index(env, data_type)

    replace(gd, "get_meta_vas_dests", endpoints)
    replace(functions, "get_measured_vantage_dst", lambda *args: {})
    replace(
        functions,
        "get_unmeasured_vantage_dst",
        lambda *args: {s: [q.destination for q in v[0]] for s, v in inventories.items()},
    )
    replace(gd, "pathGeneration_PerformanceAnalysis", compile_task)
    replace(gd, "_ensure_result_csv", ensure)
    replace(env_class, "get_exist_path_by_ASN", source_paths)
    replace(env_class, "getExistPath", index)
    try:
        yield
    finally:
        for owner, name, old in reversed(changes):
            setattr(owner, name, old)


def reuse_completed_test(run, prepared, sources, inventories):
    """Only reuse a complete byte-bound endpoint-only stage, never old labels."""
    paths = [run / "control/label_blind_test_inputs" / (s + ".json") for s in sources]
    if not any(p.exists() for p in paths):
        return False
    require(
        all(p.exists() for p in paths),
        "Resume partially completed test stages with per-source workers",
    )
    for source, path in zip(sources, paths):
        receipt = read_json(path)
        binding = inventories[source][1]
        marker = run / "control/pathgen_completed" / ("test_inputs_" + source + ".json")
        require(
            receipt["version"] == VERSION
            and receipt["prepared_sha256"] == sha256(run / "PREPARED.json")
            and receipt["compiler_sha256"] == sha256(__file__)
            and receipt["full_query_inventory_sha256"] == binding["sha256"]
            and receipt["pair_inventory_sha256"] == binding["pair_inventory_sha256"]
            and receipt["query_count"] == binding["pairs"]
            and receipt["query_count"]
            == sum(
                receipt[k]
                for k in (
                    "model_rows",
                    "direct_query_count",
                    "missing_candidate_queries",
                    "same_as_queries",
                )
            )
            and receipt["source"] == source
            and receipt["date"] == prepared["date"]
            and receipt["admitted_library_path_set_sha256"]
            == prepared["admitted_library_path_set_sha256"][source]
            and not receipt["denied_raw_route_accesses"]
            and receipt["reference_routes_read"] is False
            and receipt["labels_all_zero"] is True
            and receipt["reference_metrics_computed"] is False,
            "Cached test compilation belongs to different inputs/code/semantics",
        )
        require(
            sha256(receipt["output"]) == receipt["sha256"]
            and sha256(receipt["pg_summary_file"]) == receipt["pg_summary_sha256"]
            and sha256(receipt["dispatch_status_file"]) == receipt["dispatch_status_sha256"]
            and sha256(marker) == receipt["pathgen_completed_sha256"],
            "Cached label-blind PG output changed",
        )
        require(
            read_json(marker)["label_blind_compilation"]
            == {k: v for k, v in receipt.items() if k != "pathgen_completed_sha256"},
            "Label-blind completion receipts disagree",
        )
    return True


def verify_compilation_receipt(run, source, prepared=None):
    """Read-only resource-dispatch verifier for one completed endpoint stage."""
    run = Path(run).resolve()
    prepared = common.verify_runtime_authorization(run) if prepared is None else prepared
    source = str(source)
    inventory = load_fixed_inventory(prepared, source)
    require(
        reuse_completed_test(run, prepared, [source], {source: inventory}),
        "Label-blind compilation is incomplete",
    )
    return read_json(run / "control/label_blind_test_inputs" / (source + ".json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--stage", choices=("training_inputs", "test_inputs"), required=True)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--target_vases", required=True)
    parser.add_argument("--num_thread", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    require(
        os.environ.get("PATHVEIL_PROJECT_ROOT")
        and common.PROJECT == Path(os.environ["PATHVEIL_PROJECT_ROOT"]).resolve(),
        "Set PATHVEIL_PROJECT_ROOT before starting the isolated PG worker",
    )
    run = common.validate_run(args.experiment_dir)
    sys.dont_write_bytecode = True
    prepared = read_json(run / "PREPARED.json")
    guard = LabelBlindStageGuard(run, prepared)
    sys.addaudithook(guard.check)
    prepared = common.verify_runtime_authorization(run)
    sources = [canonical_asn(s.strip()) for s in args.target_vases.split(",")]
    require(
        len(sources) == len(set(sources))
        and set(sources) <= set(prepared["sources"])
        and args.date == prepared["date"],
        "PG requested a different source/date",
    )
    inventories = (
        {s: load_fixed_inventory(prepared, s) for s in sources}
        if args.stage == "test_inputs"
        else {}
    )
    require(
        args.stage != "test_inputs" or args.num_thread in (0, 1),
        "Use one source per CPU worker with --num_thread 1; parallelize independent source workers",
    )
    if args.check_only:
        print(
            dict(
                status="METADATA_PASS",
                stage=args.stage,
                date=args.date,
                endpoints={s: v[1] for s, v in inventories.items()},
                raw_routes_read=False,
            ),
            flush=True,
        )
        return 0
    if args.stage == "test_inputs" and reuse_completed_test(run, prepared, sources, inventories):
        print(
            dict(status="VERIFIED_TEST_INPUT_CACHE", sources=sources, raw_routes_read=False),
            flush=True,
        )
        return 0
    for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "TORCH_HOME"):
        directory = run / "runtime" / key.lower()
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    # Import the W3 preserved wrapper before changing the core namespace: its
    # bound archive stays explicit, while actual core APIs resolve to MAIN.
    from pathveil.prefix_tree import stage as observed
    import pathveil

    previous_namespace = list(pathveil.__path__)
    pathveil.__path__ = [str(common.PROJECT / "src/pathveil"), *previous_namespace]
    sys.dont_write_bytecode = True
    from pathveil.core import activate

    activate(data_manifest=prepared["data_manifest"])
    import getData as gd
    import HighFrequencyVertexBasedPathGen as hf
    import commonFunction as functions
    import environment

    require(
        Path(hf.__file__).resolve()
        == common.PROJECT / "src/pathveil/core/HighFrequencyVertexBasedPathGen.py",
        "Use the original frozen numerical PG core",
    )
    forwarded = list(sys.argv[1:] if argv is None else argv)
    forwarded = [v for v in forwarded if v != "--check-only"]
    previous_argv = sys.argv
    records = {}
    sys.argv = [str(__file__), *forwarded]
    try:
        if args.stage == "test_inputs":
            with install_test_compiler(
                gd, hf, functions, environment.Env, prepared, run, inventories, records
            ):
                observed.main()
        else:
            observed.main()
    finally:
        sys.argv = previous_argv
        pathveil.__path__ = previous_namespace
    require(not guard.denied, "PG attempted a closed route read or an out-of-run write")
    if args.stage == "test_inputs":
        require(
            set(records) == set(sources),
            "Label-blind compiler did not cover every requested source",
        )
        for source, record in records.items():
            record.update(
                prepared_sha256=sha256(run / "PREPARED.json"),
                compiler_sha256=sha256(__file__),
                denied_raw_route_accesses=list(guard.denied),
                pg_summary_sha256=sha256(record["pg_summary_file"]),
            )
            marker = run / "control/pathgen_completed" / ("test_inputs_" + source + ".json")
            completed = read_json(marker)
            require(
                completed["sha256"] == record["sha256"]
                and completed.get("experimental_prefix_generator"),
                "Original observed-prefix generator audit/CSV binding missing",
            )
            completed["label_blind_compilation"] = record
            write_json(marker, completed)
            record["pathgen_completed_sha256"] = sha256(marker)
            write_json(run / "control/label_blind_test_inputs" / (source + ".json"), record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
