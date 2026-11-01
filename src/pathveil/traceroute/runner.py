"""Resumable trace experiment with isolated heavy worker subprocesses."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pathveil.utils import file_record, project_root, read_json, sha256, utc_now, write_json
from .common import load_contract
from .score import FROZEN_TRUTH_IDENTITIES


def verify_outputs(marker):
    value = read_json(marker)
    for filename, expected in value["outputs_sha256"].items():
        if not Path(filename).is_file() or sha256(filename) != expected:
            raise ValueError("Completed traceroute stage output changed: " + filename)
    return value


def run_traceroute(
    run_dir,
    data_manifest_path,
    release_path,
    device="cuda",
    *,
    python=None,
    source="63835",
    epochs=30,
    batch_size=64,
    patience=3,
    validation_fraction=0.1,
    k=200,
    seed=None,
    date=None,
    prefix_tree_dir=None,
    pathgen_workers=1,
):
    """Run generation → BGP training → target prediction → frozen scoring.

    No measurement file is opened until the independent scorer subprocess.
    Repeated invocations reuse only stages with unchanged inputs and outputs.
    """
    data_manifest_path, release_path = (
        Path(data_manifest_path).resolve(),
        Path(release_path).resolve(),
    )
    contract = load_contract(data_manifest_path, release_path, source)
    if date is not None and str(date) != contract["date"]:
        raise ValueError("Requested traceroute date differs from the data manifest")
    date = contract["date"]
    if date not in FROZEN_TRUTH_IDENTITIES:
        raise ValueError("No frozen independent traceroute inventory registered for " + date)
    root = project_root()
    inputs = root / "data/traceroute" / date
    targets = inputs / "inputs/web_targets.csv"
    truth = inputs / "measurement/as_paths_path_inference.csv"
    # The known reference hash is recorded here, but labels stay unopened.
    identity = {
        "date": date,
        "source": str(source),
        "contract": contract,
        "targets": file_record(targets),
        "truth_expected": FROZEN_TRUTH_IDENTITIES[date],
        "configuration": {
            "device": device,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "validation_fraction": validation_fraction,
            "k": k,
            "seed": int(date) if seed is None else seed,
        },
        "code_sha256": {
            str(path): sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    prefix_arguments = []
    if prefix_tree_dir is not None:
        prefix_tree_dir = Path(prefix_tree_dir).resolve()
        tree_files = [prefix_tree_dir / 'METHOD_FROZEN.json', prefix_tree_dir / (str(source) + '.json')]
        identity['pats'] = dict(prefix_tree_dir=str(prefix_tree_dir), pathgen_workers=pathgen_workers,
                                files_sha256={str(p): sha256(p) for p in tree_files},
                                code_sha256={str(p): sha256(p) for p in (
                                    Path(__file__).resolve().parents[1] / 'prefix_tree' / name for name in (
                                        'traceroute.py', 'registered_topology.py', 'public_tree_pathgen.py',
                                        'public_tree_observed_pathgen.py'))})
        prefix_arguments = ['--prefix-tree-dir', prefix_tree_dir]
    run = Path(run_dir).resolve() / "traceroute"
    control = run / "control"
    control.mkdir(parents=True, exist_ok=True)
    identity_path = control / "IDENTITY.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Traceroute inputs/configuration changed; choose a fresh run directory")
    write_json(identity_path, identity)
    pool = run / "support_data/train"
    pool.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = pool / (str(source) + "_augmented_paths.csv"), pool / (
        str(source) + "_augmented_paths.json"
    )
    if csv_path.exists() and sha256(csv_path) != contract["sha256"]:
        raise ValueError("Run-local admitted pool CSV changed")
    if not csv_path.exists():
        shutil.copyfile(contract["path"], csv_path)
    grouped = {}
    with csv_path.open(newline="") as handle:
        for row in csv.reader(handle):
            nodes = row[0].split("->")
            grouped.setdefault(str(source) + "-" + nodes[-1], []).append(nodes)
    if json_path.exists() and read_json(json_path) != grouped:
        raise ValueError("Run-local admitted pool JSON changed")
    write_json(json_path, grouped)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PATHVEIL_TRACE_RUN_DIR"] = str(run)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for key in ["TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "TORCH_HOME"]:
        path = run / "runtime" / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    interpreter = str(python or sys.executable)

    def status(stage, state, **extra):
        write_json(
            control / "status.json",
            dict(stage=stage, status=state, updated_at_utc=utc_now(), **extra),
        )

    def stage(name, module, arguments, output_resolver):
        marker = control / (name + ".done.json")
        if marker.exists():
            return verify_outputs(marker)
        attempts = control / "attempts" / name
        attempts.mkdir(parents=True, exist_ok=True)
        outdir = attempts / ("attempt_%03d" % (len(list(attempts.glob("attempt_*.log"))) + 1))
        command = [interpreter, "-m", module, *map(str, arguments), "--outdir", str(outdir)]
        log = attempts / (outdir.name + ".log")
        status(name, "running", command=command, log=str(log))
        with log.open("x") as handle:
            result = subprocess.run(
                command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
        if result.returncode:
            status(name, "failed", returncode=result.returncode, log=str(log))
            raise RuntimeError("Traceroute stage failed: %s; inspect %s" % (name, log))
        try:
            outputs = output_resolver(outdir)
            record = dict(
                stage=name,
                completed_at_utc=utc_now(),
                outdir=str(outdir),
                outputs_sha256={str(path): sha256(path) for path in outputs},
            )
        except Exception as exc:
            status(name, "failed", error=str(exc), log=str(log))
            raise
        write_json(marker, record)
        status(name, "complete")
        return record

    base = [
        "--data-manifest",
        data_manifest_path,
        "--release",
        release_path,
        "--source-asn",
        source,
    ]
    generated = stage(
        "training_inputs",
        "pathveil.traceroute.generate_training",
        [*base, "--support-data-dir", pool.parent, "--k", k,
         *prefix_arguments, "--pathgen-workers", pathgen_workers],
        lambda out: [
            out / "TRAINING_INPUTS.json",
            Path(read_json(out / "TRAINING_INPUTS.json")["generated_csv"]),
        ],
    )
    training_input_manifest = Path(generated["outdir"]) / "TRAINING_INPUTS.json"
    training_csv = read_json(training_input_manifest)["generated_csv"]
    config = identity["configuration"]
    training = stage(
        "training",
        "pathveil.traceroute.train_bgp",
        [
            *base,
            "--date",
            date,
            "--device",
            device,
            "--generated-training-csv",
            training_csv,
            "--input-manifest",
            training_input_manifest,
            "--epochs",
            epochs,
            "--batch-size",
            batch_size,
            "--patience",
            patience,
            "--validation-fraction",
            validation_fraction,
            "--k",
            k,
            "--seed",
            config["seed"],
        ],
        lambda out: [
            out / "model.pkl",
            out / "training_manifest.json",
            out / "training_history.json",
        ],
    )
    model = Path(training["outdir"]) / "model.pkl"
    prediction = stage(
        "prediction",
        "pathveil.traceroute.predict_web",
        [
            *base,
            "--date",
            date,
            "--device",
            device,
            "--targets",
            targets,
            "--support-data-dir",
            pool.parent,
            "--selected-neighbors-csv",
            contract["selected_neighbors_csv"],
            "--recursive-anchors-csv",
            contract["recursive_anchors_csv"],
            "--model",
            model,
            "--local-knowledge-mode",
            "augmentation_only",
            "--k",
            k,
            *prefix_arguments,
        ],
        lambda out: [
            out / "pair_predictions.jsonl",
            out / "target_predictions.jsonl",
            out / "prediction_manifest.json",
        ],
    )
    score = stage(
        "evaluation",
        "pathveil.traceroute.score",
        [
            "--date",
            date,
            "--truth",
            truth,
            "--predictions",
            Path(prediction["outdir"]) / "pair_predictions.jsonl",
        ],
        lambda out: [
            out / "summary.json",
            out / "per_pair.csv",
            out / "per_path.csv",
            out / "gmpi_result.txt",
        ],
    )
    result = {
        "date": date,
        "source": str(source),
        "status": "complete",
        "model": str(model),
        "predictions": str(Path(prediction["outdir"]) / "pair_predictions.jsonl"),
        "summary": str(Path(score["outdir"]) / "summary.json"),
        "pool_contract": contract,
        "measurement_labels_used_for_training_or_selection": False,
    }
    write_json(run / "RESULT.json", result)
    status("complete", "complete", result=result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--date", help="Optional explicit date assertion (YYYYMMDD)")
    args = parser.parse_args()
    print(
        json.dumps(
            run_traceroute(
                args.run_dir, args.data_manifest, args.release, args.device, date=args.date
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
