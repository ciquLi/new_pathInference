"""Shared input-copy, pair-split and Torch runtime utilities."""
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys

from pathveil.downstream.common import require
from pathveil.downstream.device_runtime import configure_device as _production_configure_device
from pathveil.utils import read_json, sha256

_CONFIGURED_RUNTIME = None


def canonical_pair(value):
    parts = str(value).replace("->", "-").split("-")
    require(
        len(parts) == 2
        and all(
            p.isascii() and p.isdecimal() and 0 < int(p) < 2**32 and str(int(p)) == p
            for p in parts
        ),
        "Invalid query identity",
    )
    return "->".join(parts)


def fixed_split(control, train_csv, destination=None):
    """Adopt original split bytes; verify every row against admitted train CSV."""
    root = Path(control) / "surrogate_split"
    split = read_json(root / "split_manifest.json")
    require(split["input_sha256"] == sha256(train_csv), "Control split input differs")
    inventories = {name: set(split[name + "_pairs"]) for name in ("train", "validation")}
    require(inventories["train"].isdisjoint(inventories["validation"]), "Validation pair overlap")
    rows = {}
    with Path(train_csv).open(newline="") as stream:
        for row in csv.reader(stream, strict=True):
            pair = canonical_pair(row[4])
            require(len(row) == 5 and pair not in rows, "Malformed/repeated training pair")
            rows[pair] = row
    require(
        set(rows) == inventories["train"] | inventories["validation"], "Split inventory differs"
    )
    for name in ("train", "validation"):
        seen = set()
        with (root / (name + ".csv")).open(newline="") as stream:
            for row in csv.reader(stream, strict=True):
                pair = canonical_pair(row[4])
                require(
                    pair in inventories[name] and pair not in seen and row == rows[pair],
                    "Control split content differs from training input",
                )
                seen.add(pair)
        require(
            seen == inventories[name] and len(seen) == split["row_counts"][name],
            "Control split count differs",
        )
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("train.csv", "validation.csv", "split_manifest.json"):
            target = destination / name
            if target.exists():
                require(sha256(target) == sha256(root / name), "Existing split changed")
            else:
                shutil.copyfile(root / name, target)
    return split


def driver_arguments(run, source, stage, config, args, derived_audit):
    values = [
        "train_evaluate_gmpi",
        "--experiment-dir",
        str(run),
        "--source-asn",
        source,
        "--stage",
        stage,
        "--device",
        args.device,
        "--cpu-threads",
        str(args.cpu_threads),
    ]
    for key in (
        "date",
        "seed",
        "validation_fraction",
        "epochs",
        "patience",
        "batch_size",
        "workers",
        "lr",
        "k",
        "hidden_size",
    ):
        values += ["--" + key.replace("_", "-"), str(config[key])]
    if derived_audit:
        values += ["--derived-source-audit", str(derived_audit)]
    return values


def configure_runtime_once(torch, requested="auto", threads=4):
    """Configure the native Torch pools once, then verify and reuse that runtime.

    Torch 1.8 can abort in C++ when its inter-op setter is called twice. A Python
    exception handler around that setter cannot protect the process. This
    experiment adapter therefore never enters the production setter again.
    RNG state and the driver's existing seed operations are untouched.
    """
    global _CONFIGURED_RUNTIME
    require(
        requested in {"auto", "cpu", "cuda"} and 1 <= threads <= 16,
        "Invalid device/thread configuration",
    )
    if _CONFIGURED_RUNTIME is None:
        device, info = _production_configure_device(torch, requested, threads)
        _CONFIGURED_RUNTIME = dict(
            torch=torch, requested=requested, threads=threads, device=device, info=dict(info)
        )
    runtime = _CONFIGURED_RUNTIME
    require(
        runtime["torch"] is torch
        and runtime["requested"] == requested
        and runtime["threads"] == threads,
        "Runtime reuse requires the same Torch module, requested device and CPU threads",
    )
    require(
        torch.get_num_threads() == runtime["info"]["intraop_threads"]
        and torch.get_num_interop_threads() == runtime["info"]["interop_threads"],
        "Configured Torch thread pools changed before runtime reuse",
    )
    require(
        str(runtime["device"]) == runtime["info"]["effective"],
        "Configured Torch device identity changed",
    )
    return runtime["device"], dict(runtime["info"])


@contextmanager
def reuse_driver_runtime():
    """Route the frozen driver's local import through the experiment's cache."""
    from pathveil.downstream import device_runtime

    original = device_runtime.configure_device
    device_runtime.configure_device = configure_runtime_once
    try:
        yield
    finally:
        device_runtime.configure_device = original


def copy_verified(source, target, expected):
    """Copy immutable bytes atomically; a resume never overwrites different data."""
    source, target = Path(source), Path(target)
    require(source.resolve() != target.resolve(), "Control and output files must differ")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(
            not target.is_symlink() and sha256(target) == expected,
            "Existing copied input changed: " + str(target),
        )
    else:
        temporary = target.with_name(target.name + ".copy." + str(os.getpid()))
        shutil.copy2(source, temporary)
        require(sha256(temporary) == expected, "Copied bytes differ: " + str(source))
        temporary.replace(target)
    first, second = source.stat(), target.stat()
    require(
        (first.st_dev, first.st_ino) != (second.st_dev, second.st_ino),
        "A writable output may not hard-link the historical control",
    )
    return target


def rho_features(source, run_root):
    method = read_json(Path(run_root) / "METHOD_FROZEN.json")
    value = method["public_features"][str(source)]
    rho = value["rho"]
    require(
        rho is not None and math.isfinite(rho) and rho >= 0,
        "Public structure is unknown; there is no default rho branch",
    )
    return value
