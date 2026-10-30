#!/usr/bin/env python3
"""Precompute one frozen run's training cases with independent spawn workers.

This operational helper calls the unchanged numerical implementation. The
formal one-date entry point subsequently verifies and consumes these caches.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager, nullcontext
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from pathveil.utils import read_json, sha256, utc_now, write_json
from pathveil.upstream.resources import positive_int, runtime_resources

GIB = 1024**3


def memory_available():
    """Use Linux MemAvailable, including reclaimable page cache."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo does not provide MemAvailable")


def resident_bytes(pid):
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def process_start_ticks(pid):
    """PID reuse and zombies must never inherit an old memory reservation."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, ValueError, IndexError):
        return None


class WorkerMemoryCoordinator:
    """Reserve unallocated RSS growth across every date's helper process.

    The short project lock covers admission, spawn/dispatch, and atomic ledger
    writes only. A lease stays conservative while its matching worker lives;
    TTL marks a missing heartbeat but never releases a live source's budget.
    """

    def __init__(
        self,
        directory,
        memory_reader=memory_available,
        resident_reader=resident_bytes,
        identity_reader=process_start_ticks,
        lease_seconds=120,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / ".case_worker_memory.json"
        self.lock_path = self.directory / ".case_worker_memory.lock"
        self.memory_reader = memory_reader
        self.resident_reader = resident_reader
        self.identity_reader = identity_reader
        self.lease_seconds = lease_seconds
        self.owner_pid = os.getpid()
        self.owner_start = self.identity_reader(self.owner_pid)
        if self.owner_start is None:
            raise RuntimeError("Cannot identify the memory coordinator process")

    def owned(self, record):
        return (record["owner_pid"], record["owner_start_ticks"]) == (
            self.owner_pid,
            self.owner_start,
        )

    @contextmanager
    def transaction(self):
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                ledger = (
                    read_json(self.path)
                    if self.path.exists()
                    else dict(schema_version=1, reservations={})
                )
                if ledger.get("schema_version") != 1 or not isinstance(
                    ledger.get("reservations"), dict
                ):
                    raise RuntimeError("Invalid shared case-worker memory ledger")
                current = time.time()
                for key, record in list(ledger["reservations"].items()):
                    if self.identity_reader(record["worker_pid"]) != record["worker_start_ticks"]:
                        del ledger["reservations"][key]
                        continue
                    rss = self.resident_reader(record["worker_pid"]) or 0
                    record["resident_bytes"] = rss
                    record["reserved_growth_bytes"] = max(0, record["expected_rss_bytes"] - rss)
                    record["orphaned_owner"] = (
                        self.identity_reader(record["owner_pid"]) != record["owner_start_ticks"]
                    )
                    if self.owned(record):
                        record["expires_epoch"] = current + self.lease_seconds
                    record["heartbeat_overdue"] = current > record["expires_epoch"]
                yield ledger
                ledger["updated_utc"] = utc_now()
                ledger["reserved_growth_bytes"] = sum(
                    record["reserved_growth_bytes"] for record in ledger["reservations"].values()
                )
                write_json(self.path, ledger)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def record(self, process, expected_rss_bytes, phase, source=None):
        start = self.identity_reader(process.pid)
        if start is None:
            return None, None
        rss = self.resident_reader(process.pid) or 0
        key = f"{process.pid}:{start}"
        return key, dict(
            owner_pid=self.owner_pid,
            owner_start_ticks=self.owner_start,
            worker_pid=process.pid,
            worker_start_ticks=start,
            expected_rss_bytes=expected_rss_bytes,
            resident_bytes=rss,
            reserved_growth_bytes=max(0, expected_rss_bytes - rss),
            phase=phase,
            source=source,
            expires_epoch=time.time() + self.lease_seconds,
            orphaned_owner=False,
            heartbeat_overdue=False,
        )

    @staticmethod
    def reserved(ledger, excluding=None):
        return sum(
            record["reserved_growth_bytes"]
            for key, record in ledger["reservations"].items()
            if key != excluding
        )

    def spawn(self, start_process, expected_rss_bytes, min_free_bytes):
        with self.transaction() as ledger:
            if self.memory_reader() - self.reserved(ledger) < min_free_bytes + expected_rss_bytes:
                return None
            process = start_process()
            key, record = self.record(process, expected_rss_bytes, "initializing")
            if key is not None:
                ledger["reservations"][key] = record
            return process

    def ready(self, process):
        with self.transaction() as ledger:
            for record in ledger["reservations"].values():
                if self.owned(record) and record["worker_pid"] == process.pid:
                    record["phase"] = "ready_first_source"

    def dispatch(self, process, expected_rss_bytes, min_free_bytes, source, enqueue):
        with self.transaction() as ledger:
            key, record = self.record(process, expected_rss_bytes, "source", source)
            if key is None:
                return False
            if (
                self.memory_reader() - self.reserved(ledger, excluding=key)
                < min_free_bytes + record["reserved_growth_bytes"]
            ):
                return False
            # Enqueue under the admission lock: two helpers cannot both spend
            # the same unallocated headroom before publishing their leases.
            enqueue(source)
            ledger["reservations"][key] = record
            return True

    def release(self, process=None):
        with self.transaction() as ledger:
            for key, record in list(ledger["reservations"].items()):
                if self.owned(record) and (process is None or record["worker_pid"] == process.pid):
                    del ledger["reservations"][key]

    def snapshot(self):
        with self.transaction() as ledger:
            reserved = self.reserved(ledger)
            return dict(
                ledger_path=str(self.path),
                reserved_growth_bytes=reserved,
                effective_available_bytes=self.memory_reader() - reserved,
                reservations=len(ledger["reservations"]),
                orphaned_reservations=sum(
                    record["orphaned_owner"] for record in ledger["reservations"].values()
                ),
            )


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns experiment/case lock: " + str(path)) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def registered_run(path, project=PROJECT):
    run, root = Path(path).resolve(), Path(project).resolve()
    if root / "runs" not in run.parents:
        raise ValueError("--run-dir must be inside this project's runs/")
    if not run.is_dir():
        raise ValueError("Prepared experiment directory does not exist: " + str(run))
    return run


def prepare_configuration(run, cpu_threads=None, predict_threads=None):
    """Check the frozen run before admitting any worker or numerical writes."""
    from pathveil.cli import run_identity
    from pathveil.data.registry import load_registry
    from pathveil.upstream.contracts import validate_model_scope, verify
    from pathveil.upstream.method import MethodLibrary

    manifest_path = run / "data/data_manifest.json"
    bundle_path = run / "upstream/NEIGHBOR_BUNDLE.json"
    data, bundle = map(read_json, (manifest_path, bundle_path))
    frozen, split_info = map(read_json, (run / "RUN_IDENTITY.json", run / "upstream/split.json"))
    config = frozen["method_config"]
    config_path = run / "method_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Run method-config snapshot differs from RUN_IDENTITY")
    resources = runtime_resources(config)
    for name, requested in (
        ("cpu_threads", cpu_threads),
        ("case_predict_threads", predict_threads),
    ):
        if requested is not None and positive_int(requested, name) != resources[name]:
            raise ValueError(
                "Requested " + name + " differs from frozen run resources; use a new run"
            )
    if bundle.get("runtime_resources", resources) != resources:
        raise ValueError("Neighbor bundle resources differ from frozen run resources")
    current = run_identity(data["date"], load_registry(data["date"]), config)
    if current != frozen:
        raise ValueError("Frozen run code/config/date changed; refusing cache migration")
    library = MethodLibrary(PROJECT)
    receipt = library.receipt()
    previous = read_json(run / "upstream/IDENTITY.json")
    if previous["code"] != receipt or previous["method_config"] != config:
        raise ValueError("Upstream computational code or method identity changed")
    if previous["split"] != split_info or previous["date"] != data["date"]:
        raise ValueError("Upstream source split/date changed")
    if previous["sources"] != list(map(str, data["sources"])):
        raise ValueError("Formal source cohort changed")
    if previous["public_files_sha256"] != {
        str(p): sha256(p) for p in data["public_files"].values()
    }:
        raise ValueError("Dated public inputs changed")
    if sha256(data["merged_train"]) != previous["canonical_sha256"]:
        raise ValueError("Original canonical training corpus changed")
    if (
        bundle["computational_receipt"]["computational_sources_sha256"]
        != receipt["computational_sources_sha256"]
    ):
        raise ValueError("Neighbor computational code differs from frozen case implementation")
    verify(bundle["computational_receipt"].get("orchestration_sources_sha256", {}))
    validate_model_scope(bundle, data["date"], split_info["split"])
    sources = list(map(str, bundle["training_sources"]))
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("Training source queue is empty or duplicated")
    if not config_path.exists():
        write_json(config_path, config)
    return sources, dict(
        run_dir=str(run),
        project_root=str(PROJECT),
        manifest_path=str(manifest_path),
        bundle_path=str(bundle_path),
        config_path=str(config_path),
        runtime_resources=resources,
    )


class FrozenCaseBuilder:
    """One original runtime/store per independently spawned worker."""

    def __init__(self, configuration):
        from pathveil.upstream.destination import DestinationFeatureStore
        from pathveil.upstream.method import MethodLibrary
        from pathveil.upstream.neighbors import NeighborRuntime
        from pathveil.upstream.path_pipeline import construct_case

        self.config = read_json(configuration["config_path"])
        self.runtime = NeighborRuntime(
            read_json(configuration["bundle_path"]),
            read_json(configuration["manifest_path"]),
            self.config,
            method=MethodLibrary(configuration["project_root"]),
            threads=configuration.get("runtime_resources", runtime_resources(self.config))[
                "case_predict_threads"
            ],
        )
        self.store = DestinationFeatureStore(self.runtime.bundle, self.runtime.public)
        self.construct_case = construct_case
        self.run = Path(configuration["run_dir"])

    def __call__(self, source):
        output = self.run / "upstream/training_cases" / source
        existed = (output / "CASE.json").is_file()
        case = self.construct_case(source, self.runtime, self.config, self.store, output)
        return dict(
            receipt=str(output / "CASE.json"),
            receipt_sha256=sha256(output / "CASE.json"),
            rows=case["rows"],
            reused=existed,
        )


def worker_main(worker_id, configuration, inbox, events, builder_factory):
    # Spawn deliberately avoids inheriting a live OpenMP/LightGBM thread pool.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    resources = runtime_resources({"runtime_resources": configuration.get("runtime_resources", {})})
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        # CPU fitting uses cpu_threads in the parent. Independent candidate
        # workers cap their own pools to avoid workers * cpu_threads overlap.
        os.environ[name] = str(resources["case_predict_threads"])
    try:
        builder = builder_factory(configuration)
    except BaseException:
        events.put(dict(kind="init_failed", worker=worker_id, error=traceback.format_exc()))
        return
    events.put(dict(kind="ready", worker=worker_id))
    while True:
        source = inbox.get()
        if source is None:
            return
        try:
            lock = (
                Path(configuration["run_dir"]) / "upstream/training_cases" / source / ".case.lock"
            )
            with exclusive_lock(lock):
                result = builder(source)
            events.put(dict(kind="complete", worker=worker_id, source=source, result=result))
        except BaseException:
            events.put(
                dict(kind="failed", worker=worker_id, source=source, error=traceback.format_exc())
            )


def schedule_cases(
    run,
    sources,
    configuration,
    workers=2,
    min_free_bytes=12 * GIB,
    worker_memory_bytes=16 * GIB,
    memory_reader=memory_available,
    builder_factory=FrozenCaseBuilder,
    poll_seconds=1.0,
    memory_timeout_seconds=0,
    run_lock_held=False,
):
    """Finish the entire explicit source queue; errors always produce failure.

    The builder injection is for small scheduling tests, never a CLI option.
    A zero memory timeout waits indefinitely, allowing concurrent dates to free
    memory. Workers initialize one at a time to measure their resident usage.
    """
    positive_int(workers, "case_workers")
    if min_free_bytes < 0 or worker_memory_bytes < 0:
        raise ValueError("Invalid worker count or memory limits")
    sources = list(map(str, sources))
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("Training sources must be nonempty and unique")
    run = Path(run)
    progress_path = run / "upstream/CASE_PRECOMPUTE_PROGRESS.json"
    result_path = run / "upstream/CASE_PRECOMPUTE.json"
    pending, completed, failed = deque(sources), {}, {}
    pool, next_id, stopping, interrupted = {}, 0, False, None
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    previous_handlers = {}
    start, memory_wait_start = time.monotonic(), None
    resources = runtime_resources({"runtime_resources": configuration.get("runtime_resources", {})})
    resources["case_workers"] = workers
    retired_workers = 0
    estimated_next_worker_bytes = worker_memory_bytes
    coordinator = WorkerMemoryCoordinator(
        Path(configuration.get("project_root", run)) / "runs", memory_reader=memory_reader
    )
    shared_memory = coordinator.snapshot()

    def on_signal(signum, _frame):
        nonlocal stopping, interrupted
        stopping, interrupted = True, signum

    def record(status):
        value = dict(
            status=status,
            updated_utc=utc_now(),
            pid=os.getpid(),
            run_dir=str(run),
            start_method="spawn",
            requested_workers=workers,
            total_sources=len(sources),
            completed_sources=sorted(completed, key=int),
            completed=len(completed),
            pending=list(pending),
            running={str(k): v["source"] for k, v in pool.items() if v["source"]},
            failed=failed,
            memory_available_bytes=memory_reader(),
            min_free_bytes=min_free_bytes,
            worker_memory_bytes=worker_memory_bytes,
            estimated_next_worker_bytes=estimated_next_worker_bytes,
            retired_workers=retired_workers,
            runtime_resources=resources,
            run_lock_held_by_caller=bool(run_lock_held),
            shared_memory=shared_memory,
            workers={
                str(k): dict(
                    pid=v["process"].pid,
                    state=v["state"],
                    source=v["source"],
                    resident_bytes=resident_bytes(v["process"].pid),
                )
                for k, v in pool.items()
            },
            elapsed_seconds=time.monotonic() - start,
        )
        write_json(progress_path, value)
        return value

    with nullcontext() if run_lock_held else exclusive_lock(run / ".run.lock"):
        status = "FAILED"
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, on_signal)
            record("starting")
            while not stopping and (pending or any(v["source"] for v in pool.values())):
                # Drain all notifications before inspecting exited workers.
                while True:
                    try:
                        event = events.get(timeout=poll_seconds if not pool else 0.05)
                    except queue.Empty:
                        break
                    worker = pool[event["worker"]]
                    if event["kind"] == "ready":
                        worker["state"] = "idle"
                        coordinator.ready(worker["process"])
                    elif event["kind"] == "init_failed":
                        failed["worker_" + str(event["worker"])] = event["error"]
                        stopping = True
                    else:
                        source = event["source"]
                        if source != worker["source"]:
                            raise RuntimeError("Worker returned a source it did not own")
                        coordinator.release(worker["process"])
                        if event["kind"] == "complete":
                            completed[source] = event["result"]
                            print(
                                json.dumps(
                                    dict(
                                        source=source,
                                        complete=len(completed),
                                        total=len(sources),
                                        **event["result"],
                                    )
                                ),
                                flush=True,
                            )
                        else:
                            failed[source] = event["error"]
                            print(
                                json.dumps(
                                    dict(source=source, status="FAILED", error=event["error"])
                                ),
                                flush=True,
                            )
                        worker.update(state="idle", source=None)
                for worker_id, worker in list(pool.items()):
                    process = worker["process"]
                    if process.exitcode is not None:
                        process.join()
                        coordinator.release(process)
                        if worker["source"]:
                            failed[worker["source"]] = "Worker exited unexpectedly: " + str(
                                process.exitcode
                            )
                        elif worker["state"] == "initializing":
                            failed[
                                "worker_" + str(worker_id)
                            ] = "Worker exited during initialization"
                            stopping = True
                        worker["inbox"].close()
                        del pool[worker_id]
                if stopping:
                    break
                available = memory_reader()
                busy = sum(v["source"] is not None for v in pool.values())
                initializing = any(v["state"] == "initializing" for v in pool.values())
                estimated_next_worker_bytes = max(
                    [worker_memory_bytes]
                    + [resident_bytes(v["process"].pid) or 0 for v in pool.values()]
                )
                for worker in pool.values():
                    worker["peak_resident_bytes"] = max(
                        worker.get("peak_resident_bytes", 0),
                        resident_bytes(worker["process"].pid) or 0,
                    )
                shared_memory = coordinator.snapshot()
                can_spawn = (
                    shared_memory["effective_available_bytes"]
                    >= min_free_bytes + estimated_next_worker_bytes
                )
                if pending and len(pool) < workers and not initializing and can_spawn:
                    inbox = context.Queue()

                    def start_process():
                        process = context.Process(
                            target=worker_main,
                            args=(next_id, configuration, inbox, events, builder_factory),
                            name="pathveil-case-" + str(next_id),
                        )
                        process.start()
                        return process

                    process = coordinator.spawn(
                        start_process, estimated_next_worker_bytes, min_free_bytes
                    )
                    if process is not None:
                        pool[next_id] = dict(
                            process=process,
                            inbox=inbox,
                            state="initializing",
                            source=None,
                            peak_resident_bytes=0,
                        )
                        next_id += 1
                    else:
                        inbox.close()
                # Release an idle runtime at a source boundary when memory is
                # scarce. Running sources finish; their atomic caches survive.
                idle = [v for v in pool.values() if v["state"] == "idle"]
                if (
                    available < min_free_bytes
                    or shared_memory["effective_available_bytes"] < min_free_bytes
                ) and idle:
                    retiring = max(idle, key=lambda v: resident_bytes(v["process"].pid) or 0)
                    retiring["inbox"].put(None)
                    retiring["state"] = "retiring"
                    retired_workers += 1
                # Existing workers may finish their current source below the
                # reserve, but new source work waits until the reserve returns.
                dispatch_denied = False
                if available >= min_free_bytes:
                    for worker in pool.values():
                        if pending and worker["state"] == "idle":
                            source = pending[0]
                            expected_rss = max(worker_memory_bytes, worker["peak_resident_bytes"])
                            if coordinator.dispatch(
                                worker["process"],
                                expected_rss,
                                min_free_bytes,
                                source,
                                worker["inbox"].put,
                            ):
                                pending.popleft()
                                worker.update(state="running", source=source)
                            else:
                                dispatch_denied = True
                shared_memory = coordinator.snapshot()
                busy = sum(v["source"] is not None for v in pool.values())
                initializing = any(v["state"] == "initializing" for v in pool.values())
                if dispatch_denied and not busy and not initializing:
                    idle = [v for v in pool.values() if v["state"] == "idle"]
                    if idle:
                        retiring = max(idle, key=lambda v: resident_bytes(v["process"].pid) or 0)
                        retiring["inbox"].put(None)
                        retiring["state"] = "retiring"
                        retired_workers += 1
                waiting = (
                    bool(pending)
                    and not busy
                    and not initializing
                    and (
                        dispatch_denied
                        or shared_memory["effective_available_bytes"]
                        < min_free_bytes + (worker_memory_bytes if not pool else 0)
                    )
                )
                if waiting:
                    memory_wait_start = memory_wait_start or time.monotonic()
                    if (
                        memory_timeout_seconds
                        and time.monotonic() - memory_wait_start >= memory_timeout_seconds
                    ):
                        failed[
                            "memory_gate"
                        ] = "Memory reserve did not become available before timeout"
                        stopping = True
                else:
                    memory_wait_start = None
                record("waiting_for_memory" if waiting else "running")
                if pending and not pool:
                    time.sleep(poll_seconds)
            if interrupted:
                failed["interrupted"] = "Received signal " + str(interrupted)
            status = "PASS" if len(completed) == len(sources) and not failed else "FAILED"
        except BaseException:
            failed["scheduler"] = traceback.format_exc()
            stopping = True
        finally:
            # Terminate only children created here; never a process group or a
            # PID discovered by name. Completed atomic CASE markers survive.
            for worker in pool.values():
                process = worker["process"]
                if process.is_alive():
                    if stopping:
                        process.terminate()
                    else:
                        worker["inbox"].put(None)
            for worker in pool.values():
                process = worker["process"]
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                worker["inbox"].close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            events.close()
            coordinator.release()
            shared_memory = coordinator.snapshot()
        receipt = record(status)
        receipt.update(results=completed, method_arithmetic_changed=False)
        write_json(result_path, receipt)
    return 128 + interrupted if interrupted else (0 if status == "PASS" else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="Defaults to the run's registered case_workers")
    parser.add_argument("--threads", type=int, help="Must match registered cpu_threads")
    parser.add_argument(
        "--predict-threads", type=int, help="Must match registered case_predict_threads"
    )
    parser.add_argument("--min-free-gib", type=float, default=12)
    parser.add_argument("--worker-memory-gib", type=float, default=16)
    parser.add_argument("--memory-timeout-seconds", type=float, default=0)
    args = parser.parse_args()
    run = registered_run(args.run_dir)
    os.environ["PATHVEIL_PROJECT_ROOT"] = str(PROJECT)
    os.environ["PYTHONHASHSEED"] = "20261015"
    sources, configuration = prepare_configuration(run, args.threads, args.predict_threads)
    resources = configuration["runtime_resources"]
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(resources["cpu_threads"])
    return schedule_cases(
        run,
        sources,
        configuration,
        workers=args.workers if args.workers is not None else resources["case_workers"],
        min_free_bytes=int(args.min_free_gib * GIB),
        worker_memory_bytes=int(args.worker_memory_gib * GIB),
        memory_timeout_seconds=args.memory_timeout_seconds,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
