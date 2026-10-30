#!/usr/bin/env python3
"""Schedule the three approved date-local experiments through the formal entry point.

An adopted full run keeps running untouched. Other dates prepare their CPU-only
upstream concurrently; completed upstream runs enter one shared GPU slot. The
frozen method and its run identities are never rewritten by this controller.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

APPROVED_DATES = ("20260201", "20260301", "20260401")
TERMINAL = {"complete", "failed"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity(root):
    return {
        str(path.relative_to(root)): file_sha(path)
        for path in sorted((root / "src/pathveil").rglob("*.py"))
    }


def code_hash(identity):
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def process_identity(pid):
    """Return Linux start ticks, avoiding PID-reuse and zombie false positives."""
    try:
        stat = Path("/proc", str(pid), "stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable is unavailable")


def desktop_gpu_process(pid):
    """Recognize only audited desktop binaries, never arbitrary idle compute."""
    try:
        process = Path("/proc", str(int(pid)))
        executable = str((process / "exe").resolve(strict=True))
        # Chromium rewrites argv into one space-delimited process title.
        arguments = (process / "cmdline").read_bytes().replace(b"\0", b" ").split()
        if executable in ("/usr/bin/nautilus", "/usr/bin/gnome-system-monitor"):
            return True
        return executable in ("/opt/google/chrome/chrome", "/usr/share/code/code") and (
            b"--type=gpu-process" in arguments
        )
    except (OSError, ValueError):
        return False


def gpu_idle(min_free_mib, allowed_groups=()):
    try:
        free = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        active = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        devices = [
            tuple(map(int, row.split(",")))
            for row in free.stdout.splitlines()
            if row.strip()
        ]
        if not devices or any(
            len(row) != 2 or row[0] < min_free_mib for row in devices
        ):
            return False
        managed_compute = False
        for row in active.stdout.splitlines():
            if not row.strip():
                continue
            pid, memory = map(int, row.split(","))
            if memory < 0:
                return False
            try:
                managed = bool(allowed_groups) and os.getpgid(pid) in allowed_groups
            except ProcessLookupError:
                managed = False
            if managed:
                managed_compute = True
            elif memory > 1024 or not desktop_gpu_process(pid):
                return False
        return managed_compute or all(row[1] <= 10 for row in devices)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def parse_assignment(value):
    try:
        date, item = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use DATE=VALUE") from error
    if date not in APPROVED_DATES or not item:
        raise argparse.ArgumentTypeError(
            "Only " + ", ".join(APPROVED_DATES) + " are approved"
        )
    return date, item


def locked_run(run):
    path = Path(run) / ".run.lock"
    if not path.is_file():
        return False
    with path.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def active_run_dirs(root, date):
    parent = Path(root) / "runs" / date / "unified255"
    return sorted(
        path.parent for path in parent.glob("*/.run.lock") if locked_run(path.parent)
    )


def choose_default_runs(root, adopted_dates=()):
    """Initialize exactly the same identities/directories as the frozen CLI."""
    root = Path(root).resolve()
    active = {date: active_run_dirs(root, date) for date in APPROVED_DATES}
    for date, directories in active.items():
        if directories and (date not in adopted_dates or len(directories) != 1):
            raise RuntimeError(
                "Experiment "
                + date
                + " is already active; select its run and explicitly adopt its PID"
            )
    source_path = str(root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from pathveil.cli import choose_run, load_method, run_identity
    from pathveil.data.registry import load_registry

    previous_root = os.environ.get("PATHVEIL_PROJECT_ROOT")
    os.environ["PATHVEIL_PROJECT_ROOT"] = str(root)
    try:
        result = {}
        for date in APPROVED_DATES:
            if active[date]:
                result[date] = str(active[date][0])
                continue
            registry = load_registry(date, project_root=root)
            method = load_method(registry, root)
            identity = run_identity(date, registry, method)
            result[date] = str(choose_run(root, date, identity))
        return result
    finally:
        if previous_root is None:
            os.environ.pop("PATHVEIL_PROJECT_ROOT", None)
        else:
            os.environ["PATHVEIL_PROJECT_ROOT"] = previous_root


@dataclass
class Job:
    date: str
    run: Path
    stage: str = "pending_upstream"
    adopted_pid: int | None = None
    adopted_start: str | None = None
    adopted_upstream: bool = False
    precompute: bool = False
    process: subprocess.Popen | None = field(default=None, repr=False)
    log_stream: object = field(default=None, repr=False)
    error: str | None = None
    exit_code: int | None = None
    started_utc: str | None = None
    finished_utc: str | None = None

    def record(self):
        return {
            "date": self.date,
            "run_dir": str(self.run),
            "stage": self.stage,
            "pid": self.process.pid if self.process else self.adopted_pid,
            "adopted": self.adopted_pid is not None,
            "adopted_upstream": self.adopted_upstream,
            "precompute": self.precompute,
            "exit_code": self.exit_code,
            "error": self.error,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
        }


class Scheduler:
    def __init__(
        self,
        root,
        batch,
        jobs,
        max_cpu_jobs=3,
        min_available_gib=12,
        threads=4,
        case_runner=None,
        case_workers=2,
        estimated_job_gib=16,
        gpu_min_free_mib=12288,
        max_gpu_jobs=1,
    ):
        self.root = Path(root).resolve()
        self.batch = Path(batch).resolve()
        self.jobs = jobs
        self.max_cpu_jobs = max_cpu_jobs
        self.min_available_gib = min_available_gib
        self.threads = threads
        self.case_runner = Path(
            case_runner or self.root / "scripts/build_training_cases.py"
        ).resolve()
        self.case_workers = case_workers
        self.estimated_job_gib = estimated_job_gib
        self.gpu_min_free_mib = gpu_min_free_mib
        self.max_gpu_jobs = max_gpu_jobs
        self.starting_reservations = {}
        self.code = code_identity(self.root)
        if not self.code:
            raise ValueError("No formal PathVeil source files found")
        self.code_sha256 = code_hash(self.code)
        self.stop_requested = False
        self.gpu_lock = None
        self.lock = None
        self.project_lock = None
        self.waiting_for_memory = False
        self.events = []
        if not 1 <= max_cpu_jobs <= 3 or not 1 <= threads <= 20:
            raise ValueError("CPU jobs must be 1..3 and threads must be 1..20")
        if not 1 <= max_gpu_jobs <= 3:
            raise ValueError("Concurrent downstream date jobs must be 1..3")
        if min_available_gib < 0 or estimated_job_gib < 0:
            raise ValueError("Memory headroom must be nonnegative")
        if not 1 <= case_workers <= 5:
            raise ValueError("Case worker count must be 1..5")
        if not jobs or len({job.date for job in jobs}) != len(jobs):
            raise ValueError("Select distinct approved dates")
        for job in jobs:
            if job.date not in APPROVED_DATES:
                raise ValueError("Unapproved experiment date: " + job.date)
            if self.root / "runs" not in job.run.resolve().parents:
                raise ValueError("Run directories must be under project runs/")
            active = set(active_run_dirs(self.root, job.date))
            if locked_run(job.run):
                active.add(job.run)
            if active and (job.adopted_pid is None or active != {job.run}):
                raise RuntimeError(
                    "Experiment "
                    + job.date
                    + " is already active; explicitly adopt its PID and run"
                )
            self.validate_job(job)
            status = (
                read_json(job.run / "status.json")
                if (job.run / "status.json").is_file()
                else {}
            )
            if job.adopted_pid is not None:
                if (
                    status.get("pid") != job.adopted_pid
                    or status.get("date") != job.date
                ):
                    raise ValueError(
                        "Adopted PID does not match the run status: " + job.date
                    )
                job.adopted_start = process_identity(job.adopted_pid)
                if job.adopted_start is None:
                    raise ValueError(
                        "Adopted PID is not running: " + str(job.adopted_pid)
                    )
                job.stage = (
                    "adopted_upstream" if job.adopted_upstream else "adopted_full"
                )
            elif status.get("stage") == "complete" and status.get("status") == "PASS":
                job.stage = "complete"
            elif self.has_release(job):
                job.stage = "ready_downstream"
            elif job.precompute:
                if not self.case_runner.is_file():
                    raise ValueError(
                        "Case runner is unavailable: " + str(self.case_runner)
                    )
                job.stage = "pending_precompute"

    def validate_job(self, job):
        identity = read_json(job.run / "RUN_IDENTITY.json")
        if identity.get("date") != job.date or identity.get("code_sha256") != self.code:
            raise ValueError(
                "Date/source identity differs from frozen run: " + job.date
            )
        if identity.get("registry_sha256") != file_sha(
            self.root / "configs/dates" / (job.date + ".json")
        ):
            raise ValueError("Date registry changed: " + job.date)
        registry = read_json(self.root / "configs/dates" / (job.date + ".json"))
        method_path = Path(
            registry.get("method_config", "configs/method_unified255.json")
        )
        if not method_path.is_absolute():
            method_path = self.root / method_path
        if identity.get("method_config") != read_json(method_path):
            raise ValueError("Method configuration changed: " + job.date)

    def verify_code(self):
        if code_identity(self.root) != self.code:
            raise RuntimeError(
                "Frozen source changed; refusing to launch another stage"
            )

    @staticmethod
    def has_release(job):
        path = job.run / "upstream/RELEASE.json"
        return path.is_file() and read_json(path).get("status") == "PASS"

    def command(self, job, upstream, precompute=False):
        if precompute:
            return [
                sys.executable,
                str(self.case_runner),
                "--run-dir",
                str(job.run),
                "--workers",
                str(self.case_workers),
                "--min-free-gib",
                str(self.min_available_gib),
            ]
        command = ["bash", str(self.root / "scripts/run_experiment.sh"), job.date]
        command += ["--run-dir", str(job.run), "--device", "cuda"]
        if upstream:
            command.append("--upstream-only")
        return command

    def event(self, name, **details):
        value = {"utc": now(), "event": name, **details}
        self.events.append(value)
        print(json.dumps(value, sort_keys=True), flush=True)
        if self.batch.exists():
            with (self.batch / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(value, sort_keys=True) + "\n")

    def snapshot(self):
        return {
            "controller_pid": os.getpid(),
            "updated_utc": now(),
            "source_code_sha256": self.code_sha256,
            "dates": [job.date for job in self.jobs],
            "max_cpu_jobs_including_adopted": self.max_cpu_jobs,
            "max_gpu_jobs": self.max_gpu_jobs,
            "max_downstream_jobs": self.max_gpu_jobs,
            "threads_per_new_job": self.threads,
            "case_workers": self.case_workers,
            "case_runner": str(self.case_runner),
            "min_available_gib": self.min_available_gib,
            "estimated_new_job_gib": self.estimated_job_gib,
            "gpu_min_free_mib": self.gpu_min_free_mib,
            "waiting_for_memory": self.waiting_for_memory,
            "stop_requested": self.stop_requested,
            "jobs": [job.record() for job in self.jobs],
        }

    def acquire_gpu(self):
        if self.gpu_lock:
            return True
        lock = (self.root / "runs/.priority_gpu.lock").open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return False
        self.gpu_lock = lock
        return True

    def release_gpu(self):
        if self.gpu_lock:
            self.gpu_lock.close()
            self.gpu_lock = None

    def launch(self, job, upstream, precompute=False):
        self.verify_code()
        self.validate_job(job)
        env = os.environ.copy()
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = str(self.threads)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # CPU-only stages cannot accidentally reserve the shared GPU.
        if upstream or precompute:
            env["CUDA_VISIBLE_DEVICES"] = ""
        stage = "precompute" if precompute else "upstream" if upstream else "downstream"
        job.log_stream = (self.batch / (job.date + "." + stage + ".log")).open("a")
        try:
            job.process = subprocess.Popen(
                self.command(job, upstream, precompute),
                cwd=self.root,
                env=env,
                stdout=job.log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            job.log_stream.close()
            job.log_stream = None
            raise
        job.stage = stage
        job.started_utc = now()
        estimated = (
            self.estimated_job_gib * self.case_workers
            if precompute
            else self.estimated_job_gib
        )
        self.starting_reservations[job.process.pid] = (time.monotonic(), estimated)
        self.event("launched", date=job.date, stage=stage, pid=job.process.pid)

    def finish(self, job, returncode, adopted=False):
        status_file = job.run / "status.json"
        status = read_json(status_file) if status_file.is_file() else {}
        full = job.stage in ("downstream", "adopted_full")
        precompute = job.stage == "precompute"
        precompute_receipt = job.run / "upstream/CASE_PRECOMPUTE.json"
        success = (returncode in (None, 0)) and (
            precompute_receipt.is_file()
            and read_json(precompute_receipt).get("status") == "PASS"
            if precompute
            else status.get("stage") == "complete" and status.get("status") == "PASS"
            if full
            else self.has_release(job)
            and status.get("stage") == "upstream"
            and status.get("status") == "complete"
        )
        previous_stage = job.stage
        compatibility_retry = False
        metadata_error = "Only explicit date-local source-disjoint models are accepted"
        bundle_path = job.run / "upstream/neighbor_models/neighbor_bundle.json"
        normalizer = self.root / "scripts/normalize_neighbor_protocol.py"
        if (
            not success
            and metadata_error in status.get("error", "")
            and bundle_path.is_file()
            and normalizer.is_file()
            and read_json(bundle_path).get("protocol")
            == "date_local_all_eligible_source_disjoint_v2"
        ):
            # A frozen producer used the source-selection tag in the model
            # isolation field. Audit and normalize only this known metadata
            # schema; never discard or refit completed model weights.
            try:
                repaired = subprocess.run(
                    [sys.executable, str(normalizer), "--run-dir", str(job.run)],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                compatibility_retry = repaired.returncode == 0
                diagnostic = repaired.stdout if compatibility_retry else repaired.stderr
            except (OSError, subprocess.SubprocessError) as error:
                diagnostic = repr(error)
            self.event(
                "protocol_metadata_compatibility",
                date=job.date,
                normalized=compatibility_retry,
                diagnostic=diagnostic[-2000:],
            )
        if job.process:
            self.starting_reservations.pop(job.process.pid, None)
        job.exit_code = returncode
        job.finished_utc = now()
        if compatibility_retry:
            job.stage = "pending_upstream"
            job.error = None
            job.adopted_pid = None
            job.adopted_start = None
            job.adopted_upstream = False
        elif success:
            job.stage = (
                "complete"
                if full
                else "pending_upstream"
                if precompute
                else "ready_downstream"
            )
        else:
            job.stage = "failed"
            job.error = (
                status.get("error")
                or "Process exited without verified stage completion"
            )
        if job.log_stream:
            job.log_stream.close()
            job.log_stream = None
        job.process = None
        if previous_stage in ("downstream", "adopted_full") and not any(
            item.stage in ("downstream", "adopted_full") for item in self.jobs
        ):
            self.release_gpu()
        self.event(
            "finished",
            date=job.date,
            previous_stage=previous_stage,
            stage=job.stage,
            exit_code=returncode,
            adopted=adopted,
            error=job.error,
        )

    def poll(self):
        for job in self.jobs:
            if job.stage in ("adopted_full", "adopted_upstream"):
                if process_identity(job.adopted_pid) != job.adopted_start:
                    self.finish(job, None, adopted=True)
            elif job.process:
                returncode = job.process.poll()
                if returncode is not None:
                    self.finish(job, returncode)

    def tick(self):
        self.poll()
        self.waiting_for_memory = False
        if self.stop_requested:
            return
        for job in self.jobs:
            cpu_active = sum(
                item.stage
                in ("upstream", "precompute", "adopted_full", "adopted_upstream")
                for item in self.jobs
            )
            if (
                job.stage not in ("pending_upstream", "pending_precompute")
                or cpu_active >= self.max_cpu_jobs
            ):
                continue
            reserved = sum(
                estimated
                for started, estimated in self.starting_reservations.values()
                if time.monotonic() - started < 30
            )
            usable = available_gib() - reserved
            growth = self.estimated_job_gib * (
                self.case_workers if job.stage == "pending_precompute" else 1
            )
            if usable < self.min_available_gib + growth:
                self.waiting_for_memory = True
                continue
            try:
                self.launch(
                    job, upstream=True, precompute=job.stage == "pending_precompute"
                )
            except (OSError, ValueError) as error:
                job.stage, job.error = "failed", repr(error)
                self.event("launch_failed", date=job.date, error=job.error)
        gpu_jobs = [
            job for job in self.jobs if job.stage in ("adopted_full", "downstream")
        ]
        if len(gpu_jobs) < self.max_gpu_jobs:
            ready = next(
                (job for job in self.jobs if job.stage == "ready_downstream"), None
            )
            groups = set()
            for job in gpu_jobs:
                pid = job.process.pid if job.process else job.adopted_pid
                if job.process or process_identity(pid) == job.adopted_start:
                    # Formal workers inherit the controller-launched run's
                    # independent process group; unrelated compute is excluded.
                    try:
                        group = os.getpgid(pid)
                        if group == pid:
                            groups.add(group)
                    except ProcessLookupError:
                        pass
            reserved = sum(
                estimate
                for started, estimate in self.starting_reservations.values()
                if time.monotonic() - started < 30
            )
            if (
                ready
                and available_gib() - reserved
                >= self.min_available_gib + self.estimated_job_gib
                and gpu_idle(self.gpu_min_free_mib, allowed_groups=groups)
                and self.acquire_gpu()
            ):
                try:
                    self.launch(ready, upstream=False)
                except (OSError, ValueError) as error:
                    ready.stage, ready.error = "failed", repr(error)
                    if not gpu_jobs:
                        self.release_gpu()
                    self.event("launch_failed", date=ready.date, error=ready.error)

    def stop_owned(self):
        """Terminate only process groups created here; adopted jobs are untouched."""
        owned = [job for job in self.jobs if job.process and job.process.poll() is None]
        for job in owned:
            try:
                os.killpg(job.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 15
        for job in owned:
            remaining = max(0, deadline - time.monotonic())
            try:
                job.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(job.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                job.process.wait()
        self.poll()

    def run(self, poll_seconds=5):
        self.batch.mkdir(parents=True, exist_ok=True)
        self.project_lock = (self.root / "runs/.priority_scheduler.lock").open("a+")
        try:
            fcntl.flock(self.project_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.project_lock.close()
            raise RuntimeError(
                "Another priority scheduler already controls this project"
            ) from error
        self.lock = (self.batch / ".scheduler.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            self.project_lock.close()
            raise
        previous = {}

        def request_stop(signum, frame):
            self.stop_requested = True

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, request_stop)
        self.event("scheduler_started", **self.snapshot())
        try:
            while not all(job.stage in TERMINAL for job in self.jobs):
                self.tick()
                write_json(self.batch / "STATUS.json", self.snapshot())
                if self.stop_requested:
                    self.stop_owned()
                    break
                time.sleep(poll_seconds)
            exit_code = (
                130
                if self.stop_requested
                else int(any(job.stage == "failed" for job in self.jobs))
            )
            write_json(self.batch / "STATUS.json", self.snapshot())
            write_json(
                self.batch / "EXIT.json",
                {"completed_utc": now(), "exit_code": exit_code},
            )
            return exit_code
        except BaseException as error:
            self.event("scheduler_error", error=repr(error))
            write_json(
                self.batch / "STATUS.json", {**self.snapshot(), "error": repr(error)}
            )
            write_json(
                self.batch / "EXIT.json",
                {
                    "completed_utc": now(),
                    "exit_code": 1,
                    "error": repr(error),
                },
            )
            raise
        finally:
            # A controller error must never leave its owned GPU process untracked.
            self.stop_owned()
            self.release_gpu()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            self.lock.close()
            self.project_lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_assignment,
        default=[],
        metavar="DATE=RUN_DIR",
    )
    parser.add_argument(
        "--adopt",
        action="append",
        type=parse_assignment,
        default=[],
        metavar="DATE=PID",
    )
    parser.add_argument(
        "--adopt-upstream",
        action="append",
        type=parse_assignment,
        default=[],
        metavar="DATE=PID",
        help="Observe an already-running upstream-only CLI",
    )
    parser.add_argument(
        "--precompute",
        action="append",
        type=parse_assignment,
        default=[],
        metavar="DATE=RUN_DIR",
        help="Build this run's cases in CPU workers before resuming full CLI",
    )
    parser.add_argument("--case-runner", type=Path)
    parser.add_argument("--case-workers", type=int, default=2)
    parser.add_argument("--max-cpu-jobs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-available-gib", type=float, default=12)
    parser.add_argument("--estimated-job-gib", type=float, default=16)
    parser.add_argument("--gpu-min-free-mib", type=int, default=12288)
    parser.add_argument(
        "--max-downstream-jobs",
        "--max-gpu-jobs",
        dest="max_gpu_jobs",
        type=int,
        default=1,
    )
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--expected-code-sha256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve()
        runs = dict(args.run)
        adopted = dict(args.adopt)
        adopted_upstream = dict(args.adopt_upstream)
        precompute = dict(args.precompute)
        if (
            len(runs) != len(args.run)
            or len(adopted) != len(args.adopt)
            or len(adopted_upstream) != len(args.adopt_upstream)
            or len(precompute) != len(args.precompute)
        ):
            raise ValueError("Duplicate date assignment")
        if set(adopted) & set(adopted_upstream):
            raise ValueError("A run cannot be adopted as both full and upstream-only")
        adopted.update(adopted_upstream)
        for date, run in precompute.items():
            if date in adopted:
                raise ValueError("An adopted run cannot also precompute cases")
            if date in runs and Path(runs[date]).resolve() != Path(run).resolve():
                raise ValueError(
                    "Precompute and selected run directories differ: " + date
                )
            runs[date] = run
        if not runs:
            runs = choose_default_runs(root, adopted)
        if set(adopted) - set(runs):
            raise ValueError("Every adopted date needs an explicit selected run")
        if args.poll_seconds <= 0:
            raise ValueError("Poll interval must be positive")
        jobs = [
            Job(
                date,
                Path(run).resolve(),
                adopted_pid=int(adopted[date]) if date in adopted else None,
                adopted_upstream=date in adopted_upstream,
                precompute=date in precompute,
            )
            for date, run in runs.items()
        ]
        scheduler = Scheduler(
            root,
            args.batch_dir,
            jobs,
            args.max_cpu_jobs,
            args.min_available_gib,
            args.threads,
            args.case_runner,
            args.case_workers,
            args.estimated_job_gib,
            args.gpu_min_free_mib,
            args.max_gpu_jobs,
        )
        if (
            args.expected_code_sha256
            and scheduler.code_sha256 != args.expected_code_sha256
        ):
            raise ValueError("Source hash differs from requested frozen source")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        **scheduler.snapshot(),
                        "commands": [
                            {
                                "date": job.date,
                                "upstream": scheduler.command(job, True),
                                "downstream": scheduler.command(job, False),
                                "precompute": scheduler.command(job, True, True)
                                if job.precompute
                                else None,
                            }
                            for job in jobs
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        return scheduler.run(args.poll_seconds)
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
