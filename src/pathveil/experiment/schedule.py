"""Resource-bounded subprocess scheduling across independent AS stages."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pathveil.utils import project_root, read_json, sha256, utc_now, write_json
from .contracts import require
from . import worker


@dataclass(frozen=True)
class Task:
    stage: str
    source: str
    dependencies: tuple = ()

    @property
    def key(self):
        return self.stage + ":" + self.source


def downstream_tasks(sources):
    tasks = []
    for source in sources:
        previous = ()
        for stage in ("pg_train", "pg_test", "train", "annotate", "evaluate", "rescore"):
            task = Task(stage, str(source), previous)
            tasks.append(task)
            previous = (task.key,)
    return tasks


def gpu_free_mib():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        return int(result.stdout.splitlines()[0]) if result.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class Scheduler:
    def __init__(self, run, tasks, config, device="cuda"):
        self.run, self.tasks, self.config = Path(run), list(tasks), dict(config)
        require(len({t.key for t in tasks}) == len(tasks), "Repeated task identity")
        keys = {t.key for t in tasks}
        require(
            all(set(t.dependencies) <= keys for t in tasks), "Task graph has missing dependencies"
        )
        scripts = str(project_root() / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from build_training_cases import WorkerMemoryCoordinator

        self.memory = WorkerMemoryCoordinator(project_root() / "runs")
        available_gpu = gpu_free_mib() if device != "cpu" else None
        require(
            device != "cuda" or available_gpu is not None,
            "CUDA requested, but nvidia-smi is unavailable",
        )
        self.device = "cpu" if device == "cpu" or available_gpu is None else device
        self.done, self.active, self.failed = set(), {}, {}
        self.last_print = 0.0

    def gpu_task(self, task):
        return self.device != "cpu" and task.stage in ("train", "evaluate")

    def slot(self):
        # Held by this parent until its exact child has exited. All dates share slots.
        for index in range(self.config["gpu_workers"]):
            handle = (
                project_root() / "runs" / (".rho_gmpi_gpu_slot_" + str(index) + ".lock")
            ).open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                handle.close()
        return None

    def start(self, task):
        gpu = self.gpu_task(task)
        active_gpu = sum(self.gpu_task(t) for t, *_ in self.active.values())
        active_cpu = len(self.active) - active_gpu
        if (
            active_gpu >= self.config["gpu_workers"]
            if gpu
            else active_cpu >= self.config["cpu_workers"]
        ):
            return False
        gpu_slot = None
        if gpu:
            free = gpu_free_mib()
            # Account for allocations that workers have not yet made after spawn.
            pending = sum(
                max(0, self.config["gpu_worker_mib"] - gpu_memory_mib(process.pid))
                for t, process, *_ in self.active.values()
                if self.gpu_task(t)
            )
            if (
                free is None
                or free - pending
                < self.config["gpu_worker_mib"] + self.config["gpu_free_reserve_mib"]
            ):
                return False
            gpu_slot = self.slot()
            if gpu_slot is None:
                return False
        memory_gib = self.config["gpu_ram_gib"] if gpu else self.config["cpu_ram_gib"]
        if task.stage == "score":
            memory_gib = self.config["score_ram_gib"]
        if task.stage in ("annotate", "rescore"):
            memory_gib = self.config["metadata_ram_gib"]
        log = self.run / "logs" / (task.stage + "_" + task.source + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("ab")
        threads = self.config["gmpi_cpu_threads"] if task.stage in ("train", "evaluate") else 1
        command = [
            sys.executable,
            "-B",
            "-m",
            "pathveil.experiment.worker",
            "--run",
            str(self.run),
            "--stage",
            task.stage,
            "--source",
            task.source,
            "--device",
            self.device,
            "--cpu-threads",
            str(threads),
        ]
        env = dict(
            os.environ,
            PYTHONPATH=str(project_root() / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS=str(threads),
            OPENBLAS_NUM_THREADS=str(threads),
            MKL_NUM_THREADS=str(threads),
            PATHVEIL_PROJECT_ROOT=str(project_root()),
            PYTHONHASHSEED=str(
                read_json(self.run / "PROTOCOL.json")["method"]["gmpi"]["hash_seed"]
            ),
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        process = self.memory.spawn(
            lambda: subprocess.Popen(
                command,
                cwd=project_root(),
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ),
            int(memory_gib * 1024**3),
            int(self.config["free_ram_gib"] * 1024**3),
        )
        if process is None:
            stream.close()
            if gpu_slot:
                gpu_slot.close()
            return False
        self.active[task.key] = (task, process, stream, gpu_slot, str(log))
        return True

    def progress(self):
        value = dict(
            date=read_json(self.run / "PROTOCOL.json")["date"],
            status="FAILED" if self.failed else "RUNNING",
            updated_utc=utc_now(),
            pid=os.getpid(),
            phase="scores" if all(t.stage == "score" for t in self.tasks) else "downstream",
            complete=len(self.done),
            total=len(self.tasks),
            failures=self.failed,
            active=[
                dict(stage=t.stage, source=t.source, pid=p.pid, log=log)
                for t, p, _, _, log in self.active.values()
            ],
            resources=self.memory.snapshot(),
        )
        write_json(self.run / "STATUS.json", value)
        lines = [
            "# " + value["date"] + " 运行进度",
            "",
            "更新时间（UTC）：" + value["updated_utc"],
            "",
            "阶段任务：{}/{}；CPU上限{}，GPU上限{}，保留内存{} GiB。".format(
                len(self.done),
                len(self.tasks),
                self.config["cpu_workers"],
                self.config["gpu_workers"],
                self.config["free_ram_gib"],
            ),
            "",
            "| AS/起点 | 当前阶段 | PID |",
            "|---|---|---:|",
        ]
        lines += [
            "| {} | {} | {} |".format(t.source, t.stage, p.pid) for t, p, *_ in self.active.values()
        ]
        lines += ["", "已完成AS的准确率见REPORT.md和RESULTS.csv。"]
        (self.run / "PROGRESS.md").write_text("\n".join(lines) + "\n")
        if time.monotonic() - self.last_print >= 30:
            print(json.dumps({k: v for k, v in value.items() if k != "resources"}), flush=True)
            self.last_print = time.monotonic()

    def execute(self):
        try:
            return self._execute()
        except BaseException:
            # Only exact subprocess groups created by this dispatcher are stopped.
            for task, process, *_ in self.active.values():
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for task, process, stream, gpu_slot, _ in self.active.values():
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                stream.close()
                if gpu_slot:
                    gpu_slot.close()
                self.memory.release(process)
            self.active.clear()
            raise

    def _execute(self):
        for task in self.tasks:
            if worker.receipt_path(self.run, task.stage, task.source).exists():
                worker.verify(self.run, task.stage, task.source)
                self.done.add(task.key)
        while len(self.done) < len(self.tasks):
            for key, (task, process, stream, gpu_slot, log) in list(self.active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                if gpu_slot:
                    gpu_slot.close()
                self.memory.release(process)
                del self.active[key]
                if code:
                    self.failed[key] = dict(exit_code=code, log=log)
                else:
                    worker.verify(self.run, task.stage, task.source)
                    self.done.add(key)
                    if task.stage == "rescore":
                        from .report import publish

                        publish(self.run)
                    if task.stage == "train":
                        export_selected_model(self.run, task.source)
            if self.failed:
                self.progress()
                if not self.active:
                    raise RuntimeError(
                        "Numerical stage failed; completed outputs retained: "
                        + json.dumps(self.failed)
                    )
            else:
                ready = [
                    t
                    for t in self.tasks
                    if t.key not in self.done
                    and t.key not in self.active
                    and set(t.dependencies) <= self.done
                ]
                # Evaluation and ready GPU training precede starting another long CPU job.
                priority = {
                    "rescore": 0,
                    "annotate": 1,
                    "evaluate": 2,
                    "train": 3,
                    "pg_test": 4,
                    "pg_train": 5,
                    "score": 6,
                }
                for task in sorted(ready, key=lambda t: priority[t.stage]):
                    self.start(task)
                if not self.active and not ready and len(self.done) < len(self.tasks):
                    raise RuntimeError("Cyclic or unsatisfied stage dependencies")
            self.progress()
            if len(self.done) < len(self.tasks):
                time.sleep(3)
        return 0


def gpu_memory_mib(pid):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            fields = [x.strip() for x in line.split(",")]
            if fields[0] == str(pid):
                return int(fields[1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return 0


def export_selected_model(run, source):
    from .utilities import copy_verified

    run = Path(run)
    selected = read_json(run / "T-PG-B/gmpi/selections" / ("as" + source) / "SELECTION_FROZEN.json")
    protocol = read_json(run / "PROTOCOL.json")
    directory = (
        project_root() / "models" / protocol["date"] / "rho_prefix_v1" / run.name / ("as" + source)
    )
    for name, original, expected in [
        ("model.pkl", selected["chosen_model"], selected["chosen_model_sha256"]),
        (
            "training_manifest.json",
            selected["chosen_training_manifest"],
            selected["chosen_training_manifest_sha256"],
        ),
    ]:
        copy_verified(original, directory / name, expected)
    write_json(
        directory / "MODEL.json",
        dict(
            date=protocol["date"],
            source=source,
            run=str(run),
            model_sha256=selected["chosen_model_sha256"],
            selection=selected,
        ),
    )
