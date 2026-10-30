"""Exercise real spawn, dispatch, locks, failures and restart without corpora."""

import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_training_cases as scheduler


def abundant_memory():
    return 128 * scheduler.GIB


def insufficient_memory():
    return 5 * scheduler.GIB


def shared_two_worker_memory():
    return 44 * scheduler.GIB


class CoordinatedSmallBuilder:
    def __init__(self, configuration):
        self.configuration = configuration
        time.sleep(0.15)

    def __call__(self, source):
        time.sleep(0.12)
        return dict(worker_pid=os.getpid())


class BarrierMemory:
    def __init__(self, project):
        self.marker = Path(project) / "barrier_workers.txt"

    def __call__(self):
        count = len(self.marker.read_text().splitlines()) if self.marker.exists() else 0
        return (44 if count >= 3 else 128) * scheduler.GIB


class BarrierBuilder(CoordinatedSmallBuilder):
    def __init__(self, configuration):
        marker = Path(configuration["project_root"]) / "barrier_workers.txt"
        with marker.open("a") as stream:
            stream.write(str(os.getpid()) + "\n")
        deadline = time.monotonic() + 5
        while len(marker.read_text().splitlines()) < 3:
            if time.monotonic() > deadline:
                raise RuntimeError("Fixture worker barrier timed out")
            time.sleep(0.005)


def parallel_case_helper(project, run, queue, barrier=False):
    code = scheduler.schedule_cases(
        Path(run),
        list(range(10, 12 if barrier else 16)),
        dict(project_root=project, run_dir=run),
        workers=1 if barrier else 3,
        memory_reader=BarrierMemory(project) if barrier else shared_two_worker_memory,
        builder_factory=BarrierBuilder if barrier else CoordinatedSmallBuilder,
        poll_seconds=0.01,
        memory_timeout_seconds=5,
    )
    queue.put((run, code))


class SmallBuilder:
    """Observable worker fixture; cache rejects changes and preserves files."""

    def __init__(self, configuration):
        self.config = configuration
        self.run = Path(configuration["run_dir"])
        with (self.run / "initializations.log").open("a") as stream:
            stream.write(str(os.getpid()) + "\n")
        with (self.run / "runtime_initializations.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        pid=os.getpid(),
                        cpu_threads=os.environ.get("OMP_NUM_THREADS"),
                        resources=configuration.get("runtime_resources"),
                    )
                )
                + "\n"
            )

    def __call__(self, source):
        time.sleep(self.config.get("delay", 0.02))
        if source == self.config.get("fail_source"):
            raise RuntimeError("deliberate source failure")
        directory = self.run / "upstream/training_cases" / source
        payload, marker = directory / "payload.txt", directory / "CASE.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if scheduler.sha256(payload) != previous["sha"]:
                raise RuntimeError("Changed fixture payload")
            return dict(reused=True, worker_pid=os.getpid())
        with (self.run / "computations.log").open("a") as stream:
            stream.write(source + "\n")
        payload.write_text("case " + source)
        scheduler.write_json(marker, dict(sha=scheduler.sha256(payload)))
        return dict(reused=False, worker_pid=os.getpid())


def long_running_schedule(run):
    scheduler.schedule_cases(
        Path(run),
        [10, 11],
        dict(run_dir=run, project_root=run, delay=60),
        builder_factory=SmallBuilder,
        memory_reader=abundant_memory,
        poll_seconds=0.01,
    )


class PressureAfterThreeWorkers:
    def __init__(self, run):
        self.run = run
        self.recovered = False

    def __call__(self):
        path = self.run / "initializations.log"
        pids = path.read_text().splitlines() if path.exists() else []
        if len(pids) < 3 or self.recovered:
            return 128 * scheduler.GIB
        alive = 0
        for pid in pids:
            try:
                os.kill(int(pid), 0)
                alive += 1
            except ProcessLookupError:
                pass
        self.recovered = alive < 3
        return (128 if self.recovered else 5) * scheduler.GIB


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pathveil_case_schedule_")
        self.run = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_cases(self, sources, **kwargs):
        configuration = dict(
            run_dir=str(self.run),
            project_root=str(self.run),
            fail_source=kwargs.pop("fail_source", None),
            delay=kwargs.pop("delay", 0.02),
        )
        resources = kwargs.pop("runtime_resources", None)
        if resources is not None:
            configuration["runtime_resources"] = resources
        return scheduler.schedule_cases(
            self.run,
            sources,
            configuration,
            builder_factory=SmallBuilder,
            memory_reader=kwargs.pop("memory_reader", abundant_memory),
            poll_seconds=0.01,
            **kwargs,
        )

    def test_sources_are_computed_once_by_persistent_spawn_workers(self):
        self.assertEqual(self.run_cases(list(range(10, 18)), workers=2), 0)
        sources = (self.run / "computations.log").read_text().splitlines()
        self.assertEqual(sorted(sources, key=int), list(map(str, range(10, 18))))
        self.assertEqual(len(sources), len(set(sources)))
        initialized = (self.run / "initializations.log").read_text().splitlines()
        self.assertLessEqual(len(initialized), 2)
        self.assertEqual(len(initialized), 2)
        self.assertEqual(len(initialized), len(set(initialized)))
        self.assertNotIn(str(os.getpid()), initialized)
        result = scheduler.read_json(self.run / "upstream/CASE_PRECOMPUTE.json")
        self.assertEqual(result["start_method"], "spawn")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["completed"], 8)

    def test_five_workers_record_explicit_resources_without_duplicate_sources(self):
        resources = dict(cpu_threads=8, case_predict_threads=1, case_workers=5)
        self.assertEqual(
            self.run_cases(list(range(10, 34)), workers=5, delay=0.05, runtime_resources=resources),
            0,
        )
        result = scheduler.read_json(self.run / "upstream/CASE_PRECOMPUTE.json")
        self.assertEqual(result["runtime_resources"], resources)
        initialized = [
            json.loads(row)
            for row in (self.run / "runtime_initializations.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(initialized), 5)
        # Each spawned case worker owns one prediction thread; the parent's
        # eight-thread fitting budget remains in the registered resource plan.
        self.assertTrue(all(row["cpu_threads"] == "1" for row in initialized))
        self.assertTrue(all(row["resources"] == resources for row in initialized))
        sources = (self.run / "computations.log").read_text().splitlines()
        self.assertEqual(len(sources), len(set(sources)))
        self.assertEqual(result["completed"], 24)

    def test_cli_caller_can_reuse_its_run_lock_for_embedded_case_pool(self):
        with scheduler.exclusive_lock(self.run / ".run.lock"):
            self.assertEqual(self.run_cases([10, 11], run_lock_held=True), 0)
        result = scheduler.read_json(self.run / "upstream/CASE_PRECOMPUTE.json")
        self.assertTrue(result["run_lock_held_by_caller"])

    def test_memory_pressure_retires_idle_worker_without_losing_source_queue(self):
        self.assertEqual(
            self.run_cases(
                list(range(10, 34)),
                workers=3,
                delay=0.05,
                memory_reader=PressureAfterThreeWorkers(self.run),
            ),
            0,
        )
        result = scheduler.read_json(self.run / "upstream/CASE_PRECOMPUTE.json")
        self.assertGreaterEqual(result["retired_workers"], 1)
        self.assertEqual(result["completed"], 24)
        self.assertFalse(result["failed"])

    def test_three_helpers_share_initialization_and_source_growth_reservations(self):
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        helpers = []
        for date in ("20260201", "20260301", "20260401"):
            run = self.run / "runs" / date / "fixture"
            (run / "upstream").mkdir(parents=True)
            process = context.Process(
                target=parallel_case_helper, args=(str(self.run), str(run), results)
            )
            process.start()
            helpers.append(process)
        ledger_path = self.run / "runs/.case_worker_memory.json"
        max_reservations, phases = 0, set()
        deadline = time.monotonic() + 15
        while any(process.is_alive() for process in helpers) and time.monotonic() < deadline:
            if ledger_path.exists():
                ledger = scheduler.read_json(ledger_path)
                max_reservations = max(max_reservations, len(ledger["reservations"]))
                phases.update(record["phase"] for record in ledger["reservations"].values())
                self.assertLessEqual(ledger["reserved_growth_bytes"], 32 * scheduler.GIB)
            time.sleep(0.005)
        for process in helpers:
            process.join(timeout=1)
            self.assertFalse(process.is_alive(), "Fixture helper did not finish")
            self.assertEqual(process.exitcode, 0)
        self.assertEqual([results.get(timeout=2)[1] for _ in helpers], [0, 0, 0])
        results.close()
        self.assertEqual(max_reservations, 2)
        self.assertIn("initializing", phases)
        self.assertIn("source", phases)
        self.assertEqual(scheduler.read_json(ledger_path)["reservations"], {})
        for date in ("20260201", "20260301", "20260401"):
            report = scheduler.read_json(
                self.run / "runs" / date / "fixture/upstream/CASE_PRECOMPUTE.json"
            )
            self.assertEqual(report["completed_sources"], list(map(str, range(10, 16))))
            self.assertFalse(report["failed"])

    def test_ready_keeps_budget_until_source_completion_and_rss_offsets_growth(self):
        owner = os.getpid()
        identities = {owner: "owner", 101: "worker", 102: "another"}
        rss = {101: 4 * scheduler.GIB, 102: 0}
        available = [32 * scheduler.GIB]
        coordinator = scheduler.WorkerMemoryCoordinator(
            self.run / "runs",
            memory_reader=lambda: available[0],
            resident_reader=lambda pid: rss.get(pid, 0),
            identity_reader=identities.get,
        )
        worker = SimpleNamespace(pid=101)
        self.assertIs(
            coordinator.spawn(lambda: worker, 16 * scheduler.GIB, 12 * scheduler.GIB), worker
        )
        coordinator.ready(worker)
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot["reserved_growth_bytes"], 12 * scheduler.GIB)
        self.assertIsNone(
            coordinator.spawn(
                lambda: SimpleNamespace(pid=102), 16 * scheduler.GIB, 12 * scheduler.GIB
            )
        )
        enqueued = []
        self.assertTrue(
            coordinator.dispatch(
                worker, 16 * scheduler.GIB, 12 * scheduler.GIB, "10", enqueued.append
            )
        )
        self.assertEqual(enqueued, ["10"])
        self.assertEqual(coordinator.snapshot()["reserved_growth_bytes"], 12 * scheduler.GIB)
        coordinator.release(worker)
        self.assertIsNotNone(
            coordinator.spawn(
                lambda: SimpleNamespace(pid=102), 16 * scheduler.GIB, 12 * scheduler.GIB
            )
        )
        rss[102] = 16 * scheduler.GIB
        self.assertEqual(coordinator.snapshot()["reserved_growth_bytes"], 0)
        # Temporary arrays can be freed before a source finishes. Keep the
        # live lease so that its next growth is charged again after an RSS dip.
        rss[102] = 10 * scheduler.GIB
        self.assertEqual(coordinator.snapshot()["reserved_growth_bytes"], 6 * scheduler.GIB)
        coordinator.release()
        self.assertEqual(coordinator.snapshot()["reservations"], 0)

    def test_ready_helpers_recover_after_external_growth_exhausts_shared_reserve(self):
        context = multiprocessing.get_context("spawn")
        results, helpers, runs = context.Queue(), [], []
        for date in ("20260201", "20260301", "20260401"):
            run = self.run / "runs" / date / "fixture"
            (run / "upstream").mkdir(parents=True)
            runs.append(run)
            process = context.Process(
                target=parallel_case_helper, args=(str(self.run), str(run), results, True)
            )
            process.start()
            helpers.append(process)
        for process in helpers:
            process.join(timeout=10)
            self.assertFalse(process.is_alive(), "Idle initialization leases deadlocked")
            self.assertEqual(process.exitcode, 0)
        self.assertEqual([results.get(timeout=2)[1] for _ in helpers], [0, 0, 0])
        results.close()
        reports = [scheduler.read_json(run / "upstream/CASE_PRECOMPUTE.json") for run in runs]
        self.assertGreaterEqual(sum(report["retired_workers"] for report in reports), 1)
        self.assertTrue(all(report["completed_sources"] == ["10", "11"] for report in reports))
        ledger = scheduler.read_json(self.run / "runs/.case_worker_memory.json")
        self.assertEqual(ledger["reservations"], {})

    def test_dead_or_reused_pid_is_cleaned_but_expired_live_orphan_keeps_budget(self):
        owner = os.getpid()
        identities = {owner: "owner", 101: "worker"}
        coordinator = scheduler.WorkerMemoryCoordinator(
            self.run / "runs",
            memory_reader=abundant_memory,
            resident_reader=lambda pid: 0,
            identity_reader=identities.get,
        )
        worker = SimpleNamespace(pid=101)
        coordinator.spawn(lambda: worker, 16 * scheduler.GIB, 12 * scheduler.GIB)
        ledger = scheduler.read_json(coordinator.path)
        record = next(iter(ledger["reservations"].values()))
        record.update(owner_pid=999, owner_start_ticks="departed", expires_epoch=0)
        scheduler.write_json(coordinator.path, ledger)
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot["reserved_growth_bytes"], 16 * scheduler.GIB)
        self.assertEqual(snapshot["orphaned_reservations"], 1)
        identities[101] = "different-process-start"
        self.assertEqual(coordinator.snapshot()["reservations"], 0)

    def test_thread_overrides_and_method_snapshot_cannot_change_frozen_run(self):
        (self.run / "data").mkdir()
        (self.run / "upstream").mkdir()
        config = dict(runtime_resources=dict(cpu_threads=8, case_predict_threads=1, case_workers=3))
        scheduler.write_json(self.run / "RUN_IDENTITY.json", dict(method_config=config))
        for relative in (
            "data/data_manifest.json",
            "upstream/NEIGHBOR_BUNDLE.json",
            "upstream/split.json",
        ):
            scheduler.write_json(self.run / relative, {})
        with self.assertRaisesRegex(ValueError, "cpu_threads.*frozen"):
            scheduler.prepare_configuration(self.run, cpu_threads=4)
        with self.assertRaisesRegex(ValueError, "case_predict_threads.*frozen"):
            scheduler.prepare_configuration(self.run, predict_threads=2)
        scheduler.write_json(self.run / "method_config.json", {})
        with self.assertRaisesRegex(ValueError, "snapshot differs"):
            scheduler.prepare_configuration(self.run)

    def test_resume_validates_existing_case_without_recomputing(self):
        self.assertEqual(self.run_cases([10, 11]), 0)
        log = (self.run / "computations.log").read_text()
        self.assertEqual(self.run_cases([10, 11]), 0)
        self.assertEqual(log, (self.run / "computations.log").read_text())
        result = scheduler.read_json(self.run / "upstream/CASE_PRECOMPUTE.json")
        self.assertTrue(all(row["reused"] for row in result["results"].values()))
        (self.run / "upstream/training_cases/10/payload.txt").write_text("tampered")
        self.assertEqual(self.run_cases([10, 11]), 1)
        result = scheduler.read_json(self.run / "upstream/CASE_PRECOMPUTE.json")
        self.assertIn("10", result["failed"])
        self.assertEqual(result["completed_sources"], ["11"])

    def test_source_failure_does_not_omit_other_sources_or_return_success(self):
        self.assertEqual(self.run_cases([10, 11, 12], fail_source="11"), 1)
        result = scheduler.read_json(self.run / "upstream/CASE_PRECOMPUTE.json")
        self.assertEqual(result["completed_sources"], ["10", "12"])
        self.assertEqual(list(result["failed"]), ["11"])
        self.assertEqual(result["status"], "FAILED")

    def test_run_lock_prevents_workers_and_concurrent_formal_cli(self):
        with scheduler.exclusive_lock(self.run / ".run.lock"):
            with self.assertRaisesRegex(RuntimeError, "lock"):
                self.run_cases([10])
        self.assertFalse((self.run / "initializations.log").exists())
        self.assertEqual(self.run_cases([10]), 0)

    def test_case_lock_prevents_overwrite(self):
        case = self.run / "upstream/training_cases/10/.case.lock"
        with scheduler.exclusive_lock(case):
            self.assertEqual(self.run_cases([10, 11]), 1)
        self.assertFalse((case.parent / "CASE.json").exists())
        self.assertTrue((self.run / "upstream/training_cases/11/CASE.json").exists())

    def test_memory_gate_starts_no_worker_below_reserve_and_allocation(self):
        self.assertEqual(
            self.run_cases([10], memory_reader=insufficient_memory, memory_timeout_seconds=0.03),
            1,
        )
        self.assertFalse((self.run / "initializations.log").exists())
        result = scheduler.read_json(self.run / "upstream/CASE_PRECOMPUTE.json")
        self.assertIn("memory_gate", result["failed"])
        self.assertEqual(result["pending"], ["10"])

    def test_duplicate_queue_and_external_run_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.run_cases([10, 10])
        with self.assertRaisesRegex(ValueError, "inside"):
            scheduler.registered_run(self.run)

    def test_sigterm_stops_only_created_workers_and_releases_run_lock(self):
        process = multiprocessing.get_context("spawn").Process(
            target=long_running_schedule, args=(str(self.run),)
        )
        process.start()
        progress = self.run / "upstream/CASE_PRECOMPUTE_PROGRESS.json"
        deadline = time.monotonic() + 10
        try:
            while time.monotonic() < deadline:
                if progress.exists():
                    current = scheduler.read_json(progress)
                    if current["running"]:
                        break
                time.sleep(0.02)
            else:
                self.fail("Fixture worker never started")
            worker_pids = [row["pid"] for row in current["workers"].values()]
            os.kill(process.pid, signal.SIGTERM)
            process.join(timeout=10)
            self.assertFalse(process.is_alive())
            result = scheduler.read_json(self.run / "upstream/CASE_PRECOMPUTE.json")
            self.assertIn("interrupted", result["failed"])
            for pid in worker_pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            with scheduler.exclusive_lock(self.run / ".run.lock"):
                pass
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
