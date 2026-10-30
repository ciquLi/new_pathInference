"""The controller schedules subprocess stubs without running research models."""
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/run_priority_parallel.py"
SPEC = importlib.util.spec_from_file_location("priority_scheduler", MODULE_PATH)
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)

STUB = """import json, os, pathlib, sys, time
root = pathlib.Path(os.environ["STUB_ROOT"])
date = sys.argv[1]
run = pathlib.Path(sys.argv[sys.argv.index("--run-dir") + 1])
upstream = "--upstream-only" in sys.argv
phase = "upstream" if upstream else "downstream"
def record(action):
    with (root / "stub_events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(date=date, phase=phase, action=action, time=time.monotonic(),
                                    gpu_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                    threads=os.environ.get("OMP_NUM_THREADS"))) + "\\n")
def status(stage, value):
    (run / "status.json").write_text(json.dumps(dict(date=date, pid=os.getpid(), stage=stage, status=value)))
status(phase, "running")
record("start")
time.sleep(float(os.environ.get("STUB_DELAY", "0.12")))
if upstream and date == os.environ.get("STUB_FAIL_DATE"):
    status("failed", "FAILED")
    record("failed")
    sys.exit(3)
if upstream:
    (run / "upstream").mkdir(exist_ok=True)
    (run / "upstream/RELEASE.json").write_text(json.dumps(dict(status="PASS")))
    status("upstream", "complete")
else:
    status("complete", "PASS")
record("end")
"""


class ParallelScheduling(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "src/pathveil").mkdir(parents=True)
        (self.root / "src/pathveil/__init__.py").write_text("# frozen fixture\n")
        (self.root / "src/pathveil/data").mkdir()
        (self.root / "src/pathveil/data/__init__.py").write_text("")
        project = MODULE_PATH.parents[1]
        for relative in ("cli.py", "utils.py", "data/registry.py"):
            (self.root / "src/pathveil" / relative).write_text(
                (project / "src/pathveil" / relative).read_text()
            )
        (self.root / "configs/dates").mkdir(parents=True)
        (self.root / "configs/method_unified255.json").write_text(
            '{"method_id":"frozen"}'
        )
        (self.root / "scripts").mkdir()
        self.stub = self.root / "scripts/stub.py"
        self.stub.write_text(STUB)
        (self.root / "scripts/run_experiment.sh").write_text(
            '#!/bin/bash\nexec "' + sys.executable + '" "' + str(self.stub) + '" "$@"\n'
        )
        self.jobs = []
        for date in scheduler.APPROVED_DATES:
            run = self.root / "runs" / date / "fixture"
            run.mkdir(parents=True)
            registry = self.root / "configs/dates" / (date + ".json")
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "date": date,
                        "routing_date": date,
                        "auxiliary_dates": {},
                        "files_root": str(self.root / "raw"),
                        "raw_inputs": [],
                        "derived_inputs": [],
                    }
                )
            )
            identity = {
                "date": date,
                "code_sha256": scheduler.code_identity(self.root),
                "method_config": {"method_id": "frozen"},
                "registry_sha256": scheduler.file_sha(registry),
            }
            (run / "RUN_IDENTITY.json").write_text(json.dumps(identity))
            self.jobs.append(scheduler.Job(date, run))
        self.environment = patch.dict(
            "os.environ", {"STUB_ROOT": str(self.root), "STUB_DELAY": "0.12"}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.gpu_patch = patch.object(scheduler, "gpu_idle", return_value=True)
        self.gpu_patch.start()
        self.addCleanup(self.gpu_patch.stop)

    def controller(self, jobs=None):
        return scheduler.Scheduler(
            self.root,
            self.root / "runs/batch",
            jobs or self.jobs,
            max_cpu_jobs=3,
            min_available_gib=0,
            threads=4,
            estimated_job_gib=0,
        )

    def run_controller(self, controller):
        with redirect_stdout(io.StringIO()):
            return controller.run(poll_seconds=0.01)

    def events(self):
        return [
            json.loads(line)
            for line in (self.root / "stub_events.jsonl").read_text().splitlines()
        ]

    def adopt(self, upstream=False):
        job = self.jobs[0]
        command = [sys.executable, str(self.stub), job.date, "--run-dir", str(job.run)]
        if upstream:
            command.append("--upstream-only")
        process = subprocess.Popen(command, env={**os.environ, "STUB_DELAY": "0.4"})
        self.addCleanup(
            lambda: process.wait()
            if process.poll() is not None
            else (process.terminate(), process.wait())
        )
        deadline = time.monotonic() + 3
        while not (job.run / "status.json").is_file():
            if time.monotonic() > deadline:
                self.fail("Adopted subprocess did not initialize")
            time.sleep(0.01)
        job.adopted_pid = process.pid
        job.adopted_upstream = upstream
        return process

    def test_cpu_stages_overlap_and_gpu_stages_are_serial(self):
        controller = self.controller()
        self.assertEqual(self.run_controller(controller), 0)
        events = self.events()
        cpu_start = [
            e["time"]
            for e in events
            if e["phase"] == "upstream" and e["action"] == "start"
        ]
        cpu_end = [
            e["time"]
            for e in events
            if e["phase"] == "upstream" and e["action"] == "end"
        ]
        self.assertEqual(len(cpu_start), 3)
        self.assertLess(max(cpu_start), min(cpu_end))
        gpu = [e for e in events if e["phase"] == "downstream"]
        self.assertEqual([e["action"] for e in gpu], ["start", "end"] * 3)
        self.assertTrue(
            all(e["gpu_visible"] == "" for e in events if e["phase"] == "upstream")
        )
        self.assertTrue(all(e["threads"] == "4" for e in events))
        self.assertTrue(all(job.stage == "complete" for job in self.jobs))
        state = scheduler.read_json(controller.batch / "STATUS.json")
        self.assertEqual(state["source_code_sha256"], controller.code_sha256)
        self.assertEqual(
            scheduler.read_json(controller.batch / "EXIT.json")["exit_code"], 0
        )

    def test_two_gpu_slots_overlap_without_admitting_a_third(self):
        controller = self.controller()
        controller.max_gpu_jobs = 2
        self.assertEqual(self.run_controller(controller), 0)
        current = peak = 0
        for event in self.events():
            if event["phase"] == "downstream":
                current += 1 if event["action"] == "start" else -1
                peak = max(peak, current)
        self.assertEqual(peak, 2)
        self.assertEqual(current, 0)

    def test_three_downstream_dates_can_prepare_without_waiting_for_other_dates(self):
        controller = self.controller()
        controller.max_gpu_jobs = 3
        self.assertEqual(self.run_controller(controller), 0)
        current = peak = 0
        for event in self.events():
            if event["phase"] == "downstream":
                current += 1 if event["action"] == "start" else -1
                peak = max(peak, current)
        self.assertEqual(peak, 3)
        self.assertEqual(current, 0)

    def test_third_downstream_date_still_requires_memory_headroom(self):
        controller = self.controller()
        controller.max_gpu_jobs = 3
        controller.estimated_job_gib = 16
        controller.min_available_gib = 12
        self.jobs[0].stage = "ready_downstream"
        self.jobs[1].stage = "complete"
        self.jobs[2].stage = "complete"
        with patch.object(scheduler, "available_gib", return_value=27):
            controller.tick()
        self.assertEqual(self.jobs[0].stage, "ready_downstream")
        self.assertIsNone(self.jobs[0].process)

    def test_adopted_full_run_shares_one_of_two_slots(self):
        process = self.adopt()
        controller = self.controller()
        controller.max_gpu_jobs = 2
        self.assertEqual(self.run_controller(controller), 0)
        self.assertEqual(process.wait(), 0)
        events = self.events()
        adopted_end = next(
            e["time"]
            for e in events
            if e["date"] == self.jobs[0].date and e["action"] == "end"
        )
        other_starts = [
            e["time"]
            for e in events
            if e["date"] != self.jobs[0].date
            and e["phase"] == "downstream"
            and e["action"] == "start"
        ]
        self.assertLess(min(other_starts), adopted_end)

    def test_gpu_lock_survives_until_last_shared_run_finishes(self):
        controller = self.controller()
        for job in self.jobs[:2]:
            job.stage = "downstream"
            (job.run / "status.json").write_text(
                json.dumps(dict(stage="complete", status="PASS"))
            )
        with patch.object(controller, "release_gpu") as release:
            controller.finish(self.jobs[0], 0)
            release.assert_not_called()
            controller.finish(self.jobs[1], 0)
            release.assert_called_once()

    def test_adopted_full_job_is_untouched_and_blocks_new_gpu_until_finished(self):
        process = self.adopt()
        self.assertEqual(self.jobs[0].date, "20260201")
        self.assertEqual(self.run_controller(self.controller()), 0)
        self.assertEqual(process.wait(), 0)
        events = self.events()
        adopted = [e for e in events if e["date"] == self.jobs[0].date]
        self.assertEqual([e["action"] for e in adopted], ["start", "end"])
        other_cpu_start = [
            e["time"]
            for e in events
            if e["phase"] == "upstream" and e["action"] == "start"
        ]
        self.assertEqual(len(other_cpu_start), 2)
        self.assertEqual(
            {e["date"] for e in events if e["phase"] == "upstream"},
            {"20260301", "20260401"},
        )
        other_cpu_end = [
            e["time"]
            for e in events
            if e["phase"] == "upstream" and e["action"] == "end"
        ]
        self.assertLess(max(other_cpu_start), min(other_cpu_end))
        self.assertLess(max(other_cpu_start), adopted[-1]["time"])
        new_gpu_start = [
            e["time"]
            for e in events
            if e["phase"] == "downstream"
            and e["action"] == "start"
            and e["date"] != self.jobs[0].date
        ]
        self.assertGreater(min(new_gpu_start), adopted[-1]["time"])

    def test_failed_date_does_not_block_independent_dates(self):
        with patch.dict("os.environ", {"STUB_FAIL_DATE": self.jobs[0].date}):
            self.assertEqual(self.run_controller(self.controller()), 1)
        self.assertEqual(
            [job.stage for job in self.jobs], ["failed", "complete", "complete"]
        )
        self.assertEqual(self.jobs[0].exit_code, 3)

    def test_adopted_upstream_is_resumed_once_for_full_evaluation(self):
        process = self.adopt(upstream=True)
        self.assertEqual(self.run_controller(self.controller()), 0)
        process.wait()
        events = [
            event for event in self.events() if event["date"] == self.jobs[0].date
        ]
        self.assertEqual(
            [(event["phase"], event["action"]) for event in events],
            [
                ("upstream", "start"),
                ("upstream", "end"),
                ("downstream", "start"),
                ("downstream", "end"),
            ],
        )

    def test_precompute_then_upstream_release_precedes_full_gpu_resume(self):
        self.jobs[0].precompute = True
        runner = self.root / "scripts/build_training_cases.py"
        runner.write_text(
            "import json, pathlib, sys\n"
            'run = pathlib.Path(sys.argv[sys.argv.index("--run-dir") + 1])\n'
            '(run / "upstream").mkdir(exist_ok=True)\n'
            '(run / "upstream/CASE_PRECOMPUTE.json").write_text(json.dumps({"status":"PASS"}))\n'
        )
        controller = self.controller()
        self.assertEqual(controller.jobs[0].stage, "pending_precompute")
        self.assertEqual(self.run_controller(controller), 0)
        events = [
            event for event in self.events() if event["date"] == self.jobs[0].date
        ]
        self.assertEqual(
            [event["phase"] for event in events],
            ["upstream", "upstream", "downstream", "downstream"],
        )
        transitions = [
            (event.get("previous_stage"), event.get("stage"))
            for event in controller.events
            if event.get("date") == self.jobs[0].date and event["event"] == "finished"
        ]
        self.assertEqual(
            transitions,
            [
                ("precompute", "pending_upstream"),
                ("upstream", "ready_downstream"),
                ("downstream", "complete"),
            ],
        )

    def test_shutdown_terminates_owned_group_but_preserves_adopted_process(self):
        adopted = self.adopt()
        controller = self.controller()
        controller.batch.mkdir()
        with redirect_stdout(io.StringIO()):
            controller.launch(self.jobs[1], upstream=True)
            owned = self.jobs[1].process
            controller.stop_owned()
        self.assertIsNotNone(owned.poll())
        self.assertIsNone(adopted.poll())

    def test_memory_gate_waits_without_launching_or_losing_pending_jobs(self):
        controller = self.controller()
        controller.batch.mkdir()
        controller.min_available_gib = 12
        with patch.object(scheduler, "available_gib", return_value=10):
            controller.tick()
        self.assertTrue(controller.waiting_for_memory)
        self.assertTrue(all(job.stage == "pending_upstream" for job in self.jobs))
        self.assertFalse((self.root / "stub_events.jsonl").exists())

    def test_startup_growth_is_reserved_before_process_memory_becomes_visible(self):
        controller = self.controller()
        controller.batch.mkdir()
        controller.min_available_gib = 12
        controller.estimated_job_gib = 16
        with patch.object(scheduler, "available_gib", return_value=57), redirect_stdout(
            io.StringIO()
        ):
            controller.tick()
            self.assertEqual(
                [job.stage for job in self.jobs],
                ["upstream", "upstream", "pending_upstream"],
            )
            self.assertTrue(controller.waiting_for_memory)
            controller.stop_owned()

    def test_source_change_is_rejected_and_existing_files_are_preserved(self):
        controller = self.controller()
        path = self.root / "src/pathveil/__init__.py"
        path.write_text("# changed\n")
        with self.assertRaisesRegex(RuntimeError, "Frozen source changed"):
            controller.verify_code()
        with self.assertRaisesRegex(ValueError, "Date/source identity"):
            self.controller()

    def test_sigterm_of_controller_cleans_owned_tasks_and_leaves_adopted_task(self):
        adopted = self.adopt()
        batch = self.root / "runs/signal_batch"
        command = [
            sys.executable,
            str(MODULE_PATH),
            "--project-root",
            str(self.root),
            "--batch-dir",
            str(batch),
            "--min-available-gib",
            "0",
            "--estimated-job-gib",
            "0",
            "--poll-seconds",
            "0.01",
            "--adopt",
            self.jobs[0].date + "=" + str(adopted.pid),
        ]
        for job in self.jobs[:2]:
            command += ["--run", job.date + "=" + str(job.run)]
        with subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        ) as process:
            deadline = time.monotonic() + 3
            while not (batch / "STATUS.json").is_file():
                if time.monotonic() > deadline:
                    self.fail("Controller did not initialize")
                time.sleep(0.005)
            state = scheduler.read_json(batch / "STATUS.json")
            owned_pid = state["jobs"][1]["pid"]
            self.assertIsNotNone(owned_pid)
            process.terminate()
            _, errors = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 130, errors)
        self.assertIsNone(scheduler.process_identity(owned_pid))
        self.assertIsNone(adopted.poll())
        self.assertTrue(scheduler.read_json(batch / "STATUS.json")["stop_requested"])

    def test_registry_or_method_change_is_rejected(self):
        (self.root / "configs/dates/20260201.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "registry changed"):
            self.controller()

    def test_ready_release_skips_retraining_and_explicit_dates_are_required(self):
        job = self.jobs[0]
        (job.run / "upstream").mkdir()
        (job.run / "upstream/RELEASE.json").write_text('{"status":"PASS"}')
        self.assertEqual(self.controller().jobs[0].stage, "ready_downstream")
        with self.assertRaises(Exception):
            scheduler.parse_assignment("20260501=/tmp/run")
        with self.assertRaises(Exception):
            scheduler.parse_assignment("20261015=1")
        self.assertEqual(scheduler.parse_assignment("20260301=1"), ("20260301", "1"))
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.controller([job, job])

    def test_dry_run_does_not_launch_or_create_output_directory(self):
        arguments = [
            "--project-root",
            str(self.root),
            "--batch-dir",
            str(self.root / "runs/dry"),
            "--dry-run",
        ]
        for job in self.jobs:
            arguments += ["--run", job.date + "=" + str(job.run)]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(scheduler.main(arguments), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["dates"], list(scheduler.APPROVED_DATES))
        self.assertEqual(len(plan["commands"]), 3)
        self.assertFalse((self.root / "runs/dry").exists())
        self.assertFalse((self.root / "stub_events.jsonl").exists())

    def test_default_dates_initialize_and_reuse_frozen_cli_run_identities(self):
        arguments = [
            "--project-root",
            str(self.root),
            "--batch-dir",
            str(self.root / "runs/default_batch"),
            "--dry-run",
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(scheduler.main(arguments), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["dates"], list(scheduler.APPROVED_DATES))
        selected = {item["date"]: item["run_dir"] for item in plan["jobs"]}
        for date, directory in selected.items():
            identity = scheduler.read_json(Path(directory) / "RUN_IDENTITY.json")
            self.assertEqual(identity["date"], date)
            self.assertEqual(
                identity["code_sha256"], scheduler.code_identity(self.root)
            )
            current = self.root / "runs" / date / "unified255/current.json"
            self.assertEqual(scheduler.read_json(current)["run_dir"], directory)
        again = scheduler.choose_default_runs(self.root)
        self.assertEqual(again, selected)
        self.assertFalse((self.root / "stub_events.jsonl").exists())

    def test_all_three_dates_load_registered_full_source_method_without_replay(self):
        method = {
            "method_id": "frozen_full_sources",
            "source_selection": {"all_eligible_sources": True},
            "runtime_resources": {
                "cpu_threads": 8,
                "case_predict_threads": 1,
                "case_workers": 5,
            },
        }
        method_path = self.root / "configs/method_full_sources.json"
        scheduler.write_json(method_path, method)
        for date in scheduler.APPROVED_DATES:
            registry = self.root / "configs/dates" / (date + ".json")
            registered = scheduler.read_json(registry)
            registered["method_config"] = "configs/method_full_sources.json"
            scheduler.write_json(registry, registered)
        selected = scheduler.choose_default_runs(self.root)
        self.assertEqual(list(selected), ["20260201", "20260301", "20260401"])
        jobs = []
        for date, directory in selected.items():
            run = Path(directory)
            identity = scheduler.read_json(run / "RUN_IDENTITY.json")
            self.assertEqual(identity["method_config"], method)
            self.assertEqual(
                identity["registry_sha256"],
                scheduler.file_sha(self.root / "configs/dates" / (date + ".json")),
            )
            jobs.append(scheduler.Job(date, run))
        controller = self.controller(jobs)
        for job in jobs:
            command = controller.command(job, upstream=True)
            self.assertIn(job.date, command)
            self.assertIn("--upstream-only", command)
            self.assertNotIn("--migration-replay", command)
        changed = dict(method, method_id="changed_after_freeze")
        scheduler.write_json(method_path, changed)
        with self.assertRaisesRegex(ValueError, "Method configuration changed"):
            self.controller(jobs)

    def test_explicit_five_case_workers_are_supported_without_changing_default(self):
        controller = scheduler.Scheduler(
            self.root,
            self.root / "runs/five_worker_batch",
            self.jobs,
            min_available_gib=0,
            estimated_job_gib=0,
            case_workers=5,
        )
        self.assertEqual(controller.case_workers, 5)
        self.assertEqual(self.controller().case_workers, 2)
        command = controller.command(self.jobs[0], upstream=True, precompute=True)
        self.assertEqual(command[command.index("--workers") + 1], "5")
        with self.assertRaisesRegex(ValueError, "1..5"):
            scheduler.Scheduler(
                self.root, self.root / "runs/too_many", self.jobs, case_workers=6
            )

    def test_default_and_explicit_selection_reject_same_date_active_lock(self):
        # A different identity must not cause a second run for an active date.
        run = self.root / "runs/20260201/unified255/older_identity"
        run.mkdir(parents=True)
        with (run / ".run.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "explicitly adopt"):
                scheduler.choose_default_runs(self.root)
            with self.assertRaisesRegex(RuntimeError, "explicitly adopt"):
                self.controller()
        self.assertFalse(any((self.root / "runs").glob("*/unified255/current.json")))

    def test_project_lock_prevents_controller_in_another_batch_directory(self):
        controller = self.controller()
        with (self.root / "runs/.priority_scheduler.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already controls"):
                self.run_controller(controller)
        self.assertFalse((self.root / "stub_events.jsonl").exists())

    def protocol_failure_fixture(self):
        job = self.jobs[0]
        job.stage = "upstream"
        directory = job.run / "upstream/neighbor_models"
        directory.mkdir(parents=True)
        bundle = directory / "neighbor_bundle.json"
        scheduler.write_json(
            bundle, dict(protocol="date_local_all_eligible_source_disjoint_v2")
        )
        (directory / "weights.txt").write_text("immutable fitted model")
        scheduler.write_json(
            job.run / "status.json",
            dict(
                stage="failed",
                status="FAILED",
                error="UpstreamError('Only explicit date-local source-disjoint models are accepted')",
            ),
        )
        normalizer = self.root / "scripts/normalize_neighbor_protocol.py"
        normalizer.write_text(
            "import json,pathlib,sys\n"
            "run=pathlib.Path(sys.argv[sys.argv.index('--run-dir')+1])\n"
            "p=run/'upstream/neighbor_models/neighbor_bundle.json'\n"
            "d=json.loads(p.read_text());d['protocol']='date_local_source_disjoint_v1'\n"
            "p.write_text(json.dumps(d))\n"
        )
        return job, bundle, normalizer

    def test_known_metadata_failure_resumes_without_refitting_or_repeating_repair(self):
        job, bundle, _ = self.protocol_failure_fixture()
        controller = self.controller()
        controller.finish(job, 1)
        self.assertEqual(job.stage, "pending_upstream")
        self.assertEqual(
            (bundle.parent / "weights.txt").read_text(), "immutable fitted model"
        )
        self.assertEqual(
            scheduler.read_json(bundle)["protocol"], "date_local_source_disjoint_v1"
        )
        with patch.object(scheduler.subprocess, "run") as repair:
            controller.finish(job, 1)
        repair.assert_not_called()
        self.assertEqual(job.stage, "failed")

    def test_metadata_repair_failure_is_isolated_from_other_running_dates(self):
        job, _, normalizer = self.protocol_failure_fixture()
        normalizer.write_text("raise SystemExit(2)\n")
        controller = self.controller()
        controller.finish(job, 1)
        self.assertEqual(job.stage, "failed")
        self.assertEqual(self.jobs[1].stage, "pending_upstream")

    def test_metadata_repair_timeout_does_not_raise_from_controller_poll(self):
        job, _, _ = self.protocol_failure_fixture()
        controller = self.controller()
        with patch.object(
            scheduler.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("audit", 300),
        ):
            controller.finish(job, 1)
        self.assertEqual(job.stage, "failed")


class GPUAdmission(unittest.TestCase):
    def admission(self, devices="22405, 3\n", processes="", desktops=()):
        outputs = [
            subprocess.CompletedProcess([], 0, stdout=text)
            for text in (devices, processes)
        ]
        with patch.object(
            scheduler.subprocess, "run", side_effect=outputs
        ), patch.object(
            scheduler, "desktop_gpu_process", side_effect=lambda pid: pid in desktops
        ):
            return scheduler.gpu_idle(12288)

    def test_empty_compute_list_and_sufficient_memory(self):
        self.assertTrue(self.admission())

    def test_audited_desktop_contexts_do_not_block(self):
        self.assertTrue(
            self.admission(processes="14942, 94\n28187, 414\n", desktops=(14942, 28187))
        )

    def test_unknown_compute_blocks_even_when_small_and_idle(self):
        self.assertFalse(self.admission(processes="1234, 10\n"))

    def test_large_desktop_context_blocks(self):
        self.assertFalse(self.admission(processes="28187, 1025\n", desktops=(28187,)))

    def test_high_utilization_blocks(self):
        self.assertFalse(self.admission(devices="22405, 11\n"))

    def test_insufficient_memory_blocks(self):
        self.assertFalse(self.admission(devices="12000, 0\n"))

    def test_missing_or_malformed_gpu_data_blocks(self):
        for data in ("", "N/A, 0\n", "22405\n"):
            self.assertFalse(self.admission(devices=data))
        self.assertFalse(self.admission(processes="1234, [N/A]\n"))

    def test_only_managed_process_group_can_share_busy_gpu(self):
        outputs = [
            subprocess.CompletedProcess([], 0, stdout=text)
            for text in ("16000, 95\n", "1234, 6000\n")
        ]
        with patch.object(
            scheduler.subprocess, "run", side_effect=outputs
        ), patch.object(scheduler.os, "getpgid", return_value=111), patch.object(
            scheduler, "desktop_gpu_process", return_value=False
        ):
            self.assertTrue(scheduler.gpu_idle(12288, allowed_groups={111}))

    def test_unknown_compute_is_rejected_even_with_managed_run(self):
        outputs = [
            subprocess.CompletedProcess([], 0, stdout=text)
            for text in ("16000, 95\n", "1234, 6000\n5678, 10\n")
        ]
        with patch.object(
            scheduler.subprocess, "run", side_effect=outputs
        ), patch.object(scheduler.os, "getpgid", side_effect=[111, 222]), patch.object(
            scheduler, "desktop_gpu_process", return_value=False
        ):
            self.assertFalse(scheduler.gpu_idle(12288, allowed_groups={111}))


if __name__ == "__main__":
    unittest.main()
