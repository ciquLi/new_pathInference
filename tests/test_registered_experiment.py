"""Narrow metadata recovery leaves frozen CLI arguments/results untouched."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock

from pathveil.upstream.contracts import UpstreamError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_registered_experiment.py"
SPEC = importlib.util.spec_from_file_location("registered_experiment", SCRIPT)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


class RegisteredExperiment(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "runs/20260201/unified255/fixture"
        (self.run / "upstream/neighbor_models").mkdir(parents=True)
        self.config = dict(
            protocol=wrapper.FULL_PROTOCOL, source_selection=dict(policy=wrapper.SOURCE_POLICY)
        )
        self.write(self.run / "RUN_IDENTITY.json", dict(date="20260201", method_config=self.config))
        self.write(self.run / "method_config.json", self.config)
        self.bundle = dict(
            status="PASS",
            date="20260201",
            model_fit_date="20260201",
            protocol=wrapper.FULL_PROTOCOL,
        )
        self.write(self.run / "upstream/neighbor_models/neighbor_bundle.json", self.bundle)
        self.write(self.run.parent / "current.json", dict(run_dir=str(self.run)))
        self.arguments = ["20260201", "--run-dir", str(self.run), "--device", "cuda"]
        self.attempt = 0
        self.original_argv = sys.argv
        self.original_project_root = os.environ.get("PATHVEIL_PROJECT_ROOT")
        self.normalizer = Mock(side_effect=self.normalize)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def write_failure(self, message=wrapper.MODEL_GATE_REPR, pid=None):
        self.attempt += 1
        self.write(
            self.run / "status.json",
            dict(
                pid=os.getpid() if pid is None else pid,
                date="20260201",
                run_dir=str(self.run),
                stage="failed",
                status="FAILED",
                error=message,
                updated_utc=str(self.attempt),
            ),
        )

    def normalize(self, run, project_root):
        self.assertEqual(run, self.run)
        self.assertEqual(project_root, self.root)
        receipt = self.run / "normalization_receipt.json"
        self.write(receipt, dict(status="PASS"))
        return receipt

    def call(self, cli, arguments=None, normalizer=None):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return wrapper.main(
                self.arguments if arguments is None else arguments,
                cli_main=cli,
                normalizer=normalizer or self.normalizer,
                project_root=self.root,
            )

    def test_normal_cli_exit_and_return_keep_pid_arguments_and_do_not_normalize(self):
        pids = []

        def cli():
            pids.append(os.getpid())
            self.assertEqual(sys.argv[1:], self.arguments)
            self.assertEqual(os.environ["PATHVEIL_PROJECT_ROOT"], str(self.root))
            raise SystemExit(0)

        with self.assertRaises(SystemExit) as result:
            self.call(cli)
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(pids, [os.getpid()])
        self.assertIsNone(self.call(lambda: None))
        self.assertFalse(self.normalizer.called)
        self.assertIs(sys.argv, self.original_argv)
        self.assertEqual(os.environ.get("PATHVEIL_PROJECT_ROOT"), self.original_project_root)

    def test_exact_fresh_gate_normalizes_then_retries_same_cli_once(self):
        pids = []

        def cli():
            pids.append(os.getpid())
            self.assertEqual(sys.argv[1:], self.arguments)
            if len(pids) == 1:
                self.write_failure()
                raise SystemExit(1)
            raise SystemExit(0)

        with self.assertRaises(SystemExit) as result:
            self.call(cli)
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(pids, [os.getpid()] * 2)
        self.normalizer.assert_called_once_with(self.run, project_root=self.root)
        self.assertIs(sys.argv, self.original_argv)

    def test_uncaught_exact_upstream_exception_can_use_same_audited_recovery(self):
        calls = []

        def cli():
            calls.append(1)
            if len(calls) == 1:
                self.write_failure()
                raise UpstreamError(wrapper.MODEL_GATE)
            return 0

        self.assertEqual(self.call(cli), 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.normalizer.call_count, 1)

    def test_date_only_and_option_equals_locate_frozen_current_run(self):
        for arguments in (["20260201"], ["--device=cpu", "20260201", "--run-dir=" + str(self.run)]):
            with self.subTest(arguments=arguments):
                calls = []

                def cli():
                    calls.append(1)
                    self.assertEqual(sys.argv[1:], arguments)
                    if len(calls) == 1:
                        self.write_failure()
                        raise SystemExit(1)
                    return 0

                self.assertEqual(self.call(cli, arguments), 0)
                self.assertEqual(len(calls), 2)

    def test_wrong_failure_and_other_uncaught_exception_are_preserved(self):
        def cli():
            self.write_failure("UpstreamError('Different gate')")
            raise SystemExit(7)

        with self.assertRaises(SystemExit) as result:
            self.call(cli)
        self.assertEqual(result.exception.code, 7)

        def other():
            self.write_failure()
            raise ValueError(wrapper.MODEL_GATE)

        with self.assertRaises(ValueError):
            self.call(other)
        self.assertFalse(self.normalizer.called)

    def test_stale_status_and_another_process_cannot_trigger_repair(self):
        self.write_failure()
        with self.assertRaises(SystemExit):
            self.call(lambda: (_ for _ in ()).throw(SystemExit(1)))

        def another():
            self.write_failure(pid=os.getpid() + 1)
            raise SystemExit(1)

        with self.assertRaises(SystemExit):
            self.call(another)
        self.assertFalse(self.normalizer.called)

    def test_other_bundle_protocol_or_config_cannot_trigger_repair(self):
        for key, value in (
            ("protocol", "date_local_source_disjoint_v1"),
            ("model_fit_date", "20260301"),
        ):
            with self.subTest(key=key):
                self.write(
                    self.run / "upstream/neighbor_models/neighbor_bundle.json",
                    {**self.bundle, key: value},
                )

                def cli():
                    self.write_failure()
                    raise SystemExit(1)

                with self.assertRaises(SystemExit):
                    self.call(cli)
        self.write(self.run / "upstream/neighbor_models/neighbor_bundle.json", self.bundle)
        self.write(self.run / "method_config.json", {})
        with self.assertRaises(SystemExit):
            self.call(cli)
        old_config = {**self.config, "protocol": "date_local_source_disjoint_v1"}
        self.write(self.run / "method_config.json", old_config)
        self.write(self.run / "RUN_IDENTITY.json", dict(date="20260201", method_config=old_config))
        with self.assertRaises(SystemExit):
            self.call(cli)
        self.assertFalse(self.normalizer.called)

    def test_normalization_failure_or_invalid_receipt_preserves_first_exit_code(self):
        def cli():
            self.write_failure()
            raise SystemExit(7)

        failing = Mock(side_effect=RuntimeError("audit rejected"))
        with self.assertRaises(SystemExit) as result:
            self.call(cli, normalizer=failing)
        self.assertEqual(result.exception.code, 7)
        self.assertEqual(failing.call_count, 1)
        invalid_receipt = self.run / "invalid_receipt.json"
        self.write(invalid_receipt, dict(status="FAILED"))
        with self.assertRaises(SystemExit) as result:
            self.call(cli, normalizer=lambda *args, **kwargs: invalid_receipt)
        self.assertEqual(result.exception.code, 7)

    def test_retry_is_not_repaired_twice_and_keeps_its_own_failure(self):
        calls = []

        def cli():
            calls.append(1)
            self.write_failure()
            raise SystemExit(1 if len(calls) == 1 else 9)

        with self.assertRaises(SystemExit) as result:
            self.call(cli)
        self.assertEqual(result.exception.code, 9)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.normalizer.call_count, 1)

    def test_prepare_check_replay_and_unknown_flags_pass_through_without_repair(self):
        for flag in ("--check-only", "--prepare-only", "--migration-replay", "--made-up"):
            with self.subTest(flag=flag):
                arguments = [*self.arguments, flag]

                def cli():
                    self.assertEqual(sys.argv[1:], arguments)
                    self.write_failure()
                    raise SystemExit(2)

                with self.assertRaises(SystemExit) as result:
                    self.call(cli, arguments)
                self.assertEqual(result.exception.code, 2)
        self.assertFalse(self.normalizer.called)

    def test_spawn_reimport_does_not_execute_main(self):
        namespace = runpy.run_path(str(SCRIPT), run_name="__mp_main__")
        self.assertEqual(namespace["__name__"], "__mp_main__")
        self.assertTrue(callable(namespace["main"]))
        self.assertFalse(self.normalizer.called)


if __name__ == "__main__":
    unittest.main()
