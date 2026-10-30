"""Resource changes preserve features/targets and bind their execution identity."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from pathveil.upstream.contracts import UpstreamError, read
from pathveil.upstream.implementation import neighbor_fit, neighbor_struct_fit
from pathveil.upstream.implementation.distribution import DistributionHead
from pathveil.upstream.neighbor_training import _fit_family
from pathveil.upstream.resources import runtime_resources


class ReceiptOnlyModel:
    def __init__(self, threads):
        self.threads = threads
        self.booster_ = self

    def save_model(self, path):
        Path(path).write_text("model trained with " + str(self.threads))

    def current_iteration(self):
        return 240


class ThreadRecordingLearner:
    def __init__(self):
        self.calls = []

    def fit_one(self, spec, X, labels, queries, kept, columns, threads=4):
        self.calls.append(threads)
        return ReceiptOnlyModel(threads)


def small_neighbor_fixture():
    rng = np.random.RandomState(154)
    X = rng.normal(size=(240, 5)).astype(np.float32)
    labels = (X[:, 0] + 0.4 * X[:, 1] > 0).astype(np.int32)
    queries = [dict(source=source, root=True, size=48) for source in range(1, 6)]
    return X, labels, queries, ["feature_" + str(i) for i in range(5)]


class RuntimeResources(unittest.TestCase):
    def test_defaults_and_explicit_eight_thread_resource_plan(self):
        self.assertEqual(
            runtime_resources(), dict(cpu_threads=4, case_predict_threads=2, case_workers=2)
        )
        plan = dict(cpu_threads=8, case_predict_threads=1, case_workers=3)
        self.assertEqual(runtime_resources(dict(runtime_resources=plan)), plan)
        for value in (0, -1, True, 2.5, "8"):
            with self.assertRaises(ValueError):
                runtime_resources(dict(runtime_resources=dict(cpu_threads=value)))
        with self.assertRaisesRegex(ValueError, "Unknown runtime"):
            runtime_resources(dict(runtime_resources=dict(thread=8)))

    def test_resources_are_explicit_in_model_receipts_and_cache_mismatch_fails(self):
        learner = ThreadRecordingLearner()
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            arguments = (
                "root",
                np.empty((0, 1)),
                np.empty(0),
                [],
                ["x"],
                dict(kind="present"),
                learner,
                {str(s): s - 1 for s in range(1, 6)},
                work,
                "same_input_binding",
            )
            _fit_family(*arguments, threads=8)
            self.assertEqual(learner.calls, [8] * 6)
            receipt = read(work / "models/root/full.json")
            self.assertEqual(receipt["binding"]["runtime_resources"], dict(cpu_threads=8))
            before = (work / "models/root/full.txt").read_bytes()
            with self.assertRaisesRegex(UpstreamError, "cache input"):
                _fit_family(*arguments, threads=4)
            self.assertEqual(before, (work / "models/root/full.txt").read_bytes())
            self.assertEqual(learner.calls, [8] * 6)

    def test_small_real_neighbor_models_keep_features_targets_and_prediction_tolerance(self):
        X, labels, queries, names = small_neighbor_fixture()
        original_X, original_labels = X.copy(), labels.copy()
        for module in (neighbor_fit, neighbor_struct_fit):
            with self.subTest(module=module.__name__):
                one = module.fit_one(
                    dict(kind="present", root_only=True),
                    X,
                    labels,
                    queries,
                    set(range(1, 6)),
                    names,
                    threads=4,
                )
                eight = module.fit_one(
                    dict(kind="present", root_only=True),
                    X,
                    labels,
                    queries,
                    set(range(1, 6)),
                    names,
                    threads=8,
                )
                parameters_one, parameters_eight = one.get_params(), eight.get_params()
                self.assertEqual(parameters_one.pop("n_jobs"), 4)
                self.assertEqual(parameters_eight.pop("n_jobs"), 8)
                self.assertEqual(parameters_one, parameters_eight)
                scores_one = module.predict(one, X, "present", threads=4)
                scores_eight = module.predict(eight, X, "present", threads=8)
                np.testing.assert_allclose(scores_one, scores_eight, rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(
                    module.predict(one, X, "present", threads=1),
                    module.predict(one, X, "present", threads=2),
                    rtol=1e-9,
                    atol=1e-10,
                )
        np.testing.assert_array_equal(original_X, X)
        np.testing.assert_array_equal(original_labels, labels)

    def test_distribution_accepts_eight_threads_with_same_small_fixture_targets(self):
        rng = np.random.RandomState(322)
        matrix = rng.normal(size=(240, 5)).astype(np.float32)
        matrix[:, 0] = np.tile(np.arange(4), 60)
        queries = [
            dict(source=source, center=source, root=True, candidates=[701, 702, 703, 704])
            for source in range(1, 61)
        ]
        labels = {
            str(source): dict(
                root=dict(
                    branch_mass={str(701 + i): (i + 1) / 10 for i in range(4)},
                    branch_length_mass={
                        str(701 + i): {"3": (i + 1) / 20, "4": (i + 1) / 20} for i in range(4)
                    },
                ),
                conditional={},
            )
            for source in range(1, 61)
        }
        four = DistributionHead(threads=4).fit(matrix, queries, labels, set(range(1, 61)))
        eight = DistributionHead(threads=8).fit(matrix, queries, labels, set(range(1, 61)))
        self.assertEqual(four.fit_info, eight.fit_info)
        self.assertEqual(four.branch_model.n_jobs, 4)
        self.assertEqual(eight.branch_model.n_jobs, 8)
        self.assertEqual(eight.length_model.n_jobs, 8)
        self.assertEqual(eight.coverage_model.n_jobs, 8)
        for baseline, actual in zip(
            four.predict_arrays(matrix, queries), eight.predict_arrays(matrix, queries)
        ):
            np.testing.assert_allclose(baseline, actual, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            four.predict_candidate_coverage(matrix, queries),
            eight.predict_candidate_coverage(matrix, queries),
            rtol=1e-9,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
