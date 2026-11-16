"""Candidate optimization must preserve ordering, NumPy thresholds and evidence use."""
from collections import defaultdict
from pathlib import Path
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research/baselines/pathradar/performance'))
try:
    import pipeline_engine as fast
except (ImportError, FileNotFoundError):
    fast = None
from test_pathradar_native import sample, routing


@unittest.skipIf(fast is None, 'Build optional pipeline extension first')
class PipelineTests(unittest.TestCase):
    def test_random_routing_all_states_order_and_released_evidence(self):
        for seed in range(35):
            a = routing(sample(seed), fast.previous.FeatureTree)
            b = routing(sample(seed), fast.PipelineTree)
            c = routing(sample(seed), fast.PipelineTree)
            fast.previous.enable_sample(a.ms)
            fast.enable_sample(b.ms)
            fast.enable_sample(c.ms, release=True)
            for source in ('1', '2', '5', '7', '999'):
                for tree in (a, b, c):
                    tree.simulate(source)
                for name in ('AS_path', 'path_flag', 'spread_score'):
                    self.assertEqual(list(getattr(a, name).items()), list(getattr(b, name).items()), (seed, source, name))
                    self.assertEqual(list(getattr(a, name).items()), list(getattr(c, name).items()), (seed, source, name, 'released'))
                for name in ('AS_path', 'worst', 'path_flag', 'shortest'):
                    self.assertEqual(list(getattr(a.ms, name).items()), list(getattr(b.ms, name).items()))
                    self.assertEqual(list(getattr(a.ms, name).items()), list(getattr(c.ms, name).items()))

    def test_numpy_threshold_boundaries_nan_duplicates_and_candidate_order(self):
        np = fast.np
        for dtype in (np.float32, np.float64):
            for threshold in (.5, .7):
                boundary = dtype(threshold)
                values = [dtype('nan'), dtype('-inf'), np.nextafter(boundary, dtype('-inf')), boundary,
                          np.nextafter(boundary, dtype('inf')), dtype('inf'), dtype(0), dtype(-0.)]
                for initial in values:
                    for score in values:
                        routes = [('1', '2'), ('1', '3', '2'), ('1', '2'), ('1', '4', '2')]
                        results = [score, initial, score, score]
                        a = routing(sample(1), fast.previous.FeatureTree)
                        b = routing(sample(1), fast.PipelineTree)
                        for tree in (a, b):
                            tree.threshold = threshold
                            tree.AS_path = defaultdict(list, {'2': [routes[0]]})
                            tree.spread_score = {routes[0]: initial}
                            tree.get_model = lambda paths, name: results
                        left = fast.previous.previous.bridge.native.add_paths(a, {'shell1': routes}, set())
                        right = fast.native.add_paths(b, {'shell1': routes}, set())
                        self.assertEqual(list(left), list(right))
                        self.assertEqual(list(a.AS_path.items()), list(b.AS_path.items()))
                        self.assertEqual(list(a.spread_score), list(b.spread_score))
                        for path in a.spread_score:
                            np.testing.assert_equal(a.spread_score[path], b.spread_score[path])

    def test_graph_and_distance_identity_and_source_validation(self):
        a, b = sample(1), sample(2)
        fast.enable_sample(a)
        fast.enable_sample(b)
        a.simulate('1')
        tree = routing(a, fast.PipelineTree)
        tree.path_flag = {('1',): 0}
        with self.assertRaises(ValueError):
            fast.native.expand(b._candidate_graph, a._feature_distances, tree, [('1',)])
        with self.assertRaises(ValueError):
            fast.native.expand(a._candidate_graph, a._feature_distances, tree, [('2',)])

    def test_final_evaluation_reuses_tree_and_models_and_preserves_details(self):
        import evaluate_pipeline as evaluation
        a = routing(sample(1), fast.previous.FeatureTree)
        b = routing(sample(1), fast.PipelineTree)
        fast.previous.enable_sample(a.ms)
        fast.enable_sample(b.ms, release=True)
        for tree in (a, b):
            tree.rounds = 2
            tree.threshold = .7
        model = b.model
        for details in (0, 1):
            for src in ('1', '2', '999'):
                payload = (2, .7, src, {'2': [[src, '2']], '6': [[src, '3', '6']],
                                      '9999': [[src, '9999']]}, details, 1)
                with patch.object(evaluation.original, 'Routingtree', lambda *args: a), patch.object(
                        evaluation.original, 'load_models', lambda rounds: a.model):
                    expected = evaluation.original.evaluate_source_task(payload)
                with patch.object(evaluation, '_TREE', b), patch.object(
                        evaluation.original, 'Routingtree', evaluation.tree_factory), patch.object(
                        evaluation.original, 'load_models', evaluation.cached_models):
                    actual = evaluation.original.evaluate_source_task(payload)
                    self.assertIs(b, evaluation.tree_factory(2, .7, 'test', 0, 1))
                    self.assertIs(model, evaluation.cached_models(2))
                    with self.assertRaises(ValueError):
                        evaluation.tree_factory(1, .7, 'test', 0, 1)
                    with self.assertRaises(ValueError):
                        evaluation.cached_models(1)
                self.assertEqual(expected, actual)

    def test_parallel_evaluation_loads_models_once_per_worker(self):
        import evaluate_pipeline as evaluation
        tree = routing(sample(1), fast.PipelineTree)
        fast.enable_sample(tree.ms, release=True)
        tree.rounds = 2
        tree.threshold = .7
        context = fast.engine.mp.get_context('fork')
        loads = context.Value('i', 0)
        def load_models():
            with loads.get_lock():
                loads.value += 1
        tree.load_models = load_models
        tasks = [(2, .7, source, {'6': [[source, '6']], '999': [[source, '999']]}, 1, 1)
                 for source in ('1', '2', '5', '7')]
        with patch.object(evaluation, '_TREE', tree), patch.object(
                evaluation.original, 'Routingtree', evaluation.tree_factory), patch.object(
                evaluation.original, 'load_models', evaluation.cached_models):
            expected = [evaluation.original.evaluate_source_task(task) for task in tasks]
            with evaluation.original.ProcessPoolExecutor(max_workers=2, mp_context=context,
                    initializer=evaluation.initialize_worker) as executor:
                actual = list(executor.map(evaluation.original.evaluate_source_task, tasks))
        self.assertEqual(expected, actual)
        self.assertEqual(2, loads.value)


class PipelineVerificationTests(unittest.TestCase):
    def test_incomplete_proof_or_modified_code_model_dependency_or_binary_blocks_launch(self):
        import run_dates
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'runs/baselines/pathradar/performance'
            folder.mkdir(parents=True)
            proof = folder/'pipeline_equivalence_round1/REPORT.json'
            proof.parent.mkdir()
            paths = {name: root/name for name in ('code', 'model', 'dependency', 'binary', 'build')}
            for name, path in paths.items():
                path.write_text(name)
            digest = lambda path: run_dates.sha256(path)
            paths['build'].write_text(__import__('json').dumps(dict(binary=str(paths['binary']),
                binary_sha256=digest(paths['binary']),
                dependencies_sha256={str(paths['dependency']): digest(paths['dependency'])})))
            report = dict(status='PASS', comparisons=[dict(equal=True,
                pipeline_checked=dict(verified_layers=9))]*3,
                evaluation_comparisons=[dict(equal=True, pairs=2)]*3, released_evidence_equal=True,
                code_sha256={str(paths['code']): digest(paths['code'])},
                model_sha256={str(paths['model']): digest(paths['model'])},
                build_sha256={str(paths['build']): digest(paths['build'])})
            proof.write_text(__import__('json').dumps(report))
            with patch.object(run_dates, 'ROOT', root):
                self.assertIn(str(proof), run_dates.verify_pipeline())
                for path in paths.values():
                    before = path.read_bytes()
                    path.write_bytes(before+b'changed')
                    with self.assertRaises(ValueError):
                        run_dates.verify_pipeline()
                    path.write_bytes(before)
                for change in (dict(status='RUNNING'), dict(evaluation_comparisons=[]),
                               dict(released_evidence_equal=False), dict(comparisons=[])):
                    proof.write_text(__import__('json').dumps(dict(report, **change)))
                    with self.assertRaises(ValueError):
                        run_dates.verify_pipeline()


if __name__ == '__main__':
    unittest.main()
