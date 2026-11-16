"""Execution optimizations must preserve PathRadar features and output rules."""
import importlib.util
from pathlib import Path
import unittest
import sys
from unittest.mock import patch
from tempfile import TemporaryDirectory
from collections import defaultdict

FILE = Path(__file__).resolve().parents[1] / 'research/baselines/pathradar/performance/engine.py'
sys.path.insert(0,str(FILE.parent))
import engine


class FakePersistentTree:
    def __init__(self):
        self.loads=0
        self.calls=0

    def load_models(self):
        self.loads+=1

    def read_file(self,asn,sink):
        assert self.loads==1, 'models must be initialized once per worker'
        self.calls+=1
        sink.put([self.calls,0,0,0])


class PerformanceTests(unittest.TestCase):
    def test_worker_reuses_models_and_continues_after_each_destination(self):
        with TemporaryDirectory() as temporary, patch.object(engine,'DB_PATH',Path(temporary)/'absent.sqlite'), patch.object(
                engine,'_TREE',FakePersistentTree()):
            with engine.mp.get_context('fork').Pool(1,initializer=engine.initialize_worker) as pool:
                rows=list(pool.imap_unordered(engine.predict_destination,['1','2','3'],chunksize=1))
            self.assertEqual([1,2,3],[row['counts'][0] for row in rows])

    def test_flat_copy_keeps_all_shortest_paths_and_relationship_flags(self):
        from types import MethodType
        for start in ('1','2','3','4','5','6'):
            sample = engine.original.Model_Sample.__new__(engine.original.Model_Sample)
            sample.provider, sample.customer, sample.peer = (defaultdict(set) for _ in range(3))
            for provider,customer in [('1','2'),('1','3'),('2','4'),('3','4'),('5','2'),('5','6')]:
                sample.provider[customer].add(provider)
                sample.customer[provider].add(customer)
            sample.peer['1'].add('5'); sample.peer['5'].add('1')
            sample.simulate(start)
            expected = (dict(sample.AS_path), dict(sample.worst), dict(sample.path_flag), dict(sample.shortest))
            sample.stream = MethodType(engine.flat_path_stream,sample)
            sample.simulate(start)
            self.assertEqual(expected,(dict(sample.AS_path),dict(sample.worst),dict(sample.path_flag),dict(sample.shortest)))

    def test_rank_preserves_first_duplicate_missing_and_slice_semantics(self):
        values = ['3', '2', '3', '1']
        indexed = engine.IndexedOrder(values)
        for value in ['1', '2', '3', '4']:
            self.assertEqual(value in values, value in indexed)
            if value in values:
                self.assertEqual(values.index(value), indexed.index(value))
            else:
                with self.assertRaises(ValueError):
                    indexed.index(value)
        self.assertEqual(values.index('3', 1), indexed.index('3', 1))

    def test_model_loading_does_not_happen_per_destination_and_rows_match(self):
        from types import SimpleNamespace
        tree = engine.PersistentTree.__new__(engine.PersistentTree)
        tree.model = {'sentinel': object()}
        sentinel = tree.model
        tree.valid_files = {'1', '2'}
        tree.valid_dataset = 'rib'
        tree.read_dataset = lambda dataset, source: f'9|{source} 8|{source}&ignored'
        tree.rounds, tree.modes = 1, 'train'
        def simulate(source):
            path = (source, '9')
            tree.AS_path = defaultdict(list, {'9': [path]})
            tree.spread_score = {path: .75}
        tree.simulate = simulate
        tree.evaluation = lambda source, counts: counts.put([1, 0, 1, 0])
        with TemporaryDirectory() as temporary, patch.object(engine.original, 'TESTING_DIR', Path(temporary)):
            for source in ['1', '2']:
                tree.read_file(source, engine.Counts())
                self.assertIs(sentinel, tree.model)
                self.assertEqual(tree.valid_path_dict, {'9': [(source, '9')], '8': [(source, '8')]})
                output = Path(temporary)/'result_round1'/f'{source}.txt'
                self.assertEqual(f'9|{source} 0.75', output.read_text())
                self.assertFalse(output.with_suffix('.txt.tmp').exists())


if __name__ == '__main__':
    unittest.main()
