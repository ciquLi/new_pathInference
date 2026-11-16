"""Streaming initialization must preserve every evidence value and original routing."""
from collections import defaultdict
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research/baselines/pathradar/performance'))
try:
    import loader_engine as fast
except (ImportError, FileNotFoundError):
    fast = None
from test_pathradar_native import sample, routing


@unittest.skipIf(fast is None, 'Build streaming loader extension first')
class LoaderTests(unittest.TestCase):
    def test_full_frequency_content_duplicates_zero_unknown_flag_and_feature_values(self):
        with TemporaryDirectory() as temporary:
            file = Path(temporary)/'joinfreq_data.txt'
            file.write_text('prev 1|2 8\nnext 3|2 0\nprev 1|2 9\nother 1|2|3 17\n prev 1|3 -4 \r\n')
            ms = sample(42)
            ms.prev_frequency.clear()
            ms.next_frequency.clear()
            for line in file.read_text().splitlines():
                flag, key, value = line.strip().split(' ')
                (ms.prev_frequency if flag == 'prev' else ms.next_frequency)[key] = int(value)
            expected = fast.previous.native.pack(ms)
            ms.prev_frequency.clear()
            ms.next_frequency.clear()
            actual = fast.native.pack_file(ms, str(file))
            self.assertTrue(fast.native.evidence_equal(expected, actual))
            ms.prev_frequency['1|2'] = 10
            different = fast.previous.native.pack(ms)
            self.assertFalse(fast.native.evidence_equal(different, actual))
            with self.assertRaises(OSError):
                fast.native.pack_file(ms, str(file.parent/'missing'))
            for invalid in ('\n', 'prev 1|2 1 extra\n', 'prev 1|2 invalid\n', 'prev 1|2 999999999999999999999\n'):
                file.write_text(invalid)
                with self.assertRaises(ValueError):
                    fast.native.pack_file(ms, str(file))

    def test_routing_fields_match_original_loader_without_graph_or_python_frequency_tables(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = {'joinfreq_data.txt': 'prev 1|2 7\nnext 2|3 9\n',
                      'linkfreq_data.txt': '1|2 7\n2|3 6\n',
                      'linkorder_data.txt': '2 3|1|3\n1 2\n',
                      'asrel_rib.txt': '#comment\n1|2|-1\n2|3|0\n3|4|-1\n1|2|-1\n',
                      'core_shell.txt': '1 1001\n2 1000\n3 2000\n'}
            for name, value in inputs.items():
                (directory/name).write_text(value)
            with patch.object(fast.engine.original, 'INTERMEDIATE_DIR', directory):
                a = fast.engine.original.Model_Sample()
                b = fast.RoutingEvidenceSample()
                for field in ('provider', 'peer', 'customer', 'relationship', 'linkorder', 'linkfreq', 'core_cluster'):
                    self.assertEqual(getattr(a, field), getattr(b, field), field)
                self.assertEqual({}, b.prev_frequency)
                self.assertEqual({}, b.next_frequency)
                expected = fast.previous.native.pack(a)
                actual = fast.native.pack_file(b, str(directory/'joinfreq_data.txt'))
                self.assertTrue(fast.native.evidence_equal(expected, actual))

    def test_streaming_prediction_matches_reference_after_configuration(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ms = sample(42)
            (directory/'joinfreq_data.txt').write_text('prev 1|2 7\nnext 2|3 9\n')
            for mapping in (ms.prev_frequency, ms.next_frequency):
                mapping.clear()
            ms.prev_frequency['1|2'] = 7
            ms.next_frequency['2|3'] = 9
            reference = routing(ms, fast.previous.previous.FeatureTree)
            fast.previous.previous.enable_sample(ms)
            other = sample(42)
            other.prev_frequency.clear()
            other.next_frequency.clear()
            actual = routing(other, fast.previous.PipelineTree)
            with patch.object(fast.engine.original, 'INTERMEDIATE_DIR', directory), patch.object(
                    fast.previous, 'native', fast.native):
                fast.enable_sample(other)
                for source in ('1', '2', '5', '999'):
                    reference.simulate(source)
                    actual.simulate(source)
                    for field in ('AS_path', 'path_flag', 'spread_score'):
                        self.assertEqual(list(getattr(reference, field).items()), list(getattr(actual, field).items()))


class LoaderVerificationTests(unittest.TestCase):
    def test_full_evidence_proof_and_recorded_dependencies_are_required(self):
        import json
        import run_dates
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            proof = root/'runs/baselines/pathradar/performance/loader_equivalence_round1/REPORT.json'
            proof.parent.mkdir(parents=True)
            paths = {name: root/name for name in ('code', 'model', 'dependency', 'binary', 'build', 'pipeline_proof')}
            for name, path in paths.items():
                path.write_text(name)
            digest = run_dates.sha256
            paths['build'].write_text(json.dumps(dict(binary=str(paths['binary']), binary_sha256=digest(paths['binary']),
                dependencies_sha256={str(paths['dependency']): digest(paths['dependency'])})))
            report = dict(status='PASS', all_evidence_equal=True, comparisons=[dict(equal=True)]*3,
                evaluation_comparisons=[dict(equal=True, pairs=1)]*3,
                code_sha256={str(paths['code']): digest(paths['code'])},
                model_sha256={str(paths['model']): digest(paths['model'])},
                pipeline_proof_sha256={str(paths['pipeline_proof']): digest(paths['pipeline_proof'])},
                build_sha256={str(paths['build']): digest(paths['build'])})
            proof.write_text(json.dumps(report))
            with patch.object(run_dates, 'ROOT', root):
                self.assertIn(str(proof), run_dates.verify_loader())
                for path in paths.values():
                    before = path.read_bytes()
                    path.write_bytes(before+b'changed')
                    with self.assertRaises(ValueError):
                        run_dates.verify_loader()
                    path.write_bytes(before)
                for change in (dict(status='RUNNING'), dict(all_evidence_equal=False),
                               dict(comparisons=[]), dict(evaluation_comparisons=[])):
                    proof.write_text(json.dumps(dict(report, **change)))
                    with self.assertRaises(ValueError):
                        run_dates.verify_loader()
