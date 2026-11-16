"""Integer graph must reproduce PathRadar policy traversal, including ordered state."""
from pathlib import Path
import hashlib
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/baselines/pathradar/performance'))
try:
    import shortest_engine as fast
except (ImportError,FileNotFoundError):
    fast=None
from test_pathradar_native import sample,routing


@unittest.skipIf(fast is None,'Build optional integer-graph extension first')
class ShortestTests(unittest.TestCase):
    def assert_state(self,left,right):
        for name in ('AS_path','worst','path_flag','shortest'):
            self.assertEqual(list(getattr(left,name).items()),list(getattr(right,name).items()),name)

    def test_multiple_sources_reuse_immutable_graph_and_match_original_order(self):
        for seed in range(40):
            left,right=sample(seed),sample(seed)
            fast.enable_sample(right)
            capsule=right._integer_graph
            for source in ('1','2','5','7','999'):
                left.simulate(source);right.simulate(source)
                self.assert_state(left,right)
                self.assertIs(capsule,right._integer_graph)

    def test_full_routing_scores_and_candidate_order_unchanged(self):
        for seed in range(12):
            for source in ('1','2','5','7'):
                left=routing(sample(seed),fast.bridge.NativeTree)
                right=routing(sample(seed),fast.IntegerTree)
                fast.bridge.enable(left)
                fast.enable_sample(right.ms)
                left.simulate(source);right.simulate(source)
                self.assert_state(left.ms,right.ms)
                for name in ('AS_path','path_flag','spread_score'):
                    self.assertEqual(list(getattr(left,name).items()),list(getattr(right,name).items()),(seed,source,name))

    def test_shortest_traversal_has_no_nine_hop_candidate_limit(self):
        left,right=sample(9),sample(9)
        for graph in (left,right):
            for table in (graph.provider,graph.peer,graph.customer):
                table.clear()
            for n in range(1,18):
                graph.customer[str(n)].add(str(n+1))
                graph.provider[str(n+1)].add(str(n))
        fast.enable_sample(right)
        for source in ('1','18'):
            left.simulate(source);right.simulate(source)
            self.assert_state(left,right)
        self.assertEqual(18,len(right.AS_path['1']))

    def test_overlapping_relationship_sets_and_disconnected_sources(self):
        left,right=sample(42),sample(42)
        for graph in (left,right):
            graph.provider['1'].add('2')
            graph.peer['1'].add('2')
            graph.customer['1'].add('2')
        fast.enable_sample(right)
        for source in ('1','2','999'):
            left.simulate(source);right.simulate(source)
            self.assert_state(left,right)


class ShortestVerificationTests(unittest.TestCase):
    def test_code_models_and_binary_must_match_real_input_proof(self):
        import run_dates
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            folder=root/'runs/baselines/pathradar/performance'
            proof=folder/'shortest_equivalence_round1/REPORT.json'
            build=folder/'shortest_build/BUILD.json'
            proof.parent.mkdir(parents=True);build.parent.mkdir(parents=True)
            source,binary,model=(root/name for name in ('source','binary','model'))
            for path in (source,binary,model):path.write_text(path.name)
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            build.write_text(json.dumps(dict(binary=str(binary),binary_sha256=digest(binary))))
            report=dict(status='PASS',comparisons=[dict(equal=True)]*3,
                        code_sha256={str(source):digest(source)},model_sha256={str(model):digest(model)},
                        build_sha256={str(build):digest(build)})
            proof.write_text(json.dumps(report))
            with patch.object(run_dates,'ROOT',root):
                self.assertIn(str(proof),run_dates.verify_shortest())
                for path in (source,binary,model,build):
                    old=path.read_bytes();path.write_bytes(old+b'changed')
                    with self.assertRaises(ValueError):run_dates.verify_shortest()
                    path.write_bytes(old)
                for invalid in (dict(report,status='RUNNING'),dict(report,comparisons=[dict(equal=False)]*3)):
                    proof.write_text(json.dumps(invalid))
                    with self.assertRaises(ValueError):run_dates.verify_shortest()


if __name__=='__main__':
    unittest.main()
