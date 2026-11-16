"""Native feature matrices must exactly match PathRadar inputs and predictions."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/baselines/pathradar/performance'))
try:
    import features_engine as fast
except (ImportError,FileNotFoundError):
    fast=None
from test_pathradar_native import sample,routing


@unittest.skipIf(fast is None,'Build optional feature extension first')
class FeatureTests(unittest.TestCase):
    def test_random_full_routing_preserves_predictions_order_and_all_states(self):
        for seed in range(20):
            a=routing(sample(seed),fast.previous.IntegerTree)
            b=routing(sample(seed),fast.FeatureTree)
            fast.previous.enable_sample(a.ms);fast.enable_sample(b.ms)
            for source in ('1','2','5','7','999'):
                a.simulate(source);b.simulate(source)
                for name in ('AS_path','path_flag','spread_score'):
                    self.assertEqual(list(getattr(a,name).items()),list(getattr(b,name).items()),(seed,source,name))
                for name in ('AS_path','worst','path_flag','shortest'):
                    self.assertEqual(list(getattr(a.ms,name).items()),list(getattr(b.ms,name).items()),(seed,source,name))

    def test_matrix_exact_values_dtype_layout_missing_data_zero_support_and_duplicate_rank(self):
        ms=sample(42)
        ms.linkorder['1']=['2','3','2']
        ms.core_cluster={'2'}
        ms.prev_frequency.update({'1|2|3':0,'1|2':7})
        ms.next_frequency.update({'1|3|2':0,'3|2':9})
        ms.linkfreq['2','3']=2**45
        paths=[('1','2'),('1','3','2'),('1','2','3'),('1','3','2','1','2')]
        for path in paths:
            for a,b in zip(path,path[1:]):ms.relationship.setdefault((a,b),'unknown')
        ms.shortest={('1','1'):1,('1','2'):5,('1','3'):0}
        ms._feature_evidence=fast.native.pack(ms)
        ms._feature_distances=fast.native.distances(ms._feature_evidence,ms.shortest,'1')
        for path in paths:
            result=fast.matrix(ms,[path])
            self.assertEqual([ms.get_sample(path)],result.tolist())
            self.assertEqual(fast.np.dtype('int64'),result.dtype)
            self.assertTrue(result.flags.c_contiguous)
        empty=fast.matrix(ms,[])
        self.assertEqual((0,0),empty.shape)

    def test_immutable_evidence_missing_relationship_and_mixed_width_errors(self):
        ms=sample(1)
        ms.core_cluster={'2'}
        ms.relationship={('1','2'):'c2p',('1','3'):'p2p'}
        ms.shortest={('1','1'):1}
        ms._feature_evidence=fast.native.pack(ms)
        ms._feature_distances=fast.native.distances(ms._feature_evidence,ms.shortest,'1')
        with self.assertRaises(KeyError):fast.matrix(ms,[('1','999')])
        with self.assertRaises(ValueError):fast.matrix(ms,[('1',)])
        with self.assertRaises(ValueError):fast.matrix(ms,[('1','2'),('1','3')])
        with self.assertRaises(ValueError):fast.matrix(ms,[('2','1')])
        other=fast.native.pack(ms)
        with self.assertRaises(ValueError):fast.native.features(other,ms._feature_distances,[('1','2')])
        # Missing defaultdict entries need not be materialized to produce zeros.
        before=[len(ms.provider),len(ms.peer),len(ms.customer),len(ms.linkfreq)]
        fast.matrix(ms,[('1','3')])
        self.assertEqual(before,[len(ms.provider),len(ms.peer),len(ms.customer),len(ms.linkfreq)])


class FeatureVerificationTests(unittest.TestCase):
    def test_modified_binary_model_source_and_unverified_features_block_deployment(self):
        import run_dates
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            folder=root/'runs/baselines/pathradar/performance'
            proof=folder/'features_equivalence_round1/REPORT.json'
            build=folder/'features_build/BUILD.json'
            proof.parent.mkdir(parents=True);build.parent.mkdir(parents=True)
            source,binary,model=(root/name for name in ('source','binary','model'))
            for path in (source,binary,model):path.write_text(path.name)
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            build.write_text(json.dumps(dict(binary=str(binary),binary_sha256=digest(binary))))
            row=dict(equal=True,features_checked=dict(verified_values=20))
            report=dict(status='PASS',comparisons=[row]*3,
                        code_sha256={str(source):digest(source)},model_sha256={str(model):digest(model)},
                        build_sha256={str(build):digest(build)})
            proof.write_text(json.dumps(report))
            with patch.object(run_dates,'ROOT',root):
                self.assertIn(str(proof),run_dates.verify_features())
                for path in (source,binary,model,build):
                    old=path.read_bytes();path.write_bytes(old+b'changed')
                    with self.assertRaises(ValueError):run_dates.verify_features()
                    path.write_bytes(old)
                invalid=dict(equal=True,features_checked=dict(verified_values=0))
                for replacement in (dict(report,status='RUNNING'),dict(report,comparisons=[invalid]*3),dict(report,comparisons=[])):
                    proof.write_text(json.dumps(replacement))
                    with self.assertRaises(ValueError):run_dates.verify_features()


if __name__=='__main__':
    unittest.main()
