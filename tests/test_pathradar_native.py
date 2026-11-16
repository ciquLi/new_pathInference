"""Native loops must preserve numerical values, order, flags and error semantics."""
import copy
import hashlib
import json
from collections import defaultdict
from itertools import permutations
from pathlib import Path
import random
import sys
from types import MethodType
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]/'research/baselines/pathradar/performance'
sys.path.insert(0,str(HERE))
try:
    import native_engine as fast
except (ImportError,FileNotFoundError):
    fast=None


def sample(seed):
    original=fast.engine.original
    ms=original.Model_Sample.__new__(original.Model_Sample)
    ms.provider,ms.peer,ms.customer=(defaultdict(set) for _ in range(3))
    ms.linkfreq=defaultdict(int)
    ms.linkorder=defaultdict(list)
    ms.prev_frequency,ms.next_frequency=defaultdict(int),defaultdict(int)
    ms.core_cluster={'2','5'}
    ms.relationship={}
    rng=random.Random(seed)
    for a,b in permutations(map(str,range(1,8)),2):
        if int(a)>int(b) or rng.random()<.3:
            continue
        rel=rng.choice(['p2p','p2c','c2p'])
        if rel=='p2p':
            ms.peer[a].add(b);ms.peer[b].add(a)
        else:
            p,c=(a,b) if rel=='p2c' else (b,a)
            ms.customer[p].add(c);ms.provider[c].add(p)
        ms.relationship[a,b]=rel
        ms.relationship[b,a]={'p2p':'p2p','p2c':'c2p','c2p':'p2c'}[rel]
        ms.linkorder[a].append(b);ms.linkorder[b].append(a)
        ms.linkfreq[a,b]=rng.randrange(20)
        ms.linkfreq[b,a]=rng.randrange(20)
        ms.prev_frequency[a+'|'+b]=rng.randrange(20)
        ms.next_frequency[b+'|'+a]=rng.randrange(20)
    ms.shortest={}
    return ms


class FakeModel:
    def predict_proba(self,atts):
        np=fast.engine.original.np
        scores=np.asarray([(.5,.75,.25)[int(sum(row))%3] for row in atts],dtype=np.float32)
        return np.column_stack([1-scores,scores])


def routing(ms,cls):
    rt=cls.__new__(cls)
    rt.ms=ms
    rt.threshold=.5
    rt.model={kind+number:FakeModel() for kind in ('shell','core') for number in ('1','2','3')}
    return rt


@unittest.skipIf(fast is None,'Build optional PathRadar native extension first')
class NativeTests(unittest.TestCase):
    def test_shortest_and_full_routing_preserve_values_and_insertion_order(self):
        for seed in range(12):
            for source in ('1','2','5','7'):
                a=routing(sample(seed),fast.engine.original.Routingtree)
                b=routing(sample(seed),fast.NativeTree)
                fast.enable(b)
                a.simulate(source);b.simulate(source)
                for left,right in [(a.AS_path,b.AS_path),(a.path_flag,b.path_flag),
                                   (a.spread_score,b.spread_score),(a.ms.AS_path,b.ms.AS_path),
                                   (a.ms.worst,b.ms.worst),(a.ms.path_flag,b.ms.path_flag),
                                   (a.ms.shortest,b.ms.shortest)]:
                    self.assertEqual(list(left.items()),list(right.items()),(seed,source))

    def test_features_match_with_unknown_shortest_missing_frequency_and_duplicate_order(self):
        ms=sample(42)
        ms.linkorder['1']=['2','3','2']
        ms.shortest={('1','2'):2,('1','3'):5}
        paths=[('1','2'),('2','1'),('1','2','3'),('1','3','2','1','2')]
        for path in paths:
            for a,b in zip(path,path[1:]):
                ms.relationship.setdefault((a,b),'p2p')
        expected=[ms.get_sample(path) for path in paths]
        self.assertEqual(expected,fast.native.features(ms,paths))
        fast.engine.index_link_order(ms)
        self.assertEqual(expected,fast.native.features(ms,paths))
        self.assertEqual([],fast.native.features(ms,[]))
        with self.assertRaises(ValueError):
            fast.native.features(ms,[('1',)])
        del ms.relationship['1','2']
        with self.assertRaises(KeyError):
            fast.native.features(ms,[('1','2')])

    def test_nine_node_limit_and_extra_hop_limit_preserved(self):
        rt=routing(sample(1),fast.NativeTree)
        rt.path_flag={tuple(str(i) for i in range(1,10)):0,('1','3'):0}
        rt.ms.shortest={('1','2'):0}
        paths=fast.native.expand(rt,list(rt.path_flag))
        self.assertFalse(any(len(path)>9 or path==('1','3','2') for group in paths.values() for path in group))


class NativeVerificationTests(unittest.TestCase):
    def test_changed_binary_model_code_and_failed_proof_block_deployment(self):
        import run_dates
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'runs/baselines/pathradar/performance'
            build = folder/'native_build/BUILD.json'
            proof = folder/'native_equivalence_round1/REPORT.json'
            build.parent.mkdir(parents=True)
            proof.parent.mkdir(parents=True)
            source, binary, model = (root/name for name in ('native.cpp', 'native.so', 'model'))
            for path in (source, binary, model):
                path.write_text(path.name)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            build.write_text(json.dumps(dict(source_sha256=digest(source), binary_sha256=digest(binary), binary=str(binary))))
            report = dict(status='PASS', comparisons=[dict(equal=True, native_checked=dict(verified_features=1))]*3,
                          code_sha256={str(source):digest(source)}, model_sha256={str(model):digest(model)},
                          build_sha256=digest(build))
            proof.write_text(json.dumps(report))
            with patch.object(run_dates, 'ROOT', root), patch.object(run_dates, 'PERFORMANCE', root):
                self.assertIn(str(proof), run_dates.verify_native())
                for path in (binary, model, source):
                    old = path.read_bytes()
                    path.write_bytes(old+b'changed')
                    with self.assertRaises(ValueError):
                        run_dates.verify_native()
                    path.write_bytes(old)
                for invalid in (dict(report, status='RUNNING'), dict(report, comparisons=[])):
                    proof.write_text(json.dumps(invalid))
                    with self.assertRaises(ValueError):
                        run_dates.verify_native()


if __name__=='__main__':
    unittest.main()
