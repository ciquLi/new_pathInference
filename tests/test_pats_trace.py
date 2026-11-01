"""Exercise the traceroute adapter and its admitted-library boundary."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pathveil.prefix_tree import traceroute
from pathveil.prefix_tree.registered_topology import RegisteredPublicPrefixTopology as PublicPrefixTopology
from pathveil.traceroute.common import path_signature
from pathveil.utils import sha256, write_json


class PatsTraceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trees = self.root / 'trees'
        method = self.trees / 'METHOD_FROZEN.json'
        write_json(method, {'method': 'PATS'})
        self.library = [['1', '2', '3']]
        self.contract = dict(source='1', date='20261015', rows=1,
                             path_set_sha256=path_signature(self.library))
        spec = dict(source='1', date='20261015',
                    admitted_path_set_sha256=self.contract['path_set_sha256'],
                    construction_method_sha256=sha256(method), data_code_input_sha256={},
                    graph_identity_sha256='public graph',
                    prefix_classification=[{'prefix': ['1', '2']}])
        write_json(self.trees / '1.json', spec)
        self.data = self.root / 'data.json'
        write_json(self.data, dict(date='20261015', source_data_root=str(self.root)))
        index = self.root / 'data/aspath/path_by_vantage/20261015/path_without_prefix/train/pathWithKey.pkl'
        index.parent.mkdir(parents=True)
        index.write_bytes(b'training-only observed tails')
        self.hf = SimpleNamespace(load_local_knowledge_paths=Mock(return_value=self.library),
                                  networkit_sp=Mock(), generate_kpath_nosave=None)
        self.public = SimpleNamespace(
            PublicPrefixTopology=SimpleNamespace(from_manifest=Mock(return_value=SimpleNamespace(
                graph_identity_sha256='public graph'))), commercial_phases=Mock())
        self.selectors = SimpleNamespace(PublicHeadSelector=SimpleNamespace(from_records=Mock()))
        self.env = SimpleNamespace(augment_graph_for_external_nodes=Mock())
        topology = patch.object(traceroute.RegisteredPublicPrefixTopology, 'from_manifest',
                                self.public.PublicPrefixTopology.from_manifest)
        topology.start()
        self.addCleanup(topology.stop)

    def install(self):
        return traceroute.install_generator(self.trees, self.data, self.contract, self.root / 'output')

    def test_same_numerical_generator_receives_frozen_tree_and_exact_library(self):
        numerical = Mock(return_value=({'1-3': self.library}, {}))
        with patch.dict(sys.modules, HighFrequencyVertexBasedPathGen=self.hf), patch.object(
            traceroute.base, 'siblings', return_value=(Mock(), self.public, self.selectors)
        ), patch.object(traceroute.generator, 'generate_public_candidates', numerical):
            self.install()
            self.assertEqual(self.hf.generate_kpath_nosave(self.env, 200, '1', ['3']),
                             {'1-3': self.library})
            self.assertEqual(numerical.call_args.args[2:4], ('1', ['3']))
            self.assertEqual(numerical.call_args.args[10], self.library)
            self.assertIsNone(numerical.call_args.kwargs['legacy_generator'])
            self.hf.load_local_knowledge_paths.return_value = [['1', '9', '3']]
            with self.assertRaisesRegex(ValueError, 'admitted training labels'):
                self.hf.generate_kpath_nosave(self.env, 200, '1', ['3'])
            self.assertEqual(numerical.call_count, 1)

    def test_tree_mutation_is_rejected_before_generation(self):
        with patch.dict(sys.modules, HighFrequencyVertexBasedPathGen=self.hf), patch.object(
            traceroute.base, 'siblings', return_value=(Mock(), self.public, self.selectors)
        ):
            self.install()
            write_json(self.trees / '1.json', {'changed': True})
            with self.assertRaises(ValueError):
                self.hf.generate_kpath_nosave(self.env, 200, '1', ['3'])


class ExplicitPublicSnapshotTests(unittest.TestCase):
    def test_manifest_admits_only_the_registered_older_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes, links, rel = [root / name for name in ('nodes.jsonl', 'links.jsonl', 'rel.txt')]
            nodes.write_text('{"asn": "1", "date": "2026-10-01"}\n')
            links.write_text('')
            rel.write_text('')
            data = dict(date='20261015', public_files=dict(
                asrank_nodes=str(nodes), asrank_links=str(links), asrel=str(rel)))
            with self.assertRaisesRegex(ValueError, 'row date differs'):
                PublicPrefixTopology.from_manifest(data)
            data['auxiliary_dates'] = {'asrank': '20261001'}
            graph = PublicPrefixTopology.from_manifest(data)
            self.assertEqual(graph.date, '20261015')
            self.assertEqual(graph.asrank_date, '20261001')
            data['auxiliary_dates']['asrank'] = '20261016'
            with self.assertRaisesRegex(ValueError, 'on or before'):
                PublicPrefixTopology.from_manifest(data)


if __name__ == '__main__':
    unittest.main()
