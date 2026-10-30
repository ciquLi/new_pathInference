from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from pathveil.upstream.contracts import UpstreamError, sha, validate_model_scope
from pathveil.upstream.destination import DestinationFeatureStore
from pathveil.upstream.method import MethodLibrary
from pathveil.upstream.witnesses import WitnessStore, prepare_witnesses


class UpstreamScopeTests(unittest.TestCase):
    def test_later_model_cannot_be_relabelled_as_an_earlier_local_model(self):
        bundle = dict(
            date="20260201", model_fit_date="20260301", protocol="date_local_source_disjoint_v1"
        )
        with self.assertRaisesRegex(UpstreamError, "fitted on this same date"):
            validate_model_scope(bundle, "20260201", {})

    def test_method_is_self_contained_and_keeps_222_plus_33_columns(self):
        library = MethodLibrary()
        c = library.core()
        self.assertEqual(
            222,
            len(c.pathbase.FEATURE_NAMES)
            + len(c.final.prefix_features.FEATURE_NAMES)
            + len(c.final.next_features.FEATURE_NAMES),
        )
        self.assertEqual(33, len(c.destination.APPENDED_NAMES))
        self.assertTrue(library.receipt()["self_contained"])
        self.assertTrue(
            all(
                "pathveil/upstream" in p and "releases" not in p
                for p in library.receipt()["computational_sources_sha256"]
            )
        )

    def test_destination_donor_counts_mass_and_receiver_independent_source_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donors = {}
            expected = {}
            for source, fold, dest, alternatives, role in [
                (1, 0, 90, [20, 30], 0),
                (2, 1, 90, [20], 0),
                (3, 2, 90, [30], 2),
                (4, 3, 90, [30], 4),
                (5, 4, 91, [20], 0),
            ]:
                path = root / (str(source) + ".npz")
                np.savez_compressed(
                    path,
                    ND_N=np.array([10], np.uint32),
                    ND_D=np.array([dest], np.uint32),
                    ND_total_unique_full_paths=np.array([len(alternatives)], np.uint32),
                    ND_role_code=np.array([role], np.uint8),
                    NAD_nd_index=np.zeros(len(alternatives), np.uint32),
                    NAD_A=np.array(alternatives, np.uint32),
                    NAD_unique_full_path_count=np.ones(len(alternatives), np.uint32),
                )
                donors[str(source)] = str(path)
                expected[str(path)] = sha(path)
            public = SimpleNamespace(
                relations={(100, 10): "c2p", (200, 10): "p2c"},
                asrel_relations={(100, 10): "c2p", (200, 10): "p2c"},
                links={},
            )
            bundle = dict(
                donor_sources=[1, 2, 3, 4, 5],
                donor_shards=donors,
                source_to_fold={str(i): i - 1 for i in range(1, 6)},
                inputs_sha256=expected,
            )
            store = DestinationFeatureStore(bundle, public)
            args = [np.array([10], np.uint32), np.array([20], np.uint32), np.array([90], np.uint32)]
            full = store.batch_query(99, *args, np.array([100], np.uint32))[0]
            held = store.batch_query(1, *args, np.array([100], np.uint32))[0]
            other_receiver = store.batch_query(1, *args, np.array([200], np.uint32))[0]
            self.assertAlmostEqual(0.5, float(full[2]))
            self.assertAlmostEqual(0.375, float(full[3]))
            self.assertAlmostEqual(1 / 3, float(held[2]), places=6)
            self.assertAlmostEqual(1 / 3, float(held[3]), places=6)
            self.assertEqual(1.0, held[8])
            self.assertEqual(1.0, held[9])
            np.testing.assert_array_equal(held[:6], other_receiver[:6])
            self.assertEqual(0.0, other_receiver[8])
            self.assertEqual(1.0, other_receiver[11])

    def test_witnesses_keep_terminal_and_exclude_formal_origin_before_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged = root / "merged.txt"
            merged.write_text("1->2->3->4\n9->2->3->4\n7->2->3\n9->2->2->3->4\n")
            split = dict(split=dict(train=[9], validation=[7], formal_test=[1]))
            path = prepare_witnesses(
                dict(date="20260501", merged_train=str(merged)), split, [1, 2, 3], root / "output"
            )
            store = WitnessStore(path, [9], {"9": 0})
            try:
                self.assertEqual({("2", "3", "4"), ("2", "3")}, store.source(2))
                self.assertEqual({("2", "3", "4")}, store.source(2, training_only=True))
                self.assertEqual(set(), store.source(1))
                self.assertNotIn(("3", "2"), store.source(3))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
