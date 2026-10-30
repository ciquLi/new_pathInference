"""Origin masking, explicit public selection, and dated policy boundaries."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest

from pathveil.upstream.contracts import UpstreamError, sha
from pathveil.upstream.evidence import load_policies, make_public, prepare_evidence
from pathveil.upstream.method import MethodLibrary


class NeighborBoundaries(unittest.TestCase):
    def test_actual_origin_mask_before_suffix_expansion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Its filename deliberately names a formal source. Actual first
            # ASNs identify which rows may enter residual evidence and labels.
            canonical = root / "99.csv"
            canonical.write_text(
                "10->701->800\n10->900\n11->702->801\n12->703->802\n"
                "13->704->803\n14->705->804\n98->810->820\n99->830->840\n"
                "200->10->300\n200->10->300\n201->400->500\n"
            )
            split = {
                "split": {"train": [10, 11, 12, 13, 14], "validation": [98], "formal_test": [99]}
            }
            result = prepare_evidence(
                {"date": "20260501", "merged_train": str(canonical)},
                split,
                root / "output",
                MethodLibrary().core(),
            )
            with Path(result["evidence_pickle"]).open("rb") as stream:
                evidence = pickle.load(stream)
            self.assertEqual(evidence["starts"], {200: 2, 201: 1})
            self.assertEqual(evidence["edges"][10, 300], 2)
            self.assertNotIn((10, 701), evidence["edges"])
            with Path(result["safe_index_pickle"]).open("rb") as stream:
                segments = pickle.load(stream)
            self.assertIn("10->300", segments["index"]["10"])
            self.assertNotIn("300->10", segments["index"].get("300", {}))
            refs = json.loads(Path(result["training_references"]).read_text())
            self.assertEqual(set(refs), {"10", "11", "12", "13", "14"})
            self.assertEqual(refs["10"]["counts"]["paths"], 2)
            self.assertFalse(result["formal_labels_read"])
            self.assertFalse(result["validation_labels_read"])

    def test_public_parser_uses_only_explicit_named_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {k: root / k for k in ("asrel", "asrank_links", "asrank_nodes", "peeringdb")}
            files["asrel"].write_text("10|20|-1\n")
            files["asrank_links"].write_text(
                json.dumps(
                    {
                        "date": "2026-05-01",
                        "asn0": {"asn": "10"},
                        "asn1": {"asn": "20"},
                        "relationship": "customer",
                        "numberPaths": 3,
                    }
                )
                + "\n"
            )
            files["asrank_nodes"].write_text(
                json.dumps({"date": "2026-05-01", "asn": "10", "rank": 1}) + "\n"
            )
            files["peeringdb"].write_text("{}")
            data = {
                "date": "20260501",
                "public_files": {k: str(p) for k, p in files.items()},
                "public_input_sha256": {k: sha(p) for k, p in files.items()},
                "auxiliary_dates": {
                    "asrank": "20260501",
                    "relationships": "20260501",
                    "peeringdb": "20260501",
                },
            }
            public = make_public(data, MethodLibrary().core())
            self.assertEqual(public.paths["asrank_nodes"], str(files["asrank_nodes"]))
            self.assertEqual(public.relations[10, 20], "p2c")
            self.assertEqual(public.links[10, 20]["number_paths"], 3.0)
            data["auxiliary_dates"]["asrank"] = "20260601"
            with self.assertRaises(UpstreamError):
                make_public(data, MethodLibrary().core())

    def test_missing_policy_is_unknown_and_future_policy_is_rejected(self):
        records, receipt = load_policies({"date": "20260501"}, [10])
        self.assertFalse(records["10"]["valid"])
        self.assertEqual(receipt["status"], "unknown_missing_dated_rpsl")
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "policies.json"
            p.write_text(
                json.dumps({"10": dict(records["10"], snapshot_from="2026-06-01T00:00:00Z")})
            )
            with self.assertRaises(UpstreamError):
                load_policies({"date": "20260501", "parsed_policy_file": str(p)}, [10])


if __name__ == "__main__":
    unittest.main()
