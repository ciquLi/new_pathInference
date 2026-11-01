"""Verify source holdout and immutable reuse of the date-local BGP view."""
import json
from pathlib import Path
import tempfile
import unittest

from pathveil.data.bgp_holdout import ensure_cohort
from pathveil.data.prepare import read_double_json


class BgpHoldoutTests(unittest.TestCase):
    def test_rebuilt_graph_and_vectors_exclude_heldout_originals_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external"
            original = external / "data/aspath/path_by_vantage/20261015/path_without_prefix"
            original.mkdir(parents=True)
            for source, routes in {"1": "1->9->3\n", "2": "2->4\n", "9": "9->777->888\n"}.items():
                (original / (source + ".csv")).write_text(routes)
            asrank = root / "asrank.jsonl"
            asrank.write_text(json.dumps(dict(asn="1", date="2026-10-01")) + "\n")
            registry_file = root / "registry.json"
            registry_file.write_text("{}\n")
            local = root / "data/cohorts/20261015/files"
            registry = dict(date="20261015", project_root=str(root), files_root=str(external),
                routing_files_root=str(local), test_sources=["9"],
                rebuild_bgp_holdout={"policy": "original_source_holdout_v1"},
                registry_path=str(registry_file), auxiliary_dates={"asrank": "20261001"},
                raw_inputs=[dict(id="asrank_asns", type="file", candidates=[str(asrank)])])
            before = {str(p): p.read_bytes() for p in original.glob("*.csv")}
            phases = []
            self.assertEqual(ensure_cohort(registry, phases.append), local)
            paths = local / "data/aspath/path_by_vantage/20261015/path_without_prefix"
            self.assertFalse((paths / "train/9.csv").exists())
            self.assertEqual((paths / "test/9.csv").read_text(), "9->777->888\n")
            self.assertNotIn("777", (paths / "train/expand_paths.txt").read_text())
            self.assertIn("9->3\n", (paths / "train/expand_paths.txt").read_text())
            graph = paths.parent / "RL_Graph"
            self.assertEqual(set(read_double_json(graph / "edge_frequency.json")), {"1-9", "9-3", "2-4"})
            vector_nodes = {line.split()[0] for line in
                (graph / "20261015_246_embeddings_collectedPath.emb").read_text().splitlines()[1:]}
            self.assertEqual(vector_nodes, {"1", "2", "3", "4", "9"})
            self.assertEqual(before, {str(p): p.read_bytes() for p in original.glob("*.csv")})
            outputs = {str(p): p.stat().st_mtime_ns for p in local.rglob("*") if p.is_file()}
            ensure_cohort(registry, phases.append)
            self.assertEqual(outputs, {str(p): p.stat().st_mtime_ns for p in local.rglob("*") if p.is_file()})
            self.assertIn("cohort_EMBEDDING_reused", phases)
            (graph / "edge_frequency.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Cohort artifact changed"):
                ensure_cohort(registry, phases.append)


if __name__ == "__main__":
    unittest.main()
