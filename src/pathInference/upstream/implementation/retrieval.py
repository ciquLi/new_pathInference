"""Unified255 retrieval computations."""
from collections import defaultdict
from .neighbor_features import FeatureBuilder as LegacyFeatureBuilder


def reverse_presence(mapping):
    reverse = defaultdict(set)
    for source, locations in mapping.items():
        for location in locations:
            reverse[location].add(source)
    return reverse


class PDBRetrievalBuilder(LegacyFeatureBuilder):
    def __init__(self, public, evidence, cap=4096):
        super().__init__(public, evidence, cap)
        self.ix_members = reverse_presence(public._ixps)
        self.fac_members = reverse_presence(public._facilities)

    def candidates(self, source, center):
        p = self.public
        e = self.evidence
        graph = e["adjacency"].get(center, set()) | p.adjacency.get(center, set())
        coloc = set()
        for loc in p._ixps.get(center, set()):
            coloc.update(self.ix_members[loc])
        for loc in p._facilities.get(center, set()):
            coloc.update(self.fac_members[loc])
        candidates = (graph | coloc) - {source, center}

        def priority(v):
            graph_present = int(v in graph)
            location_count = len(p._ixps.get(center, set()) & p._ixps.get(v, set())) + len(
                p._facilities.get(center, set()) & p._facilities.get(v, set())
            )
            return (
                -graph_present,
                -e["edges"][center, v],
                -p.links.get((center, v), {}).get("number_paths", 0),
                -e["edges"][v, center],
                -location_count,
                v,
            )

        return (sorted(candidates, key=priority)[: self.cap], len(candidates))
