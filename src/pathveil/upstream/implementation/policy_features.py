"""Unified255 policy features computations."""
from datetime import datetime
import math
from .retrieval import PDBRetrievalBuilder


def rpsl_features(policy, neighbor):
    valid = policy["valid"]
    peers = policy["neighbors"]
    p = peers.get(str(neighbor), {})
    has_import = p.get("import_count", 0) > 0
    has_export = p.get("export_count", 0) > 0
    prefs = {n: min(v["pref_values"]) for (n, v) in peers.items() if v["pref_values"]}
    pref = prefs.get(str(neighbor))
    ordered = sorted(set(prefs.values()))
    t = policy.get("snapshot_from")
    age = (
        (datetime(2026, 3, 1) - datetime.fromisoformat(t.replace("Z", ""))).total_seconds() / 86400
        if t
        else 0.0
    )
    f = dict(
        rpsl_source_valid=float(valid),
        rpsl_source_unknown=float(not valid),
        rpsl_source_policy_rows_log1p=math.log1p(policy["rows"]),
        rpsl_source_declared_peers_log1p=math.log1p(len(peers)),
        rpsl_source_unresolved_terms_log1p=math.log1p(policy["unresolved_terms"]),
        rpsl_source_unresolved_fraction=policy["unresolved_terms"] / max(1, policy["rows"]),
        rpsl_source_snapshot_age_days_log1p=math.log1p(max(0.0, age)),
        rpsl_peer_import_declared=float(has_import),
        rpsl_peer_export_declared=float(has_export),
        rpsl_peer_bidirectional=float(has_import and has_export),
        rpsl_peer_import_only=float(has_import and (not has_export)),
        rpsl_peer_export_only=float(has_export and (not has_import)),
        rpsl_peer_default_declared=float(p.get("default_count", 0) > 0),
        rpsl_peer_pref_missing=float(pref is None),
        rpsl_peer_pref_log1p=math.log1p(pref or 0),
        rpsl_peer_pref_rank_inverse=1 / (ordered.index(pref) + 1) if pref is not None else 0.0,
        rpsl_peer_import_share=p.get("import_count", 0)
        / max(1, sum((v["import_count"] for v in peers.values()))),
        rpsl_peer_export_share=p.get("export_count", 0)
        / max(1, sum((v["export_count"] for v in peers.values()))),
    )
    for field in [
        "import_count",
        "export_count",
        "default_count",
        "import_any",
        "export_any",
        "import_none",
        "export_none",
        "ipv4",
        "ipv6",
        "import_origin_only",
        "import_other_origin",
        "import_as_set",
        "import_prefix_list",
        "import_path_regex",
        "export_origin_only",
        "export_other_origin",
        "export_as_set",
        "export_prefix_list",
        "export_path_regex",
    ]:
        f["rpsl_peer_" + field + "_log1p"] = math.log1p(p.get(field, 0))
    return f


class PolicyBuilder(PDBRetrievalBuilder):
    def __init__(self, public, evidence, policies):
        super().__init__(public, evidence, 4096)
        self.policies = policies

    def candidates(self, source, center):
        (base, uncapped) = super().candidates(source, center)
        if source != center:
            return (base, uncapped)
        extra = (
            set(map(int, self.policies[str(source)]["neighbors"])) - set(base) - {source, center}
        )
        return (base + sorted(extra), uncapped + len(extra))
