"""Unified255 prefix features computations."""
import math


FEATURE_NAMES = tuple(
    sorted(
        (
            "prefix_local_p_known_count",
            "prefix_local_p_missing_count",
            "prefix_local_p_missing_fraction",
            "prefix_local_p_min",
            "prefix_local_p_max",
            "prefix_local_p_mean",
            "prefix_local_p_log_sum",
            "prefix_local_p_zero_count",
            "prefix_local_p_first",
            "prefix_local_p_first_missing",
            "prefix_local_p_last",
            "prefix_local_p_last_missing",
            "prefix_local_p_uncalibrated_count",
            "prefix_local_p_calibration_unknown_count",
            "prefix_local_rank_last",
            "prefix_local_rank_last_missing",
            "prefix_inferred_edge_count",
            "prefix_inferred_edge_fraction",
            "prefix_public_edge_fraction",
            "prefix_phase_possible_count",
            "prefix_phase_can_up",
            "prefix_phase_can_after_peer",
            "prefix_phase_can_down",
            "prefix_phase_conflict",
            "prefix_phase_unknown_relation_count",
            "firsthop_observed_transit_raw_p",
            "firsthop_observed_transit_missing",
            "firsthop_observed_transit_uncalibrated",
            "firsthop_observed_transit_calibration_unknown",
            "firsthop_role_requires_transit",
            "firsthop_role_transit_deficit_soft",
            "firsthop_role_transit_deficit_missing",
        )
    )
)


def enrich_prefix_proofs(candidate, plan):
    """Restore calibration/rank provenance from the matching frozen prefix."""
    result = dict(candidate)
    prefix = tuple(map(str, candidate["prefix"]))
    proofs = []
    known = plan.get("prefix_edge_provenance", {})
    queries = plan.get("frontier_prediction_queries", {})
    for i, proof in enumerate(candidate.get("prefix_edge_evidence", []), 1):
        proof = dict(proof)
        text = "->".join(prefix[: i + 1])
        raw = known.get(text, {})
        if "probability_calibrated" in raw:
            proof["probability_calibrated"] = bool(raw["probability_calibrated"])
        query = queries.get("->".join(prefix[:i]), {})
        for rank, row in enumerate(query.get("candidates", []), 1):
            if str(row["asn"]) == prefix[i]:
                proof["local_filtered_rank"] = rank
                break
        proofs.append(proof)
    result["prefix_edge_evidence"] = proofs
    return result


def _probability(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Raw local-support scores must lie in[0,1]")
    return value


def phase_set(relations):
    states = {"up"}
    for relation in relations:
        roles = (
            ("c2p", "p2p", "p2c") if relation not in ("c2p", "p2p", "p2c", "s2s") else (relation,)
        )
        nxt = set()
        for state in states:
            for role in roles:
                if role == "s2s":
                    nxt.add(state)
                elif state == "up":
                    nxt.add({"c2p": "up", "p2p": "after_peer", "p2c": "down"}[role])
                elif role == "p2c":
                    nxt.add("down")
        states = nxt
    return states


def features(candidate, firsthop_observed_transit=None):
    prefix = candidate["prefix"]
    expected = len(prefix) - 1
    proofs = candidate.get("prefix_edge_evidence", [])
    if len(proofs) != expected:
        raise ValueError("Explicit prefix-edge proof/missingness required")
    probabilities = [_probability(p.get("observed_support_probability")) for p in proofs]
    values = [p for p in probabilities if p is not None]
    missing = expected - len(values)
    phases = phase_set([p.get("relationship", "unknown") for p in proofs])
    role = dict(firsthop_observed_transit or {})
    if "probability" not in role:
        role["probability"] = None if role.get("role_missing") else role.get("role_p")
    if "probability_calibrated" not in role and "role_probability_calibrated" in role:
        role["probability_calibrated"] = role["role_probability_calibrated"]
    rp = _probability(role.get("probability"))
    requires_transit = len(candidate.get("path", prefix)) > 2
    rank = proofs[-1].get("local_filtered_rank")
    inferred = sum((bool(p.get("inferred_only")) for p in proofs))
    out = dict(
        prefix_local_p_known_count=len(values),
        prefix_local_p_missing_count=missing,
        prefix_local_p_missing_fraction=missing / expected,
        prefix_local_p_min=min(values) if values else 0.0,
        prefix_local_p_max=max(values) if values else 0.0,
        prefix_local_p_mean=sum(values) / len(values) if values else 0.0,
        prefix_local_p_log_sum=sum((math.log(max(p, 1e-06)) for p in values)),
        prefix_local_p_zero_count=sum((p == 0 for p in values)),
        prefix_local_p_first=probabilities[0] if probabilities[0] is not None else 0.0,
        prefix_local_p_first_missing=float(probabilities[0] is None),
        prefix_local_p_last=probabilities[-1] if probabilities[-1] is not None else 0.0,
        prefix_local_p_last_missing=float(probabilities[-1] is None),
        prefix_local_p_uncalibrated_count=sum(
            (
                p.get("probability_calibrated") is False and v is not None
                for (p, v) in zip(proofs, probabilities)
            )
        ),
        prefix_local_p_calibration_unknown_count=sum(
            (
                "probability_calibrated" not in p and v is not None
                for (p, v) in zip(proofs, probabilities)
            )
        ),
        prefix_local_rank_last=float(rank or 0),
        prefix_local_rank_last_missing=float(rank is None),
        prefix_inferred_edge_count=inferred,
        prefix_inferred_edge_fraction=inferred / expected,
        prefix_public_edge_fraction=sum((bool(p.get("public_edge")) for p in proofs)) / expected,
        prefix_phase_possible_count=len(phases),
        prefix_phase_can_up=float("up" in phases),
        prefix_phase_can_after_peer=float("after_peer" in phases),
        prefix_phase_can_down=float("down" in phases),
        prefix_phase_conflict=float(not phases),
        prefix_phase_unknown_relation_count=sum(
            (p.get("relationship", "unknown") not in ("c2p", "p2p", "p2c", "s2s") for p in proofs)
        ),
        firsthop_observed_transit_raw_p=rp if rp is not None else 0.0,
        firsthop_observed_transit_missing=float(rp is None),
        firsthop_observed_transit_uncalibrated=float(
            rp is not None and role.get("probability_calibrated") is False
        ),
        firsthop_observed_transit_calibration_unknown=float(
            rp is not None and "probability_calibrated" not in role
        ),
        firsthop_role_requires_transit=float(requires_transit),
        firsthop_role_transit_deficit_soft=1 - rp if requires_transit and rp is not None else 0.0,
        firsthop_role_transit_deficit_missing=float(requires_transit and rp is None),
    )
    assert set(out) == set(FEATURE_NAMES)
    return {name: float(out[name]) for name in FEATURE_NAMES}
