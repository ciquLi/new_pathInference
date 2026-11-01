VALID_RELATIONSHIPS = {"c2p", "p2p", "p2c", "s2s"}
VALID_BRANCH_QUALITIES = {"normal", "low"}
VALID_RECURSIVE_POLICIES = {"auto", "force", "never"}


def _as_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value, default):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def make_entry(
    relationship,
    priority=1,
    confidence=0.0,
    selection_reason="",
    branch_quality="normal",
    path_count=0,
    path_share=0.0,
    recursive_policy="auto",
    recursive_score=0.0,
    recursive_budget_hint="none",
    recursive_reasons=None,
    preferred_second_hop_roles=None,
):
    relationship = str(relationship or "s2s")
    if relationship not in VALID_RELATIONSHIPS:
        relationship = "s2s"

    branch_quality = str(branch_quality or "normal")
    if branch_quality not in VALID_BRANCH_QUALITIES:
        branch_quality = "normal"

    recursive_policy = str(recursive_policy or "auto")
    if recursive_policy not in VALID_RECURSIVE_POLICIES:
        recursive_policy = "auto"

    priority = max(1, _as_int(priority, 1))
    return {
        "relationship": relationship,
        "priority": priority,
        "confidence": _as_float(confidence, 0.0),
        "selection_reason": str(selection_reason or "selected"),
        "branch_quality": branch_quality,
        "path_count": max(0, _as_int(path_count, 0)),
        "path_share": _as_float(path_share, 0.0),
        "recursive_policy": recursive_policy,
        "recursive_score": _as_float(recursive_score, 0.0),
        "recursive_budget_hint": str(recursive_budget_hint or "none"),
        "recursive_reasons": list(recursive_reasons or []),
        "preferred_second_hop_roles": list(preferred_second_hop_roles or []),
    }


def normalize_prior(prior):
    normalized = {}
    for vas, neighbors in (prior or {}).items():
        vas_key = str(vas)
        normalized[vas_key] = {}
        for neighbor, meta in (neighbors or {}).items():
            meta = meta or {}
            normalized[vas_key][str(neighbor)] = make_entry(
                relationship=meta.get("relationship"),
                priority=meta.get("priority"),
                confidence=meta.get("confidence"),
                selection_reason=meta.get("selection_reason"),
                branch_quality=meta.get("branch_quality"),
                path_count=meta.get("path_count", 0),
                path_share=meta.get("path_share", 0.0),
                recursive_policy=meta.get("recursive_policy", "auto"),
                recursive_score=meta.get("recursive_score", 0.0),
                recursive_budget_hint=meta.get("recursive_budget_hint", "none"),
                recursive_reasons=meta.get("recursive_reasons", []),
                preferred_second_hop_roles=meta.get("preferred_second_hop_roles", []),
            )
    return normalized
