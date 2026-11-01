"""Unified255 prior spec computations."""


def prediction_spec(source, prediction, public):
    """Accept old nested rankings plus explicit new selected_neighbors."""
    method = prediction.get("method", "rank_pdb_legacy")
    root = prediction["root"]
    selected = prediction.get("selected_neighbors", root.get("selected_neighbors"))
    if selected is None:
        selected = root["rankings"][method][:10]
    if isinstance(selected, dict):
        selected = list(selected)
    selected = [str(n["neighbor"] if isinstance(n, dict) else n) for n in selected]
    prefixes = []
    proofs = {}
    gateway = {}
    recursive = {}
    degree = int(prediction["visible_degree"])
    for rank, n in enumerate(selected, 1):
        prefixes.append((str(source), n))
        proof = root.get("provenance", {}).get(n, {})
        proofs[str(source), n] = proof
        gateway[n] = dict(
            pred_rank=rank,
            priority=rank,
            vas_graph_neighbor_count=degree,
            relationship=public.relations.get((int(source), int(n)), "unknown"),
        )
        if degree > 10:
            continue
        query = prediction.get("recursive", {}).get(n, {})
        anchors = query.get("selected_neighbors")
        if anchors is None:
            anchors = query.get("rankings", {}).get(method, [])[:10]
        for arank, a in enumerate(anchors, 1):
            a = str(a["neighbor"] if isinstance(a, dict) else a)
            if a in (str(source), n):
                continue
            prefixes.append((str(source), n, a))
            proofs[n, a] = query.get("provenance", {}).get(a, {})
            recursive.setdefault(n, {})[a] = dict(
                center_evidence={str(source): dict(pred_rank=arank)}
            )
    return (prefixes, proofs, dict(gateway={str(source): gateway}, recursive=recursive), selected)
