"""Unified255 joint selector computations."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import math
import numpy as np


VERSION = "predicted_branch_length_selection_v3_branch_quality_and_global_novel_floor"


@dataclass(frozen=True)
class SelectionPolicy:
    budget: int = 2000
    minimum_paths: int = 1000
    minimum_score: float = 0.0
    novel_reserve: int = 1
    minimum_novel_paths: int = 0
    minimum_branch_paths: int = 1
    major_probability: float = 0.1
    major_topk: int = 3
    major_minimum_ratio: float = 0.5
    major_minimum_paths: int = 30
    branch_strength: float = 1.0
    length_strength: float = 1.0
    joint_strength: float = 0.5
    conditional_strength: float = 0.5
    destination_repeat_strength: float = 0.1
    probability_floor: float = 1e-12


def length_aware_representatives(case):
    """Retain best overall AND best novel for every(N,D,length,conditional).

    This must run on the full structurally checked candidate pool, before the
    old best_union mask. Otherwise its one/two representatives per(N,D) may
    already have discarded all longer alternatives. No source label is used.
    Different destinations have no hard quota, and no route is lengthened.
    """
    paths = [tuple(map(str, p)) for p in case["paths"]]
    scores = np.asarray(case["scores"], dtype=float)
    novel = np.asarray(case["novel"], dtype=bool)
    base = np.asarray(case.get("structural_eligible", np.ones(len(paths), bool)), dtype=bool)
    keys = list(map(str, case.get("branch_keys", [p[1] if len(p) > 1 else "" for p in paths])))
    conditional = list(map(str, case.get("conditional_keys", [""] * len(paths))))
    if any((v.shape != (len(paths),) for v in (scores, novel, base))):
        raise ValueError("Representative candidate arrays differ")
    if len(keys) != len(paths) or len(conditional) != len(paths) or (not np.isfinite(scores).all()):
        raise ValueError("Invalid representative keys/scores")
    best = {}
    for i, path in enumerate(paths):
        if not base[i] or not 3 <= len(path) <= 20 or len(path) != len(set(path)):
            continue
        key = (keys[i], path[-1], str(len(path)), conditional[i])
        priority = (-scores[i], path, i)
        groups = [(key, "all")]
        if novel[i]:
            groups.append((key, "novel"))
        for group in groups:
            if group not in best or priority < best[group][0]:
                best[group] = (priority, i)
    take = np.zeros(len(paths), bool)
    if best:
        take[list({i for (_, i) in best.values()})] = True
    return take


def _probabilities(values, label):
    values = {str(k): float(v) for (k, v) in values.items()}
    if (
        not values
        or any((not math.isfinite(v) or v < 0 for v in values.values()))
        or sum(values.values()) <= 0
    ):
        raise ValueError(f"Missing/nonpositive learned {label} distribution; no uniform fallback")
    total = sum(values.values())
    return {k: v / total for (k, v) in values.items()}


def _increment(count, expected):
    if count == 0:
        return -math.log(expected)
    return math.log(count + 1) + count * math.log1p(1.0 / count) - math.log(expected)


def destination_equal_distribution(paths, positions):
    """positions supplies a row's branch key, including arbitrary recursive keys."""
    grouped = defaultdict(set)
    for path, branch in zip(paths, positions):
        grouped[str(path[-1])].add((tuple(path), str(branch)))
    result = Counter()
    for rows in grouped.values():
        for _, branch in rows:
            result[branch] += 1.0 / (len(grouped) * len(rows))
    return dict(result)


def select_joint_budget(
    case,
    predicted_branch,
    predicted_length,
    policy=None,
    predicted_joint=None,
    predicted_conditional=None,
):
    """Return row-aligned mask, branch records, auditable diagnostics.

    Required case fields: paths, scores, novel. Optional eligible defaults true;
    best_union is accepted for the old frozen candidate-cache adapter. Branch
    keys default to firsthop. conditional_keys can instead name the next AS at
    the FIRST REAL BRANCH after a fixed chain, supplied from evidence alone.
    The function does not pretend a common fixed prefix is concentration.

    predicted_joint maps branch->{total-length:joint probability}; optional
    predicted_conditional maps branch->{conditional_key:conditional probability}.
    All predicted distributions must be fitted without the evaluated source's
    labels. The caller owns and freezes that provenance; this pure function has
    deliberately no label or file argument.
    """
    policy = policy or SelectionPolicy()
    for key in (
        "budget",
        "minimum_paths",
        "novel_reserve",
        "minimum_novel_paths",
        "minimum_branch_paths",
        "major_topk",
        "major_minimum_paths",
    ):
        if getattr(policy, key) < 0 or int(getattr(policy, key)) != getattr(policy, key):
            raise ValueError("Count controls must be nonnegative integers")
    if policy.budget < policy.minimum_paths:
        raise ValueError("Nominal budget is below declared quantity goal")
    if not 0 <= policy.minimum_score <= 1 or not 0 < policy.probability_floor < 1:
        raise ValueError("Invalid score threshold or probability floor")
    strengths = [
        policy.branch_strength,
        policy.length_strength,
        policy.joint_strength,
        policy.conditional_strength,
        policy.destination_repeat_strength,
    ]
    if any((not math.isfinite(v) or v < 0 for v in strengths)):
        raise ValueError("Penalty strengths must be finite and nonnegative")
    if not 0 <= policy.major_probability <= 1 or not 0 <= policy.major_minimum_ratio <= 1:
        raise ValueError("Invalid major-branch safeguards")
    paths = [tuple((str(v) for v in path)) for path in case["paths"]]
    n = len(paths)
    scores = np.asarray(case["scores"], dtype=float)
    novel = np.asarray(case["novel"], dtype=bool)
    eligible = np.asarray(
        case.get("eligible", case.get("best_union", np.ones(n, bool))), dtype=bool
    )
    if any((v.shape != (n,) for v in (scores, novel, eligible))):
        raise ValueError("Candidate arrays are not row-aligned")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite observation-support probabilities")
    branches = list(map(str, case.get("branch_keys", [p[1] if len(p) > 1 else "" for p in paths])))
    conditional = list(map(str, case.get("conditional_keys", [""] * n)))
    if len(branches) != n or len(conditional) != n:
        raise ValueError("Branch keys are not row-aligned")
    root = _probabilities(predicted_branch, "branch")
    lengths = _probabilities(predicted_length, "length")
    if any((branch not in root for (branch, valid) in zip(branches, eligible) if valid)):
        raise ValueError("Candidate branch missing from learned selected-branch prior")
    if predicted_joint is not None:
        joint = {
            (str(b), str(length)): float(v)
            for (b, row) in predicted_joint.items()
            for (length, v) in row.items()
        }
        _probabilities({repr(k): v for (k, v) in joint.items()}, "joint")
        total = sum(joint.values())
        joint = {k: v / total for (k, v) in joint.items()}
    else:
        joint = {(b, length): p * q for (b, p) in root.items() for (length, q) in lengths.items()}
    cond = {
        str(b): _probabilities(row, "conditional")
        for (b, row) in (predicted_conditional or {}).items()
    }
    valid = eligible & (scores >= policy.minimum_score)
    malformed = []
    deduplicated = {}
    for row, path in enumerate(paths):
        if not valid[row]:
            continue
        if not 3 <= len(path) <= 20 or len(set(path)) != len(path):
            valid[row] = False
            malformed.append(row)
            continue
        old = deduplicated.get(path)
        if old is None or scores[row] > scores[old]:
            if old is not None:
                valid[old] = False
            deduplicated[path] = row
        else:
            valid[row] = False
    by_branch = defaultdict(list)
    for row in np.flatnonzero(valid):
        by_branch[branches[row]].append(int(row))
    for rows in by_branch.values():
        rows.sort(key=lambda i: (-scores[i], paths[i], i))
    capacity = int(valid.sum())
    target = min(policy.budget, capacity)
    selected = np.zeros(n, bool)
    mandatory = set()
    requirements = {}
    unmet = []
    major_top = set(sorted(root, key=lambda b: (-root[b], b))[: policy.major_topk])
    for branch, probability in root.items():
        rows = by_branch.get(branch, [])
        is_major = probability > 0 and (
            probability >= policy.major_probability or branch in major_top
        )
        floor = (
            max(
                policy.minimum_branch_paths,
                policy.major_minimum_paths,
                math.ceil(policy.major_minimum_ratio * probability * target),
            )
            if is_major
            else policy.minimum_branch_paths
        )
        new_rows = [i for i in rows if novel[i]][: policy.novel_reserve]
        mandatory.update(new_rows)
        requirements[branch] = floor
        if len(rows) < floor:
            unmet.append(dict(branch=branch, required=floor, eligible=len(rows)))
    required_total = sum(
        (
            max(
                sum((branches[i] == branch for i in mandatory)),
                min(requirements[branch], len(by_branch.get(branch, []))),
            )
            for branch in root
        )
    )
    if required_total > target:
        target = required_total
    counts = Counter()

    def memberships(row):
        (branch, length) = (branches[row], str(len(paths[row])))
        result = [
            (("branch", branch), policy.branch_strength, root[branch]),
            (("length", length), policy.length_strength, lengths.get(length, 0.0)),
            (("joint", branch, length), policy.joint_strength, joint.get((branch, length), 0.0)),
        ]
        if branch in cond and conditional[row]:
            result.append(
                (
                    ("conditional", branch, conditional[row]),
                    policy.conditional_strength,
                    root[branch] * cond[branch].get(conditional[row], 0.0),
                )
            )
        return result

    memberships_cache = {int(row): memberships(int(row)) for row in np.flatnonzero(valid)}
    dest_counts = Counter()

    def gain(row):
        value = math.log(max(scores[row], policy.probability_floor))
        for group, strength, probability in memberships_cache[row]:
            value -= strength * _increment(
                counts[group], max(target * probability, policy.probability_floor)
            )
        value -= policy.destination_repeat_strength * _increment(dest_counts[paths[row][-1]], 1.0)
        return value

    def commit(row):
        selected[row] = True
        for group, _, _ in memberships_cache[row]:
            counts[group] += 1
        dest_counts[paths[row][-1]] += 1

    for row in sorted(mandatory):
        commit(row)
    selected_count = len(mandatory)
    floor_counts = {b: min(requirements[b], len(by_branch.get(b, []))) for b in root}
    floor_deficit = sum((max(0, need - counts["branch", b]) for (b, need) in floor_counts.items()))
    floor_heap = [
        (-gain(int(row)), paths[row], int(row))
        for row in np.flatnonzero(valid & ~selected)
        if counts["branch", branches[row]] < floor_counts[branches[row]]
    ]
    heapq.heapify(floor_heap)
    while floor_deficit:
        if not floor_heap:
            raise AssertionError("Available major-branch floor capacity exhausted")
        (_, path, row) = heapq.heappop(floor_heap)
        if counts["branch", branches[row]] >= floor_counts[branches[row]]:
            continue
        current = gain(row)
        if floor_heap and current < -floor_heap[0][0] - 1e-12:
            heapq.heappush(floor_heap, (-current, path, row))
            continue
        commit(row)
        floor_deficit -= 1
        selected_count += 1
    mandatory_after_floors = selected_count
    novel_before = int(novel[selected].sum())
    novel_deficit = max(0, policy.minimum_novel_paths - novel_before)
    novel_rows = np.flatnonzero(valid & ~selected & novel)
    achievable_novel = min(novel_deficit, len(novel_rows))
    target = max(target, selected_count + achievable_novel)
    novel_heap = [(-gain(int(row)), paths[row], int(row)) for row in novel_rows]
    heapq.heapify(novel_heap)
    remaining_novel = achievable_novel
    while remaining_novel:
        if not novel_heap:
            raise AssertionError("Available novel capacity exhausted")
        (_, path, row) = heapq.heappop(novel_heap)
        current = gain(row)
        if novel_heap and current < -novel_heap[0][0] - 1e-12:
            heapq.heappush(novel_heap, (-current, path, row))
            continue
        commit(row)
        remaining_novel -= 1
        selected_count += 1
    novel_floor_available = novel_before + len(novel_rows)
    heap = [(-gain(int(row)), paths[row], int(row)) for row in np.flatnonzero(valid & ~selected)]
    heapq.heapify(heap)
    while selected_count < target:
        if not heap:
            raise AssertionError("Valid candidate capacity exhausted")
        (_, path, row) = heapq.heappop(heap)
        current = gain(row)
        if heap and current < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-current, path, row))
            continue
        commit(row)
        selected_count += 1
    rows = list(map(int, np.flatnonzero(selected)))
    chosen = [paths[i] for i in rows]
    records = []
    for branch in root:
        take = [i for i in rows if branches[i] == branch]
        records.append(
            dict(
                branch=branch,
                predicted_probability=root[branch],
                eligible_paths=len(by_branch.get(branch, [])),
                selected_paths=len(take),
                selected_novel=sum((bool(novel[i]) for i in take)),
                major_minimum=requirements[branch],
                unknown_prior=False,
            )
        )
    diag = dict(
        version=VERSION,
        nominal_budget=policy.budget,
        selected_paths=len(rows),
        eligible_unique_paths=capacity,
        quantity_goal_met=len(rows) >= policy.minimum_paths,
        mandatory_paths=mandatory_after_floors,
        expanded_for_safeguards=target > policy.budget,
        unmet_major_branch_floors=unmet,
        malformed_or_out_of_range_paths=len(malformed),
        selected_novel=sum((bool(novel[i]) for i in rows)),
        minimum_novel_paths=policy.minimum_novel_paths,
        novel_quantity_goal_met=int(novel[selected].sum()) >= policy.minimum_novel_paths,
        novel_floor_available=novel_floor_available,
        destination_equal_branch_mass=destination_equal_distribution(
            chosen, [branches[i] for i in rows]
        ),
        destination_equal_length_mass=destination_equal_distribution(
            chosen, [str(len(paths[i])) for i in rows]
        ),
        length_mean=float(np.mean([len(p) for p in chosen])) if chosen else None,
        predicted_branch_mass=root,
        predicted_length_mass=lengths,
        selected_branch_row_mass={
            b: counts["branch", b] / len(rows) if rows else 0.0 for b in root
        },
        empty_selected_branches=[b for b in root if not by_branch.get(b)],
        source_references_read=False,
        labels_read=False,
        routes_invented=False,
        distribution_control_unit="Path-row group penalties; destination-equal diagnostics reported separately",
        optimization="Lazy greedy marginal gains; no global optimality guarantee for overlapping groups",
    )
    return (selected, records, diag)
