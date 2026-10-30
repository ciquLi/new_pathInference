"""Unified255 next features computations."""
import math


FEATURE_NAMES = tuple(
    sorted(
        (
            "actual_next_raw_support_p",
            "actual_next_support_missing",
            "actual_next_rank_log1p",
            "actual_next_rank_inverse",
            "actual_next_rank_missing",
            "actual_next_uncalibrated",
            "actual_next_calibration_unknown",
            "actual_next_phase_conflict",
            "actual_next_phase_ambiguous",
        )
    )
)


def features(candidate, conditional_store):
    path = candidate["path"]
    result = (
        conditional_store.next(path[0], path[1], path[2]) if len(path) > 2 else dict(missing=True)
    )
    missing = bool(result.get("missing", True))
    p = None if missing else result.get("probability")
    rank = None if missing else result.get("rank")
    if p is not None and (not math.isfinite(float(p)) or not 0 <= p <= 1):
        raise ValueError("Bad raw local support")
    out = dict(
        actual_next_raw_support_p=float(p or 0.0),
        actual_next_support_missing=float(p is None),
        actual_next_rank_log1p=math.log1p(rank or 0),
        actual_next_rank_inverse=1 / rank if rank else 0.0,
        actual_next_rank_missing=float(rank is None),
        actual_next_uncalibrated=float(
            p is not None and result.get("probability_calibrated") is False
        ),
        actual_next_calibration_unknown=float(
            p is not None and "probability_calibrated" not in result
        ),
        actual_next_phase_conflict=float(result.get("phase_status") == "conflict"),
        actual_next_phase_ambiguous=float(result.get("phase_status") == "ambiguous"),
    )
    assert set(out) == set(FEATURE_NAMES)
    return out
