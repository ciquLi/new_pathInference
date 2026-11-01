"""PATS admission when a legal candidate pool cannot reach its quantity target."""
from pathveil.experiment.contracts import quality_selection as strict_quality_selection
from pathveil.experiment.contracts import source_relative_scores


def quality_selection(paths, raw_scores, threshold, minimum_paths=None):
    if minimum_paths is None or len(paths) >= minimum_paths:
        return strict_quality_selection(paths, raw_scores, threshold, minimum_paths)
    # Validate the requested minimum through the existing contract. The normal
    # gate still rejects invalid scores and pools without any positive evidence.
    strict_quality_selection(paths, raw_scores, threshold, 1)
    if isinstance(minimum_paths, bool) or not isinstance(minimum_paths, int) or minimum_paths <= 0:
        raise ValueError("Minimum path count must be a positive integer")
    relative = source_relative_scores(paths, raw_scores)
    original_count = sum(value >= threshold for value in relative)
    effective = min(float(threshold), min(relative))
    return list(paths), dict(
        calibrated_threshold=float(threshold),
        effective_threshold=float(effective),
        requested_minimum=minimum_paths,
        original_quality_paths=int(original_count),
        completion_paths=len(paths) - int(original_count),
        cutoff_ties="retain all",
        final_quality_paths=len(paths),
        threshold_relaxed=effective < threshold,
        insufficient_legal_pool=True,
        insufficient_legal_pool_policy="retain_all",
        available_legal_paths=len(paths),
        minimum_shortfall=minimum_paths - len(paths),
    )
