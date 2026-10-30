"""Unified255 donor stats computations."""
from collections import Counter, defaultdict


def parse(raw, source):
    try:
        original = tuple((int(x) for x in raw.decode("ascii").strip().split("->")))
    except (ValueError, UnicodeDecodeError):
        return (None, False, "malformed")
    if not original or original[0] != source or any((not 0 < x < 2**32 for x in original)):
        return (None, False, "source_or_asn_invalid")
    route = tuple((x for (i, x) in enumerate(original) if i == 0 or x != original[i - 1]))
    if len(route) < 2 or len(route) != len(set(route)):
        return (None, False, "loop_or_short")
    return (route, len(original) == len(route), None)


def nd_stats(routes):
    result = defaultdict(Counter)
    direct = set()
    for p in routes:
        if len(p) == 2:
            direct.add(p[1])
        else:
            result[p[1], p[-1]][p[2]] += 1
    return (result, direct)
