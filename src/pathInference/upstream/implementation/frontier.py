"""Unified255 frontier computations."""


PHASES = {"UP", "AFTER_PEER", "DOWN"}
RELATIONS = {"c2p", "p2p", "p2c", "s2s"}


def advance_phases(phases, relation):
    """Exact valley-free state set under a known or unresolved relationship."""
    if relation == "s2s":
        return set(phases)
    relations = [relation] if relation in RELATIONS else ["c2p", "p2p", "p2c"]
    output = set()
    for phase in phases:
        for rel in relations:
            if phase == "UP":
                output.add({"c2p": "UP", "p2p": "AFTER_PEER", "p2c": "DOWN"}[rel])
            elif phase in {"AFTER_PEER", "DOWN"} and rel == "p2c":
                output.add("DOWN")
    return output
