"""Run the original evaluator with shared verified evidence and per-worker models."""
import functools
import gc
import json
import multiprocessing as mp
import sys
import time

import pipeline_engine as fast
import evaluate_top1_pairs as original

_TREE = None


def tree_factory(rounds, threshold, modes, quick_mode, workers):
    if _TREE is None or (_TREE.rounds, _TREE.threshold) != (rounds, threshold):
        raise ValueError('Evaluation tree identity mismatch')
    return _TREE


def initialize_worker():
    # Models load once per worker; immutable native evidence remains fork-shared.
    _TREE.load_models()
    gc.enable()


def cached_models(rounds):
    if _TREE is None or _TREE.rounds != rounds:
        raise ValueError('Evaluation model round mismatch')
    return _TREE.model


def main():
    global _TREE
    args = original.parse_args()
    started = time.monotonic()
    _TREE = fast.PipelineTree(args.rounds, args.threshold, 'test', 0, args.workers)
    fast.enable_sample(_TREE.ms, release=True)
    if _TREE.store:
        _TREE.store.close()
        _TREE.store = None
    original.Routingtree = tree_factory
    original.load_models = cached_models
    original.ProcessPoolExecutor = functools.partial(original.ProcessPoolExecutor,
        mp_context=mp.get_context('fork'), initializer=initialize_worker)
    print(json.dumps(dict(event='evaluation_initialized', seconds=time.monotonic()-started,
                          workers=args.workers, shared_evidence=True)), flush=True)
    gc.collect()
    gc.freeze()
    try:
        # Pair selection, tie breaks, scoring, accumulation and output schema are upstream.
        original.main()
    finally:
        gc.unfreeze()


if __name__ == '__main__':
    main()
