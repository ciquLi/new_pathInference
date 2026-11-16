"""Profile the verified C++ adapter without writing into live prediction caches."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import time

import native_engine as fast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asns', nargs='+', default=['19529', '328840', '151333'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tree = fast.enable(fast.NativeTree(1, .5, 'train', 0, 1))
    tree.valid_dataset = 'rib'
    tree.valid_files = set(tree.list_dataset('rib'))
    tree.load_models()
    fast.engine.original.TESTING_DIR = args.output/'predictions'
    print(json.dumps(dict(event='initialized', seconds=time.monotonic()-started)), flush=True)
    results = []
    profile = cProfile.Profile()
    for asn in args.asns:
        sink = fast.engine.Counts()
        started = time.monotonic()
        profile.runcall(tree.read_file, asn, sink)
        result = dict(asn=asn, seconds=time.monotonic()-started, counts=sink.value)
        results.append(result)
        print(json.dumps(result), flush=True)
        profile.dump_stats(str(args.output/'native.pstats'))
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
        (args.output/'PROFILE.txt').write_text(stream.getvalue())
        (args.output/'TIMINGS.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
