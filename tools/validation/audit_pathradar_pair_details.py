"""Independently check persisted PathRadar query rows against dated test inventory."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(file):
    value = hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def audit_rows(rows, expected, summary):
    seen = set()
    covered = exact = 0
    scores = []
    for row in rows:
        pair = row['src'], row['dst']
        if pair in seen:
            raise ValueError('Duplicate query in persisted CSV')
        seen.add(pair)
        valid = int(row['valid_prediction'])
        match = int(row['exact_match'])
        score = float(row['jaccard'])
        if valid not in (0, 1) or match not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid persisted flag or score')
        if valid:
            path = row['predicted_path'].split('->')
            if len(path) < 2 or (path[0], path[-1]) != pair:
                raise ValueError('Prediction endpoints differ from query')
        elif row['predicted_path'] or score or match:
            raise ValueError('Uncovered query has a prediction or nonzero score')
        covered += valid
        exact += match
        scores.append(score)
    if seen != expected or len(seen) != summary['requested_pairs']:
        raise ValueError('Persisted queries differ from the complete dated test inventory')
    mean = math.fsum(scores) / len(scores)
    if (covered != summary['coverage_count'] or exact != summary['exact_match_count']
            or not math.isclose(mean, summary['mean_jaccard_over_requested'], rel_tol=0, abs_tol=1e-12)):
        raise ValueError('CSV totals differ from evaluation summary')
    return dict(status='PASS', records=len(seen), test_queries=len(expected),
                valid_predictions=covered, invalid_predictions=len(seen)-covered,
                exact_matches=exact, jaccard_mean=mean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('date')
    args = parser.parse_args()
    run = ROOT / 'runs/baselines/pathradar' / args.date
    control = run / 'control/pair_details_backfill'
    completed = control / 'REEVALUATION_COMPLETE.json'
    if completed.exists():
        pointer = json.loads(completed.read_text())
        output = Path(pointer['detail_output'])
    else:
        output = run / 'testing/top1_eval_round2_full.json'
    receipt = json.loads(output.with_suffix('.pairs.audit.json').read_text())
    if receipt['status'] != 'PASS' or digest(output) != receipt['evaluation_sha256']:
        raise ValueError('Evaluation output differs from its validated identity')
    config = json.loads((ROOT / 'configs/dates' / (args.date + '.json')).read_text())
    tests = ROOT.parent / 'files/data/aspath/path_by_vantage' / args.date / 'path_without_prefix/test'
    files = sorted(tests.glob('*.csv'))
    if {file.stem for file in files} != set(config['test_sources']):
        raise ValueError('Dated test source inventory differs from registration')
    expected = set()
    for file in files:
        with file.open(encoding='utf-8', errors='ignore') as stream:
            for line in stream:
                path = line.strip().split('->')
                if len(path) >= 2:
                    expected.add((path[0], path[-1]))
    summary = json.loads(output.read_text())
    file = Path(receipt['details_file'])
    if digest(file) != receipt['details_sha256']:
        raise ValueError('Persisted CSV identity changed')
    with file.open(newline='') as stream:
        result = audit_rows(csv.DictReader(stream), expected, summary)
    result.update(csv_sha256=receipt['details_sha256'], evaluation_file=str(output),
                  details_file=str(file), historical_aggregate_preserved=True,
                  private_temporary_evidence_absent=not (control / 'evidence').exists())
    identity = json.loads((run / 'IDENTITY.json').read_text())
    if identity.get('cached_intermediate'):
        result['external_evidence_preserved'] = (ROOT.parent / 'pathRadar' /
            (args.date + '_devoid') / 'intermediate').is_dir()
        if not result['external_evidence_preserved']:
            raise ValueError('Original external evidence cache is missing')
    (control / 'INDEPENDENT_DETAILS_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
