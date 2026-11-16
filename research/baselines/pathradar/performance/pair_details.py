"""Validate and persist the evaluator's existing per-query output schema."""
import csv
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate(result, expected_pairs=None):
    requested = result['requested_pairs']
    rows = result.get('pair_details', [])
    if (not result.get('include_pair_details') or len(rows) != requested
            or result['pairs_with_ground_truth'] != requested
            or result.get('pairs_missing_ground_truth', 0)):
        raise ValueError('Per-query details do not cover every requested query')
    pairs = set()
    covered = exact = 0
    scores = []
    for row in rows:
        pair = row['src'], row['dst']
        if pair in pairs:
            raise ValueError('Duplicate query: ' + str(pair))
        pairs.add(pair)
        path = row['predicted_path']
        valid = row['covered']
        score = row['jaccard']
        if type(valid) is not bool or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid prediction flag or Jaccard')
        if valid:
            if not isinstance(path, list) or len(path) < 2 or (path[0], path[-1]) != pair:
                raise ValueError('Prediction endpoints differ from query')
        elif path is not None or score != 0 or row['exact_match']:
            raise ValueError('Uncovered query must have a null path and zero score')
        covered += valid
        exact += bool(row['exact_match'])
        scores.append(score)
    if expected_pairs is not None and pairs != set(expected_pairs):
        raise ValueError('Detail query inventory differs from dated test queries')
    if covered != result['coverage_count'] or exact != result['exact_match_count']:
        raise ValueError('Detail coverage/exact totals differ from aggregate')
    if not requested or not math.isclose(math.fsum(scores) / requested,
            result['mean_jaccard_over_requested'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Detail Jaccard mean differs from aggregate')
    return dict(status='PASS', records=len(rows), requested_pairs=requested,
                unique_pairs=len(pairs), coverage_count=covered, exact_match_count=exact,
                test_inventory_checked=expected_pairs is not None)


def test_pairs(directory):
    pairs = set()
    for file in sorted(Path(directory).glob('*.csv')):
        if '_country' in file.name or '_continent' in file.name:
            continue
        with file.open(encoding='utf-8', errors='ignore') as stream:
            for line in stream:
                path = line.strip().split('->')
                if len(path) >= 2:
                    pairs.add((path[0], path[-1]))
    return pairs


def persist(output, result, expected_pairs):
    output = Path(output)
    check = validate(result, expected_pairs)
    destination = output.with_suffix('.pairs.csv')
    temporary = destination.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['src', 'dst', 'predicted_path', 'jaccard', 'valid_prediction', 'exact_match'])
        for row in result['pair_details']:
            writer.writerow([row['src'], row['dst'],
                '->'.join(row['predicted_path']) if row['covered'] else '',
                repr(row['jaccard']), int(row['covered']), int(row['exact_match'])])
    temporary.replace(destination)
    with destination.open(newline='', encoding='utf-8') as stream:
        stored = csv.DictReader(stream)
        count = covered = 0
        for row, original in zip_longest(stored, result['pair_details']):
            if row is None or original is None:
                raise ValueError('Persisted CSV count differs from requested queries')
            if (row['src'], row['dst']) != (original['src'], original['dst']):
                raise ValueError('Persisted query order differs from evaluated queries')
            count += 1
            covered += int(row['valid_prediction'])
        if count != check['records'] or covered != check['coverage_count']:
            raise ValueError('Persisted CSV count differs from requested queries')
    check.update(evaluation_file=str(output), evaluation_sha256=sha256(output),
                 details_file=str(destination), details_sha256=sha256(destination))
    receipt = output.with_suffix('.pairs.audit.json')
    temporary = receipt.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(check, indent=2) + '\n')
    temporary.replace(receipt)
    return check
