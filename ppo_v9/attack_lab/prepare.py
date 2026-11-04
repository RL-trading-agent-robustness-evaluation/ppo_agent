"""Offline observation perturbation exercise. Only implement perturb_values()."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import traceback

HERE = Path(__file__).resolve().parent


def perturb_values(values, rng, config):
    perturbed=[]

    for val in values:
        noise=rng.gauss(0.0, sigma=config['sigma'])
        new_val=val+noise 
        perturbed.append(new_val)

    return perturbed 

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(rows, config, mode):
    columns = config['columns']
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Choose a nonempty, unique list of attacked feature columns')
    if any(c not in rows[0] for c in columns):
        raise ValueError('An attacked column is missing from the CSV')
    # This exercise supports additive perturbations of standardized features only.
    # Raw prices require a different contract (units, OHLC consistency, feature rebuilding).
    if config['representation'] != 'standardized_observation_features':
        raise ValueError('This scaffold expects standardized observation features')
    forbidden = {'date', 'observed_close', 'open_raw', 'close_raw',
                 'ma_fast_above_slow', 'trend_above_sma_200'}
    if set(columns) & forbidden:
        raise ValueError('This scaffold does not perturb identifiers, prices or gate flags')
    if mode == 'attack':
        if not config.get('spec_reference') or 'TODO' in config['spec_reference']:
            raise ValueError('Fill in the attack specification reference first')
        sigma = config.get('sigma')
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
            raise ValueError('sigma must be a finite, nonnegative number')
    original = copy.deepcopy(rows)
    output = copy.deepcopy(rows)
    rng = random.Random(config['noise_seed'])
    deltas = {c: [] for c in columns}
    for row_index, row in enumerate(rows):
        values = [float(row[c]) for c in columns]
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f'Nonfinite input at row {row_index}')
        given = list(values)
        changed = list(values) if mode == 'identity' else list(perturb_values(given, rng, config))
        if given != values:
            raise AssertionError('Your function mutated its input list')
        if len(changed) != len(columns) or not all(math.isfinite(x) for x in changed):
            raise ValueError(f'Wrong shape or nonfinite output at row {row_index}')
        for column, old, new in zip(columns, values, changed):
            if mode == 'attack' and config['sigma'] == 0 and new != old:
                raise AssertionError('sigma=0 must leave values unchanged')
            # Preserve exact CSV text when the numeric value is unchanged.
            output[row_index][column] = row[column] if new == old else repr(float(new))
            deltas[column].append(new - old)
    if rows != original:
        raise AssertionError('Clean data was mutated')
    for before, after in zip(original, output):
        if any(before[c] != after[c] for c in before if c not in columns):
            raise AssertionError('A protected column changed')
    stats = {c: {'mean_delta': statistics.mean(d), 'std_delta': statistics.pstdev(d),
                 'max_abs_delta': max(map(abs, d)), 'changed_values': sum(x != 0 for x in d)}
             for c, d in deltas.items()}
    return output, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--demo', action='store_true', help='Use 8 synthetic rows; no licensed data')
    parser.add_argument('--mode', choices=['identity', 'attack'], default='identity')
    parser.add_argument('--limit', type=int, help='Debug only: first N rows; recorded in manifest')
    parser.add_argument('--output', type=Path, required=True, help='A NEW output directory')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if not args.demo and (args.input is None or args.config is None):
        parser.error('provide --input and --config, or use --demo')
    config = (json.loads(args.config.read_text(encoding='utf-8')) if args.config else
              {'spec_reference': 'synthetic exercise only',
               'representation': 'standardized_observation_features',
               'columns': ['feature_a', 'feature_b'], 'sigma': 0.05, 'noise_seed': 123})
    if args.demo:
        rows = [{'date': f'2020-01-{i+1:02}', 'feature_a': str(i/10),
                 'feature_b': str(-i/5), 'protected_flag': str(i % 2)} for i in range(8)]
        source_sha = None
    else:
        source_sha = digest(args.input)
        with args.input.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('Empty input')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        def write_csv(name, data):
            with (args.output / name).open('x', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        write_csv('clean.csv', rows)
        output, stats = transform(rows, config, args.mode)
        # Replay checks that your function uses the supplied RNG reproducibly.
        replay, _ = transform(rows, config, args.mode)
        if replay != output:
            raise AssertionError('Same noise seed did not reproduce identical output')
        if args.input and digest(args.input) != source_sha:
            raise AssertionError('Source file changed during the run')
        write_csv('perturbed.csv', output)
        preview = [{'row': i, 'column': c, 'clean': rows[i][c], 'perturbed': output[i][c],
                    'delta': float(output[i][c])-float(rows[i][c])}
                   for i in range(min(5, len(rows))) for c in config['columns']]
        with (args.output / 'manifest.json').open('x', encoding='utf-8') as handle:
            json.dump({'status': 'PASS', 'mode': args.mode, 'synthetic': args.demo,
                       'rows': len(rows), 'debug_row_limit': args.limit, 'config': config,
                       'source': str(args.input.resolve()) if args.input else None,
                       'source_sha256': source_sha, 'script_sha256': digest(__file__),
                       'clean_sha256': digest(args.output/'clean.csv'),
                       'perturbed_sha256': digest(args.output/'perturbed.csv'),
                       'statistics': stats, 'preview': preview,
                       'formal_attack_spec_verified': False}, handle, indent=2, allow_nan=False)
        print(json.dumps({'rows': len(rows), 'statistics': stats, 'preview': preview}, indent=2))
        print(f'Artifacts: {args.output.resolve()}')
    except BaseException:
        with (args.output / 'FAILURE.txt').open('x', encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        raise


if __name__ == '__main__':
    main()
