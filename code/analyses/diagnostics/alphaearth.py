#!/usr/bin/env python3
"""Audit one real AlphaEarth ACS fit without changing paper outputs."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_hash(value):
    import numpy as np
    a = np.array(value, dtype='<f8', order='C', copy=True)
    a[np.isnan(a)] = np.nan  # Normalize NaN payloads across readers.
    return hashlib.sha256(a.tobytes()).hexdigest()


def compare(left, right):
    a, b = [json.loads((Path(p) / 'report.json').read_text()) for p in (left, right)]
    keys = ['source_sha256', 'preparation_sha256', 'fit_function_sha256',
            'input_sha256', 'features', 'n_tracts', 'state_counts',
            'ordered_ids_sha256', 'sorted_ids_sha256', 'ordered_x_sha256',
            'sorted_x_sha256', 'sorted_y_sha256', 'feature_sha256',
            'fold_assignment_sha256', 'folds', 'environment', 'r2_mean', 'r2_std']
    for key in keys:
        same = a[key] == b[key]
        print(f"{'MATCH' if same else 'DIFFER'}: {key}")
        if not same and key in ('r2_mean', 'r2_std', 'environment'):
            print(f'  left: {a[key]}\n  right: {b[key]}')
    return 0 if all(a[k] == b[k] for k in keys) else 2


def run(args):
    import numpy as np
    import polars as pl
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
    from threadpoolctl import threadpool_info

    root = Path(__file__).resolve().parents[3]
    source = root / 'code/analyses/analyses_sherlock.py'
    raw = source.read_text()
    tree = ast.parse(raw)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'lgbm_cv_r2')
    fit_source = ast.get_source_segment(raw, function)
    start = raw.index('_meta     =')
    prep = raw[start:raw.index('def lgbm_cv_r2(')]
    stats = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'STAT_SUFFIXES' for t in n.targets))
    if set(stats) != {'MEAN', 'MINIMUM', 'MAXIMUM', 'STD'}:
        raise ValueError(f'Unexpected aggregation statistics: {stats}')
    data = args.data_root.resolve()
    paths = {'embeddings': args.embeddings.resolve() if args.embeddings else data / 'alphaearth/alphaearth_embeddings.csv',
             'acs': args.acs.resolve() if args.acs else root / 'data/acs.csv',
             'places': data / 'PLACES__Local_Data_for_Better_Health__Census_Tract_Data__2025_release.csv',
             'readi': data / 'social_risk_indices/ReADI_CT_2022.csv'}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Refuse to overwrite an earlier audit.
    ctx = {'pl': pl, 'np': np, 'ACS_CSV': paths['acs'], 'SRI_DIR': paths['readi'].parent,
           'EMBEDDINGS_PATH': paths['embeddings'], 'AREA_SOURCE': paths['embeddings'],
           'PLACES_CSV': paths['places'], 'STAT_SUFFIXES': stats,
           '_exclude_fips': {'02', '15'}, '_exclude_abbrs': {'AK', 'HI'},
           'args': SimpleNamespace(model='alphaearth', stage='all')}
    exec(compile(prep, str(source), 'exec'), ctx)
    # Preserve the original runner's pandas conversion, row order, and missingness rule.
    frame = ctx['df'].to_pandas()
    features = ctx['FEATURE_COLS']
    mask = frame['median_family_income'].notna().to_numpy()
    X = frame[features].to_numpy(dtype=float)[mask]
    y = frame.loc[mask, 'median_family_income'].to_numpy(dtype=float)
    ids = frame.loc[mask, 'tract_fips'].to_numpy()
    states = frame.loc[mask, 'tract_fips'].str[:2].to_numpy()
    if len(set(ids)) != len(ids) or {'02', '15'} & set(states):
        raise ValueError('Duplicate tracts or excluded states in analysis sample')
    if any(c.endswith('_MEDIAN') for c in features):
        raise ValueError('Median embedding features present')
    folds, models = [], []

    class RecordedGroupKFold(GroupKFold):
        def split(self, *values, **kwargs):
            for tr, va in super().split(*values, **kwargs):
                folds.append((tr, va))
                yield tr, va

    def recorded_model(**kwargs):
        model = lgb.LGBMRegressor(**kwargs)
        models.append(model)
        return model

    fit_ctx = {'np': np, 'GroupKFold': RecordedGroupKFold,
               'lgb': SimpleNamespace(LGBMRegressor=recorded_model), 'r2_score': r2_score}
    exec(compile(fit_source, str(source), 'exec'), fit_ctx)
    mean, sd = fit_ctx['lgbm_cv_r2'](X, y, states)
    assignment = np.full(len(ids), -1, dtype=int)
    predictions = np.full(len(ids), np.nan)
    fold_reports = []
    for k, ((tr, va), model) in enumerate(zip(folds, models), 1):
        assignment[va] = k
        predictions[va] = model.predict(X[va])
        fold_reports.append({'fold': k, 'n_train': len(tr), 'n_test': len(va),
                             'test_states': sorted(set(states[va])),
                             'r2': float(r2_score(y[va], predictions[va])),
                             'model_sha256': hashlib.sha256(model.booster_.model_to_string().encode()).hexdigest(),
                             'parameters': model.booster_.params})
    if np.any(assignment < 1):
        raise ValueError('Incomplete held-out predictions')
    tracts = pl.DataFrame({'row': np.arange(len(ids)), 'tract_fips': ids.tolist(),
                           'state_fips': states.tolist(), 'test_fold': assignment,
                           'observed_median_family_income': y, 'prediction': predictions})
    tracts.write_csv(out / 'tracts.csv')
    order = np.argsort(ids)
    encode_ids = lambda values: hashlib.sha256('\n'.join(values).encode()).hexdigest()
    sorted_folds = '\n'.join(f'{ids[i]},{assignment[i]}' for i in order)
    report = {'schema_version': 1, 'target': 'median_family_income', 'excluded_states': ['AK', 'HI'],
              'statistics': stats, 'source_sha256': file_hash(source),
              'diagnostic_sha256': file_hash(__file__),
              'preparation_sha256': hashlib.sha256(prep.encode()).hexdigest(),
              'fit_function_sha256': hashlib.sha256(fit_source.encode()).hexdigest(),
              'input_sha256': {k: file_hash(p) for k, p in paths.items()},
              'features': features, 'n_tracts': len(ids),
              'state_counts': {s: int((states == s).sum()) for s in sorted(set(states))},
              'ordered_ids_sha256': encode_ids(ids), 'sorted_ids_sha256': encode_ids(ids[order]),
              'ordered_x_sha256': array_hash(X), 'sorted_x_sha256': array_hash(X[order]),
              'sorted_y_sha256': array_hash(y[order]),
              'feature_sha256': {name: array_hash(X[order, j]) for j, name in enumerate(features)},
              'fold_assignment_sha256': hashlib.sha256(sorted_folds.encode()).hexdigest(),
              'folds': fold_reports, 'r2_mean': mean, 'r2_std': sd,
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'packages': {name: importlib.metadata.version(name) for name in
                                           ['numpy', 'polars', 'pandas', 'scikit-learn', 'lightgbm', 'scipy']},
                              'thread_variables': {k: os.environ.get(k) for k in
                                                   ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
                              'threadpools': threadpool_info()},
              'git_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'R2 mean={mean:.12f}, SD={sd:.12f}\nDiagnostic saved to {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', nargs=2, metavar=('LOCAL_REPORT_DIR', 'SHERLOCK_REPORT_DIR'))
    parser.add_argument('--data-root', type=Path, help='Directory containing the original Sherlock input layout')
    parser.add_argument('--embeddings', type=Path, help='Optional override for the AlphaEarth CSV')
    parser.add_argument('--acs', type=Path, help='Optional override for the ACS CSV')
    parser.add_argument('--output', type=Path, help='New directory; existing directories are refused')
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.data_root is None or args.output is None:
        parser.error('--data-root and --output are required for a fit')
    run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
