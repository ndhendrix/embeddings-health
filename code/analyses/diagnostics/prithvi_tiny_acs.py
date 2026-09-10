#!/usr/bin/env python3
"""Compare Tiny ACS preparation, state folds, and predictor precision on Sherlock."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

TARGETS = ['median_family_income', 'income_disparity_ratio']


def combine_states(directory, destination):
    """Preserve CSV values and within-file order; record sorted source order."""
    paths = sorted(directory.glob('*.csv'))
    if not paths:
        raise FileNotFoundError(f'No state CSVs in {directory}')
    sources = []
    header = None
    with destination.open('xb') as output:
        for path in paths:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                first = stream.readline()
                digest.update(first)
                if not first:
                    raise ValueError(f'Empty source CSV: {path}')
                if header is None:
                    header = first.rstrip(b'\r\n')
                    output.write(header + b'\n')
                elif first.rstrip(b'\r\n') != header:
                    raise ValueError(f'State CSV column order/header differs: {path}')
                last = b''
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    output.write(block)
                    last = block[-1:]
                if last and last != b'\n':
                    output.write(b'\n')
            sources.append({'path': str(path), 'sha256': digest.hexdigest()})
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduction', type=Path, required=True)
    parser.add_argument('--original-repo', type=Path, required=True)
    parser.add_argument('--original-data', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True, help='Combined CSV or directory of state CSVs')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output directory')
    sys.path.insert(0, str(args.reproduction.resolve()))
    import numpy as np
    import polars as pl
    from sklearn.model_selection import GroupKFold
    from analysis import base_frame, valid_values, acs_fold_labels, cross_validation
    from acs import prepare_acs
    from replication import verify_inputs, fingerprint, atomic_json
    from prithvi_precision import compare_arrays
    config = json.loads((args.reproduction/'config.json').read_text())
    lock = json.loads((args.reproduction/'data-lock.json').read_text())
    names = ['analysis_samples.parquet', 'tract_area.parquet', 'embeddings_prithvi_tiny_2022.parquet', 'acs_source.parquet', 'acs_folds.json']
    verify_inputs(args.data_dir, {'files': {n:lock['files'][n] for n in names}})
    args.output.mkdir(parents=True)
    fp, runtime = fingerprint(lock, config, args.threads)
    report = {'status': 'running', 'scope': 'Two Tiny ACS targets; not full validation',
              'runtime': runtime, 'fingerprint': fp, 'threads': args.threads,
              'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'targets': {}}
    atomic_json(args.output/'report.json', report)
    source = args.original_repo/'code/analyses/analyses_sherlock.py'
    raw = source.read_text()
    tree = ast.parse(raw)
    stats = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'STAT_SUFFIXES' for t in n.targets))
    if set(stats) != {'MEAN', 'MINIMUM', 'MAXIMUM', 'STD'}:
        raise ValueError(f'Unexpected original statistics: {stats}')
    prep = raw[raw.index('_meta     ='):raw.index('def lgbm_cv_r2(')]
    fit_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'lgbm_cv_r2')
    # Preserve the source definition in the diagnostic, without running its fits.
    report['original_fit_source'] = ast.get_source_segment(raw, fit_function)
    embedding_path = args.embeddings
    if embedding_path.is_dir():
        combined = args.output/'tiny_full_combined.csv'
        report['state_sources_in_order'] = combine_states(embedding_path, combined)
        report['combination_note'] = 'Sorted filenames; original within-file order preserved. This does not establish the historical combined-file order.'
        embedding_path = combined
        atomic_json(args.output/'report.json', report)
    paths = {'embeddings': embedding_path,
             'acs': args.original_repo/'data/acs.csv',
             'places': args.original_data/'PLACES__Local_Data_for_Better_Health__Census_Tract_Data__2025_release.csv',
             'readi': args.original_data/'social_risk_indices/ReADI_CT_2022.csv',
             'area': args.original_data/'alphaearth/alphaearth_embeddings.csv'}
    for path in paths.values():
        if not path.is_file(): raise FileNotFoundError(path)
    report['original_source_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
    report['original_preparation_sha256'] = hashlib.sha256(prep.encode()).hexdigest()
    report['original_inputs'] = {}
    for name, path in paths.items():
        with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        report['original_inputs'][name] = {'path':str(path), 'sha256':digest}
    ctx = {'pl':pl, 'np':np, 'ACS_CSV':paths['acs'], 'SRI_DIR':paths['readi'].parent,
           'EMBEDDINGS_PATH':paths['embeddings'], 'AREA_SOURCE':paths['area'],
           'PLACES_CSV':paths['places'], 'STAT_SUFFIXES':stats,
           '_exclude_fips':set(), '_exclude_abbrs':set(),
           'args':SimpleNamespace(model='prithvi_tiny', stage='acs')}
    exec(compile(prep, str(source), 'exec'), ctx)
    original = ctx['df']; original_features = ctx['FEATURE_COLS']
    base, features = base_frame(args.data_dir, config, 'prithvi_tiny', q2=False)
    acs = prepare_acs(args.data_dir/'acs_source.parquet').rename({'tract_fips':'GEOID'})
    report['features'] = features
    report['original_features'] = original_features
    for target in TARGETS:
        frame = valid_values(base.join(acs.select('GEOID',target), on='GEOID', how='left', validate='1:1').sort('analysis_order'), target)
        ids = frame['GEOID'].to_list()
        X = frame.select(features).to_numpy(); y = frame[target].to_numpy()
        frozen = acs_fold_labels(args.data_dir, 'prithvi_tiny', target, ids)
        groups = np.array([i[:2] for i in ids])
        current = np.zeros(len(ids), dtype=int)
        for k, (_, test) in enumerate(GroupKFold(n_splits=5).split(X,y,groups),1):current[test]=k
        orig = valid_values(original, target)
        orig_ids = orig['tract_fips'].to_list()
        item = {'n_tracts':len(ids), 'original_n_tracts':len(orig_ids),
                'same_membership':set(ids)==set(orig_ids), 'same_order':ids==orig_ids,
                'same_features':features==original_features,
                'frozen_vs_current_folds_equal':bool(np.array_equal(frozen,current)),
                'fold_states': {str(k):{'frozen':sorted(set(groups[frozen==k])), 'current':sorted(set(groups[current==k]))} for k in range(1,6)},
                'fits':{}}
        if item['same_membership'] and item['same_features']:
            aligned = frame.select('GEOID','analysis_order').join(orig.select('tract_fips',target,*features).rename({'tract_fips':'GEOID'}),on='GEOID',how='left',validate='1:1').sort('analysis_order')
            item['original_predictors'] = compare_arrays(X,aligned.select(features).to_numpy())
            item['original_targets'] = compare_arrays(y,aligned[target].to_numpy())
        report['targets'][target] = item
        atomic_json(args.output/'report.json', report)
        variants = [('release_float64_frozen',X,frozen), ('release_float32_frozen',X.astype(np.float32),frozen)]
        if not item['frozen_vs_current_folds_equal']:variants.append(('release_float64_current_folds',X,current))
        for label, matrix, folds in variants:
            mean, sd, scores = cross_validation(matrix,y,config,args.threads,fold_labels=folds)
            item['fits'][label] = {'r2_mean':mean,'r2_std':sd,'fold_scores':scores}
            print(target,label,mean,sd,flush=True)
            atomic_json(args.output/'report.json',report)
        # Current original preparation can differ from the released preparation.
        if not (item['same_order'] and item['same_features'] and item.get('original_predictors',{}).get('equal') and item.get('original_targets',{}).get('equal') and item['frozen_vs_current_folds_equal']):
            mean,sd,scores = cross_validation(orig.select(original_features).to_numpy().astype(float),orig[target].to_numpy().astype(float),config,args.threads,groups=np.array([i[:2] for i in orig_ids]))
            item['fits']['original_preparation_current_folds'] = {'r2_mean':mean,'r2_std':sd,'fold_scores':scores}
        else:
            item['original_preparation_fit_equivalent_to'] = 'release_float64_frozen'
        reference = pl.read_csv(args.reproduction/'reference/acs_prithvi_tiny.csv').filter(pl.col('variable')==target).row(0,named=True)
        item['paper_reference'] = {k:reference[k] for k in ['r2_mean','r2_std']}
        for fit in item['fits'].values():
            fit['matches_paper_3dp'] = all(round(fit[k],3)==round(reference[k],3) for k in ['r2_mean','r2_std'])
        pl.DataFrame({'tract_fips':ids,'target_value':y,'frozen_fold':frozen,'current_fold':current}).write_csv(args.output/f'{target}_tracts.csv')
        atomic_json(args.output/'report.json',report)
    report['status'] = 'comparison_complete'
    atomic_json(args.output/'report.json',report)
    print(f'Return {args.output / "report.json"}',flush=True)


if __name__ == '__main__':
    main()
