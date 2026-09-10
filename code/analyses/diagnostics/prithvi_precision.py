#!/usr/bin/env python3
"""Paired ACCESS2 fits: change predictor precision only; preserve release inputs."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


def compare_arrays(a, b):
    import numpy as np
    if a.shape != b.shape:
        return {'same_shape': False, 'equal': False}
    equal = (a == b) | (np.isnan(a) & np.isnan(b))
    return {'same_shape': True, 'equal': bool(equal.all()), 'different_values': int((~equal).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reproduction', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.output.exists(): p.error('Choose a new output directory')
    sys.path.insert(0, str(args.reproduction.resolve()))
    import numpy as np
    import polars as pl
    from sklearn.metrics import r2_score
    from analysis import base_frame, target_frame, valid_values, estimator
    from replication import verify_inputs, fingerprint, atomic_json
    config = json.loads((args.reproduction/'config.json').read_text())
    lock = json.loads((args.reproduction/'data-lock.json').read_text())
    required = ['analysis_samples.parquet', 'tract_area.parquet', 'embeddings_prithvi_300m_tl_2022.parquet', 'places_outcomes.parquet']
    verify_inputs(args.data_dir, {'files': {k: lock['files'][k] for k in required}})
    args.output.mkdir(parents=True)
    fp, runtime = fingerprint(lock, config, args.threads)
    frame, features = base_frame(args.data_dir, config, 'prithvi_300m_tl')
    frame = valid_values(target_frame(frame, args.data_dir, 'ACCESS2'), 'observed_prevalence_pct')
    ids = frame['GEOID'].to_list()
    y = frame['observed_prevalence_pct'].to_numpy()
    train = ~frame['state_fips'].is_in(config['holdout_states']).to_numpy()
    X64 = frame.select(features).to_numpy().astype(np.float64, copy=False)
    del frame
    gc.collect()
    report = {'status': 'running', 'runtime': runtime, 'fingerprint': fp,
              'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'outcome': 'ACCESS2', 'model': 'prithvi_300m_tl', 'threads': args.threads,
              'parameters': config['parameters'], 'features': features,
              'n_train': int(train.sum()), 'n_test': int((~train).sum()),
              'ordered_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(), 'fits': {}}
    atomic_json(args.output/'report.json', report)
    # Compare the original prepared data without executing its source code.
    saved = args.prepared_dir/'analysis_data.parquet'
    meta = json.loads((args.prepared_dir/'metadata.json').read_text())
    original_ids = (pl.scan_parquet(saved).filter(pl.col('index_eligible'))
                    .filter(pl.col('ACCESS2').is_not_null() & ~pl.col('ACCESS2').is_nan())
                    .sort('index_order').select('tract_fips', 'ACCESS2').collect())
    report['original_sample'] = {
        'same_membership': set(original_ids['tract_fips'].to_list()) == set(ids),
        'same_order': original_ids['tract_fips'].to_list() == ids,
        'same_holdout_states': set(meta['shared_holdout_states']) == set(config['holdout_states']),
        'same_features': meta['feature_columns'] == features,
        'declared_feature_dtype': meta.get('feature_dtype')}
    alignment = pl.DataFrame({'tract_fips': ids}).with_row_index('diagnostic_order')
    different = 0
    for start in range(0, len(features), 128):
        cols = features[start:start+128]
        block = pl.read_parquet(saved, columns=['tract_fips', *cols])
        block = alignment.join(block, on='tract_fips', how='left', validate='1:1').sort('diagnostic_order')
        comparison = compare_arrays(X64[:, start:start+128].astype(np.float32), block.select(cols).to_numpy())
        different += comparison.get('different_values', 0)
    report['original_predictors_vs_cast_float32'] = {'equal': different == 0, 'different_values': different}
    aligned_y = alignment.join(original_ids, on='tract_fips', how='left', validate='1:1').sort('diagnostic_order')['ACCESS2'].to_numpy()
    report['original_target'] = compare_arrays(y, aligned_y)
    atomic_json(args.output/'report.json', report)
    del block, alignment, original_ids
    gc.collect()
    predictions = {}
    for dtype in ['float64', 'float32']:
        X = X64 if dtype == 'float64' else X64.astype(np.float32)
        start = time.monotonic()
        model = estimator(config, args.threads)
        model.fit(X[train], y[train])
        pred = model.predict(X[~train])
        predictions[dtype] = pred
        report['fits'][dtype] = {'r2': float(r2_score(y[~train], pred)), 'elapsed_seconds': time.monotonic()-start}
        print(dtype, report['fits'][dtype], flush=True)
        atomic_json(args.output/'report.json', report)
        del model, X
        gc.collect()
    # Read comparison targets only after both independent fits are complete.
    reference = pl.read_csv(args.reproduction/'reference/table_s1_figure2_r2_by_outcome.csv').filter(
        (pl.col('embedding_model') == config['models']['prithvi_300m_tl']['label']) &
        (pl.col('outcome') == 'ACCESS2') & (pl.col('predictor_type') == 'Embeddings'))
    expected = reference['r2'].item()
    report['paper_r2'] = expected
    for fit in report['fits'].values(): fit['matches_paper_4dp'] = round(fit['r2'], 4) == round(expected, 4)
    report['status'] = 'comparison_complete'
    report['scope'] = 'One direct PLACES fit per precision; not full model validation'
    atomic_json(args.output/'report.json', report)
    print(json.dumps(report['fits'], indent=2), flush=True)
    print(f'Return {args.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
