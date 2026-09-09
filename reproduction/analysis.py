"""Numerical analyses. Expected paper results are never read by this module."""
import json
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from acs import prepare_acs


def estimator(config, threads, small=False):
    return lgb.LGBMRegressor(**config['heterogeneity_parameters' if small else 'parameters'], n_jobs=threads)


def task_id(task):
    return f"{task['model']}__{task['stage']}__{task['target']}"


def base_frame(data, config, model, q2=True):
    sample = pl.read_parquet(data / 'analysis_samples.parquet').filter(pl.col('model_id') == model)
    if q2:
        sample = sample.filter(pl.col('q2_member'))
        if sample['q2_order'].null_count():
            raise ValueError('Missing frozen PLACES analysis order')
    sample = sample.with_columns(pl.col('q2_order' if q2 else 'source_order').alias('analysis_order'))
    emb = pl.read_parquet(data / f'embeddings_{model}_2022.parquet')
    area = pl.read_parquet(data / 'tract_area.parquet')
    frame = sample.join(emb, on='GEOID', how='left', validate='1:1').join(area, on='GEOID', how='left', validate='1:1').sort('analysis_order')
    if frame.height != sample.height:
        raise ValueError('Sample membership changed during feature join')
    features = config['models'][model]['features'] + ['ALAND', 'AWATER']
    if set(features) - set(frame.columns):
        raise ValueError('Required feature columns missing')
    return frame.with_columns(pl.col('GEOID').str.slice(0, 2).alias('state_fips')), features


def target_frame(frame, data, outcome):
    y = pl.read_parquet(data / 'places_outcomes.parquet').filter(pl.col('outcome') == outcome).select(
        pl.col('tract_fips').alias('GEOID'), 'observed_prevalence_pct')
    return frame.join(y, on='GEOID', how='left', validate='1:1').sort('analysis_order')


def valid_values(frame, column):
    return frame.filter(pl.col(column).is_not_null() & ~pl.col(column).is_nan())


def cross_validation(X, y, config, threads, groups=None, small=False, fold_labels=None):
    splitter = KFold(n_splits=2, shuffle=True, random_state=42) if small else GroupKFold(n_splits=5)
    scores = []
    if fold_labels is None:
        splits = splitter.split(X, y, groups)
    else:
        labels = np.asarray(fold_labels)
        if len(labels) != len(y) or set(labels) != {1, 2, 3, 4, 5}:
            raise ValueError('Invalid frozen cross-validation folds')
        splits = ((np.flatnonzero(labels != k), np.flatnonzero(labels == k)) for k in range(1, 6))
    for train, test in splits:
        fitted = estimator(config, threads, small)
        fitted.fit(X[train], y[train])
        scores.append(float(r2_score(y[test], fitted.predict(X[test]))))
    return float(np.mean(scores)), float(np.std(scores)), scores


def acs_fold_labels(data, model, target, ids):
    """Read validated state assignments; never regenerate version-sensitive ties."""
    manifest = json.loads((data / 'acs_folds.json').read_text())
    entries = [r for r in manifest['assignments'] if r['model'] == model and r['target'] == target]
    if len(entries) != 1:
        raise ValueError(f'Missing or duplicate frozen ACS folds for {model}/{target}; import the Sherlock fold export')
    entry = entries[0]
    digest = hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
    counts = dict(Counter(i[:2] for i in ids))
    if digest != entry['sorted_ids_sha256'] or counts != entry['state_counts'] or len(ids) != entry['n_tracts']:
        raise ValueError(f'ACS sample differs from frozen folds: {model}/{target}')
    mapping = entry['state_to_fold']
    if set(mapping) != set(counts) or set(mapping.values()) != {1, 2, 3, 4, 5}:
        raise ValueError('Incomplete frozen state assignments')
    return np.array([mapping[i[:2]] for i in ids], dtype=int)


def run_task(task, data, config, threads):
    model, stage, target = task['model'], task['stage'], task['target']
    label = config['models'][model]['label']
    frame, features = base_frame(data, config, model, q2=stage != 'acs')
    if stage == 'places':
        frame = valid_values(target_frame(frame, data, target), 'observed_prevalence_pct')
        train = ~frame['state_fips'].is_in(config['holdout_states']).to_numpy()
        X = frame.select(features).to_numpy()
        y = frame['observed_prevalence_pct'].to_numpy()
        m = estimator(config, threads)
        m.fit(X[train], y[train])
        direct = {'embedding_model': label, 'outcome': target, 'predictor_type': 'Embeddings',
                  'r2': float(r2_score(y[~train], m.predict(X[~train]))), 'n_train': int(train.sum()), 'n_test': int((~train).sum())}
        residual = pl.scan_parquet(data / f'residuals_{model}.parquet').filter(pl.col('outcome') == target).collect()
        rows = []
        for index in ['ReADI', 'SDI', 'SVI']:
            r = residual.filter(pl.col('social_risk_index') == index).rename({'tract_fips': 'GEOID'})
            r = frame.select('GEOID', 'analysis_order', *features).join(r, on='GEOID', how='inner', validate='1:1').sort('analysis_order')
            if r.height != frame.height:
                raise ValueError('Residual and direct analysis samples differ')
            tr = r['split'].eq('train').to_numpy()
            if not np.array_equal(train, tr):
                raise ValueError('Residual and direct holdouts differ')
            Xr = r.select(features).to_numpy(); yr = r['residual_prevalence_points'].to_numpy()
            fitted = estimator(config, threads); fitted.fit(Xr[tr], yr[tr])
            residual_r2 = float(r2_score(yr[~tr], fitted.predict(Xr[~tr])))
            stage1 = float(r2_score(r.filter(pl.col('split') == 'test')['observed_prevalence_pct'].to_numpy(),
                                   r.filter(pl.col('split') == 'test')['stage1_predicted_prevalence_pct'].to_numpy()))
            delta = max(0., residual_r2) * max(0., 1. - stage1)
            rows.append({'embedding_model': label, 'index': index, 'outcome': target, 'r2_index': stage1,
                         'r2_resid_explained_by_emb': residual_r2, 'delta_r2': delta,
                         'pct_unexplained_var': delta / (1. - stage1) if stage1 != 1 else None,
                         'r2_sequential_combined': stage1 + delta, 'n_train': int(tr.sum()), 'n_test': int((~tr).sum())})
        return {'places': [direct], 'residuals': rows}
    if stage == 'acs':
        acs = prepare_acs(data / 'acs_source.parquet').rename({'tract_fips': 'GEOID'})
        if target not in acs.columns:
            raise ValueError(f'ACS target unavailable: {target}')
        frame = valid_values(frame.join(acs.select('GEOID', target), on='GEOID', how='left', validate='1:1').sort('analysis_order'), target)
        labels = acs_fold_labels(data, model, target, frame['GEOID'].to_list())
        mean, std, scores = cross_validation(frame.select(features).to_numpy(), frame[target].to_numpy(), config, threads, fold_labels=labels)
        return {'acs': [{'embedding_model': label, 'variable': target, 'r2_mean': mean, 'r2_std': std, 'n_tracts': frame.height, 'fold_scores': scores}]}
    if stage in ('state', 'decile'):
        if stage == 'state':
            frame = frame.filter(pl.col('state_fips') == target)
        else:
            area = frame['ALAND'].to_numpy().astype(float)
            edges = np.nanpercentile(area, np.linspace(0, 100, 11)); edges[0] -= 1
            frame = frame.with_columns(pl.Series('_decile', np.searchsorted(edges[1:-1], area) + 1))
            frame = frame.filter(pl.col('_decile') == int(target))
        if frame.height < 50:
            raise ValueError('Insufficient tracts for requested heterogeneity task')
        scores = []
        outcomes = sorted(pl.read_parquet(data / 'places_outcomes.parquet', columns=['outcome'])['outcome'].unique())
        # Read the outcome table only once for these 40 fits.
        places = pl.read_parquet(data / 'places_outcomes.parquet').pivot(on='outcome', index='tract_fips', values='observed_prevalence_pct').rename({'tract_fips':'GEOID'})
        frame = frame.join(places, on='GEOID', how='left', validate='1:1').sort('analysis_order')
        for outcome in outcomes:
            valid = valid_values(frame, outcome)
            if valid.height < 50:
                continue
            mean, _, folds = cross_validation(valid.select(features).to_numpy(), valid[outcome].to_numpy(), config, threads, small=True)
            scores.append({'outcome': outcome, 'r2': mean, 'fold_scores': folds, 'n_tracts': valid.height})
        if not scores:
            raise ValueError('No eligible outcomes')
        row = {'embedding_model': label, 'r2_emb': float(np.nanmean([r['r2'] for r in scores])), 'n_outcomes': len(scores), 'n_tracts': frame.height}
        if stage == 'state': row['state_fips'] = target
        else: row.update(decile=int(target), median_tract_area_km2=float(np.median(frame['ALAND'].to_numpy()) / 1e6))
        return {stage: [row], 'fold_details': scores}
    raise ValueError(f'Unknown stage: {stage}')
