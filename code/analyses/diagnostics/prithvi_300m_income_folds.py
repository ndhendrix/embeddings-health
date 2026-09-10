#!/usr/bin/env python3
"""Test the tied-state fold swap for Prithvi-300M income disparity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def swapped_labels(labels, ids):
    import numpy as np
    result = np.array(labels, copy=True)
    states = np.array([value[:2] for value in ids])
    ma, tn = states == '25', states == '47'
    if not ma.any() or ma.sum() != tn.sum():
        raise ValueError('Expected equal nonzero Massachusetts/Tennessee counts')
    if set(result[ma]) != {1} or set(result[tn]) != {5}:
        raise ValueError('Expected baseline MA fold 1 and TN fold 5; refusing another change')
    result[ma], result[tn] = 5, 1
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reproduction', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.output.exists(): p.error('Choose a new output directory')
    sys.path.insert(0, str(args.reproduction.resolve()))
    import polars as pl
    from acs import prepare_acs
    from analysis import base_frame, valid_values, predictor_matrix, acs_fold_labels, cross_validation
    from replication import verify_inputs, fingerprint, atomic_json
    config = json.loads((args.reproduction/'config.json').read_text())
    lock = json.loads((args.reproduction/'data-lock.json').read_text())
    names = ['analysis_samples.parquet','tract_area.parquet','embeddings_prithvi_300m_tl_2022.parquet','acs_source.parquet','acs_folds.json']
    verify_inputs(args.data_dir, {'files': {n:lock['files'][n] for n in names}})
    model, target = 'prithvi_300m_tl', 'income_disparity_ratio'
    frame, features = base_frame(args.data_dir, config, model, q2=False)
    acs = prepare_acs(args.data_dir/'acs_source.parquet').rename({'tract_fips':'GEOID'})
    frame = valid_values(frame.join(acs.select('GEOID',target),on='GEOID',how='left',validate='1:1').sort('analysis_order'),target)
    ids = frame['GEOID'].to_list()
    original = acs_fold_labels(args.data_dir,model,target,ids)
    swapped = swapped_labels(original,ids)
    X = predictor_matrix(frame,features,model)
    if str(X.dtype) != 'float32': raise ValueError('Expected corrected Float32 model inputs; pull current code')
    y = frame[target].to_numpy()
    fp,runtime = fingerprint(lock,config,args.threads)
    report = {'status':'running','model':model,'target':target,'n_tracts':len(ids),
              'predictor_dtype':str(X.dtype),'threads':args.threads,'fingerprint':fp,'runtime':runtime,
              'diagnostic_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'ordered_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
              'sorted_ids_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
              'fits':{}, 'fold_maps':{}}
    args.output.mkdir(parents=True)
    atomic_json(args.output/'report.json',report)
    for name,labels in [('frozen',original),('ma_tn_swapped',swapped)]:
        mapping = {}
        for tract,fold in zip(ids,labels):
            state=tract[:2]
            if state in mapping and mapping[state]!=int(fold):raise ValueError('State split across folds')
            mapping[state]=int(fold)
        report['fold_maps'][name]=mapping
        mean,sd,scores=cross_validation(X,y,config,args.threads,fold_labels=labels)
        report['fits'][name]={'r2_mean':mean,'r2_std':sd,'fold_scores':scores}
        atomic_json(args.output/'report.json',report)
        print(name,mean,sd,flush=True)
    expected=pl.read_csv(args.reproduction/'reference/acs_prithvi_300m_tl.csv').filter(pl.col('variable')==target).row(0,named=True)
    report['paper_reference']={k:expected[k] for k in ['r2_mean','r2_std']}
    for fit in report['fits'].values():fit['matches_paper_3dp']=all(round(fit[k],3)==round(expected[k],3) for k in ['r2_mean','r2_std'])
    report['status']='comparison_complete'
    report['scope']='Single ACS target, paired fold comparison; no release assignments changed'
    atomic_json(args.output/'report.json',report)
    print(f'Return {args.output / "report.json"}',flush=True)


if __name__ == '__main__':main()
