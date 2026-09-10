#!/usr/bin/env python3
"""Inspect saved Prithvi full-run inputs without fitting or changing results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re

SECRET = re.compile(r'password|secret|token|credential|api_key', re.I)
META = re.compile(r'manifest|config|metadata|plan|schema|info|validation|params|feature', re.I)


def summarize(value, key=''):
    if SECRET.search(key):
        return '<REDACTED>'
    if isinstance(value, dict):
        return {k: summarize(v, k) for k, v in value.items()}
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value) and ('feature' in key.lower() or 'column' in key.lower()):
            return value
        if len(value) > 50:
            return {'items': len(value), 'preview': [summarize(v) for v in value[:3]]}
        return [summarize(v) for v in value]
    return value


def inspect_file(path):
    result = {'path': str(path), 'bytes': path.stat().st_size}
    suffix = path.suffix.lower()
    try:
        if suffix == '.parquet':
            import polars as pl
            result['schema'] = {k: str(v) for k, v in pl.read_parquet_schema(path).items()}
        elif suffix == '.csv':
            with path.open(newline='') as stream:
                result['columns'] = next(csv.reader(stream))
        elif suffix == '.npy':
            import numpy as np
            # Never unpickle unknown saved arrays.
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            result.update(shape=list(array.shape), dtype=str(array.dtype))
        elif suffix == '.npz':
            import numpy as np
            with np.load(path, allow_pickle=False) as archive:
                result['array_names'] = archive.files
        elif suffix == '.json' and META.search(path.name) and path.stat().st_size < 2_000_000:
            result['metadata'] = summarize(json.loads(path.read_text()))
        elif suffix in {'.py', '.sh', '.sbatch'} and path.stat().st_size < 1_000_000:
            content = path.read_text()
            result['sha256'] = hashlib.sha256(content.encode()).hexdigest()
            # Relevant source lines, not full scripts or environment dumps.
            signal = re.compile(r'feature|EMB_COLS|FEATURE_COLS|LGBM|random_state|seed|n_estimators|num_leaves|learning_rate|subsample|colsample|n_jobs|sort|GroupKFold|KFold|train_test_split|pca|prepared', re.I)
            result['relevant_lines'] = [{'line': i, 'text': line} for i, line in enumerate(content.splitlines(), 1)
                                        if signal.search(line) and not SECRET.search(line)][:250]
    except Exception as error:
        result['inspection_error'] = f'{type(error).__name__}: {error}'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--original-code', type=Path, required=True)
    parser.add_argument('--reproduction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new report path')
    report = {'purpose': 'Full Prithvi-300M lineage inspection; no fits performed',
              'python': platform.python_version(), 'locations': {}, 'files': [], 'source_candidates': []}
    report['locations']['original_code'] = {'path': str(args.original_code.resolve()),
                                           'exists': args.original_code.is_dir()}
    for name, directory in [('prepared', args.prepared_dir), ('original_run', args.run_root)]:
        directory = directory.resolve()
        report['locations'][name] = {'path': str(directory), 'exists': directory.is_dir()}
        if directory.is_dir():
            paths = sorted(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink())
            report['locations'][name]['file_count'] = len(paths)
            report['locations'][name]['inventory_truncated'] = len(paths) > 2000
            report['files'].extend(inspect_file(p) for p in paths[:2000])
    # Locate likely preparation/sharded scripts in the original code checkout.
    if args.original_code.is_dir():
        for p in sorted(args.original_code.rglob('*')):
            if p.is_file() and not p.is_symlink() and p.suffix in {'.py', '.sh', '.sbatch'} and (re.search(r'prithvi|shard|prepare', p.name, re.I) or p.name == 'analyses_sherlock.py'):
                report['source_candidates'].append(inspect_file(p))
    config = json.loads((args.reproduction/'config.json').read_text())
    features = config['models']['prithvi_300m_tl']['features']
    report['reproduction_features'] = features
    report['reproduction_feature_count'] = len(features)
    # Compare declared feature lists when saved metadata contains them.
    def lists(value, location):
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, list) and v and all(isinstance(x, str) for x in v) and ('feature' in k.lower() or 'column' in k.lower()):
                    emb = [x for x in v if re.match(r'^PR\d+_(MEAN|MINIMUM|MAXIMUM|STD)$', x)]
                    if emb:
                        report.setdefault('feature_comparisons', []).append({'location': location+'.'+k,
                            'embedding_features': len(emb), 'same_membership': set(emb)==set(features),
                            'same_order': emb==features, 'first_features': emb[:12]})
                lists(v, location+'.'+k)
        elif isinstance(value, list):
            for i, v in enumerate(value): lists(v, f'{location}[{i}]')
    for item in report['files']:
        lists(item.get('metadata', {}), item['path'])
        if 'schema' in item or 'columns' in item:
            lists({'columns': item.get('columns', list(item.get('schema', {})))}, item['path'])
    report['inspection_status'] = 'complete' if all(v['exists'] for v in report['locations'].values()) and not any('inspection_error' in f for f in report['files'] + report['source_candidates']) and not any(v.get('inventory_truncated') for v in report['locations'].values()) else 'incomplete'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(f'Report: {args.output}')
    print(json.dumps(report['locations'], indent=2))
    print('Missing locations and inspection errors are recorded, not treated as a successful comparison.')
    return 0 if report['inspection_status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
