#!/usr/bin/env python3
"""Export state-to-fold assignments on the verified Sherlock environment; no fits."""
import argparse
import contextlib
import io
import os
import sys
import hashlib
import json
from pathlib import Path
import platform


def export(request_path, output):
    import numpy as np
    import sklearn
    from sklearn.model_selection import GroupKFold
    request_path, output = Path(request_path), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    diagnostic = output.with_suffix('.diagnostic.json')
    if diagnostic.exists():
        raise FileExistsError(f'Refusing to overwrite {diagnostic}')
    runtime = io.StringIO()
    with contextlib.redirect_stdout(runtime):
        if hasattr(np, 'show_runtime'):
            np.show_runtime()
    cpu = {}
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            key, sep, value = line.partition(':')
            if sep and key.strip() in {'model name', 'vendor_id', 'flags'}:
                cpu.setdefault(key.strip(), value.strip())
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                   'python_executable': sys.executable, 'cpu': cpu,
                   'numpy_runtime': runtime.getvalue(),
                   'job': {k: os.environ.get(k) for k in ['SLURM_JOB_ID', 'SLURMD_NODENAME']},
                   'cpu_dispatch_controls': {k: os.environ.get(k) for k in
                       ['NPY_DISABLE_CPU_FEATURES', 'NPY_ENABLE_CPU_FEATURES']}}
    print('Fold export environment: ' + json.dumps(environment), flush=True)
    request = json.loads(request_path.read_text())
    assignments = []
    for item in request['requests']:
        counts = item['state_counts']
        groups = np.concatenate([np.repeat(state, counts[state]) for state in sorted(counts)])
        if len(groups) != item['n_tracts']:
            raise ValueError('State counts do not sum to the requested sample size')
        mapping = {}
        for fold, (_, test) in enumerate(GroupKFold(n_splits=request['n_splits']).split(groups, groups=groups), 1):
            for state in np.unique(groups[test]):
                if str(state) in mapping:
                    raise ValueError('State appears in multiple test folds')
                mapping[str(state)] = fold
        assignments.append({**item, 'state_to_fold': mapping})
    case = request['confirmed_case']
    confirmed = [a for a in assignments if a['model'] == case['model'] and a['target'] == case['target']]
    observed = confirmed[0]['state_to_fold'] if len(confirmed) == 1 else {}
    differences = {state: {'expected': expected, 'observed': observed.get(state)}
                   for state, expected in case['state_to_fold'].items()
                   if observed.get(state) != expected}
    passed = len(confirmed) == 1 and observed == case['state_to_fold']
    result = {'schema_version': 1, 'source': 'Sherlock GroupKFold export; no models fitted',
              'source_request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
              'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'environment': environment,
              'validation': {'status': 'passed' if passed else 'failed',
                             'case': {'model': case['model'], 'target': case['target']},
                             'state_differences': differences},
              'assignments': assignments}
    if not passed:
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.write_text(json.dumps(result, indent=2) + '\n')
        raise ValueError(f'Environment does not reproduce the confirmed Sherlock folds. Candidate assignments and environment saved to {diagnostic}; this is not a release fold manifest. Differing states: {differences}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Exported {len(assignments)} model/target fold assignments to {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', type=Path, default=Path(__file__).with_name('acs_fold_requests.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    export(args.requests, args.output)
