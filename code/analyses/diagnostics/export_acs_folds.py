#!/usr/bin/env python3
"""Export state-to-fold assignments on the verified Sherlock environment; no fits."""
import argparse
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
    if len(confirmed) != 1 or confirmed[0]['state_to_fold'] != case['state_to_fold']:
        raise ValueError('This environment does not reproduce the confirmed Sherlock folds. Run this exporter in the same Sherlock environment as the successful diagnostic.')
    result = {'schema_version': 1, 'source': 'Sherlock GroupKFold export; no models fitted',
              'source_request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'numpy': np.__version__, 'scikit-learn': sklearn.__version__},
              'assignments': assignments}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Exported {len(assignments)} model/target fold assignments to {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', type=Path, default=Path(__file__).with_name('acs_fold_requests.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    export(args.requests, args.output)
