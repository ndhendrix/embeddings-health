#!/usr/bin/env python3
"""Locate saved Tiny run evidence without executing historical code or models."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SECRET = re.compile(r'password|secret|token|credential|api[_-]?key|authorization', re.I)
SIGNAL = re.compile(r'tiny|full|pca|embedding|cache|Q1|ACS|python|numpy|scikit|sklearn|lightgbm|polars|gcc|module|version|threads|cpus|fold|state|median_family_income|income_disparity|error|traceback|warning|output|command|job.id', re.I)
SUFFIXES = {'.log', '.out', '.err', '.sbatch', '.sh', '.json'}
PRUNE = {'.git', '.venv', '__pycache__', 'site-packages', 'node_modules', 'prithvi_embeddings', 'prithvi_aggregated', 'prithvi_aggregated_full'}


def metadata(path):
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size,
            'modified_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def excerpt(path):
    record = metadata(path)
    # Bound every read. Modified times may reflect copying, not original execution.
    limit = 4_000_000
    with path.open('rb') as stream: data = stream.read(limit + 1)
    record['read_truncated'] = len(data) > limit
    if not record['read_truncated']: record['sha256'] = hashlib.sha256(data).hexdigest()
    lines = data[:limit].decode('utf-8', errors='replace').splitlines()
    matches = [{'line': i, 'text': line[:1500]} for i, line in enumerate(lines, 1)
               if SIGNAL.search(line) and not SECRET.search(line)]
    record['matching_lines'] = len(matches)
    record['excerpt_truncated'] = len(matches) > 250
    record['excerpt'] = matches[:250]
    return record


def candidates(roots):
    seen = set()
    for root in roots:
        if not root.is_dir(): continue
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [n for n in dirs if n not in PRUNE and not n.startswith('venv') and not (Path(current)/n).is_symlink()]
            for name in sorted(files):
                p = Path(current)/name
                if p.is_symlink() or p.suffix.lower() not in SUFFIXES: continue
                if not ('tiny' in str(p).lower() or name.startswith('slurm-')): continue
                resolved = str(p.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-repo', type=Path, required=True)
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output report')
    repo, scratch = args.original_repo, args.scratch_root
    roots = [repo/'code/analyses', repo/'logs', scratch/'logs', scratch/'outputs/prithvi_tiny_full', scratch/'outputs/prithvi_tiny']
    # Inspect Tiny-specific saved cache folders, without walking other model data.
    cache = scratch/'cache'
    if cache.is_dir(): roots.extend(p for p in cache.iterdir() if p.is_dir() and 'tiny' in p.name.lower() and not p.name.startswith('venv'))
    report = {'purpose': 'Historical Tiny evidence discovery; no fits performed',
              'status': 'inspection_complete', 'limitations': [
                  'Current scripts/environments may differ from those used for the historical run.',
                  'File modification times can reflect copying; they do not establish job dates.',
                  'Matching source excerpts are not complete logs; retain originals for follow-up.'],
              'searched_locations': [{'path': str(p), 'exists': p.is_dir()} for p in roots],
              'evidence': [], 'result_files': [], 'current_environment_metadata': [], 'errors': []}
    paths = sorted(candidates(roots), key=lambda p: p.stat().st_mtime, reverse=True)
    # Include scheduler logs saved directly in either project root.
    for directory in [repo, scratch]:
        if directory.is_dir():
            paths.extend(p for p in directory.glob('slurm-*.out') if not p.is_symlink())
    report['candidate_count'] = len(paths)
    report['inventory_truncated'] = len(paths) > 300
    for path in paths[:300]:
        try: report['evidence'].append(excerpt(path))
        except Exception as error: report['errors'].append({'path':str(path),'error':str(error)})
    for output_root in [repo/'outputs', scratch/'outputs']:
        if not output_root.is_dir(): continue
        for folder in output_root.glob('*tiny*'):
            for path in folder.glob('*acs*correlations*.csv'):
                try:
                    row = metadata(path)
                    with path.open(newline='') as stream:
                        row['targets'] = [r for r in csv.DictReader(stream) if r.get('variable') in {'median_family_income','income_disparity_ratio'}]
                    row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    report['result_files'].append(row)
                except Exception as error: report['errors'].append({'path':str(path),'error':str(error)})
    # Read distribution metadata as text. Do not execute cached interpreters.
    for env in [cache/'venv-3.11-analysis',cache/'venv-3.11-cpu']:
        for package in ['numpy','scikit_learn','lightgbm','polars','scipy']:
            for path in env.glob(f'lib/python*/site-packages/{package}-*.dist-info/METADATA'):
                with path.open() as stream:
                    version = next((line.strip() for line in stream if line.startswith('Version:')), None)
                report['current_environment_metadata'].append({'path':str(path),'version':version})
    try:
        result = subprocess.run(['git','log','-20','--format=%h %ad %s','--date=iso-strict','--','code/analyses/analyses_sherlock.py','code/analyses/slurm/*tiny*'],cwd=repo,capture_output=True,text=True,timeout=20)
        report['git_history'] = {'exit_code':result.returncode,'lines':[line for line in result.stdout.splitlines() if not SECRET.search(line)]}
    except Exception as error: report['git_history'] = {'error':str(error)}
    if not report['evidence'] or report['errors'] or report['inventory_truncated']:
        report['status'] = 'inspection_incomplete'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2);stream.write('\n')
    print(f"{report['status']}: {len(report['evidence'])} evidence files, {len(report['result_files'])} ACS result files")
    print(args.output)
    return 0 if report['status'] == 'inspection_complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
