"""Single entry point: verified downloads, local/Slurm task execution, reporting."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from importlib.metadata import version

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def verify_inputs(data, lock, record_id=None):
    data.mkdir(parents=True, exist_ok=True)
    for name, spec in lock['files'].items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Unsafe data-lock filename')
        path = data / name
        if not path.exists():
            if record_id is None:
                raise FileNotFoundError(f'Missing {name}. Put the Zenodo files in {data}, or supply --record-id.')
            if not re.fullmatch(r'[0-9]+', str(record_id)):
                raise ValueError('Use the numeric ID of the exact Zenodo record version')
            # Fetch only filenames pinned in the local code release, never arbitrary API filenames.
            url = f'https://zenodo.org/records/{record_id}/files/{urllib.parse.quote(name)}?download=1'
            temporary = path.with_suffix(path.suffix + '.partial')
            print(f'Downloading {name}', flush=True)
            try:
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
                        stream.write(block)
                if temporary.stat().st_size != spec['bytes'] or digest(temporary) != spec['sha256']:
                    raise ValueError(f'Download checksum mismatch: {name}')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}. Refusing to fit different data.')
        print(f'Verified {name}', flush=True)


def fingerprint(data_lock, config, threads):
    h = hashlib.sha256(json.dumps({'lock':data_lock,'config':config,'threads':threads},sort_keys=True).encode())
    for path in sorted(ROOT.glob('*.py')):
        h.update(path.name.encode()); h.update(path.read_bytes())
    runtime = {'python':platform.python_version(),'machine':platform.machine(),'system':platform.system(),
               **{n:version(n) for n in ['numpy','polars','scipy','lightgbm','scikit-learn']}}
    h.update(json.dumps(runtime,sort_keys=True).encode())
    return h.hexdigest(), runtime


def make_tasks(data, config, models=None, stages=None, smoke=False):
    import polars as pl
    models = models or list(config['models'])
    stages = stages or ['places','acs','state','decile']
    if smoke:
        return [{'model':'alphaearth','stage':s,'target':t} for s,t in
                [('places','ACCESS2'),('acs','median_family_income'),('state','01'),('decile','1')] if s in stages]
    outcomes = sorted(pl.read_parquet(data/'places_outcomes.parquet',columns=['outcome'])['outcome'].unique())
    sample = pl.read_parquet(data/'analysis_samples.parquet')
    result=[]
    for model in models:
        if model not in config['models']: raise ValueError(f'Unknown model {model}')
        for stage in stages:
            if stage=='places': targets=outcomes
            elif stage=='acs': targets=[c['variable'] for c in config['acs_concepts']]
            elif stage=='decile': targets=[str(i) for i in range(1,11)]
            elif stage=='state':
                counts=(sample.filter((pl.col('model_id')==model)&pl.col('q2_member'))
                        .group_by(pl.col('GEOID').str.slice(0,2).alias('state')).len().filter(pl.col('len')>=50))
                targets=sorted(counts['state'])
            else: raise ValueError(f'Unknown stage {stage}')
            result.extend({'model':model,'stage':stage,'target':t} for t in targets)
    return result


def execute(task, data, config, output, run_fingerprint, threads, resume=True):
    from analysis import run_task, task_id
    path=Path(output)/'tasks'/f'{task_id(task)}.json'
    if resume and path.exists():
        previous=json.loads(path.read_text())
        encoded=json.dumps(previous.get('result'),sort_keys=True,allow_nan=False).encode()
        if previous.get('fingerprint')==run_fingerprint and previous.get('result_sha256')==hashlib.sha256(encoded).hexdigest() and previous.get('task')==task:
            return {'id':task_id(task),'status':'reused'}
    if path.exists():
        # A failed fresh fit must not leave an old successful result available
        # to the collector. Preserve it outside the active task namespace.
        previous = path.parent / 'superseded'
        previous.mkdir(exist_ok=True)
        path.replace(previous / f'{time.time_ns()}-{path.name}')
    start=time.monotonic()
    result=run_task(task,Path(data),config,threads)
    encoded=json.dumps(result,sort_keys=True,allow_nan=False).encode()
    atomic_json(path,{'task':task,'fingerprint':run_fingerprint,'elapsed_seconds':time.monotonic()-start,
                      'result_sha256':hashlib.sha256(encoded).hexdigest(),'result':result})
    return {'id':task_id(task),'status':'fitted'}


def finish(plan, output):
    from reporting import assemble
    return assemble(plan, output, ROOT/'reference')


def main():
    p=argparse.ArgumentParser(description='Reproduce the analyses supported by the shared data.')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    p.add_argument('--record-id',help='Exact published Zenodo record version ID')
    p.add_argument('--models',nargs='+')
    p.add_argument('--stages',nargs='+',choices=['places','acs','state','decile'])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--smoke',action='store_true',help='Four AlphaEarth checks, not a full replication')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--fresh',action='store_true',help='Refit rather than reuse matching completed tasks')
    p.add_argument('--slurm',action='store_true')
    p.add_argument('--partition')
    p.add_argument('--account')
    p.add_argument('--array-limit',type=int,default=12)
    p.add_argument('--memory',default='32G')
    p.add_argument('--time-limit',default='12:00:00')
    p.add_argument('--worker-plan',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--task-index',type=int,help=argparse.SUPPRESS)
    p.add_argument('--collect-plan',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.collect_plan:
        plan=json.loads(args.collect_plan.read_text())
        return finish(plan,Path(plan['output_dir']))
    if args.worker_plan:
        plan=json.loads(args.worker_plan.read_text()); task=plan['tasks'][args.task_index]
        config=json.loads((ROOT/'config.json').read_text()); lock=json.loads((ROOT/'data-lock.json').read_text())
        current,_=fingerprint(lock,config,plan['threads'])
        if current!=plan['fingerprint']: raise ValueError('Worker environment/code differs from submitted plan')
        # The launcher verifies all files before submission. Each worker checks
        # the pinned files it reads, so later changes cannot silently pass.
        names=['analysis_samples.parquet','tract_area.parquet',f"embeddings_{task['model']}_2022.parquet"]
        names+=['acs_source.parquet','acs_folds.json'] if task['stage']=='acs' else ['places_outcomes.parquet']
        if task['stage']=='places': names += [f"residuals_{task['model']}.parquet"]
        verify_inputs(Path(plan['data_dir']),{'files':{n:lock['files'][n] for n in names}})
        print(execute(task,plan['data_dir'],config,plan['output_dir'],current,plan['threads'],not plan['fresh']))
        return 0
    if min(args.threads,args.workers,args.array_limit)<1: p.error('Thread, worker and array counts must be positive')
    config=json.loads((ROOT/'config.json').read_text()); lock=json.loads((ROOT/'data-lock.json').read_text())
    args.data_dir=args.data_dir.resolve(); args.output_dir=args.output_dir.resolve()
    verify_inputs(args.data_dir,lock,args.record_id or config.get('zenodo_record_id'))
    print('All required input hashes match this code release.',flush=True)
    if args.check_only: return 0
    tasks=make_tasks(args.data_dir,config,args.models,args.stages,args.smoke)
    run_fingerprint,runtime=fingerprint(lock,config,args.threads)
    plan={'tasks':tasks,'fingerprint':run_fingerprint,'runtime':runtime,'threads':args.threads,
          'data_dir':str(args.data_dir),'output_dir':str(args.output_dir),'fresh':args.fresh,
          'scope':'selected_checks' if args.smoke or args.models or args.stages else 'all_included_analyses',
          'excluded':config['excluded'],'config':config}
    # Separate plans are immutable while Slurm tasks run.
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plan_path=args.output_dir/f'plan-{time.time_ns()}.json'; atomic_json(plan_path,plan)
    print(f'{len(tasks)} analysis tasks; scope={plan["scope"]}',flush=True)
    if args.slurm:
        worker=args.output_dir/f'worker-{plan_path.stem}.sh'
        worker.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.quote(sys.executable)+' replication.py --worker-plan '+shlex.quote(str(plan_path))+' --task-index "$SLURM_ARRAY_TASK_ID"\n')
        command=['sbatch','--parsable',f'--array=0-{len(tasks)-1}%{args.array_limit}',f'--cpus-per-task={args.threads}',f'--mem={args.memory}',f'--time={args.time_limit}',f'--output={args.output_dir}/slurm-%A_%a.log']
        if args.partition: command+=['--partition',args.partition]
        if args.account: command+=['--account',args.account]
        job=subprocess.check_output(command+[str(worker)],text=True).strip().split(';')[0]
        if not job.isdigit(): raise ValueError(f'Unexpected sbatch job ID {job}')
        collector=args.output_dir/f'collect-{plan_path.stem}.sh'
        collector.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.quote(sys.executable)+' replication.py --collect-plan '+shlex.quote(str(plan_path))+'\n')
        collect=['sbatch','--parsable',f'--dependency=afterany:{job}','--mem=4G','--time=00:30:00',f'--output={args.output_dir}/validation-%j.log']
        if args.partition: collect+=['--partition',args.partition]
        if args.account: collect+=['--account',args.account]
        collect_job=subprocess.check_output(collect+[str(collector)],text=True).strip()
        print(f'Submitted array {job}; validation job {collect_job}. Submission is not a completed replication.')
        return 0
    failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute,t,args.data_dir,config,args.output_dir,run_fingerprint,args.threads,not args.fresh):t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            try: print(f.result(),flush=True)
            except Exception as error:
                failures.append({'task':futures[f],'error':str(error)}); print(f'FAILED: {futures[f]}: {error}',flush=True)
    atomic_json(args.output_dir/'failures.json',failures)
    return finish(plan,args.output_dir)


if __name__=='__main__':
    try: sys.exit(main())
    except Exception as error:
        print(f'ERROR: {error}',file=sys.stderr); sys.exit(1)
