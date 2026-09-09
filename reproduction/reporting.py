"""Assemble only computed results; use paper tables strictly for validation."""
import hashlib
import json
from pathlib import Path
import math
import numpy as np
import polars as pl


def rounded_equal(actual, expected, digits):
    if actual is None or expected is None: return actual is expected
    return math.isfinite(float(actual)) and math.isfinite(float(expected)) and round(float(actual),digits)==round(float(expected),digits)


def paper_residual_rows(rows):
    """Mirror tables.R's sequential transmute without altering raw fit results."""
    result = []
    for original in rows:
        row = dict(original)
        index = round(row['r2_index'], 4)
        increment = row['delta_r2']
        row.update(r2_index=index,
                   r2_resid_explained_by_emb=round(row['r2_resid_explained_by_emb'], 4),
                   delta_r2=round(increment, 4),
                   pct_unexplained_var=round(increment / (1 - index), 4) if index != 1 else None,
                   r2_sequential_combined=round(index + increment, 4))
        result.append(row)
    return result


def check_rows(actual, expected, keys, metrics, name):
    checks=[]
    lookup={tuple(r[k] for k in keys):r for r in expected}
    if len(lookup)!=len(expected): raise ValueError(f'Duplicate reference keys in {name}')
    seen=set()
    for row in actual:
        key=tuple(row[k] for k in keys)
        if key in seen: raise ValueError(f'Duplicate computed keys in {name}')
        seen.add(key); ref=lookup.get(key)
        if ref is None:
            checks.append({'table':name,'key':list(key),'metric':'row','status':'unexpected','actual':None,'expected':None}); continue
        for metric,digits in metrics.items():
            a,b=row.get(metric),ref.get(metric)
            ok=(a==b) if digits is None else rounded_equal(a,b,digits)
            checks.append({'table':name,'key':list(key),'metric':metric,'status':'match' if ok else 'mismatch','actual':a,'expected':b})
    return checks


def write_table(rows,path,labels=None):
    if not rows: return
    df=pl.DataFrame(rows)
    if labels and 'outcome' in df.columns:
        df=df.with_columns(pl.col('outcome').replace_strict(labels).alias('outcome_label'))
    for c,dtype in df.schema.items():
        if dtype==pl.Float64: df=df.with_columns(pl.col(c).round(4))
    df.write_csv(path)


def figures(tables, output, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors=['#0072B2','#56B4E9','#CC79A7','#E69F00','#D55E00','#009E73']
    palette={m['label']:color for m,color in zip(config['models'].values(),colors)}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    destination=output/'figures'; destination.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(destination/f'{name}.png',dpi=200,bbox_inches='tight')
        fig.savefig(destination/f'{name}.pdf',bbox_inches='tight'); plt.close(fig)
    if tables['places']:
        rows=tables['places']; outcomes=sorted({r['outcome'] for r in rows})
        fig,ax=plt.subplots(figsize=(9,max(3,len(outcomes)*.26)))
        for label,color in palette.items():
            subset=[r for r in rows if r['embedding_model']==label]
            if subset: ax.scatter([r['r2'] for r in subset],[outcomes.index(r['outcome']) for r in subset],label=label,color=color,s=25)
        ax.set_yticks(range(len(outcomes)),[config['outcome_labels'][o] for o in outcomes]); ax.invert_yaxis()
        ax.axvline(0,color='grey',lw=.7); ax.set_xlabel('Held-out-state R²'); ax.set_title('PLACES outcomes: embeddings only')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False); save(fig,'figure2_embeddings_only')
    if tables['residuals']:
        rows=tables['residuals']; outcomes=sorted({r['outcome'] for r in rows})
        fig,axes=plt.subplots(1,3,figsize=(14,max(3,len(outcomes)*.26)),sharey=True)
        for ax,index in zip(axes,['ReADI','SDI','SVI']):
            for label,color in palette.items():
                subset=[r for r in rows if r['index']==index and r['embedding_model']==label]
                if subset: ax.scatter([r['delta_r2'] for r in subset],[outcomes.index(r['outcome']) for r in subset],color=color,s=22)
            ax.set_title(f'Beyond {index}'); ax.set_xlabel('Additional variance explained (ΔR²)'); ax.axvline(0,color='grey',lw=.7)
        axes[0].set_yticks(range(len(outcomes)),[config['outcome_labels'][o] for o in outcomes]); axes[0].invert_yaxis()
        fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l) for l,c in palette.items() if any(r['embedding_model']==l for r in rows)],loc='upper center',bbox_to_anchor=(.5,1.05),ncol=3,frameon=False)
        save(fig,'figure3_residuals')
    if tables['decile']:
        fig,ax=plt.subplots(figsize=(9,5))
        for label,color in palette.items():
            rows=sorted([r for r in tables['decile'] if r['embedding_model']==label],key=lambda r:r['decile'])
            if rows: ax.plot([r['decile'] for r in rows],[r['r2_emb'] for r in rows],marker='o',label=label,color=color)
        ax.set(xlabel='Tract land-area decile',ylabel='Mean cross-validated R² across PLACES outcomes',title='Performance by tract size: embeddings only')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False); save(fig,'figure4_embeddings_only')
    if tables['acs']:
        rows=tables['acs']; variables=[c['variable'] for c in config['acs_concepts'] if any(r['variable']==c['variable'] for r in rows)]
        descriptions={c['variable']:c['concept'] for c in config['acs_concepts']}
        fig,ax=plt.subplots(figsize=(10,max(3,len(variables)*.38)))
        for label,color in palette.items():
            subset=[r for r in rows if r['embedding_model']==label]
            if subset: ax.errorbar([r['r2_mean'] for r in subset],[variables.index(r['variable']) for r in subset],xerr=[r['r2_std'] for r in subset],fmt='o',color=color,label=label,ms=4,alpha=.8)
        ax.set_yticks(range(len(variables)),[descriptions[v] for v in variables]); ax.invert_yaxis()
        ax.set(xlabel='Mean state-grouped cross-validation R² (bars: fold SD)',title='Available ACS targets')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False); save(fig,'acs_included')
    if tables['state']:
        rows=tables['state']; states=sorted({r['state'] for r in rows})
        fig,ax=plt.subplots(figsize=(10,max(3,len(states)*.22)))
        for label,color in palette.items():
            subset=[r for r in rows if r['embedding_model']==label]
            if subset: ax.scatter([r['r2_emb'] for r in subset],[states.index(r['state']) for r in subset],color=color,label=label,s=18)
        ax.set_yticks(range(len(states)),states); ax.invert_yaxis(); ax.set(xlabel='Mean cross-validated R² across PLACES outcomes',title='Performance by state: embeddings only')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False); save(fig,'state_embeddings_only')


def assemble(plan,output,reference):
    output=Path(output); config=plan['config']
    tables={k:[] for k in ['places','residuals','acs','state','decile']}; missing=[]
    for task in plan['tasks']:
        from analysis import task_id
        path=output/'tasks'/f'{task_id(task)}.json'
        if not path.exists(): missing.append(task_id(task)); continue
        obj=json.loads(path.read_text())
        payload_hash=hashlib.sha256(json.dumps(obj['result'],sort_keys=True,allow_nan=False).encode()).hexdigest()
        if obj['fingerprint']!=plan['fingerprint'] or obj.get('result_sha256')!=payload_hash or obj['task']!=task:
            missing.append(task_id(task)); continue
        stage = task['stage']
        expected_counts = {'places': {'places': 1, 'residuals': 3},
                           'acs': {'acs': 1}, 'state': {'state': 1}, 'decile': {'decile': 1}}[stage]
        if any(len(obj['result'].get(kind, [])) != count for kind, count in expected_counts.items()):
            missing.append(task_id(task)); continue
        for key in tables: tables[key].extend(obj['result'].get(key,[]))
    for row in tables['state']: row['state']=config['states'][row.pop('state_fips')]
    raw_residuals = tables['residuals']
    tables['residuals'] = paper_residual_rows(raw_residuals)
    target=output/'tables'; target.mkdir(exist_ok=True)
    file_map={'places':'table_s1_embeddings_only.csv','residuals':'table_s2_figure3_incremental_r2.csv',
              'state':'table_s4_embeddings_only.csv','decile':'table_s3_embeddings_only.csv'}
    for key,name in file_map.items(): write_table(tables[key],target/name,config['outcome_labels'])
    if tables['acs']:
        write_table([{k:v for k,v in row.items() if k!='fold_scores'} for row in tables['acs']],target/'acs_included_long.csv')
    checks=[]
    specs=[('places','table_s1_figure2_r2_by_outcome.csv',['embedding_model','outcome','predictor_type'],{'r2':4,'n_train':None,'n_test':None}),
           ('residuals','table_s2_figure3_incremental_r2.csv',['embedding_model','index','outcome'],{'r2_index':4,'r2_resid_explained_by_emb':4,'delta_r2':4,'pct_unexplained_var':4,'r2_sequential_combined':4,'n_train':None,'n_test':None}),
           ('state','table_s4_state_level_r2.csv',['embedding_model','state'],{'r2_emb':4,'n_outcomes':None,'n_tracts':None}),
           ('decile','table_s3_figure4_tract_size_decile.csv',['embedding_model','decile'],{'r2_emb':4,'median_tract_area_km2':4,'n_tracts':None})]
    for kind,name,keys,metrics in specs:
        checks+=check_rows(tables[kind],pl.read_csv(reference/name).to_dicts(),keys,metrics,name)
    for model,spec in config['models'].items():
        rows=[r for r in tables['acs'] if r['embedding_model']==spec['label']]
        expected=pl.read_csv(reference/f'acs_{model}.csv').to_dicts()
        checks+=check_rows(rows,expected,['variable'],{'r2_mean':3,'r2_std':3},f'ACS:{model}')
    # S5 uses the already rounded S2 entries in the original tables.R.
    summary=[]
    for label in [m['label'] for m in config['models'].values()]:
        for index in ['ReADI','SDI','SVI']:
            rows=[r for r in tables['residuals'] if r['embedding_model']==label and r['index']==index]
            if len(rows)==40:
                summary.append({'embedding_model':label,'index':index,'mean_r2_index':float(np.mean([round(r['r2_index'],4) for r in rows])),
                                'mean_delta_r2':float(np.mean([round(r['delta_r2'],4) for r in rows])),
                                'mean_r2_sequential_combined':float(np.mean([round(r['r2_sequential_combined'],4) for r in rows])),'n_outcomes':40})
    write_table(summary,target/'table_s5_sequential_r2_summary.csv')
    checks+=check_rows(summary,pl.read_csv(reference/'table_s5_sequential_r2_summary.csv').to_dicts(),['embedding_model','index'],{'mean_r2_index':4,'mean_delta_r2':4,'mean_r2_sequential_combined':4,'n_outcomes':None},'table_s5')
    # Preserve semantic labels and formatting, while computing all numeric cells.
    wide=[]
    for concept in config['acs_concepts']:
        rows=[r for r in tables['acs'] if r['variable']==concept['variable']]
        if not rows: continue
        row={'concept':concept['concept']}
        membership = config.get('acs_membership', {}).get(concept['variable'], [])
        row.update({key: key in membership for key in ['ADI', 'SDI', 'SVI']})
        for r in rows: row[r['embedding_model']]=f"{r['r2_mean']:.3f} ({r['r2_std']:.3f})"
        if len(rows)==len(config['models']):
            values=[r['r2_mean'] for r in rows]; row['Mean R² (range across 6 models)']=f'{np.mean(values):.3f} ({min(values):.3f}–{max(values):.3f})'
        wide.append(row)
    if wide:
        pl.DataFrame(wide,infer_schema_length=None).write_csv(target/'table_s6_acs_included.csv')
        expected_s6 = pl.read_csv(reference/'table_s6_acs_index_correlations.csv').to_dicts()
        for row in wide:
            metrics = {k: None for k in row if k != 'concept'}
            checks += check_rows([row], expected_s6, ['concept'], metrics, 'table_s6_formatted')
    for index in ['ReADI','SDI','SVI']:
        rows=[r for r in tables['residuals'] if r['index']==index]
        if rows:
            df=pl.DataFrame(rows).with_columns(pl.col('outcome').replace_strict(config['outcome_labels']).alias('outcome_label'),
                pl.col('pct_unexplained_var').round(4).map_elements(lambda x:f'{x*100:.1f}%',return_dtype=pl.String))
            df.select('outcome','outcome_label','embedding_model','pct_unexplained_var').pivot(on='embedding_model',index=['outcome','outcome_label'],values='pct_unexplained_var').sort('outcome_label').write_csv(target/f'table_s2_wide_{index}.csv')
    failures=[c for c in checks if c['status']!='match']
    report={'scope':plan['scope'],'status':'passed' if not missing and not failures and checks else 'failed',
            'planned_tasks':len(plan['tasks']),'missing_or_invalid_tasks':missing,'numeric_checks':len(checks),
            'matched_checks':len(checks)-len(failures),'discrepancies':failures,'excluded':plan['excluded'],
            'runtime':plan['runtime'],'fingerprint':plan['fingerprint'],
            'precision_contract':'Exact equality at paper reporting precision; not binary file identity.',
            'figures':'Scoped plots from recomputed data; excluded index curves are absent. Original image bytes and layout are not reproduced.'}
    from replication import atomic_json
    atomic_json(output/'validation.json',report)
    if checks:
        pl.DataFrame([{**c,'key':json.dumps(c['key'])} for c in checks],infer_schema_length=None).write_csv(output/'validation_checks.csv')
    figures({**tables, 'residuals': raw_residuals},output,config)
    lines=['# Replication report','',f"Status: **{report['status']}**. Scope: **{report['scope']}**.",'',
           f"{report['matched_checks']}/{len(checks)} numerical checks match the paper at its reported precision.",
           f"{len(missing)} tasks missing or invalid; {len(failures)} discrepancies.",'',
           'See validation_checks.csv for individual comparisons. A selected-check run is not a full reproduction.',
           'Tables contain only recalculated included results. Figures use these results and omit excluded curves.',
           '', '## Intentionally excluded','']+['- '+x for x in plan['excluded']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f"Validation {report['status']}: {report['matched_checks']}/{len(checks)} checks, {len(missing)} missing tasks. See {output/'REPORT.md'}",flush=True)
    return 0 if report['status']=='passed' else 2
