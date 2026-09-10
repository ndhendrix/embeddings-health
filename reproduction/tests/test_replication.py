import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from replication import verify_inputs, execute
from reporting import rounded_equal, check_rows, assemble


class IntegrityTests(unittest.TestCase):
    def test_s2_uses_rounded_index_but_unrounded_increment(self):
        from reporting import paper_residual_rows
        original = {'r2_index': -.03581805771741675,
                    'r2_resid_explained_by_emb': .2641941169952231,
                    'delta_r2': .27365703712635997,
                    'pct_unexplained_var': .2641941169952231,
                    'r2_sequential_combined': .23783897940894322}
        row = paper_residual_rows([original])[0]
        self.assertEqual(row['r2_sequential_combined'], .2379)
        self.assertEqual(row['pct_unexplained_var'], round(original['delta_r2'] / 1.0358, 4))
        self.assertEqual(row['delta_r2'], .2737)
        self.assertEqual(original['r2_sequential_combined'], .23783897940894322)

    def test_corrupted_manual_download_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'x').write_bytes(b'bad')
            lock={'files':{'x':{'bytes':3,'sha256':hashlib.sha256(b'good').hexdigest()}}}
            with self.assertRaisesRegex(ValueError,'checksum mismatch'): verify_inputs(p,lock)

    def test_path_traversal_in_lock_fails(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                verify_inputs(Path(d),{'files':{'../outside':{'bytes':0,'sha256':''}}})

    def test_resume_checks_input_code_fingerprint_and_payload(self):
        with tempfile.TemporaryDirectory() as d:
            task={'model':'m','stage':'places','target':'x'}
            with patch('analysis.run_task',return_value={'places':[{'r2':.2}]} ) as fit:
                self.assertEqual(execute(task,d,{},d,'a',1)['status'],'fitted')
                self.assertEqual(execute(task,d,{},d,'a',1)['status'],'reused')
                self.assertEqual(fit.call_count,1)
                self.assertEqual(execute(task,d,{},d,'b',1)['status'],'fitted')
                p=Path(d)/'tasks/m__places__x.json'; obj=json.loads(p.read_text()); obj['result']['places'][0]['r2']=.9; p.write_text(json.dumps(obj))
                self.assertEqual(execute(task,d,{},d,'b',1)['status'],'fitted')
                self.assertEqual(fit.call_count,3)

    def test_precision_is_reported_precision_not_loose_tolerance(self):
        self.assertTrue(rounded_equal(.202155931930033,.2022,4))
        self.assertFalse(rounded_equal(.202149,.2022,4))
        self.assertFalse(rounded_equal(float('nan'),.2,4))

    def test_wrong_sample_is_a_failure_even_if_score_matches(self):
        a=[{'model':'m','r2':.2,'n':10}]; b=[{'model':'m','r2':.2,'n':11}]
        checks=check_rows(a,b,['model'],{'r2':4,'n':None},'test')
        self.assertEqual([c['status'] for c in checks],['match','mismatch'])

    def test_no_expected_results_are_imported_by_fitting_module(self):
        source=(Path(__file__).resolve().parents[1]/'analysis.py').read_text()
        self.assertNotIn('read_csv',source)
        self.assertNotIn('reference/',source)



class WorkflowTests(unittest.TestCase):
    def test_fresh_failure_cannot_reuse_previous_success(self):
        with tempfile.TemporaryDirectory() as d:
            task={'model':'m','stage':'places','target':'x'}
            with patch('analysis.run_task',return_value={'places':[]}): execute(task,d,{},d,'a',1)
            with patch('analysis.run_task',side_effect=RuntimeError('fit failed')):
                with self.assertRaisesRegex(RuntimeError,'fit failed'): execute(task,d,{},d,'a',1,resume=False)
            self.assertFalse((Path(d)/'tasks/m__places__x.json').exists())
            self.assertEqual(len(list((Path(d)/'tasks/superseded').glob('*.json'))),1)

    def test_download_is_pinned_and_verified_without_token(self):
        import io
        content=b'verified release data'
        lock={'files':{'data.parquet':{'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}}}
        with tempfile.TemporaryDirectory() as d, patch('urllib.request.urlopen',return_value=io.BytesIO(content)) as request:
            verify_inputs(Path(d),lock,'12345')
            self.assertEqual((Path(d)/'data.parquet').read_bytes(),content)
            self.assertIn('zenodo.org/records/12345/files/data.parquet',request.call_args.args[0])
            self.assertFalse(list(Path(d).glob('*.partial')))

    def test_slurm_uses_same_worker_and_afterany_collection(self):
        from replication import main
        with tempfile.TemporaryDirectory() as d:
            argv=['replication.py','--slurm','--smoke','--data-dir',d,'--output-dir',d,'--partition','compute']
            with patch.object(sys,'argv',argv), patch('replication.verify_inputs'), patch('replication.make_tasks',return_value=[{'model':'alphaearth','stage':'places','target':'ACCESS2'}]), patch('replication.fingerprint',return_value=('fp',{})), patch('subprocess.check_output',side_effect=['123;cluster\n','124\n']) as submit:
                self.assertEqual(main(),0)
                self.assertIn('--array=0-0%12',submit.call_args_list[0].args[0])
                self.assertIn('--dependency=afterany:123',submit.call_args_list[1].args[0])
                worker=next(Path(d).glob('worker-*.sh')).read_text()
                self.assertIn('replication.py --worker-plan',worker)
                self.assertIn('"$SLURM_ARRAY_TASK_ID"',worker)

    def test_missing_task_is_reported_as_failure(self):
        from replication import ROOT
        config=json.loads((ROOT/'config.json').read_text())
        plan={'tasks':[{'model':'alphaearth','stage':'places','target':'ACCESS2'}], 'fingerprint':'fp','config':config,'scope':'selected_checks','runtime':{},'excluded':config['excluded']}
        with tempfile.TemporaryDirectory() as d, patch('reporting.figures'):
            self.assertEqual(assemble(plan,Path(d),ROOT/'reference'),2)
            report=json.loads((Path(d)/'validation.json').read_text())
            self.assertEqual(report['missing_or_invalid_tasks'],['alphaearth__places__ACCESS2'])
            self.assertEqual(report['status'],'failed')

class FrozenFoldTests(unittest.TestCase):
    def manifest(self, path, ids):
        from collections import Counter
        item={'model':'m','target':'income','n_tracts':len(ids),
              'state_counts':dict(Counter(i[:2] for i in ids)),
              'sorted_ids_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
              'state_to_fold':{f'{i:02d}':i for i in range(1,6)}}
        (path/'acs_folds.json').write_text(json.dumps({'assignments':[item]}))

    def test_frozen_folds_follow_states_after_row_reordering(self):
        from analysis import acs_fold_labels
        ids=[f'{i:02d}001000100' for i in range(1,6)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); self.manifest(p,ids)
            self.assertEqual(acs_fold_labels(p,'m','income',ids[::-1]).tolist(),[5,4,3,2,1])
            with self.assertRaisesRegex(ValueError,'sample differs'):
                acs_fold_labels(p,'m','income',ids[:-1]+['05001000999'])
            with self.assertRaisesRegex(ValueError,'Missing or duplicate'):
                acs_fold_labels(p,'m','other',ids)

    def test_frozen_folds_never_call_version_sensitive_splitter(self):
        import numpy as np
        from analysis import cross_validation
        class Model:
            def fit(self,X,y): return self
            def predict(self,X): return X[:,0]
        y=np.arange(10,dtype=float); labels=np.tile(np.arange(1,6),2)
        with patch('analysis.GroupKFold') as splitter, patch('analysis.estimator',return_value=Model()):
            mean,sd,scores=cross_validation(y[:,None],y,{},1,fold_labels=labels)
            splitter.return_value.split.assert_not_called()
            self.assertEqual((mean,sd),(1.0,0.0))

    def test_q2_order_is_distinct_from_acs_source_order(self):
        import polars as pl
        from analysis import base_frame
        sample=pl.DataFrame({'model_id':['m','m'],'GEOID':['01001000100','01001000200'],
                             'q2_member':[True,True],'source_order':[0,1],'q2_order':[1,0]})
        emb=pl.DataFrame({'GEOID':sample['GEOID'],'x':[1.,2.]})
        area=pl.DataFrame({'GEOID':sample['GEOID'],'ALAND':[1.,2.],'AWATER':[0.,0.]})
        config={'models':{'m':{'features':['x']}}}
        with patch('analysis.pl.read_parquet',side_effect=[sample,emb,area,sample,emb,area]):
            q2,_=base_frame(Path('.'),config,'m',True)
            acs,_=base_frame(Path('.'),config,'m',False)
        self.assertEqual(q2['GEOID'].to_list(),sample['GEOID'].to_list()[::-1])
        self.assertEqual(acs['GEOID'].to_list(),sample['GEOID'].to_list())


class PredictorPrecisionTests(unittest.TestCase):
    def test_prithvi_matrix_casts_embeddings_and_area_without_changing_source(self):
        import numpy as np
        import polars as pl
        from analysis import predictor_matrix
        frame = pl.DataFrame({'PR0000_MEAN': [1.00000001],
                              'ALAND': [16777217.0], 'AWATER': [0.1],
                              'outcome': [1.00000001]})
        features = ['PR0000_MEAN', 'ALAND', 'AWATER']
        original = frame.select(features).to_numpy()
        matrix = predictor_matrix(frame, features, 'prithvi_300m_tl')
        self.assertEqual(matrix.dtype, np.dtype('float32'))
        np.testing.assert_array_equal(matrix, original.astype(np.float32))
        for model in ['alphaearth', 'clay', 'olmoearth_base', 'olmoearth_nano', 'prithvi_tiny']:
            other = predictor_matrix(frame, features, model)
            self.assertEqual(other.dtype, original.dtype)
            np.testing.assert_array_equal(other, original)
        self.assertEqual(frame['ALAND'][0], 16777217.0)
        self.assertEqual(frame['outcome'][0], 1.00000001)


if __name__=='__main__': unittest.main()
