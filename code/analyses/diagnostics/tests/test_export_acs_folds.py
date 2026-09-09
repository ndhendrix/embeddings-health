import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_acs_folds import export


class ExportTests(unittest.TestCase):
    def request(self, path, mismatch=False):
        counts={f'{i:02d}':i for i in range(1,11)}
        groups=np.concatenate([np.repeat(s,n) for s,n in counts.items()])
        mapping={str(s):fold for fold,(_,test) in enumerate(GroupKFold(5).split(groups,groups=groups),1)
                 for s in np.unique(groups[test])}
        if mismatch: mapping['01']=mapping['01']%5+1
        request={'schema_version':1,'n_splits':5,
                 'requests':[{'model':'test','target':'income','n_tracts':55,'state_counts':counts}],
                 'confirmed_case':{'model':'test','target':'income','state_to_fold':mapping}}
        path.write_text(json.dumps(request))

    def test_failure_preserves_evidence_but_not_release_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);request=p/'request.json';output=p/'folds.json';self.request(request,True)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'not a release'):
                export(request,output)
            self.assertFalse(output.exists())
            report=json.loads(output.with_suffix('.diagnostic.json').read_text())
            self.assertEqual(report['validation']['status'],'failed')
            self.assertIn('01',report['validation']['state_differences'])
            self.assertIn('numpy',report['environment'])
            self.assertEqual(len(report['assignments']),1)
            with self.assertRaises(FileExistsError):export(request,output)

    def test_success_creates_only_passed_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);request=p/'request.json';output=p/'folds.json';self.request(request)
            with contextlib.redirect_stdout(io.StringIO()):export(request,output)
            self.assertEqual(json.loads(output.read_text())['validation']['status'],'passed')
            self.assertFalse(output.with_suffix('.diagnostic.json').exists())


if __name__=='__main__':unittest.main()
