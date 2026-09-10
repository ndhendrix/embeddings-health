import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'prithvi_full.py'
spec = importlib.util.spec_from_file_location('prithvi_full', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrithviInspectionTests(unittest.TestCase):
    def test_feature_order_difference_and_read_only_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['prepared', 'run', 'code', 'reproduction']:
                (root/name).mkdir()
            features = ['PR0000_MEAN', 'PR0000_STD']
            metadata = root/'prepared/metadata.json'
            metadata.write_text(json.dumps({'features': features[::-1], 'api_key': 'not-a-real-key'}))
            before = metadata.read_bytes()
            (root/'reproduction/config.json').write_text(json.dumps({'models': {'prithvi_300m_tl': {'features': features}}}))
            command = [sys.executable, str(SCRIPT), '--prepared-dir', str(root/'prepared'), '--run-root', str(root/'run'), '--original-code', str(root/'code'), '--reproduction', str(root/'reproduction'), '--output', str(root/'report.json')]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((root/'report.json').read_text())
            self.assertTrue(report['feature_comparisons'][0]['same_membership'])
            self.assertFalse(report['feature_comparisons'][0]['same_order'])
            self.assertNotIn('not-a-real-key', (root/'report.json').read_text())
            self.assertEqual(metadata.read_bytes(), before)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            command[-1] = str(root/'missing-report.json')
            command[command.index('--prepared-dir')+1] = str(root/'missing')
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(json.loads((root/'missing-report.json').read_text())['inspection_status'], 'incomplete')

    def test_long_feature_list_is_not_truncated(self):
        values = [f'PR{i:04d}_MEAN' for i in range(1024)]
        self.assertEqual(module.summarize({'features': values})['features'], values)
