import json
import os
import subprocess
import sys
import tempfile
import unittest

from role_prompt_tta_definitions import VARIANT_NAMES


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RolePromptTTAStaticTest(unittest.TestCase):

    def test_preflight_without_server_dependencies(self):
        result = subprocess.run(
            [sys.executable, 'tools/preflight_role_prompt_tta.py'],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(
            list(payload['datasets']),
            ['udd5', 'vdd', 'vaihingen', 'potsdam',
             'openearthmap', 'loveda'])
        self.assertIn('new method is constructor-default-off', payload['checks'])

    def test_isaid_is_not_in_experiment_configs(self):
        directory = os.path.join(ROOT, 'configs', 'experiments')
        for name in os.listdir(directory):
            if not name.endswith('_role_prompt_tta.py'):
                continue
            with open(os.path.join(directory, name), encoding='utf-8') as handle:
                self.assertNotIn('isaid', handle.read().lower())

    def test_exact_confusion_summarizer(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, 'role_prompt_tta.rank0.jsonl')
            out_dir = os.path.join(directory, 'summary')
            variants = {}
            for name in VARIANT_NAMES:
                matrix = (
                    [[9, 1], [1, 9]]
                    if name == 'full_regrounded_e2e'
                    else [[8, 2], [1, 9]])
                variants[name] = {
                    'confusion': {'matrix': matrix},
                    'changed_pixels': 1 if name == 'full_regrounded_e2e' else 0,
                    'improved_pixels': 1 if name == 'full_regrounded_e2e' else 0,
                    'harmed_pixels': 0,
                    'wrong_to_wrong_pixels': 0,
                    'help_minus_harm': 1 if name == 'full_regrounded_e2e' else 0,
                }
            record = {
                'schema_version': 1,
                'rank': 0,
                'dataset_name': 'udd5',
                'img_path': '/tmp/image.tif',
                'class_names': ['foreground', 'background'],
                'valid_pixels': 20,
                'primary_variant': 'full_regrounded_e2e',
                'variants': variants,
                'views': [{
                    'baseline_reconstruction_max_abs': 0.0,
                    'presence_gate': [1.0, 0.0],
                    'seed_counts': [2, 0],
                    'seed_entropy': [0.1, None],
                    'visual_affinity': [[0.1] * 5, [0.0] * 5],
                    'anchor_entropy_mean': 0.2,
                    'remoteclip_global_norm': 1.0,
                    'candidate_stats': [[], []],
                    'weight_stats': [],
                    'head_change': [],
                    'language_fusion_stats': {},
                    'seed_gt_diagnostics': [
                        {'seed_purity': 1.0, 'valid_seed_pixels': 2,
                         'correct_seed_pixels': 2},
                        {'seed_purity': 0.0, 'valid_seed_pixels': 0,
                         'correct_seed_pixels': 0},
                    ],
                }],
            }
            with open(input_path, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            result = subprocess.run(
                [sys.executable, 'tools/summarize_role_prompt_tta.py',
                 '--inputs', input_path, '--out-dir', out_dir,
                 '--expected-datasets', 'udd5',
                 '--positive-dataset-gate', '1'],
                cwd=ROOT, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(out_dir, 'summary.json'),
                      encoding='utf-8') as handle:
                summary = json.load(handle)
            self.assertTrue(summary['gate_passed'])
            self.assertGreater(
                summary['datasets'][0]['primary_delta'], 0.0)
            self.assertTrue(os.path.isfile(
                os.path.join(out_dir, 'component_attribution.csv')))


if __name__ == '__main__':
    unittest.main()
