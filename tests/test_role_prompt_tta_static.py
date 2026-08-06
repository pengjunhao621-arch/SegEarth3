import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest

from role_prompt_tta_definitions import VARIANT_NAMES


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RolePromptTTAStaticTest(unittest.TestCase):

    def test_surrogate_optimizer_reenables_grad_locally(self):
        path = os.path.join(ROOT, 'role_prompt_tta.py')
        with open(path, encoding='utf-8') as handle:
            source = handle.read()
        tree = ast.parse(source, filename=path)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'optimize_surrogate_weights')
        function_source = ast.get_source_segment(source, function)
        self.assertIn('with torch.enable_grad():', function_source)
        self.assertIn('loss.backward()', function_source)

    def test_e2e_memory_controls_are_wired(self):
        role_path = os.path.join(ROOT, 'role_prompt_tta.py')
        with open(role_path, encoding='utf-8') as handle:
            role_source = handle.read()
        self.assertIn('def resolve_remoteclip_device(', role_source)
        self.assertIn("requested == 'aux'", role_source)
        self.assertIn('state[\'backbone_out\'][key] = cloned', role_source)
        e2e_call = role_source.index(
            'e2e_weights, e2e_trajectory, e2e_classes')
        diagnostic_reground = role_source.index(
            'anchor_reground, anchor_fusion', e2e_call)
        self.assertLess(e2e_call, diagnostic_reground)
        runner_path = os.path.join(ROOT, 'tools', 'run_role_prompt_tta_v1.sh')
        with open(runner_path, encoding='utf-8') as handle:
            runner_source = handle.read()
        self.assertIn('REMOTECLIP_DEVICE', runner_source)
        self.assertIn('PYTORCH_CUDA_ALLOC_CONF', runner_source)

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
                    'cuda_memory': {
                        'main': {
                            'allocated_mb': 100.0,
                            'reserved_mb': 120.0,
                            'peak_allocated_mb': 200.0,
                            'peak_reserved_mb': 220.0,
                        },
                        'remoteclip': {
                            'allocated_mb': 50.0,
                            'reserved_mb': 60.0,
                            'peak_allocated_mb': 70.0,
                            'peak_reserved_mb': 80.0,
                        },
                    },
                    'e2e_trajectory': [{
                        'class_index': 0,
                        'steps': [{
                            'loss': 0.3,
                            'grad_norm': 0.2,
                            'cuda_memory': {
                                'before_forward': {
                                    'peak_allocated_mb': 150.0},
                                'after_forward': {
                                    'peak_allocated_mb': 180.0},
                                'after_backward': {
                                    'peak_allocated_mb': 200.0},
                            },
                        }],
                    }],
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
            mechanism = summary['datasets'][0]['mechanism_row']
            self.assertEqual(
                mechanism['cuda_main_peak_allocated_mb'], 200.0)
            self.assertEqual(
                mechanism['e2e_cuda_after_backward_peak_allocated_mb'],
                200.0)


if __name__ == '__main__':
    unittest.main()
