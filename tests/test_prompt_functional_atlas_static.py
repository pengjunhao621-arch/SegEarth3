import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest

from prompt_functional_atlas_definitions import (
    PROMPT_SLOTS,
    SCHEMA_VERSION,
    VARIANT_NAMES,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PromptFunctionalAtlasStaticTest(unittest.TestCase):

    def test_preflight_without_server_dependencies(self):
        result = subprocess.run(
            [sys.executable, 'tools/preflight_prompt_functional_atlas.py'],
            cwd=ROOT, check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertEqual(report['dataset'], 'udd5')
        self.assertEqual(report['prompts_per_class'], 5)
        self.assertEqual(report['variants'], len(VARIANT_NAMES))

    def test_protocol_is_default_off_and_dispatch_is_explicit(self):
        with open(os.path.join(ROOT, 'segearthov3_segmentor.py'),
                  encoding='utf-8') as handle:
            segmentor = handle.read()
        with open(os.path.join(ROOT, 'role_prompt_tta.py'),
                  encoding='utf-8') as handle:
            role = handle.read()
        self.assertIn("role_prompt_tta_protocol='v1'", segmentor)
        self.assertIn('role_prompt_tta_e2e_measure_post=False', segmentor)
        self.assertIn(
            "self.role_prompt_tta_protocol == 'prompt_functional_atlas_v2'",
            role)
        self.assertIn('post_update_loss', role)
        self.assertIn('_rpt_reground_anchor_residual', role)

    def test_anchor_residual_restores_native_language_dtype(self):
        with open(os.path.join(ROOT, 'role_prompt_tta.py'),
                  encoding='utf-8') as handle:
            source = handle.read()
        tree = ast.parse(source, filename='role_prompt_tta.py')
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == '_rpt_reground_anchor_residual')
        method_source = ast.get_source_segment(source, method)
        self.assertIn('anchor[overlap].float()', method_source)
        self.assertIn('target[overlap].float()', method_source)
        self.assertIn('updated.to(dtype=fused.dtype)', method_source)

    def test_synthetic_summary_preserves_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, 'prompt_atlas.rank0.jsonl')
            out_dir = os.path.join(directory, 'summary')
            variants = {}
            for name in VARIANT_NAMES:
                matrix = (
                    [[9, 1], [1, 9]]
                    if name == 'full_bg_literal_output'
                    else [[8, 2], [1, 9]])
                variants[name] = dict(
                    confusion=dict(matrix=matrix),
                    changed_pixels=(1 if name != 'baseline' else 0),
                    improved_pixels=(1 if name == 'full_bg_literal_output' else 0),
                    harmed_pixels=0,
                    wrong_to_wrong_pixels=0,
                    help_minus_harm=(
                        1 if name == 'full_bg_literal_output' else 0),
                )
            candidate_stats = []
            for class_name in ('foreground', 'background'):
                candidate_stats.append([
                    dict(
                        prompt=(class_name if idx == 0
                                else f'{class_name} prompt {idx}'),
                        token_count=idx + 1,
                        word_count=idx + 1,
                        character_count=10 + idx,
                        presence=1.0 / (idx + 1),
                        raw_candidate_count=10,
                        kept_candidate_count=5 - idx,
                        semantic_mean=0.5,
                        instance_mean=0.25,
                        final_mean=0.4,
                    )
                    for idx in range(len(PROMPT_SLOTS))
                ])
            record = dict(
                schema_version=SCHEMA_VERSION,
                rank=0,
                dataset_name='udd5',
                img_path='/tmp/example.tif',
                class_names=['foreground', 'background'],
                primary_variant='baseline',
                settings=dict(protocol='prompt_functional_atlas_v2'),
                variants=variants,
                views=[dict(
                    baseline_reconstruction_max_abs=0.0,
                    literal_reground_parity_max_abs=dict(
                        final=0.0, semantic=0.0, semantic_raw=0.0,
                        instance=0.0, presence=0.0),
                    candidate_stats=candidate_stats,
                    presence_selected_indices=[0, 0],
                    head_change=[],
                    e2e_trajectory=[dict(
                        class_index=0,
                        steps=[dict(
                            loss=0.4,
                            grad_norm=0.2,
                            post_update_loss=0.3,
                            post_update_weight_l1=0.01)])],
                )],
            )
            with open(input_path, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            result = subprocess.run(
                [sys.executable,
                 'tools/summarize_prompt_functional_atlas.py',
                 '--inputs', input_path,
                 '--out-dir', out_dir,
                 '--expected-dataset', 'udd5'],
                cwd=ROOT, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(out_dir, 'summary.json'),
                      encoding='utf-8') as handle:
                summary = json.load(handle)
            self.assertTrue(summary['controls']['literal_identity_passed'])
            self.assertEqual(len(summary['prompt_rows']), 10)
            self.assertLess(summary['e2e_rows'][0]['loss_delta'], 0.0)
            self.assertEqual(
                summary['directional_evidence']['best_prompt_role']['role'],
                'literal')
            for name in (
                    'dataset_variants.csv', 'per_class.csv',
                    'prompt_function.csv', 'head_path.csv',
                    'e2e_update.csv', 'comparisons.csv',
                    'correlations.csv', 'correlations_per_class.csv',
                    'prompt_role_summary.csv', 'decision_report.md'):
                self.assertTrue(os.path.isfile(os.path.join(out_dir, name)))


if __name__ == '__main__':
    unittest.main()
