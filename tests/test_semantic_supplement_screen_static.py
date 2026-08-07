import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

from semantic_supplement_definitions import (
    FIXED_FAMILY_VARIANTS,
    MAX_CANDIDATES,
    PROTOCOL,
    SCHEMA_VERSION,
    VARIANT_NAMES,
    candidate_variant_name,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SemanticSupplementScreenStaticTest(unittest.TestCase):

    def test_preflight_without_server_dependencies(self):
        result = subprocess.run(
            [sys.executable,
             'tools/preflight_semantic_supplement_screen.py'],
            cwd=ROOT, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        self.assertEqual(
            list(payload['datasets']),
            ['udd5', 'vdd', 'vaihingen', 'potsdam',
             'openearthmap', 'loveda'])
        self.assertEqual(payload['variants'], 24)
        self.assertGreater(
            payload['datasets']['openearthmap']['lexical_candidates'], 0)

    def test_formula_keeps_structural_anchor_and_bounds_semantic_delta(self):
        path = os.path.join(ROOT, 'semantic_supplement_screen.py')
        with open(path, encoding='utf-8') as handle:
            source = handle.read()
        tree = ast.parse(source, filename=path)
        bounded = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == '_ss_bounded_semantic')
        bounded_source = ast.get_source_segment(source, bounded)
        self.assertIn(
            'candidate_semantic.float() - anchor_semantic.float()',
            bounded_source)
        self.assertIn('role_prompt_tta_semantic_residual_clip', bounded_source)
        self.assertIn('role_prompt_tta_semantic_residual_alpha', bounded_source)
        compose = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == '_ss_compose_semantic')
        compose_source = ast.get_source_segment(source, compose)
        self.assertIn('anchor,', compose_source)
        self.assertIn('instance_override=anchor_instance', compose_source)

    def test_variant_contract_separates_families_and_oracle_is_not_variant(self):
        self.assertEqual(MAX_CANDIDATES, 3)
        self.assertEqual(len(VARIANT_NAMES), 24)
        self.assertEqual(
            FIXED_FAMILY_VARIANTS,
            ('lexical_mean_semantic_residual',
             'modifier_mean_semantic_residual'))
        self.assertFalse(any('oracle' in name for name in VARIANT_NAMES))
        for family in ('lexical', 'modifier'):
            for slot in range(1, MAX_CANDIDATES + 1):
                for path in (
                        'shared_native', 'semantic_replace',
                        'semantic_residual'):
                    self.assertIn(
                        candidate_variant_name(family, slot, path),
                        VARIANT_NAMES)

    def test_synthetic_summary_marks_fixed_and_oracle_results(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, 'screen.rank0.jsonl')
            duplicate_path = os.path.join(directory, 'screen.rank1.jsonl')
            out_dir = os.path.join(directory, 'summary')
            variants = {}
            for name in VARIANT_NAMES:
                matrix = [[8, 2], [2, 8]]
                if name == 'lexical_mean_semantic_residual':
                    matrix = [[9, 1], [1, 9]]
                variants[name] = dict(
                    confusion=dict(matrix=matrix),
                    changed_pixels=(0 if name == 'baseline' else 2),
                    improved_pixels=(
                        2 if name == 'lexical_mean_semantic_residual' else 0),
                    harmed_pixels=0,
                    wrong_to_wrong_pixels=0,
                    help_minus_harm=(
                        2 if name == 'lexical_mean_semantic_residual' else 0),
                )
            candidate_rows = []
            family_rows = []
            for class_idx, class_name in enumerate(('foreground', 'background')):
                for family in ('lexical', 'modifier'):
                    family_rows.append(dict(
                        class_index=class_idx,
                        class_name=class_name,
                        family=family,
                        active_candidate_count=2,
                        aggregate='mean',
                        semantic_signed_delta_mean=0.1,
                        semantic_abs_delta_mean=0.2,
                        semantic_clip_ratio=0.1,
                        clipped_abs_delta_mean=0.1,
                        replace_final_mean=0.4,
                        residual_final_mean=0.3,
                    ))
                    for slot in range(1, MAX_CANDIDATES + 1):
                        candidate_rows.append(dict(
                            class_index=class_idx,
                            class_name=class_name,
                            family=family,
                            slot=slot,
                            available=(slot < 3),
                            anchor_prompt=class_name,
                            candidate_prompt=f'{class_name} {family} {slot}',
                            anchor_token_count=1,
                            candidate_token_count=2,
                            candidate_word_count=3,
                            anchor_presence=0.8,
                            candidate_presence=0.7,
                            presence_delta=-0.1,
                            anchor_semantic_mean=0.3,
                            candidate_semantic_mean=0.4,
                            candidate_semantic_area_050=0.2,
                            semantic_signed_delta_mean=0.1,
                            semantic_abs_delta_mean=0.2,
                            semantic_positive_delta_ratio=0.6,
                            semantic_negative_delta_ratio=0.4,
                            semantic_clip_ratio=0.1,
                            clipped_abs_delta_mean=0.1,
                            anchor_instance_mean=0.2,
                            candidate_instance_mean=0.1,
                            anchor_native_kept_count=2,
                            candidate_native_kept_count=1,
                            candidate_raw_object_score_mean=0.4,
                            candidate_raw_object_score_max=0.8,
                            native_recomposition_max_abs=0.0,
                            native_instance_recomposition_max_abs=0.0,
                            path_final_means={
                                'shared_native': 0.2,
                                'semantic_replace': 0.3,
                                'semantic_residual': 0.25,
                            },
                        ))
            record = dict(
                schema_version=SCHEMA_VERSION,
                rank=0,
                dataset_name='udd5',
                img_path='/tmp/example.tif',
                class_names=['foreground', 'background'],
                valid_pixels=20,
                prob_thd=0.0,
                confidence_threshold=0.5,
                primary_variant='baseline',
                prompt_count=7,
                settings=dict(protocol=PROTOCOL),
                variants=variants,
                views=[dict(
                    baseline_reconstruction_max_abs=0.0,
                    raw_recomposition_max_abs=0.0,
                    anchor_vs_official_max_abs=0.0,
                    diagnostic_cpu_bytes=100,
                    diagnostic_variant_count=len(VARIANT_NAMES),
                    synonym_query_count=0,
                    candidate_rows=candidate_rows,
                    family_rows=family_rows,
                )],
            )
            with open(input_path, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            duplicate = copy.deepcopy(record)
            duplicate['rank'] = 1
            with open(duplicate_path, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps(duplicate) + '\n')
            result = subprocess.run(
                [sys.executable,
                 'tools/summarize_semantic_supplement_screen.py',
                 '--inputs', input_path, duplicate_path,
                 '--out-dir', out_dir,
                 '--expected-datasets', 'udd5',
                 '--allow-incomplete'],
                cwd=ROOT, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(out_dir, 'summary.json'),
                      encoding='utf-8') as handle:
                summary = json.load(handle)
            self.assertTrue(summary['all_integrity_passed'])
            self.assertEqual(
                summary['distributed_padding_duplicates_removed'],
                {'udd5': 1})
            self.assertTrue(all(
                row['oracle_only']
                for row in summary['oracle_diagnostics']))
            for name in (
                    'dataset_variants.csv', 'per_class.csv',
                    'candidate_effects.csv',
                    'per_class_candidate_effects.csv',
                    'family_effects.csv', 'prompt_response.csv',
                    'family_response.csv', 'oracle_diagnostics.csv',
                    'integrity.csv', 'cross_dataset_variants.csv',
                    'decision_report.md'):
                self.assertTrue(os.path.isfile(os.path.join(out_dir, name)))


if __name__ == '__main__':
    unittest.main()
