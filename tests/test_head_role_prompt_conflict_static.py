import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest

from head_role_prompt_conflict_definitions import (
    DESCRIPTION_SLOTS,
    HEAD_PATHS,
    SCHEMA_VERSION,
    VARIANT_NAMES,
    variant_name,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HeadRolePromptConflictStaticTest(unittest.TestCase):

    def test_preflight_without_server_dependencies(self):
        result = subprocess.run(
            [sys.executable, 'tools/preflight_head_role_prompt_conflict.py'],
            cwd=ROOT, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        self.assertEqual(
            list(payload['datasets']),
            ['udd5', 'vdd', 'vaihingen', 'potsdam',
             'openearthmap', 'loveda'])
        self.assertEqual(payload['variants'], 22)

    def test_protocol_is_default_off_and_raw_gate_is_recomputed(self):
        segmentor_path = os.path.join(ROOT, 'segearthov3_segmentor.py')
        role_path = os.path.join(ROOT, 'role_prompt_tta.py')
        with open(segmentor_path, encoding='utf-8') as handle:
            segmentor = handle.read()
        with open(role_path, encoding='utf-8') as handle:
            role = handle.read()
        self.assertIn("role_prompt_tta_protocol='v1'", segmentor)
        self.assertIn('role_prompt_variant_class_logits', segmentor)
        tree = ast.parse(role, filename=role_path)
        helper = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == '_rpt_head_role_instance_from_raw')
        source = ast.get_source_segment(role, helper)
        self.assertIn('raw_scores_device * presence_device', source)
        self.assertIn('self.processor.confidence_threshold', source)
        self.assertIn('raw_masks[:count].to(self.device)[keep]', source)
        self.assertIn('sam3_interpolate(', source)
        self.assertNotIn('F.interpolate(', source)

        imports = [
            node for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == 'sam3.model.data_misc'
        ]
        self.assertEqual(len(imports), 1)
        self.assertTrue(any(
            alias.name == 'interpolate'
            and alias.asname == 'sam3_interpolate'
            for alias in imports[0].names
        ))

    def test_variant_contract_covers_five_paths_per_description(self):
        self.assertEqual(
            len(VARIANT_NAMES),
            2 + len(DESCRIPTION_SLOTS) * len(HEAD_PATHS))
        for slot, _ in DESCRIPTION_SLOTS:
            for path, _, _, _ in HEAD_PATHS:
                self.assertIn(variant_name(slot, path), VARIANT_NAMES)

    def test_synthetic_summary_reports_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(
                directory, 'head_role_conflict.rank0.jsonl')
            out_dir = os.path.join(directory, 'summary')
            variants = {}
            for name in VARIANT_NAMES:
                matrix = [[8, 2], [2, 8]]
                if name == variant_name(1, 'semantic_only'):
                    matrix = [[9, 1], [1, 9]]
                elif name == variant_name(
                        1, 'semantic_instance_literal_presence'):
                    matrix = [[9, 1], [2, 8]]
                variants[name] = dict(
                    confusion=dict(matrix=matrix),
                    changed_pixels=(0 if name == 'baseline' else 2),
                    improved_pixels=(
                        2 if name == variant_name(1, 'semantic_only') else 0),
                    harmed_pixels=0,
                    wrong_to_wrong_pixels=0,
                    help_minus_harm=(
                        2 if name == variant_name(1, 'semantic_only') else 0),
                )
            head_rows = []
            for class_idx, class_name in enumerate(('foreground', 'background')):
                for slot, role in DESCRIPTION_SLOTS:
                    head_rows.append(dict(
                        class_index=class_idx,
                        class_name=class_name,
                        prompt_slot=slot,
                        prompt_role=role,
                        literal_prompt=class_name,
                        description_prompt=f'{class_name} description {slot}',
                        literal_presence=0.8,
                        description_presence=0.4,
                        presence_delta=-0.4,
                        literal_semantic_mean=0.4,
                        description_semantic_mean=0.6,
                        description_semantic_area_050=0.5,
                        literal_instance_mean=0.2,
                        description_instance_mean=0.1,
                        semantic_abs_change=0.2,
                        instance_abs_change_native=0.1,
                        semantic_instance_iou=0.3,
                        raw_object_score_mean=0.4,
                        raw_object_score_max=0.9,
                        raw_candidate_count=200,
                        native_kept_count=2,
                        description_instance_literal_presence_kept_count=4,
                        literal_instance_description_presence_kept_count=1,
                        native_recomposition_max_abs=0.0,
                        native_instance_recomposition_max_abs=0.0,
                        path_final_means={path: 0.2 for path, _, _, _ in HEAD_PATHS},
                        path_kept_counts={path: 2 for path, _, _, _ in HEAD_PATHS},
                    ))
            record = dict(
                schema_version=SCHEMA_VERSION,
                rank=0,
                dataset_name='udd5',
                img_path='/tmp/example.tif',
                class_names=['foreground', 'background'],
                primary_variant='baseline',
                settings=dict(protocol='head_role_prompt_conflict_v1'),
                variants=variants,
                views=[dict(
                    baseline_reconstruction_max_abs=0.0,
                    raw_recomposition_max_abs=0.0,
                    literal_vs_official_max_abs=0.0,
                    synonym_query_count=0,
                    head_role_rows=head_rows,
                )],
            )
            with open(input_path, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            result = subprocess.run(
                [sys.executable,
                 'tools/summarize_head_role_prompt_conflict.py',
                 '--inputs', input_path,
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
                summary['conflict_counts'][
                    'datasets_with_semantic_positive_native_nonpositive'], 1)
            for name in (
                    'dataset_variants.csv', 'per_class.csv',
                    'dataset_mechanisms.csv', 'per_class_mechanisms.csv',
                    'head_response.csv', 'integrity.csv',
                    'cross_dataset_variants.csv', 'decision_report.md'):
                self.assertTrue(os.path.isfile(os.path.join(out_dir, name)))


if __name__ == '__main__':
    unittest.main()
