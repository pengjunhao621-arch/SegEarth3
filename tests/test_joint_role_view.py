import ast
import json
import os
import unittest

from role_functional_text_definitions import (
    JOINT_ROLE_VIEW_OPERATOR_SPECS,
    JOINT_ROLE_VIEW_OPERATOR_NAMES,
    load_joint_role_view_registry,
)
from tools.summarize_joint_role_view import _expected_variants, summarize


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(
    ROOT, 'configs', 'experiments', 'joint_role_view_profiles_v1.json')
DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


class JointRoleViewTest(unittest.TestCase):

    def test_single_image_control_keeps_roles_and_never_reads_neighbors(self):
        config_dir = os.path.join(ROOT, 'configs', 'experiments')
        with open(os.path.join(
                config_dir, 'role_visual_field_single_image_v1.json')) as handle:
            fields = json.load(handle)['datasets']
        with open(os.path.join(
                config_dir, 'role_visual_field_v1.json')) as handle:
            original_fields = json.load(handle)['datasets']

        # Exercise the actual pure view helpers without loading SAM3/PyTorch.
        path = os.path.join(ROOT, 'role_visual_field.py')
        with open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef)
                   and node.name == 'RoleVisualFieldMixin')
        names = {'_rvf_grid_boxes', '_rvf_context_box', '_rvf_visual_units',
                 '_jrv_global_source', '_rvf_fine_matches_official',
                 '_jrv_reference_endpoint'}
        cls.body = [node for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name in names]

        def forbidden_neighbor_index(*args, **kwargs):
            raise AssertionError('Single-image control accessed neighboring files.')

        namespace = dict(os=os, CoordinateTileIndex=forbidden_neighbor_index)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), path, 'exec'),
             namespace)

        class Image:
            width = height = 512
            size = (512, 512)

            def crop(self, box):
                value = Image()
                value.width, value.height = box[2] - box[0], box[3] - box[1]
                value.size = (value.width, value.height)
                return value

        self.assertEqual(set(fields), {'potsdam', 'vaihingen'})
        for dataset, config in fields.items():
            original = load_joint_role_view_registry(REGISTRY, dataset)
            control = load_joint_role_view_registry(os.path.join(
                config_dir, 'joint_role_view_single_image_v1.json'), dataset)
            self.assertEqual(control['role_candidates'], original['role_candidates'])
            self.assertIsNone(control['prior_view_miou'])
            self.assertEqual(original_fields[dataset]['source_mode'], 'coordinate_tiles')
            helper = namespace['RoleVisualFieldMixin']()
            helper._rvf_config = config
            helper.slide_crop = helper.slide_stride = 0
            helper._rvf_tile_indices = {}
            image = Image()
            self.assertEqual(helper._jrv_global_source(image), 'full_image')
            self.assertEqual(helper._jrv_reference_endpoint(image), 'global')
            units = helper._rvf_visual_units(image, '/not-a-directory/current.png')
            self.assertEqual([unit['target_box'] for unit in units], [
                (0, 0, 256, 256), (256, 0, 512, 256),
                (0, 256, 256, 512), (256, 256, 512, 512)])
            for unit in units:
                self.assertEqual(unit['fine'].size, (256, 256))
                self.assertEqual(unit['context'].size, (512, 512))
                self.assertEqual(unit['metadata']['contributor_tiles'], 1)
                self.assertEqual(unit['context_roi'], unit['target_box'])

    def test_registry_keeps_anchor_current_and_positive_alternatives(self):
        for dataset in DATASETS:
            value = load_joint_role_view_registry(REGISTRY, dataset)
            self.assertIn(value['anchor_candidate'], value['_candidate_ids'])
            self.assertIn(
                value['current_role_candidate'], value['_candidate_ids'])
            self.assertGreaterEqual(len(value['role_candidates']), 6)
            self.assertTrue(any(
                candidate['id'] not in (
                    value['anchor_candidate'], value['current_role_candidate'])
                and candidate['source_miou']
                > value['role_candidates'][0]['source_miou']
                for candidate in value['role_candidates']))

    def test_all_datasets_share_the_same_view_operator_family(self):
        self.assertEqual(JOINT_ROLE_VIEW_OPERATOR_NAMES[:2], (
            'local', 'global'))
        self.assertIn('linear_g050', JOINT_ROLE_VIEW_OPERATOR_NAMES)
        self.assertIn('power2_g050', JOINT_ROLE_VIEW_OPERATOR_NAMES)
        self.assertIn('power4_g050', JOINT_ROLE_VIEW_OPERATOR_NAMES)
        self.assertIn('logit_g050', JOINT_ROLE_VIEW_OPERATOR_NAMES)
        self.assertEqual(JOINT_ROLE_VIEW_OPERATOR_NAMES[-1], 'max')

    def test_dynamic_variant_contract_is_complete_and_unique(self):
        for dataset in DATASETS:
            value = load_joint_role_view_registry(REGISTRY, dataset)
            payload = dict(
                role_candidates=[dict(
                    id=candidate['id'], slots=list(candidate['slots']),
                    admission=candidate['admission'])
                    for candidate in value['role_candidates']],
                anchor_candidate=value['anchor_candidate'])
            names = _expected_variants(payload)
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(
                len(names),
                1 + len(value['role_candidates'])
                * len(JOINT_ROLE_VIEW_OPERATOR_NAMES))

    def test_summary_reports_four_way_and_joint_profiles(self):
        candidates = [
            dict(id='anchor', slots=[0, 0, 0], admission='native',
                 source_miou=50.0),
            dict(id='current', slots=[1, 1, 0], admission='native',
                 source_miou=55.0),
        ]
        operator_specs = [dict(
            name=name, family=family, global_weight=weight, rho=rho)
            for name, family, weight, rho
            in JOINT_ROLE_VIEW_OPERATOR_SPECS]
        variants = {
            'jrv_official': dict(
                candidate_id='anchor', slots=[0, 0, 0],
                admission='native', operator='local',
                operator_family='reference', global_weight=0.0, rho=None,
                confusion=dict(matrix=[[5, 1], [1, 5]]),
                changed_pixels=0, improved_pixels=0, harmed_pixels=0,
                help_minus_harm=0),
        }
        for candidate in candidates:
            prefix = 'anchor' if candidate['id'] == 'anchor' else candidate['id']
            for operator in operator_specs:
                matrix = ([[5, 1], [1, 5]] if candidate['id'] == 'anchor'
                          else [[6, 0], [1, 5]])
                variants[f'jrv_{prefix}__{operator["name"]}'] = dict(
                    candidate_id=candidate['id'], slots=candidate['slots'],
                    admission=candidate['admission'],
                    operator=operator['name'],
                    operator_family=operator['family'],
                    global_weight=operator['global_weight'], rho=operator['rho'],
                    confusion=dict(matrix=matrix), changed_pixels=1,
                    improved_pixels=1, harmed_pixels=0, help_minus_harm=1)
        payload = dict(
            schema_version=1, protocol='joint_role_view_profile_v1',
            global_source='aligned_context', reference_endpoint='local',
            role_candidates=candidates, anchor_candidate='anchor',
            current_role_candidate='current', prior_view_operator='max',
            prior_view_miou=54.54545454545454,
            reference_identity_max_abs=0.0,
            native_prompt_parity_max_abs=0.0,
            operator_specs=operator_specs, variants=variants,
            complementarity={
                candidate['id']: dict(
                    local_only_correct_pixels=1,
                    global_only_correct_pixels=1,
                    disagreement_pixels=2,
                    oracle_confusion=dict(matrix=[[6, 0], [0, 6]]))
                for candidate in candidates},
        )
        second_payload = dict(payload, global_source='full_image')
        result = summarize([
            dict(dataset_name='demo', img_path='demo-a.png',
                 class_names=['background', 'target'],
                 joint_role_view=payload),
            dict(dataset_name='demo', img_path='demo-b.png',
                 class_names=['background', 'target'],
                 joint_role_view=second_payload),
        ])
        profiles = {
            value['profile'] for value in result['profiles']}
        self.assertEqual(profiles, {
            'official', 'role_only', 'role_only_best', 'view_only_best',
            'sequential_role_view', 'current_role_best_view',
            'joint_role_view_best'})
        self.assertEqual(len(result['profile_macros']), 7)
        self.assertEqual(len(result['reproduction']), 1)
        self.assertAlmostEqual(
            result['reproduction'][0]['reference_identity_max_abs'], 0.0)
        self.assertEqual(
            result['reproduction'][0]['global_source_counts'],
            'aligned_context:1;full_image:1')


if __name__ == '__main__':
    unittest.main()
