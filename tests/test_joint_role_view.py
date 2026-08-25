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
                anchor_candidate=value['anchor_candidate'],
                operator_specs=[
                    {'name': name}
                    for name in JOINT_ROLE_VIEW_OPERATOR_NAMES])
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
        result = summarize([dict(
            dataset_name='demo', img_path='demo.png',
            class_names=['background', 'target'],
            joint_role_view=payload)])
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

    def test_role_only_summary_accepts_global_operator_subset(self):
        candidates = [
            dict(id='anchor', slots=[0, 0, 0], admission='native',
                 source_miou=None),
            dict(id='selected', slots=[1, 2, 1], admission='native',
                 source_miou=None),
        ]
        operator_specs = [dict(
            name='global', family='endpoint', global_weight=1.0, rho=None)]
        variants = {
            'jrv_official': dict(
                candidate_id='anchor', slots=[0, 0, 0],
                admission='native', operator='global',
                operator_family='reference', global_weight=1.0, rho=None,
                confusion=dict(matrix=[[5, 1], [1, 5]]),
                changed_pixels=0, improved_pixels=0, harmed_pixels=0,
                help_minus_harm=0),
            'jrv_anchor__global': dict(
                candidate_id='anchor', slots=[0, 0, 0],
                admission='native', operator='global',
                operator_family='endpoint', global_weight=1.0, rho=None,
                confusion=dict(matrix=[[5, 1], [1, 5]]),
                changed_pixels=0, improved_pixels=0, harmed_pixels=0,
                help_minus_harm=0),
            'jrv_selected__global': dict(
                candidate_id='selected', slots=[1, 2, 1],
                admission='native', operator='global',
                operator_family='endpoint', global_weight=1.0, rho=None,
                confusion=dict(matrix=[[6, 0], [1, 5]]),
                changed_pixels=1, improved_pixels=1, harmed_pixels=0,
                help_minus_harm=1),
        }
        payload = dict(
            schema_version=1, protocol='joint_role_view_profile_v1',
            global_source='official_observation', reference_endpoint='global',
            role_candidates=candidates, anchor_candidate='anchor',
            current_role_candidate='anchor', prior_view_operator='global',
            prior_view_miou=None, reference_identity_max_abs=0.0,
            native_prompt_parity_max_abs=0.0,
            operator_specs=operator_specs, variants=variants,
            complementarity={
                candidate['id']: dict(
                    local_only_correct_pixels=0,
                    global_only_correct_pixels=0,
                    disagreement_pixels=0,
                    oracle_confusion=dict(matrix=[[5, 1], [1, 5]]))
                for candidate in candidates},
        )
        result = summarize([dict(
            dataset_name='isaid', img_path='tile.png',
            class_names=['background', 'target'],
            joint_role_view=payload)])
        profiles = {row['profile']: row for row in result['profiles']}
        self.assertEqual(profiles['role_only_best']['candidate_id'], 'selected')
        self.assertEqual(profiles['joint_role_view_best']['operator'], 'global')
        self.assertEqual(result['complementarity'], [])
        self.assertEqual([row['operator'] for row in result['operators']], [
            'global'])


if __name__ == '__main__':
    unittest.main()
