import unittest

from tools.plan_isaid_resume import plan_resume
from tools.summarize_isaid_sequential_role_view import (
    expected_variants,
    summarize,
)


class ISAIDSequentialRoleViewTest(unittest.TestCase):

    def test_resume_starts_at_first_missing_and_allows_overlap(self):
        images = [f'/data/tile_{index}.png' for index in range(6)]
        result = plan_resume(
            images, {images[0], images[1], images[2], images[4]})
        self.assertEqual(result['first_missing_index'], 3)
        self.assertEqual(result['resume_tail_images'], 3)
        self.assertEqual(result['dataset_indices'], -3)
        self.assertEqual(result['expected_overlap'], 1)

    def test_global_only_role_summary_selects_best_non_anchor(self):
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
            execution='role_only', global_source='official_observation',
            reference_endpoint='global', role_candidates=candidates,
            anchor_candidate='anchor', current_role_candidate='anchor',
            prior_view_operator='global', prior_view_miou=None,
            reference_identity_max_abs=0.0,
            native_prompt_parity_max_abs=0.0,
            operator_specs=operator_specs, variants=variants,
            complementarity={},
        )
        self.assertEqual(expected_variants(payload), tuple(variants))
        result = summarize([dict(
            dataset_name='isaid', img_path='tile.png',
            class_names=['background', 'target'],
            joint_role_view=payload)])
        profiles = {row['profile']: row for row in result['profiles']}
        self.assertEqual(profiles['role_only_best']['candidate_id'], 'selected')
        self.assertGreater(
            profiles['role_only_best']['miou'], profiles['official']['miou'])


if __name__ == '__main__':
    unittest.main()
