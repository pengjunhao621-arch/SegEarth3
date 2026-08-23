import os
import unittest

from role_functional_text_definitions import (
    load_joint_role_view_final_registry,
)
from tools.summarize_joint_role_view_final import (
    summarize_accuracy,
    summarize_efficiency,
)

try:
    import torch
    from role_visual_field import RoleVisualFieldMixin
except ModuleNotFoundError:
    torch = None
    RoleVisualFieldMixin = None


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(
    ROOT, 'configs', 'experiments', 'joint_role_view_final_v1.json')
DATASETS = ('udd5', 'vdd', 'vaihingen', 'potsdam', 'openearthmap', 'loveda')


class JointRoleViewFinalTest(unittest.TestCase):

    def test_registry_and_compiled_candidate_subset(self):
        for dataset in DATASETS:
            value = load_joint_role_view_final_registry(REGISTRY, dataset)
            self.assertEqual(value['primary_profile'], 'joint_residual')
            self.assertIn('joint_direct', value['profiles'])
            self.assertIn('fast_joint_residual', value['profiles'])
            if RoleVisualFieldMixin is not None:
                mixin = object.__new__(RoleVisualFieldMixin)
                mixin._jrv_final_config = value
                mixin.role_prompt_tta_final_profile = 'joint_residual'
                candidates = mixin._jrv_final_candidates()
                self.assertLessEqual(len(candidates), 2)
                self.assertEqual(candidates[0]['id'], 'anchor')

    @unittest.skipIf(torch is None, 'PyTorch is only available on the server.')
    def test_view_operators_and_reference(self):
        local = torch.tensor([[[0.2, 0.8]]])
        global_value = torch.tensor([[[0.6, 0.4]]])
        endpoints = dict(local=local, global_value=global_value)
        mixin = object.__new__(RoleVisualFieldMixin)
        self.assertTrue(torch.equal(
            mixin._jrv_final_fuse(endpoints, 'reference', 'local'), local))
        self.assertTrue(torch.equal(
            mixin._jrv_final_fuse(endpoints, 'max', 'global'),
            torch.maximum(local, global_value)))

    def test_accuracy_and_efficiency_summary(self):
        official = dict(
            candidate='anchor', update='residual', operator='reference',
            confusion=[[2, 1], [1, 2]], changed_pixels=0,
            improved_pixels=0, harmed_pixels=0,
            foreground_added_pixels=0, foreground_removed_pixels=0,
            foreground_class_switch_pixels=0)
        profile = dict(
            candidate='full', update='residual', operator='max',
            confusion=[[3, 0], [1, 2]], changed_pixels=1,
            improved_pixels=1, harmed_pixels=0,
            foreground_added_pixels=0, foreground_removed_pixels=0,
            foreground_class_switch_pixels=1)
        records = [dict(
            dataset_name='toy', img_path='image.png',
            class_names=['a', 'b'],
            joint_role_view_final=dict(
                official=official,
                profiles={'joint_residual': profile}))]
        profiles, classes, duplicates = summarize_accuracy(records)
        self.assertEqual(len(profiles), 2)
        self.assertEqual(len(classes), 4)
        self.assertEqual(duplicates, {})
        efficiency = summarize_efficiency([
            dict(dataset='toy', profile='joint_residual', warmup=True,
                 latency_seconds=9.0, peak_allocated_bytes=1,
                 peak_reserved_bytes=2),
            dict(dataset='toy', profile='joint_residual', warmup=False,
                 latency_seconds=2.0, peak_allocated_bytes=3,
                 peak_reserved_bytes=4, image_encoder_calls=2,
                 grounding_calls=8),
        ])
        self.assertEqual(efficiency[0]['measured_iterations'], 1)
        self.assertEqual(efficiency[0]['latency_mean_seconds'], 2.0)


if __name__ == '__main__':
    unittest.main()
