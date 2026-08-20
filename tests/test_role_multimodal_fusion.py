import unittest

import torch

from role_functional_text_definitions import (
    ROLE_MULTIMODAL_FUSION_VARIANT_NAMES,
)
from role_multimodal_fusion import RoleMultimodalFusionMixin


class RoleMultimodalFusionTest(unittest.TestCase):

    def test_variant_contract_excludes_external_exemplars(self):
        joined = ' '.join(ROLE_MULTIMODAL_FUSION_VARIANT_NAMES).lower()
        self.assertNotIn('v3', joined)
        self.assertNotIn('exemplar', joined)
        self.assertNotIn('support', joined)
        for family in ('v2', 'v4', 'v5', 'v6', 'v7', 'v8'):
            self.assertIn(family, joined)

    def test_full_roi_alignment_preserves_toy_memory(self):
        memory = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
        aligned = RoleMultimodalFusionMixin._rmf_align_memory(
            memory, ((2, 2),), (0, 0, 20, 20), (20, 20), ((2, 2),))
        self.assertTrue(torch.allclose(memory, aligned))

    def test_operator_outputs_are_finite_and_shape_preserving(self):
        fine = torch.randn(4, 1, 8)
        context = torch.randn(4, 1, 8)
        for method in ('v4_transport', 'v6_qk', 'v7_value', 'v8_full'):
            value, stats = RoleMultimodalFusionMixin._rmf_operator(
                method, fine, context, ((2, 2),))
            self.assertEqual(tuple(value.shape), tuple(fine.shape))
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreaterEqual(
                stats['normalized_attention_entropy'], 0.0)
            self.assertLessEqual(
                stats['normalized_attention_entropy'], 1.0 + 1e-5)

    def test_zero_role_residual_makes_agreement_identity(self):
        fine = torch.randn(4, 1, 8)
        context = torch.randn(4, 1, 8)
        value, stats = RoleMultimodalFusionMixin._rmf_agreement_operator(
            fine, context, fine.clone(), context.clone())
        self.assertTrue(torch.allclose(value, fine))
        self.assertEqual(stats['positive_agreement_fraction'], 0.0)


if __name__ == '__main__':
    unittest.main()
