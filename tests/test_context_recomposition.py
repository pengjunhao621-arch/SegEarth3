import unittest

import torch
from PIL import Image

from context_recomposition import ContextRecompositionMixin
from role_functional_text_definitions import (
    CONTEXT_RECOMPOSITION_VARIANT_SPECS,
)


class ContextRecompositionTest(unittest.TestCase):

    def test_variant_contract_covers_r0_through_r14(self):
        research_ids = {value[1] for value in CONTEXT_RECOMPOSITION_VARIANT_SPECS}
        for index in range(15):
            self.assertIn(f'r{index}', research_ids)
        joined = ' '.join(value[0] for value in CONTEXT_RECOMPOSITION_VARIANT_SPECS)
        self.assertNotIn('exemplar', joined)
        self.assertNotIn('retrieval', joined)

    def test_prior_and_spatial_reconstruct_residual(self):
        value = torch.randn(3, 7, 9)
        prior, spatial = ContextRecompositionMixin._cr_prior_spatial(value)
        self.assertTrue(torch.allclose(prior + spatial, value, atol=1e-6))
        self.assertTrue(torch.allclose(
            spatial.mean(dim=(-2, -1)), torch.zeros(3), atol=1e-6))

    def test_probability_and_logit_zero_residual_are_identity(self):
        value = torch.rand(3, 7, 9) * 0.8 + 0.1
        probability = ContextRecompositionMixin._cr_apply_probability(
            value, torch.zeros_like(value))
        logit = ContextRecompositionMixin._cr_apply_logit(
            value, torch.zeros_like(value))
        self.assertTrue(torch.allclose(probability, value, atol=1e-6))
        self.assertTrue(torch.allclose(logit, value, atol=1e-6))

    def test_context_ring_masks_target_and_near_masks_far_area(self):
        context = Image.new('RGB', (16, 16), (255, 0, 0))
        roi = (4, 4, 12, 12)
        ring = ContextRecompositionMixin._cr_context_view(
            context, roi, 'ring')
        near = ContextRecompositionMixin._cr_context_view(
            context, roi, 'near')
        self.assertEqual(ring.getpixel((8, 8)), (128, 128, 128))
        self.assertEqual(ring.getpixel((1, 1)), (255, 0, 0))
        self.assertEqual(near.getpixel((8, 8)), (128, 128, 128))
        self.assertEqual(near.getpixel((1, 1)), (128, 128, 128))
        self.assertEqual(near.getpixel((3, 8)), (255, 0, 0))

    def test_foveated_canvas_contains_fine_once_at_target_roi(self):
        fine = Image.new('RGB', (8, 8), (0, 255, 0))
        context = Image.new('RGB', (16, 16), (255, 0, 0))
        canvas, roi = ContextRecompositionMixin._cr_foveated_canvas(
            fine, context, (4, 4, 12, 12), 'ring', resolution=32)
        self.assertEqual(canvas.size, (32, 32))
        self.assertEqual(roi, (8, 8, 24, 24))
        self.assertEqual(canvas.getpixel((16, 16)), (0, 255, 0))
        self.assertEqual(canvas.getpixel((2, 2)), (255, 0, 0))

    def test_geometry_permutation_preserves_non_target_block_content(self):
        context = Image.new('RGB', (16, 16), (0, 0, 0))
        pixels = context.load()
        for y in range(16):
            for x in range(16):
                pixels[x, y] = (x // 4 * 40, y // 4 * 40, 0)
        roi = (4, 4, 12, 12)
        ring = ContextRecompositionMixin._cr_context_view(
            context, roi, 'ring')
        permuted = ContextRecompositionMixin._cr_context_view(
            context, roi, 'permuted')
        outside = [
            (x, y) for y in range(16) for x in range(16)
            if not (roi[0] <= x < roi[2] and roi[1] <= y < roi[3])
        ]
        self.assertEqual(
            sorted(ring.getpixel(point) for point in outside),
            sorted(permuted.getpixel(point) for point in outside))
        self.assertTrue(any(
            ring.getpixel(point) != permuted.getpixel(point)
            for point in outside))

    def test_lowpass_is_shape_preserving_and_finite(self):
        value = torch.randn(4, 31, 29)
        output = ContextRecompositionMixin._cr_lowpass(value)
        self.assertEqual(tuple(output.shape), tuple(value.shape))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
