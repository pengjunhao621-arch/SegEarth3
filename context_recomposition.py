"""Training-free R0--R14 Context recomposition screen.

The protected Native Fine/Role-Text path remains the prediction base.  Extra
views are used only to form matched counterfactual residuals; no external
images, labels, trainable parameters, or dataset-specific weights are used.
"""

import os
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from role_functional_text_definitions import (
    CONTEXT_RECOMPOSITION_BLEND,
    CONTEXT_RECOMPOSITION_LOGIT_CLIP,
    CONTEXT_RECOMPOSITION_LOWPASS_DIVISOR,
    CONTEXT_RECOMPOSITION_NEAR_RATIO,
    CONTEXT_RECOMPOSITION_PROB_CLIP,
    CONTEXT_RECOMPOSITION_PROTOCOL,
    CONTEXT_RECOMPOSITION_SCHEMA_VERSION,
    CONTEXT_RECOMPOSITION_VARIANT_NAMES,
    CONTEXT_RECOMPOSITION_VARIANT_SPECS,
)


CR_PROTOCOL = CONTEXT_RECOMPOSITION_PROTOCOL
CR_SCHEMA_VERSION = CONTEXT_RECOMPOSITION_SCHEMA_VERSION
CR_BLEND = CONTEXT_RECOMPOSITION_BLEND
CR_PROB_CLIP = CONTEXT_RECOMPOSITION_PROB_CLIP
CR_LOGIT_CLIP = CONTEXT_RECOMPOSITION_LOGIT_CLIP
CR_NEAR_RATIO = CONTEXT_RECOMPOSITION_NEAR_RATIO
CR_LOWPASS_DIVISOR = CONTEXT_RECOMPOSITION_LOWPASS_DIVISOR
CR_VARIANTS = CONTEXT_RECOMPOSITION_VARIANT_NAMES
CR_VARIANT_SPECS = CONTEXT_RECOMPOSITION_VARIANT_SPECS


class ContextRecompositionMixin:
    """Compose R0--R14 from a small shared bank of frozen SAM3 views."""

    @staticmethod
    def _cr_clip_box(box, size):
        width, height = size
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return x1, y1, x2, y2

    @classmethod
    def _cr_expand_box(cls, box, size, ratio=CR_NEAR_RATIO):
        x1, y1, x2, y2 = cls._cr_clip_box(box, size)
        width, height = x2 - x1, y2 - y1
        return cls._cr_clip_box((
            x1 - ratio * width,
            y1 - ratio * height,
            x2 + ratio * width,
            y2 + ratio * height,
        ), size)

    @staticmethod
    def _cr_fill_box(image, box, value=(128, 128, 128)):
        result = image.copy()
        ImageDraw.Draw(result).rectangle(
            (box[0], box[1], box[2] - 1, box[3] - 1), fill=value)
        return result

    @classmethod
    def _cr_context_view(cls, context, target_roi, mode):
        """Build full, ring, near-ring, or geometry-permuted Context."""
        roi = cls._cr_clip_box(target_roi, context.size)
        neutral = (128, 128, 128)
        if mode == 'full':
            return context.copy()
        if mode == 'ring':
            return cls._cr_fill_box(context, roi, neutral)
        if mode == 'near':
            expanded = cls._cr_expand_box(roi, context.size)
            result = Image.new('RGB', context.size, neutral)
            result.paste(context.crop(expanded), expanded[:2])
            return cls._cr_fill_box(result, roi, neutral)
        if mode != 'permuted':
            raise ValueError(f'Unknown Context mode: {mode}.')

        # Move coarse non-target blocks while preserving their local texture.
        # This is a deterministic counterfactual, not a candidate natural view.
        source = cls._cr_fill_box(context, roi, neutral)
        grid = 4
        xs = [int(round(index * context.width / grid))
              for index in range(grid + 1)]
        ys = [int(round(index * context.height / grid))
              for index in range(grid + 1)]
        boxes = [
            (xs[column], ys[row], xs[column + 1], ys[row + 1])
            for row in range(grid) for column in range(grid)
        ]
        crops = [source.crop(box) for box in boxes]
        movable = [
            index for index, box in enumerate(boxes)
            if (box[2] <= roi[0] or box[0] >= roi[2]
                or box[3] <= roi[1] or box[1] >= roi[3])
        ]
        shift = max(1, len(movable) // 2) if len(movable) > 1 else 0
        sources = {
            destination: movable[(position + shift) % len(movable)]
            for position, destination in enumerate(movable)
        }
        result = Image.new('RGB', context.size, neutral)
        for index, destination in enumerate(boxes):
            crop = crops[sources.get(index, index)]
            size = (destination[2] - destination[0],
                    destination[3] - destination[1])
            if crop.size != size:
                resampling = getattr(Image, 'Resampling', Image).BILINEAR
                crop = crop.resize(size, resampling)
            result.paste(crop, destination[:2])
        return cls._cr_fill_box(result, roi, neutral)

    @classmethod
    def _cr_foveated_canvas(
            cls, fine, context, target_roi, context_mode,
            resolution=1008):
        """Place Fine once at its geospatial ROI inside a Context canvas."""
        if context_mode == 'blank':
            base = Image.new('RGB', context.size, (128, 128, 128))
        else:
            base = cls._cr_context_view(
                context, target_roi, context_mode)
        resampling = getattr(Image, 'Resampling', Image).BILINEAR
        canvas = base.resize((resolution, resolution), resampling)
        x1, y1, x2, y2 = cls._cr_clip_box(target_roi, context.size)
        roi = cls._cr_clip_box((
            x1 * resolution / context.width,
            y1 * resolution / context.height,
            x2 * resolution / context.width,
            y2 * resolution / context.height,
        ), (resolution, resolution))
        fine_resized = fine.resize(
            (roi[2] - roi[0], roi[3] - roi[1]), resampling)
        canvas.paste(fine_resized, roi[:2])
        return canvas, roi

    @staticmethod
    def _cr_lowpass(value, divisor=CR_LOWPASS_DIVISOR):
        value = torch.as_tensor(value).float()
        height, width = value.shape[-2:]
        pooled = F.adaptive_avg_pool2d(
            value.unsqueeze(0),
            (max(1, height // int(divisor)),
             max(1, width // int(divisor))))
        return F.interpolate(
            pooled, size=(height, width), mode='bilinear',
            align_corners=False).squeeze(0)

    @staticmethod
    def _cr_prior_spatial(value):
        value = torch.as_tensor(value).float()
        prior = value.mean(dim=(-2, -1), keepdim=True)
        return prior.expand_as(value), value - prior

    @staticmethod
    def _cr_logit(value, eps=1e-4):
        value = torch.as_tensor(value).float().clamp(eps, 1.0 - eps)
        return torch.log(value) - torch.log1p(-value)

    @staticmethod
    def _cr_residual_stats(value):
        value = torch.as_tensor(value).float()
        return dict(
            mean_abs_residual=float(value.abs().mean().item()),
            positive_fraction=float((value > 0).float().mean().item()),
            negative_fraction=float((value < 0).float().mean().item()),
            mean_signed_residual=float(value.mean().item()),
        )

    def _cr_ground_outputs(self, image, roi, output_shape):
        view = self._rmf_compact_view(self._rvf_ground_view(image))
        outputs = self._rmf_view_outputs(view, roi, output_shape)
        return outputs

    def _cr_process_unit(self, unit):
        output_shape = (unit['fine'].height, unit['fine'].width)
        native_view = self._rmf_compact_view(
            self._rvf_ground_view(unit['fine']))
        native_query = self._rmf_resize(
            self._rvf_crop(
                native_view['query_final'], unit['fine_roi']),
            output_shape)
        native = self._rmf_view_outputs(
            native_view, unit['fine_roi'], output_shape)
        del native_view

        full_context = self._cr_context_view(
            unit['context'], unit['context_roi'], 'full')
        ring_context = self._cr_context_view(
            unit['context'], unit['context_roi'], 'ring')
        near_context = self._cr_context_view(
            unit['context'], unit['context_roi'], 'near')

        side_images = {}
        side_images['blank'], side_roi = self._rmf_canvas(unit['fine'])
        side_images['full'], _ = self._rmf_canvas(
            unit['fine'], full_context)
        side_images['ring'], _ = self._rmf_canvas(
            unit['fine'], ring_context)
        side_images['near'], _ = self._rmf_canvas(
            unit['fine'], near_context)
        side = {
            name: self._cr_ground_outputs(image, side_roi, output_shape)
            for name, image in side_images.items()
        }

        foveated_images = {}
        foveated_rois = {}
        for name, mode in (
                ('blank', 'blank'), ('ring', 'ring'),
                ('permuted', 'permuted')):
            image, roi = self._cr_foveated_canvas(
                unit['fine'], unit['context'], unit['context_roi'], mode)
            foveated_images[name], foveated_rois[name] = image, roi
        foveated = {
            name: self._cr_ground_outputs(
                image, foveated_rois[name], output_shape)
            for name, image in foveated_images.items()
        }

        side_anchor = side['full']['anchor'] - side['blank']['anchor']
        side_s = side['full']['s'] - side['blank']['s']
        side_i = side['full']['i'] - side['blank']['i']
        side_psi = side['full']['psi'] - side['blank']['psi']
        side_did_s = side_s - side_anchor
        side_did_i = side_i - side_anchor
        side_did_psi = side_psi - side_anchor

        foveated_anchor = (
            foveated['ring']['anchor'] - foveated['blank']['anchor'])
        foveated_did_s = (
            foveated['ring']['s'] - foveated['blank']['s']
            - foveated_anchor)
        foveated_did_i = (
            foveated['ring']['i'] - foveated['blank']['i']
            - foveated_anchor)
        foveated_did_psi = (
            foveated['ring']['psi'] - foveated['blank']['psi']
            - foveated_anchor)

        prior, spatial = self._cr_prior_spatial(side_did_psi)
        probability_residuals = OrderedDict((
            ('cr_r1_final_residual', side_psi),
            ('cr_r2_semantic_residual', side_s),
            ('cr_r3_instance_residual', side_i),
            ('cr_r4_si_native_presence', side_did_s + side_did_i),
            ('cr_r5_lowfreq_semantic', self._cr_lowpass(side_did_s)),
            ('cr_r6_foveated_si', foveated_did_s + foveated_did_i),
            ('cr_r7_foveated_ring', foveated_did_psi),
            ('cr_r8_role_context_did', side_did_psi),
            ('cr_r9_surround_only',
             side['ring']['psi'] - side['blank']['psi']),
            ('cr_r9_target_scale_only',
             side['full']['psi'] - side['ring']['psi']),
            ('cr_r10_near_only',
             side['near']['psi'] - side['blank']['psi']),
            ('cr_r10_far_only',
             side['ring']['psi'] - side['near']['psi']),
            ('cr_r11_prior_only', prior),
            ('cr_r11_spatial_only', spatial),
            ('cr_r12_add_only', side_did_psi.clamp_min(0.0)),
            ('cr_r12_suppress_only', side_did_psi.clamp_max(0.0)),
            ('cr_r13_geometry_only',
             foveated['ring']['psi'] - foveated['permuted']['psi']),
            ('cr_r13_scene_only',
             foveated['permuted']['psi'] - foveated['blank']['psi']),
        ))

        log_side = (
            self._cr_logit(side['full']['psi'])
            - self._cr_logit(side['blank']['psi']))
        log_anchor = (
            self._cr_logit(side['full']['anchor'])
            - self._cr_logit(side['blank']['anchor']))
        logit_residuals = OrderedDict((
            ('cr_r14_logodds', log_side),
            ('cr_r14_logodds_did', log_side - log_anchor),
        ))
        expected_probability = set(CR_VARIANTS) - {
            'cr_official', 'cr_r0_native_role_text',
            *logit_residuals.keys(),
        }
        if set(probability_residuals) != expected_probability:
            raise RuntimeError('R1--R13 probability contract drifted.')

        stats = {
            name: self._cr_residual_stats(value)
            for name, value in (
                list(probability_residuals.items())
                + list(logit_residuals.items()))
        }
        metadata = dict(
            target_box=list(unit['target_box']),
            fine_size=[unit['fine'].width, unit['fine'].height],
            context_size=[unit['context'].width, unit['context'].height],
            context_roi=list(unit['context_roi']),
            side_roi=list(side_roi),
            foveated_roi=list(foveated_rois['ring']),
            source_prefix=unit['metadata']['source_prefix'],
            source_canvas=unit['metadata']['source_canvas'],
            context_box=unit['metadata']['context_box'],
            contributor_tiles=unit['metadata']['contributor_tiles'],
        )
        return native_query, native, probability_residuals, logit_residuals, \
            stats, metadata

    @staticmethod
    def _cr_apply_probability(reference, residual):
        return (
            torch.as_tensor(reference).float()
            + float(CR_BLEND) * torch.as_tensor(residual).float().clamp(
                -float(CR_PROB_CLIP), float(CR_PROB_CLIP))
        ).clamp(0.0, 1.0)

    @classmethod
    def _cr_apply_logit(cls, reference, residual):
        value = (
            cls._cr_logit(reference)
            + float(CR_BLEND) * torch.as_tensor(residual).float().clamp(
                -float(CR_LOGIT_CLIP), float(CR_LOGIT_CLIP)))
        return torch.sigmoid(value)

    def _cr_predict_image(self, image, image_path):
        height, width = image.height, image.width
        counts = torch.zeros((1, height, width), dtype=torch.float32)
        query_sum = torch.zeros(
            (self.num_queries, height, width), dtype=torch.float32)
        residual_sum = OrderedDict(
            (name, torch.zeros(
                (self.num_cls, height, width), dtype=torch.float32))
            for name in CR_VARIANTS
            if name not in ('cr_official', 'cr_r0_native_role_text'))
        native_text_sum = torch.zeros(
            (self.num_cls, height, width), dtype=torch.float32)
        unit_rows, stats_sum, stats_count = [], defaultdict(
            lambda: defaultdict(float)), defaultdict(int)

        for unit_index, unit in enumerate(
                self._rvf_visual_units(image, image_path)):
            (query, native, probability, logit,
             stats, unit_metadata) = self._cr_process_unit(unit)
            x1, y1, x2, y2 = unit['target_box']
            query_sum[:, y1:y2, x1:x2] += query
            native_text_sum[:, y1:y2, x1:x2] += native['psi']
            counts[:, y1:y2, x1:x2] += 1.0
            for name, value in list(probability.items()) + list(logit.items()):
                residual_sum[name][:, y1:y2, x1:x2] += value
            for name, row in stats.items():
                stats_count[name] += 1
                for field, value in row.items():
                    stats_sum[name][field] += float(value)
            unit_rows.append(dict(unit_index=unit_index, **unit_metadata))

        if torch.any(counts == 0):
            raise RuntimeError('Context recomposition left uncovered pixels.')
        query_mean = query_sum / counts
        native_text = native_text_sum / counts
        residual_mean = {
            name: value / counts for name, value in residual_sum.items()}

        if self._rvf_fine_matches_official(image):
            official_query = query_mean
            official = self._aggregate_query_logits_to_classes(query_mean)
            best_text = native_text
            reused_fine = True
        else:
            official_query, official, best_text = (
                self._rvf_accumulate_official_units(image))
            reused_fine = False

        variants = OrderedDict()
        variants['cr_official'] = official
        variants['cr_r0_native_role_text'] = best_text
        for name in CR_VARIANTS[2:]:
            if name.startswith('cr_r14_'):
                variants[name] = self._cr_apply_logit(
                    best_text, residual_mean[name])
            else:
                variants[name] = self._cr_apply_probability(
                    best_text, residual_mean[name])
        if tuple(variants) != CR_VARIANTS:
            raise RuntimeError('R0--R14 variant order drifted.')

        identity_error = max(
            float((variants['cr_official'] - official).abs().max()),
            float((variants['cr_r0_native_role_text']
                   - best_text).abs().max()))
        if (self.role_prompt_tta_strict_integrity
                and identity_error
                > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                'Context recomposition protected endpoint drifted: '
                f'{identity_error}.')

        mechanism = {
            name: {
                field: value / max(stats_count[name], 1)
                for field, value in fields.items()
            }
            for name, fields in stats_sum.items()
        }
        slots = self._rpt_role_selection['_best_overall_slots']
        metadata = dict(
            schema_version=CR_SCHEMA_VERSION,
            protocol=CR_PROTOCOL,
            variants=list(CR_VARIANTS),
            variant_specs=[list(value) for value in CR_VARIANT_SPECS],
            blend=float(CR_BLEND),
            probability_clip=float(CR_PROB_CLIP),
            logit_clip=float(CR_LOGIT_CLIP),
            near_ratio=float(CR_NEAR_RATIO),
            lowpass_divisor=int(CR_LOWPASS_DIVISOR),
            source_mode=self._rvf_config['source_mode'],
            fine_size=int(self._rvf_config['fine_size']),
            context_size=int(self._rvf_config['context_size']),
            selected_slots=list(slots),
            selected_admission=self._rpt_role_selection[
                'best_overall']['admission'],
            official_reused_fine=bool(reused_fine),
            protected_identity_max_abs=identity_error,
            excluded_methods=[
                'external_exemplar', 'retrieval_bank', 'training',
                'dataset_specific_weight', 'confidence_gate'],
            mechanism_summaries=mechanism,
            units=unit_rows,
        )
        return official_query.to(self.device), variants, metadata

    def _cr_record_image(
            self, variants, metadata, data_sample, image_path):
        if not self.dump_role_prompt_tta_stats:
            return
        gt = data_sample.gt_sem_seg.data.squeeze().detach().long().cpu()
        valid = gt != 255
        predictions = {
            name: self._rpt_threshold(value).to(torch.int16)
            for name, value in variants.items()
        }
        rows = OrderedDict()
        for name in CR_VARIANTS:
            if name == 'cr_official':
                reference_name = 'cr_official'
            elif name == 'cr_r0_native_role_text':
                reference_name = 'cr_official'
            else:
                reference_name = 'cr_r0_native_role_text'
            prediction = predictions[name]
            reference = predictions[reference_name]
            changed = valid & (prediction != reference)
            improved = changed & (prediction == gt) & (reference != gt)
            harmed = changed & (prediction != gt) & (reference == gt)
            reference_fg = reference != self.bg_idx
            prediction_fg = prediction != self.bg_idx
            oracle = reference.clone()
            oracle[improved] = prediction[improved]
            rows[name] = dict(
                reference_variant=reference_name,
                confusion=self._rpt_confusion(prediction, gt, valid),
                oracle_confusion=self._rpt_confusion(oracle, gt, valid),
                changed_pixels=int(changed.sum()),
                improved_pixels=int(improved.sum()),
                harmed_pixels=int(harmed.sum()),
                help_minus_harm=int(improved.sum()) - int(harmed.sum()),
                background_to_foreground_pixels=int((
                    changed & ~reference_fg & prediction_fg).sum()),
                foreground_to_background_pixels=int((
                    changed & reference_fg & ~prediction_fg).sum()),
                foreground_to_foreground_pixels=int((
                    changed & reference_fg & prediction_fg).sum()),
            )
        record = dict(
            schema_version=CR_SCHEMA_VERSION,
            rank=int(os.environ.get('RANK', 0)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            dataset_name=self.role_prompt_tta_dataset_name,
            img_path=image_path,
            class_names=list(self.class_names),
            valid_pixels=int(valid.sum()),
            prob_thd=float(self.prob_thd),
            context_recomposition=dict(metadata, variants=rows),
        )
        self._rpt_write_stats(record)
