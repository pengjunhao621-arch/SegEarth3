"""Same-forward semantic/instance fusion formula bank for SegEarth-OV3.

The module is diagnostic-only.  It reuses one set of SAM3 decoder outputs to
evaluate predeclared fusion graphs and never changes the returned baseline
prediction.  Ground truth is consumed only after inference to measure exact
pixel transitions and confusion matrices.
"""

import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from dual_head_fusion_definitions import (
    CONTRASTS,
    SCHEMA_VERSION,
    VARIANT_NAMES,
)


def _ranked_jsonl_path(path):
    rank = int(os.environ.get('RANK', 0))
    root, ext = os.path.splitext(path)
    return f'{root}.rank{rank}{ext or ".jsonl"}'


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return _finite(float(numerator) / float(denominator))


def _masked_mean(value, mask):
    if not isinstance(value, torch.Tensor):
        return None
    if not isinstance(mask, torch.Tensor) or not bool(mask.any()):
        return None
    return _finite(value[mask].detach().float().mean().item())


class DualHeadFusionDiagnosticMixin:
    """Build and evaluate fixed fusion graphs from one decoder output."""

    def _uses_dual_head_fusion_diagnostic(self):
        return bool(self.dump_dual_head_fusion_stats)

    def _dhf_validate_runtime_contract(self):
        errors = []
        if not bool(self.use_sem_seg):
            errors.append('use_sem_seg must be True')
        if not bool(self.use_transformer_decoder):
            errors.append('use_transformer_decoder must be True')
        if not bool(self.use_presence_score):
            errors.append('use_presence_score must be True')
        if str(self.instance_score_type).lower() != 'presence':
            errors.append("instance_score_type must be 'presence'")
        if bool(self.use_reject_aware_calibration):
            errors.append('use_reject_aware_calibration must be False')
        if errors:
            raise RuntimeError(
                'Dual-head fusion bank requires the unchanged SegEarth-OV3 '
                'baseline contract: ' + '; '.join(errors))

    def _dhf_aggregate_query_logits_to_classes(self, query_logits):
        """Exact prompt max without a class-by-prompt broadcast tensor."""
        if int(self.num_cls) == int(self.num_queries):
            return query_logits
        query_to_class = [
            int(value) for value in self.query_idx.detach().cpu().tolist()
        ]
        class_maps = []
        for class_index in range(int(self.num_cls)):
            prompt_indices = [
                index
                for index, mapped_class in enumerate(query_to_class)
                if mapped_class == class_index
            ]
            if not prompt_indices:
                raise RuntimeError(
                    'Dual-head fusion query mapping has no prompt for class '
                    f'{class_index}.')
            class_maps.append(
                query_logits[prompt_indices].max(dim=0)[0])
        return torch.stack(class_maps, dim=0)

    @staticmethod
    def _dhf_area(values, threshold):
        return _safe_div(
            int((values.detach().float() >= float(threshold)).sum().item()),
            int(values.numel()),
        )

    @staticmethod
    def _dhf_update_winner(
            current_value, current_mask, candidate_value, candidate_mask):
        replace = candidate_value > current_value
        return (
            torch.where(replace, candidate_value, current_value),
            torch.where(replace, candidate_mask, current_mask),
        )

    def _dhf_reduce_native_queries(
            self, inference_state, semantic_map, output_shape):
        raw_masks = inference_state.get('raw_masks_logits_lowres')
        raw_scores = inference_state.get('raw_object_score')
        keep_mask = inference_state.get('raw_keep_mask')
        if (
                not isinstance(raw_masks, torch.Tensor)
                or not isinstance(raw_scores, torch.Tensor)
                or not isinstance(keep_mask, torch.Tensor)):
            raise RuntimeError(
                'Dual-head fusion bank requires raw SAM3 masks, raw object '
                'scores, and the native keep mask.')

        count = min(
            int(raw_masks.shape[0]),
            int(raw_scores.numel()),
            int(keep_mask.numel()),
        )
        raw_masks = raw_masks[:count]
        raw_scores = raw_scores[:count].detach().float().clamp(0.0, 1.0)
        keep_mask = keep_mask[:count].detach().bool()
        raw_score_max_all = (
            float(raw_scores.max().item())
            if raw_scores.numel() > 0 else 0.0)
        selected = torch.nonzero(
            keep_mask, as_tuple=False).flatten()

        device = semantic_map.device
        zeros = torch.zeros(
            output_shape, device=device, dtype=torch.float32)
        object_instance = zeros.clone()
        object_winner_mask = zeros.clone()
        object_winner_contrast = zeros.clone()
        raw_instance = zeros.clone()
        corrected_instance = zeros.clone()
        corrected_winner_mask = zeros.clone()
        representative_mask = zeros.clone()

        if selected.numel() == 0:
            return dict(
                raw_candidate_count=count,
                native_keep_count=0,
                object_instance=object_instance,
                object_winner_mask=object_winner_mask,
                object_winner_contrast=object_winner_contrast,
                raw_instance=raw_instance,
                corrected_instance=corrected_instance,
                corrected_winner_mask=corrected_winner_mask,
                representative_mask=representative_mask,
                representative_score=0.0,
                raw_score_max_all=raw_score_max_all,
                selected_scores=raw_scores.new_empty((0,)),
                selected_presence_scores=raw_scores.new_empty((0,)),
                semantic_contrast=raw_scores.new_empty((0,)),
                corrected_scores=raw_scores.new_empty((0,)),
                selected_indices=selected,
                winner_switch_fraction=0.0,
            )

        presence = inference_state['presence_score'].detach().float().reshape(())
        selected_scores = raw_scores[selected]
        selected_presence_scores = selected_scores * presence
        representative_local = int(selected_scores.argmax().item())
        representative_index = int(selected[representative_local].item())
        representative_score = float(
            selected_scores[representative_local].item())

        semantic = semantic_map.detach().float()
        semantic_flat = semantic.reshape(1, -1)
        contrast_values = []
        corrected_values = []
        chunk_size = max(1, int(self.dual_head_fusion_chunk_size))
        ring_kernel = max(1, int(self.dual_head_fusion_ring_kernel))
        if ring_kernel % 2 == 0:
            ring_kernel += 1
        padding = ring_kernel // 2
        eps = float(self.dual_head_fusion_eps)
        eta = float(self.dual_head_fusion_eta)

        for start in range(0, int(selected.numel()), chunk_size):
            indices = selected[start:start + chunk_size]
            masks = raw_masks[indices].detach().float().unsqueeze(1)
            masks = torch.sigmoid(self._interpolate_float32(
                masks, output_shape)).squeeze(1)
            scores = raw_scores[indices].view(-1, 1, 1)

            object_values = masks * scores
            chunk_object_value, chunk_object_index = (
                object_values.max(dim=0))
            chunk_object_mask = torch.gather(
                masks,
                0,
                chunk_object_index.unsqueeze(0),
            ).squeeze(0)
            object_replace = chunk_object_value > object_instance
            object_instance, object_winner_mask = self._dhf_update_winner(
                object_instance,
                object_winner_mask,
                chunk_object_value,
                chunk_object_mask,
            )

            chunk_raw_value = masks.max(dim=0)[0]
            raw_instance = torch.maximum(
                raw_instance, chunk_raw_value)

            flat_masks = masks.flatten(1)
            inside = (
                flat_masks * semantic_flat
            ).sum(dim=1) / flat_masks.sum(dim=1).clamp_min(eps)

            binary = masks >= float(
                self.dual_head_fusion_mask_threshold)
            if ring_kernel <= 1:
                dilated = binary
            else:
                padded = F.pad(
                    binary.float().unsqueeze(1),
                    (padding, padding, padding, padding),
                    value=0.0,
                )
                dilated = F.max_pool2d(
                    padded,
                    kernel_size=ring_kernel,
                    stride=1,
                    padding=0,
                ).squeeze(1) > 0
            ring = dilated & ~binary
            ring_flat = ring.flatten(1)
            ring_count = ring_flat.sum(dim=1)
            ring_mean = (
                ring_flat.float() * semantic_flat
            ).sum(dim=1) / ring_count.clamp_min(1).float()
            ring_mean = torch.where(
                ring_count > 0, ring_mean, inside)
            contrast = (
                (inside - ring_mean)
                / (inside + ring_mean + eps)
            ).clamp(-1.0, 1.0)
            corrected = (
                raw_scores[indices] * (1.0 + eta * contrast)
            ).clamp(0.0, 1.0)
            contrast_values.append(contrast)
            corrected_values.append(corrected)
            chunk_object_contrast = torch.gather(
                contrast.view(-1, 1, 1).expand_as(masks),
                0,
                chunk_object_index.unsqueeze(0),
            ).squeeze(0)
            object_winner_contrast = torch.where(
                object_replace,
                chunk_object_contrast,
                object_winner_contrast,
            )

            corrected_maps = masks * corrected.view(-1, 1, 1)
            chunk_corrected_value, chunk_corrected_index = (
                corrected_maps.max(dim=0))
            chunk_corrected_mask = torch.gather(
                masks,
                0,
                chunk_corrected_index.unsqueeze(0),
            ).squeeze(0)
            (
                corrected_instance,
                corrected_winner_mask,
            ) = self._dhf_update_winner(
                corrected_instance,
                corrected_winner_mask,
                chunk_corrected_value,
                chunk_corrected_mask,
            )

            representative_matches = torch.nonzero(
                indices == representative_index,
                as_tuple=False,
            ).flatten()
            if representative_matches.numel() > 0:
                representative_mask = masks[
                    int(representative_matches[0].item())]

        semantic_contrast = torch.cat(contrast_values)
        corrected_scores = torch.cat(corrected_values)
        support = torch.maximum(
            object_instance, corrected_instance) >= float(
                self.dual_head_fusion_support_threshold)
        winner_switch = (
            (object_winner_mask - corrected_winner_mask).abs()
            > float(self.dual_head_fusion_integrity_tolerance)
        ) & support

        return dict(
            raw_candidate_count=count,
            native_keep_count=int(selected.numel()),
            object_instance=object_instance,
            object_winner_mask=object_winner_mask,
            object_winner_contrast=object_winner_contrast,
            raw_instance=raw_instance,
            corrected_instance=corrected_instance,
            corrected_winner_mask=corrected_winner_mask,
            representative_mask=representative_mask,
            representative_score=representative_score,
            raw_score_max_all=raw_score_max_all,
            selected_scores=selected_scores,
            selected_presence_scores=selected_presence_scores,
            semantic_contrast=semantic_contrast,
            corrected_scores=corrected_scores,
            selected_indices=selected,
            winner_switch_fraction=_safe_div(
                int(winner_switch.sum().item()),
                int(support.sum().item()),
            ) or 0.0,
        )

    def _dhf_build_prompt_variants(
            self, inference_state, semantic_map, native_instance_map,
            baseline_map, output_shape):
        """Compute every fusion graph for one text prompt."""
        self._dhf_validate_runtime_contract()
        presence_tensor = inference_state.get('presence_score')
        if (
                not isinstance(presence_tensor, torch.Tensor)
                or presence_tensor.numel() != 1):
            raise RuntimeError(
                'Dual-head fusion bank expects one Presence scalar per prompt.')

        native_semantic = semantic_map.detach()
        native_instance = native_instance_map.detach()
        native_baseline = baseline_map.detach()
        native_presence = presence_tensor.detach().reshape(())
        p0_reconstructed_native = native_presence * torch.maximum(
            native_semantic, native_instance)

        semantic = native_semantic.float().clamp(0.0, 1.0)
        native_instance = native_instance.float().clamp(0.0, 1.0)
        baseline = native_baseline.float().clamp(0.0, 1.0)
        presence = native_presence.float().clamp(0.0, 1.0)
        reduced = self._dhf_reduce_native_queries(
            inference_state, semantic, output_shape)

        object_instance = reduced['object_instance']
        object_winner = reduced['object_winner_mask']
        object_winner_contrast = reduced['object_winner_contrast']
        raw_instance = reduced['raw_instance']
        corrected_instance = reduced['corrected_instance']
        corrected_winner = reduced['corrected_winner_mask']
        representative_mask = reduced['representative_mask']
        representative_score = semantic.new_tensor(
            reduced['representative_score'])
        raw_score_max_all = semantic.new_tensor(
            reduced['raw_score_max_all'])

        semantic_only = presence * semantic
        instance_p0_path = presence * native_instance
        p1_role_once = presence * torch.maximum(
            semantic, object_instance)
        winner_score_once = presence * torch.maximum(
            semantic, representative_score * representative_mask)

        proc_eps = 1e-4
        proc_base = torch.maximum(semantic, native_instance)
        proc_agreement = (
            1.0 - (semantic - proc_base).abs()).clamp(0.0, 1.0)
        proc_advantage = (proc_base - semantic).clamp_min(0.0)
        # ProC's public reducer defines max_instance_score from p*r, then
        # _pgrf_prompt_fusion multiplies it by p again before the final p.
        proc_max_instance_score = presence * raw_score_max_all
        proc_gate = (
            presence * proc_max_instance_score * proc_agreement
        ).clamp(0.0, 1.0)
        proc_pgrf = presence * (
            semantic
            + proc_gate * proc_advantage
        ).clamp(proc_eps, 1.0 - proc_eps)
        proc_gate_debiased = presence * (
            semantic
            + raw_score_max_all * proc_agreement * proc_advantage
        ).clamp(proc_eps, 1.0 - proc_eps)
        proc_role_base = torch.maximum(semantic, object_instance)
        proc_role_agreement = (
            1.0 - (semantic - proc_role_base).abs()
        ).clamp(0.0, 1.0)
        proc_role_advantage = (
            proc_role_base - semantic).clamp_min(0.0)
        proc_role_once = presence * (
            semantic
            + raw_score_max_all
            * proc_role_agreement
            * proc_role_advantage
        ).clamp(proc_eps, 1.0 - proc_eps)

        object_advantage = (object_instance - semantic).clamp_min(0.0)
        uni_floor_agreement = (
            1.0
            - float(self.dual_head_fusion_agreement_gamma)
            * (semantic - object_winner).abs()
        ).clamp(0.0, 1.0)
        uni_full_agreement = (
            1.0 - (semantic - object_winner).abs()).clamp(0.0, 1.0)
        uni_rcrf_floor = presence * (
            semantic + uni_floor_agreement * object_advantage)
        region_agreement = (
            1.0 + 0.5 * object_winner_contrast
        ).clamp(0.5, 1.0)
        uni_rcrf_region = presence * (
            semantic
            + uni_floor_agreement
            * region_agreement
            * object_advantage)
        uni_rcrf_unfloored = presence * (
            semantic + uni_full_agreement * object_advantage)

        corrected_agreement = (
            1.0
            - float(self.dual_head_fusion_agreement_gamma)
            * (semantic - corrected_winner).abs()
        ).clamp(0.0, 1.0)
        bi_s2i_only = presence * (
            semantic
            + corrected_agreement
            * (corrected_instance - semantic).clamp_min(0.0))

        beta = float(self.dual_head_fusion_beta)
        object_boundary = (
            4.0 * object_winner * (1.0 - object_winner)
        ).clamp(0.0, 1.0)
        corrected_boundary = (
            4.0 * corrected_winner * (1.0 - corrected_winner)
        ).clamp(0.0, 1.0)
        semantic_i2s = (
            semantic
            + beta * object_boundary * (object_winner - semantic)
        ).clamp(0.0, 1.0)
        semantic_full = (
            semantic
            + beta * corrected_boundary * (corrected_winner - semantic)
        ).clamp(0.0, 1.0)
        i2s_delta = semantic_i2s - semantic
        full_i2s_delta = semantic_full - semantic
        i2s_agreement = (
            1.0
            - float(self.dual_head_fusion_agreement_gamma)
            * (semantic_i2s - object_winner).abs()
        ).clamp(0.0, 1.0)
        full_agreement = (
            1.0
            - float(self.dual_head_fusion_agreement_gamma)
            * (semantic_full - corrected_winner).abs()
        ).clamp(0.0, 1.0)
        bi_i2s_only = presence * (
            semantic_i2s
            + i2s_agreement
            * (object_instance - semantic_i2s).clamp_min(0.0))
        bi_full = presence * (
            semantic_full
            + full_agreement
            * (corrected_instance - semantic_full).clamp_min(0.0))

        soft_or = presence * (
            1.0 - (1.0 - semantic) * (1.0 - object_instance))
        convex_25 = presence * (
            0.75 * semantic + 0.25 * object_instance)
        boundary_residual = presence * (
            semantic + object_boundary * object_advantage)
        interior_residual = presence * (
            semantic + (1.0 - object_boundary) * object_advantage)
        raw_mask_residual = presence * torch.maximum(
            semantic, raw_instance)

        variants = dict(
            semantic_only=semantic_only,
            instance_p0_path=instance_p0_path,
            p1_role_once=p1_role_once,
            winner_score_once=winner_score_once,
            proc_pgrf=proc_pgrf,
            proc_gate_debiased=proc_gate_debiased,
            proc_role_once=proc_role_once,
            uni_rcrf_floor=uni_rcrf_floor,
            uni_rcrf_region=uni_rcrf_region,
            uni_rcrf_unfloored=uni_rcrf_unfloored,
            bi_s2i_only=bi_s2i_only,
            bi_i2s_only=bi_i2s_only,
            bi_full=bi_full,
            soft_or=soft_or,
            convex_25=convex_25,
            boundary_residual=boundary_residual,
            interior_residual=interior_residual,
            raw_mask_residual=raw_mask_residual,
        )
        for name, values in variants.items():
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError(
                    f'Dual-head fusion variant {name!r} contains non-finite '
                    'values.')
            variants[name] = values.clamp(0.0, 1.0)

        scores = reduced['selected_scores']
        corrected_scores = reduced['corrected_scores']
        contrast = reduced['semantic_contrast']
        support_threshold = float(
            self.dual_head_fusion_support_threshold)
        score_delta = (
            corrected_scores - scores
            if scores.numel() > 0 else scores)
        stats = dict(
            presence=_finite(presence.item()),
            raw_candidate_count=int(reduced['raw_candidate_count']),
            native_keep_count=int(reduced['native_keep_count']),
            object_score_max=(
                _finite(scores.max().item())
                if scores.numel() > 0 else None),
            object_score_mean=(
                _finite(scores.mean().item())
                if scores.numel() > 0 else None),
            presence_object_score_max=(
                _finite(reduced['selected_presence_scores'].max().item())
                if scores.numel() > 0 else None),
            native_gate_margin_min=(
                _finite(
                    reduced['selected_presence_scores'].min().item()
                    - float(self.confidence_threshold))
                if scores.numel() > 0 else None),
            semantic_contrast_mean=(
                _finite(contrast.mean().item())
                if contrast.numel() > 0 else None),
            semantic_contrast_min=(
                _finite(contrast.min().item())
                if contrast.numel() > 0 else None),
            semantic_contrast_max=(
                _finite(contrast.max().item())
                if contrast.numel() > 0 else None),
            object_score_delta_mean=(
                _finite(score_delta.mean().item())
                if score_delta.numel() > 0 else None),
            object_score_delta_abs_mean=(
                _finite(score_delta.abs().mean().item())
                if score_delta.numel() > 0 else None),
            object_score_delta_min=(
                _finite(score_delta.min().item())
                if score_delta.numel() > 0 else None),
            object_score_delta_max=(
                _finite(score_delta.max().item())
                if score_delta.numel() > 0 else None),
            winner_switch_fraction=_finite(
                reduced['winner_switch_fraction']),
            semantic_area=self._dhf_area(
                semantic, support_threshold),
            native_instance_area=self._dhf_area(
                native_instance, support_threshold),
            object_instance_area=self._dhf_area(
                object_instance, support_threshold),
            corrected_instance_area=self._dhf_area(
                corrected_instance, support_threshold),
            boundary_area=self._dhf_area(
                object_boundary, support_threshold),
            p1_residual_area=self._dhf_area(
                object_advantage, support_threshold),
            p1_residual_mean=_finite(
                object_advantage.mean().item()),
            p1_residual_max=_finite(
                object_advantage.max().item()),
            proc_residual_area=self._dhf_area(
                proc_advantage, support_threshold),
            proc_residual_mean=_finite(
                proc_advantage.mean().item()),
            proc_residual_max=_finite(
                proc_advantage.max().item()),
            proc_role_residual_area=self._dhf_area(
                proc_role_advantage, support_threshold),
            proc_official_max_instance_score=_finite(
                proc_max_instance_score.item()),
            raw_object_score_max_all=_finite(
                raw_score_max_all.item()),
            region_agreement_mean=_finite(
                region_agreement.mean().item()),
            region_agreement_min=_finite(
                region_agreement.min().item()),
            i2s_positive_area=self._dhf_area(
                i2s_delta.clamp_min(0.0), support_threshold),
            i2s_negative_area=self._dhf_area(
                (-i2s_delta).clamp_min(0.0), support_threshold),
            i2s_abs_delta_mean=_finite(
                i2s_delta.abs().mean().item()),
            full_i2s_positive_area=self._dhf_area(
                full_i2s_delta.clamp_min(0.0), support_threshold),
            full_i2s_negative_area=self._dhf_area(
                (-full_i2s_delta).clamp_min(0.0), support_threshold),
            full_i2s_abs_delta_mean=_finite(
                full_i2s_delta.abs().mean().item()),
            baseline_reconstruction_max_abs=_finite(
                (
                    p0_reconstructed_native.float() - baseline
                ).abs().max().item()),
            baseline_reconstruction_mae=_finite(
                (
                    p0_reconstructed_native.float() - baseline
                ).abs().mean().item()),
        )
        # These maps are diagnostic products, not inputs to another SAM3
        # forward. Keeping them on CUDA until the next text prompt makes peak
        # memory scale with ``num_variants * image_area`` and can prevent SAM3
        # from upsampling the next prompt's masks on large VDD images.
        # CPU float32 changes only storage device; formulas and thresholds are
        # unchanged.
        variants_cpu = {
            name: values.detach().float().cpu()
            for name, values in variants.items()
        }
        return variants_cpu, stats

    def _dhf_threshold_prediction(self, class_logits):
        prediction = class_logits.argmax(dim=0)
        max_score = class_logits.max(dim=0)[0]
        prediction = prediction.clone()
        prediction[max_score < float(self.prob_thd)] = int(self.bg_idx)
        return prediction

    def _dhf_confusion(self, prediction, ground_truth, valid_mask):
        class_count = int(self.num_cls)
        gt = ground_truth[valid_mask].long()
        pred = prediction[valid_mask].long()
        index = gt * class_count + pred
        return torch.bincount(
            index,
            minlength=class_count * class_count,
        ).reshape(class_count, class_count)

    @staticmethod
    def _dhf_change_stats(reference, candidate, ground_truth, valid_mask):
        changed = valid_mask & (reference != candidate)
        reference_correct = reference == ground_truth
        candidate_correct = candidate == ground_truth
        improved = changed & ~reference_correct & candidate_correct
        harmed = changed & reference_correct & ~candidate_correct
        wrong_to_wrong = (
            changed & ~reference_correct & ~candidate_correct)
        return dict(
            changed_pixels=int(changed.sum().item()),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            wrong_to_wrong_pixels=int(wrong_to_wrong.sum().item()),
            net_correct_pixels=(
                int(improved.sum().item()) - int(harmed.sum().item())),
        )

    def _dhf_high_agreement_stats(
            self, base_logits, base_prediction, ground_truth, valid_mask,
            class_logits, class_presence, class_object_score):
        semantic = class_logits['semantic_only']
        instance = class_logits['instance_p0_path']
        gather_index = base_prediction.unsqueeze(0)
        predicted_semantic = torch.gather(
            semantic, 0, gather_index).squeeze(0)
        predicted_instance = torch.gather(
            instance, 0, gather_index).squeeze(0)
        gap = (predicted_semantic - predicted_instance).abs()
        both_strength = torch.minimum(
            predicted_semantic, predicted_instance)
        wrong = valid_mask & (base_prediction != ground_truth)
        correct = valid_mask & (base_prediction == ground_truth)
        predicted_presence = class_presence[
            base_prediction.clamp(0, int(self.num_cls) - 1)]
        predicted_object = class_object_score[
            base_prediction.clamp(0, int(self.num_cls) - 1)]
        top_values = torch.topk(
            base_logits, k=min(2, int(self.num_cls)), dim=0).values
        margin = (
            top_values[0] - top_values[1]
            if top_values.shape[0] > 1 else top_values[0])

        rows = []
        evidence_threshold = float(
            self.dual_head_fusion_high_agreement_evidence)
        for gap_threshold in self._parse_float_list(
                self.dual_head_fusion_high_agreement_gaps,
                '0.05,0.10,0.20'):
            selected = (
                valid_mask
                & (both_strength >= evidence_threshold)
                & (gap <= float(gap_threshold))
            )
            selected_wrong = selected & wrong
            selected_correct = selected & correct
            rows.append(dict(
                gap_threshold=float(gap_threshold),
                evidence_threshold=evidence_threshold,
                selected_pixels=int(selected.sum().item()),
                wrong_pixels=int(selected_wrong.sum().item()),
                correct_pixels=int(selected_correct.sum().item()),
                wrong_fraction=_safe_div(
                    int(selected_wrong.sum().item()),
                    int(selected.sum().item())),
                image_wrong_coverage=_safe_div(
                    int(selected_wrong.sum().item()),
                    int(wrong.sum().item())),
                mean_wrong_presence=_masked_mean(
                    predicted_presence, selected_wrong),
                mean_wrong_object_score=_masked_mean(
                    predicted_object, selected_wrong),
                mean_wrong_margin=_masked_mean(
                    margin, selected_wrong),
                mean_wrong_semantic=_masked_mean(
                    predicted_semantic, selected_wrong),
                mean_wrong_instance=_masked_mean(
                    predicted_instance, selected_wrong),
                low_presence_wrong_pixels=int((
                    selected_wrong
                    & (
                        predicted_presence
                        <= float(
                            self.dual_head_fusion_low_presence_threshold)
                    )
                ).sum().item()),
                low_object_wrong_pixels=int((
                    selected_wrong
                    & (
                        predicted_object
                        <= float(
                            self.dual_head_fusion_low_object_threshold)
                    )
                ).sum().item()),
            ))
        return rows

    def _dhf_artifact_shape(self, height, width):
        max_side = max(
            16, int(self.dual_head_fusion_artifact_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _dhf_save_artifact(
            self, image_path, ground_truth, valid_mask, class_logits,
            predictions, high_agreement_stats):
        if not bool(self.dual_head_fusion_save_npz):
            return None
        if (
                self._dual_head_fusion_saved_images
                >= int(self.dual_head_fusion_max_saved_images)):
            return None
        changed = any(
            bool((predictions[name] != predictions['p0_baseline']).any())
            for name in VARIANT_NAMES
        )
        if not changed:
            return None
        height, width = ground_truth.shape[-2:]
        shape = self._dhf_artifact_shape(height, width)
        arrays = {}
        arrays['ground_truth'] = F.interpolate(
            ground_truth.float().view(1, 1, height, width),
            size=shape,
            mode='nearest',
        ).squeeze().byte().cpu().numpy()
        arrays['valid_mask'] = F.interpolate(
            valid_mask.float().view(1, 1, height, width),
            size=shape,
            mode='nearest',
        ).squeeze().bool().cpu().numpy()
        for name, prediction in predictions.items():
            arrays[f'pred_{name}'] = F.interpolate(
                prediction.float().view(1, 1, height, width),
                size=shape,
                mode='nearest',
            ).squeeze().byte().cpu().numpy()
        for name, logits in class_logits.items():
            resized = self._interpolate_float32(
                logits.detach().float().unsqueeze(0), shape).squeeze(0)
            arrays[f'logits_{name}'] = resized.half().cpu().numpy()
        arrays['high_agreement_stats'] = np.asarray(
            [json.dumps(high_agreement_stats, sort_keys=True)],
            dtype=np.str_,
        )
        arrays['class_names'] = np.asarray(
            self.class_names, dtype=np.str_)

        artifact_dir = (
            self.dual_head_fusion_artifact_dir
            or './work_dirs/dual_head_fusion/artifacts')
        rank = int(os.environ.get('RANK', 0))
        rank_dir = os.path.join(artifact_dir, f'rank{rank}')
        os.makedirs(rank_dir, exist_ok=True)
        digest = hashlib.sha1(
            str(image_path).encode('utf-8')).hexdigest()[:12]
        path = os.path.join(rank_dir, f'{digest}.npz')
        np.savez_compressed(path, **arrays)
        self._dual_head_fusion_saved_images += 1
        return path

    def _build_dual_head_fusion_diagnostic(
            self, base_class_logits, base_prediction, query_base_logits,
            components, data_sample, image_path):
        """Aggregate formula-bank outputs and write one exact image record."""
        self._dhf_validate_runtime_contract()
        if components is None:
            raise RuntimeError(
                'Dual-head fusion bank requires inference components.')
        class_variants = components.get(
            'dual_head_fusion_class_logits')
        class_presence = components.get('presence_scores')
        prompt_stats = components.get(
            'dual_head_fusion_prompt_stats', [])
        if not isinstance(class_variants, dict):
            raise RuntimeError(
                'Missing dual_head_fusion_class_logits.')
        if not isinstance(class_presence, torch.Tensor):
            raise RuntimeError(
                'Missing class-level Presence scores.')
        for name in VARIANT_NAMES:
            if name not in class_variants:
                raise RuntimeError(
                    f'Missing dual-head fusion variant {name!r}.')

        reconstructed_base = self._aggregate_query_logits_to_classes(
            query_base_logits.detach().float())
        base_error = (
            reconstructed_base - base_class_logits.detach().float()).abs()
        reconstructed_prediction = self._dhf_threshold_prediction(
            reconstructed_base)
        prediction_mismatch = int(
            (reconstructed_prediction != base_prediction).sum().item())
        max_prompt_error = max(
            (
                float(row.get('baseline_reconstruction_max_abs', 0.0))
                for row in prompt_stats
                if row.get('baseline_reconstruction_max_abs') is not None
            ),
            default=0.0,
        )
        integrity = dict(
            class_logit_max_abs=_finite(base_error.max().item()),
            class_logit_mae=_finite(base_error.mean().item()),
            prompt_reconstruction_max_abs=_finite(max_prompt_error),
            baseline_prediction_mismatch_pixels=prediction_mismatch,
        )
        tolerance = float(
            self.dual_head_fusion_integrity_tolerance)
        if bool(self.dual_head_fusion_strict_integrity):
            if (
                    integrity['class_logit_max_abs'] > tolerance
                    or integrity['prompt_reconstruction_max_abs'] > tolerance
                    or prediction_mismatch > 0):
                raise RuntimeError(
                    'Dual-head fusion baseline reconstruction failed: '
                    f'{integrity}')

        # Formula-bank maps are deliberately offloaded after each prompt.
        # Keep all diagnostic comparisons on that device instead of copying
        # every full-resolution variant back to CUDA.
        analysis_device = class_variants[VARIANT_NAMES[0]].device
        class_logits = {
            'p0_baseline': base_class_logits.detach().float().to(
                analysis_device)}
        for name in VARIANT_NAMES:
            logits = class_variants[name].detach().float().to(
                analysis_device)
            if int(logits.shape[0]) != int(self.num_cls):
                raise RuntimeError(
                    f'Dual-head fusion variant {name!r} has '
                    f'{int(logits.shape[0])} classes; expected '
                    f'{int(self.num_cls)}.')
            class_logits[name] = logits
        predictions = {
            'p0_baseline': base_prediction.detach().long().to(
                analysis_device)}
        for name in VARIANT_NAMES:
            predictions[name] = self._dhf_threshold_prediction(
                class_logits[name])

        gt = data_sample.gt_sem_seg.data
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        gt = gt.to(analysis_device).long()
        valid = gt != 255
        valid_pixels = int(valid.sum().item())

        variant_stats = {}
        for name, prediction in predictions.items():
            row = dict(
                confusion=self._dhf_confusion(
                    prediction, gt, valid).detach().cpu().tolist(),
                valid_pixels=valid_pixels,
            )
            if name != 'p0_baseline':
                row.update(self._dhf_change_stats(
                    predictions['p0_baseline'],
                    prediction,
                    gt,
                    valid,
                ))
            variant_stats[name] = row

        contrast_stats = []
        for reference_name, candidate_name, hypothesis in CONTRASTS:
            row = dict(
                reference=reference_name,
                candidate=candidate_name,
                hypothesis=hypothesis,
            )
            row.update(self._dhf_change_stats(
                predictions[reference_name],
                predictions[candidate_name],
                gt,
                valid,
            ))
            contrast_stats.append(row)

        class_presence = class_presence.detach().float().to(
            analysis_device).clamp(0.0, 1.0)
        class_object_score = torch.zeros_like(class_presence)
        for prompt_row in prompt_stats:
            class_index = int(prompt_row.get('class_index', -1))
            value = prompt_row.get('object_score_max')
            if (
                    0 <= class_index < int(self.num_cls)
                    and value is not None):
                class_object_score[class_index] = torch.maximum(
                    class_object_score[class_index],
                    class_object_score.new_tensor(float(value)),
                )
        high_agreement_stats = self._dhf_high_agreement_stats(
            class_logits['p0_baseline'],
            predictions['p0_baseline'],
            gt,
            valid,
            class_logits,
            class_presence,
            class_object_score,
        )

        class_stats = []
        for class_index, class_name in enumerate(self.class_names):
            gt_mask = valid & (gt == int(class_index))
            row = dict(
                class_index=int(class_index),
                class_name=str(class_name),
                gt_pixels=int(gt_mask.sum().item()),
                presence_max=_finite(
                    class_presence[class_index].item()),
                object_score_max=_finite(
                    class_object_score[class_index].item()),
                variants={},
            )
            for name, prediction in predictions.items():
                pred_mask = valid & (
                    prediction == int(class_index))
                row['variants'][name] = dict(
                    predicted_pixels=int(pred_mask.sum().item()),
                    correct_pixels=int(
                        (pred_mask & gt_mask).sum().item()),
                )
            class_stats.append(row)

        artifact_path = self._dhf_save_artifact(
            image_path,
            gt,
            valid,
            class_logits,
            predictions,
            high_agreement_stats,
        )
        record = dict(
            schema_version=SCHEMA_VERSION,
            rank=int(os.environ.get('RANK', 0)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            dataset_name=(
                self.dual_head_fusion_dataset_name
                or self.seed_dataset_name
                or 'unknown'),
            img_path=str(image_path),
            valid_pixels=valid_pixels,
            class_names=list(self.class_names),
            prob_thd=float(self.prob_thd),
            confidence_threshold=float(self.confidence_threshold),
            variant_names=['p0_baseline'] + list(VARIANT_NAMES),
            execution_contract=dict(
                same_decoder_output_per_prompt=True,
                extra_sam3_forward_calls=0,
                ground_truth_used_in_fusion=False,
                diagnostic_storage='cpu_float32',
                candidate_policy='native_keep_presence_times_object_score',
                class_prompt_reduction='max',
            ),
            formula_parameters=dict(
                agreement_gamma=float(
                    self.dual_head_fusion_agreement_gamma),
                eta=float(self.dual_head_fusion_eta),
                beta=float(self.dual_head_fusion_beta),
                ring_kernel=int(
                    self.dual_head_fusion_ring_kernel),
                mask_threshold=float(
                    self.dual_head_fusion_mask_threshold),
            ),
            integrity=integrity,
            variant_stats=variant_stats,
            contrast_stats=contrast_stats,
            high_agreement_stats=high_agreement_stats,
            class_stats=class_stats,
            prompt_stats=prompt_stats,
            artifact_path=artifact_path,
        )
        self._write_dual_head_fusion_stats(record)
        return record

    def _write_dual_head_fusion_stats(self, record):
        if not bool(self.dump_dual_head_fusion_stats):
            return
        if self._dual_head_fusion_stats_file is None:
            path = (
                self.dual_head_fusion_stats_path
                or './work_dirs/dual_head_fusion/'
                   'dual_head_fusion.jsonl')
            path = _ranked_jsonl_path(path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self._dual_head_fusion_stats_file = open(
                path, 'a', buffering=1)
        self._dual_head_fusion_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')
