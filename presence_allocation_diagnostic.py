"""Training-free audit of decoder-presence placement in SegEarth-OV3.

The official SAM3 image post-processor uses decoder presence to calibrate
object-query scores.  SegEarth-OV3 additionally multiplies the fused semantic
and instance map by the same scalar.  Under the default baseline this means
that decoder presence:

1. participates in the hard object-query keep decision;
2. scales the retained instance evidence; and
3. scales the final fused map, so the instance path contains presence twice.

This module reconstructs predeclared counterfactual computation graphs from a
single SAM3 forward.  It never selects top-1/top-2 alternatives and never
changes the returned baseline prediction.  Ground truth is used only to
measure the exact effect of each structural intervention.
"""

import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = 'presence-allocation-v1'
VARIANT_NAMES = (
    'p1_branch_once',
    'p2_delayed_gate',
    'p3_logit_prior',
    'p4_instance_scope',
)
CONTRASTS = (
    ('p0_baseline', 'p1_branch_once', 'remove_duplicate_instance_presence'),
    ('p1_branch_once', 'p2_delayed_gate', 'delay_presence_candidate_gate'),
    ('p2_delayed_gate', 'p3_logit_prior', 'probability_to_logit_prior'),
    ('p1_branch_once', 'p4_instance_scope', 'remove_semantic_presence'),
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


class PresenceAllocationDiagnosticMixin:
    """Build exact, prediction-preserving presence-allocation counterfactuals."""

    def _uses_presence_allocation_diagnostic(self):
        return bool(self.dump_presence_allocation_stats)

    def _pa_validate_runtime_contract(self):
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
                'Presence-allocation audit requires the unchanged '
                'SegEarth-OV3 baseline contract: ' + '; '.join(errors))

    def _pa_raw_gate_threshold(self):
        value = self.presence_allocation_raw_gate_threshold
        return (
            float(self.confidence_threshold)
            if value is None else float(value)
        )

    def _pa_reduce_raw_instances(
            self, raw_masks, raw_scores, keep_mask, output_shape):
        """Max-reduce selected raw masks without materializing all full maps."""
        if (
                not isinstance(raw_masks, torch.Tensor)
                or not isinstance(raw_scores, torch.Tensor)
                or not isinstance(keep_mask, torch.Tensor)):
            return torch.zeros(
                output_shape, device=self.device, dtype=torch.float32)
        count = min(
            int(raw_masks.shape[0]),
            int(raw_scores.numel()),
            int(keep_mask.numel()),
        )
        if count <= 0:
            return torch.zeros(
                output_shape, device=self.device, dtype=torch.float32)
        selected = torch.nonzero(
            keep_mask[:count].detach().bool(), as_tuple=False).flatten()
        result = torch.zeros(
            output_shape, device=self.device, dtype=torch.float32)
        if selected.numel() == 0:
            return result
        chunk_size = max(1, int(self.presence_allocation_chunk_size))
        for start in range(0, int(selected.numel()), chunk_size):
            indices = selected[start:start + chunk_size]
            masks = raw_masks[:count][indices].detach().float().unsqueeze(1)
            masks = torch.sigmoid(self._interpolate_float32(
                masks, output_shape)).squeeze(1)
            scores = raw_scores[:count][indices].detach().float().view(
                -1, 1, 1)
            result = torch.maximum(
                result, (masks * scores).max(dim=0)[0])
        return result

    @staticmethod
    def _pa_logit(probability, eps):
        probability = probability.float().clamp(eps, 1.0 - eps)
        return torch.log(probability) - torch.log1p(-probability)

    def _pa_area(self, values, threshold):
        return _safe_div(
            int((values.float() >= float(threshold)).sum().item()),
            int(values.numel()),
        )

    def _pa_build_prompt_variants(
            self, inference_state, semantic_map, native_instance_map,
            baseline_map, output_shape):
        """Build one prompt's structural variants from the same SAM3 output."""
        self._pa_validate_runtime_contract()
        presence_tensor = inference_state.get('presence_score')
        if not isinstance(presence_tensor, torch.Tensor):
            raise RuntimeError(
                'Presence-allocation audit requires presence_score.')
        if presence_tensor.numel() != 1:
            raise RuntimeError(
                'Presence-allocation audit expects one decoder-presence '
                f'scalar per text prompt, got shape {presence_tensor.shape}.')

        # Reconstruct P0 before dtype conversion.  This mirrors the exact
        # arithmetic used by the baseline and prevents bfloat16-to-float32
        # conversion noise from being mistaken for a structural mismatch.
        native_semantic = semantic_map.detach()
        native_instance = native_instance_map.detach()
        native_baseline = baseline_map.detach()
        native_presence = presence_tensor.detach().reshape(())
        p0_reconstructed_native = native_presence * torch.maximum(
            native_semantic, native_instance)

        semantic_map = native_semantic.float()
        native_instance_map = native_instance.float()
        baseline_map = native_baseline.float()
        presence = native_presence.float().clamp(0.0, 1.0)

        raw_masks = inference_state.get('raw_masks_logits_lowres')
        raw_scores = inference_state.get('raw_object_score')
        native_keep = inference_state.get('raw_keep_mask')
        if (
                not isinstance(raw_masks, torch.Tensor)
                or not isinstance(raw_scores, torch.Tensor)
                or not isinstance(native_keep, torch.Tensor)):
            raise RuntimeError(
                'Presence-allocation audit requires raw SAM3 masks, scores, '
                'and the native keep mask.')
        count = min(
            int(raw_masks.shape[0]),
            int(raw_scores.numel()),
            int(native_keep.numel()),
        )
        raw_masks = raw_masks[:count]
        raw_scores = raw_scores[:count].detach().float()
        native_keep = native_keep[:count].detach().bool()
        raw_gate = raw_scores > self._pa_raw_gate_threshold()
        presence_deleted = raw_gate & ~native_keep

        native_raw_instance = self._pa_reduce_raw_instances(
            raw_masks, raw_scores, native_keep, output_shape)
        raw_gate_instance = self._pa_reduce_raw_instances(
            raw_masks, raw_scores, raw_gate, output_shape)
        deleted_raw_instance = self._pa_reduce_raw_instances(
            raw_masks, raw_scores, presence_deleted, output_shape)

        semantic_presence = semantic_map * presence
        # P0 reconstruction is kept only as an integrity check.  The returned
        # baseline map remains the authoritative P0 value.
        # P1: semantic and instance branches each contain presence once.
        p1_branch_once = torch.maximum(
            semantic_presence, native_instance_map)
        # P2: remove presence from the irreversible candidate gate, then apply
        # it once to both branches after spatial evidence has been formed.
        p2_delayed_gate = presence * torch.maximum(
            semantic_map, raw_gate_instance)
        # P3: same raw evidence as P2, but presence enters once as a global
        # log-odds prior instead of a probability-space multiplier.
        eps = float(self.presence_allocation_logit_eps)
        raw_fusion = torch.maximum(semantic_map, raw_gate_instance)
        p3_logit_prior = torch.sigmoid(
            self._pa_logit(raw_fusion, eps)
            + self._pa_logit(presence, eps)
        )
        # P4: preserve SAM3's native object-query presence calibration but do
        # not export the decoder scalar into the semantic branch.
        p4_instance_scope = torch.maximum(
            semantic_map, native_instance_map)

        prob_threshold = float(self.prob_thd)
        support_threshold = float(
            self.presence_allocation_support_threshold)
        stats = dict(
            presence=_finite(presence.item()),
            raw_candidate_count=count,
            native_keep_count=int(native_keep.sum().item()),
            raw_gate_keep_count=int(raw_gate.sum().item()),
            presence_deleted_count=int(presence_deleted.sum().item()),
            presence_deleted_ratio=_safe_div(
                int(presence_deleted.sum().item()),
                int(raw_gate.sum().item()),
            ),
            raw_score_max=(
                _finite(raw_scores.max().item()) if count > 0 else None),
            raw_score_mean=(
                _finite(raw_scores.mean().item()) if count > 0 else None),
            native_raw_instance_area=self._pa_area(
                native_raw_instance, support_threshold),
            raw_gate_instance_area=self._pa_area(
                raw_gate_instance, support_threshold),
            deleted_raw_instance_area=self._pa_area(
                deleted_raw_instance, support_threshold),
            semantic_raw_area=self._pa_area(
                semantic_map, support_threshold),
            semantic_presence_area=self._pa_area(
                semantic_presence, support_threshold),
            semantic_threshold_suppressed_area=_safe_div(
                int((
                    (semantic_map >= prob_threshold)
                    & (semantic_presence < prob_threshold)
                ).sum().item()),
                int(semantic_map.numel()),
            ),
            instance_double_suppressed_area=_safe_div(
                int((
                    (native_instance_map >= prob_threshold)
                    & (native_instance_map * presence < prob_threshold)
                ).sum().item()),
                int(native_instance_map.numel()),
            ),
            baseline_reconstruction_max_abs=_finite(
                (
                    p0_reconstructed_native.float() - baseline_map
                ).abs().max().item()),
            baseline_reconstruction_mae=_finite(
                (
                    p0_reconstructed_native.float() - baseline_map
                ).abs().mean().item()),
        )
        variants = dict(
            p1_branch_once=p1_branch_once,
            p2_delayed_gate=p2_delayed_gate,
            p3_logit_prior=p3_logit_prior,
            p4_instance_scope=p4_instance_scope,
        )
        return variants, stats

    def _pa_threshold_prediction(self, class_logits):
        prediction = class_logits.argmax(dim=0)
        max_score = class_logits.max(dim=0)[0]
        prediction = prediction.clone()
        prediction[max_score < float(self.prob_thd)] = int(self.bg_idx)
        return prediction

    def _pa_confusion(self, prediction, ground_truth, valid_mask):
        class_count = int(self.num_cls)
        gt = ground_truth[valid_mask].long()
        pred = prediction[valid_mask].long()
        index = gt * class_count + pred
        return torch.bincount(
            index,
            minlength=class_count * class_count,
        ).reshape(class_count, class_count)

    @staticmethod
    def _pa_change_stats(reference, candidate, ground_truth, valid_mask):
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

    def _pa_artifact_shape(self, height, width):
        max_side = max(
            16, int(self.presence_allocation_artifact_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _pa_save_artifact(
            self, image_path, ground_truth, valid_mask, class_logits,
            predictions, class_presence):
        if not bool(self.presence_allocation_save_npz):
            return None
        if (
                self._presence_allocation_saved_images
                >= int(self.presence_allocation_max_saved_images)):
            return None
        changed = any(
            bool((predictions[name] != predictions['p0_baseline']).any())
            for name in VARIANT_NAMES
        )
        if not changed:
            return None
        height, width = ground_truth.shape[-2:]
        shape = self._pa_artifact_shape(height, width)
        arrays = {}
        gt = ground_truth.float().view(1, 1, height, width)
        arrays['ground_truth'] = F.interpolate(
            gt, size=shape, mode='nearest').squeeze().byte().cpu().numpy()
        valid = valid_mask.float().view(1, 1, height, width)
        arrays['valid_mask'] = F.interpolate(
            valid, size=shape, mode='nearest').squeeze().bool().cpu().numpy()
        for name, prediction in predictions.items():
            pred = prediction.float().view(1, 1, height, width)
            arrays[f'pred_{name}'] = F.interpolate(
                pred, size=shape, mode='nearest').squeeze().byte().cpu().numpy()
        for name, logits in class_logits.items():
            resized = self._interpolate_float32(
                logits.detach().float().unsqueeze(0), shape).squeeze(0)
            arrays[f'logits_{name}'] = (
                resized.half().cpu().numpy())
        arrays['class_presence_max'] = (
            class_presence.detach().float().half().cpu().numpy())
        arrays['class_names'] = np.asarray(
            self.class_names, dtype=np.str_)

        artifact_dir = (
            self.presence_allocation_artifact_dir
            or './work_dirs/presence_allocation/artifacts')
        rank = int(os.environ.get('RANK', 0))
        rank_dir = os.path.join(artifact_dir, f'rank{rank}')
        os.makedirs(rank_dir, exist_ok=True)
        digest = hashlib.sha1(
            str(image_path).encode('utf-8')).hexdigest()[:12]
        path = os.path.join(rank_dir, f'{digest}.npz')
        np.savez_compressed(path, **arrays)
        self._presence_allocation_saved_images += 1
        return path

    def _build_presence_allocation_diagnostic(
            self, base_class_logits, base_prediction, query_base_logits,
            components, data_sample, image_path):
        """Aggregate prompt variants, evaluate exact effects, and write one row."""
        self._pa_validate_runtime_contract()
        if components is None:
            raise RuntimeError(
                'Presence-allocation audit requires inference components.')
        query_variants = components.get(
            'presence_allocation_query_logits')
        class_presence = components.get('presence_scores')
        prompt_stats = components.get(
            'presence_allocation_prompt_stats', [])
        if not isinstance(query_variants, dict):
            raise RuntimeError(
                'Missing presence_allocation_query_logits.')
        if not isinstance(class_presence, torch.Tensor):
            raise RuntimeError(
                'Missing class-level presence scores.')
        for name in VARIANT_NAMES:
            if name not in query_variants:
                raise RuntimeError(
                    f'Missing presence-allocation variant {name!r}.')

        reconstructed_base = self._aggregate_query_logits_to_classes(
            query_base_logits.detach().float())
        base_logit_error = (
            reconstructed_base - base_class_logits.detach().float()).abs()
        simple_base_prediction = self._pa_threshold_prediction(
            reconstructed_base)
        base_prediction_mismatch = int(
            (simple_base_prediction != base_prediction).sum().item())
        max_prompt_reconstruction = max(
            (
                float(row.get('baseline_reconstruction_max_abs', 0.0))
                for row in prompt_stats
                if row.get('baseline_reconstruction_max_abs') is not None
            ),
            default=0.0,
        )
        integrity = dict(
            class_logit_max_abs=_finite(base_logit_error.max().item()),
            class_logit_mae=_finite(base_logit_error.mean().item()),
            prompt_reconstruction_max_abs=_finite(
                max_prompt_reconstruction),
            baseline_prediction_mismatch_pixels=base_prediction_mismatch,
        )
        tolerance = float(self.presence_allocation_integrity_tolerance)
        if bool(self.presence_allocation_strict_integrity):
            if (
                    integrity['class_logit_max_abs'] > tolerance
                    or integrity['prompt_reconstruction_max_abs'] > tolerance
                    or base_prediction_mismatch > 0):
                raise RuntimeError(
                    'Presence-allocation baseline reconstruction failed: '
                    f'{integrity}')

        class_logits = {'p0_baseline': base_class_logits.detach().float()}
        for name in VARIANT_NAMES:
            class_logits[name] = self._aggregate_query_logits_to_classes(
                query_variants[name].detach().float())
        predictions = {'p0_baseline': base_prediction.detach().long()}
        for name in VARIANT_NAMES:
            predictions[name] = self._pa_threshold_prediction(
                class_logits[name])

        gt = data_sample.gt_sem_seg.data
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        gt = gt.to(base_prediction.device).long()
        valid = gt != 255
        valid_pixels = int(valid.sum().item())
        class_presence = class_presence.detach().float()

        variant_stats = {}
        for name, prediction in predictions.items():
            confusion = self._pa_confusion(prediction, gt, valid)
            row = dict(
                confusion=confusion.detach().cpu().tolist(),
                valid_pixels=valid_pixels,
            )
            if name != 'p0_baseline':
                row.update(self._pa_change_stats(
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
            row.update(self._pa_change_stats(
                predictions[reference_name],
                predictions[candidate_name],
                gt,
                valid,
            ))
            contrast_stats.append(row)

        class_stats = []
        for class_index, class_name in enumerate(self.class_names):
            gt_mask = valid & (gt == int(class_index))
            row = dict(
                class_index=int(class_index),
                class_name=str(class_name),
                gt_pixels=int(gt_mask.sum().item()),
                presence_max=_finite(
                    class_presence[class_index].item()),
                variants={},
            )
            for name, prediction in predictions.items():
                pred_mask = valid & (prediction == int(class_index))
                row['variants'][name] = dict(
                    predicted_pixels=int(pred_mask.sum().item()),
                    correct_pixels=int((pred_mask & gt_mask).sum().item()),
                )
            class_stats.append(row)

        artifact_path = self._pa_save_artifact(
            image_path,
            gt,
            valid,
            class_logits,
            predictions,
            class_presence,
        )
        record = dict(
            schema_version=SCHEMA_VERSION,
            rank=int(os.environ.get('RANK', 0)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            dataset_name=(
                self.presence_allocation_dataset_name
                or self.seed_dataset_name
                or 'unknown'),
            img_path=str(image_path),
            valid_pixels=valid_pixels,
            class_names=list(self.class_names),
            prob_thd=float(self.prob_thd),
            confidence_threshold=float(self.confidence_threshold),
            raw_gate_threshold=float(self._pa_raw_gate_threshold()),
            support_threshold=float(
                self.presence_allocation_support_threshold),
            integrity=integrity,
            variant_stats=variant_stats,
            contrast_stats=contrast_stats,
            class_stats=class_stats,
            prompt_stats=prompt_stats,
            artifact_path=artifact_path,
        )
        self._write_presence_allocation_stats(record)
        return record

    def _write_presence_allocation_stats(self, record):
        if not bool(self.dump_presence_allocation_stats):
            return
        if self._presence_allocation_stats_file is None:
            path = (
                self.presence_allocation_stats_path
                or './work_dirs/presence_allocation/'
                   'presence_allocation.jsonl')
            path = _ranked_jsonl_path(path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self._presence_allocation_stats_file = open(
                path, 'a', buffering=1)
        self._presence_allocation_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')
