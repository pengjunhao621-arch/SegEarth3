"""Prediction-preserving provenance diagnostics for SAM3 object queries.

The baseline combines SAM3 evidence hierarchically:

1. object queries are max-reduced inside each text prompt and image view;
2. sliding-window views are averaged for the same text prompt;
3. synonym prompts are max-reduced into one semantic class; and
4. the semantic and instance heads compete before final class prediction.

This module preserves that hierarchy for reconstruction auditing while also
describing the atomic query sources that form each predicted connected region.
Ground truth is used only after prediction to measure region quality and
counterfactual action oracles.  Nothing in this module changes model outputs.
"""

import hashlib
import json
import math
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = 'query-topology-v1'


def _ranked_jsonl_path(path):
    rank = int(os.environ.get('RANK', 0))
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext or '.jsonl'}"


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return _finite(float(numerator) / float(denominator))


class QueryTopologyDiagnosticMixin:
    """Trace final regions back to SAM3 object-query source topology."""

    def _uses_query_topology_diagnostic(self):
        return bool(self.dump_query_topology_stats)

    def _qtd_shape(self, height, width):
        max_side = max(16, int(self.query_topology_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    @staticmethod
    def _qtd_components(mask, min_pixels):
        array = mask.detach().bool().cpu().numpy().astype(np.uint8)
        count, labels = cv2.connectedComponents(array, connectivity=8)
        rows = []
        for label in range(1, count):
            component = torch.from_numpy(labels == label).to(mask.device)
            area = int(component.sum().item())
            if area >= int(min_pixels):
                rows.append((area, component))
        rows.sort(key=lambda item: item[0], reverse=True)
        return rows

    def _qtd_reconstruct_mask(
            self, raw_mask, crop_box, full_shape, diag_shape):
        full_h, full_w = full_shape
        diag_h, diag_w = diag_shape
        if crop_box is None:
            x1, y1, x2, y2 = 0, 0, full_w, full_h
        else:
            x1, y1, x2, y2 = [int(value) for value in crop_box]
        dx1 = max(0, min(diag_w - 1, int(round(x1 * diag_w / full_w))))
        dy1 = max(0, min(diag_h - 1, int(round(y1 * diag_h / full_h))))
        dx2 = max(dx1 + 1, min(diag_w, int(round(x2 * diag_w / full_w))))
        dy2 = max(dy1 + 1, min(diag_h, int(round(y2 * diag_h / full_h))))
        local = torch.sigmoid(self._interpolate_float32(
            raw_mask.to(self.device).float().view(
                1, 1, *raw_mask.shape[-2:]),
            (dy2 - dy1, dx2 - dx1),
        )).squeeze()
        result = torch.zeros(
            diag_shape, device=self.device, dtype=torch.float32)
        result[dy1:dy2, dx1:dx2] = local
        coverage = torch.zeros(
            diag_shape, device=self.device, dtype=torch.bool)
        coverage[dy1:dy2, dx1:dx2] = True
        return result, coverage

    @staticmethod
    def _qtd_prompt_presence(raw_scores, presence_scores):
        valid = raw_scores > 1e-8
        if not valid.any():
            return 0.0
        ratios = (
            presence_scores[valid].float()
            / raw_scores[valid].float().clamp_min(1e-8)
        )
        return float(ratios.median().clamp(0.0, 1.0).item())

    def _qtd_collect_sources(
            self, candidates, full_shape, diag_shape, query_count):
        """Reconstruct the hierarchy and retain atomic kept-query sources."""
        diag_h, diag_w = diag_shape
        prompt_sum = torch.zeros(
            (query_count, diag_h, diag_w),
            device=self.device,
            dtype=torch.float32,
        )
        prompt_count = torch.zeros_like(prompt_sum)
        source_scores = []
        source_binary_masks = []
        source_metadata = []
        selected_kept = 0
        reported_kept = 0
        reported_candidates = 0

        for record_idx, record in enumerate(candidates or []):
            raw_masks = record.get('raw_masks_lowres')
            raw_scores = record.get('raw_scores')
            presence_scores = record.get('raw_presence_scores')
            keep_mask = record.get('raw_keep_mask')
            selected_indices = record.get('selected_indices')
            if (
                    not isinstance(raw_masks, torch.Tensor)
                    or not isinstance(raw_scores, torch.Tensor)
                    or not isinstance(presence_scores, torch.Tensor)
                    or not isinstance(keep_mask, torch.Tensor)
                    or raw_masks.numel() == 0):
                continue
            count = min(
                int(raw_masks.shape[0]),
                int(raw_scores.numel()),
                int(presence_scores.numel()),
                int(keep_mask.numel()),
            )
            if count <= 0:
                continue
            query_index = int(record.get('query_index', -1))
            if not 0 <= query_index < query_count:
                continue
            reported_kept += int(record.get('raw_kept_count', 0))
            reported_candidates += int(record.get(
                'raw_candidate_count', count))
            raw_scores = raw_scores[:count].float()
            presence_scores = presence_scores[:count].float()
            keep_mask = keep_mask[:count].bool()
            selected_kept += int(keep_mask.sum().item())
            presence = self._qtd_prompt_presence(
                raw_scores, presence_scores)

            # Every view contributes to the sliding-window denominator even
            # when it has no retained instance.
            _, coverage = self._qtd_reconstruct_mask(
                raw_masks[0],
                record.get('crop_box'),
                full_shape,
                diag_shape,
            )
            prompt_count[query_index] += coverage.float()
            kept_local = torch.nonzero(
                keep_mask, as_tuple=False).flatten().tolist()
            if not kept_local:
                continue

            local_pre_scores = []
            local_final_scores = []
            local_soft_masks = []
            local_metadata = []
            for local_index in kept_local:
                soft_mask, _ = self._qtd_reconstruct_mask(
                    raw_masks[local_index],
                    record.get('crop_box'),
                    full_shape,
                    diag_shape,
                )
                score = (
                    float(raw_scores[local_index].item())
                    if str(self.instance_score_type).lower() == 'raw'
                    else float(presence_scores[local_index].item())
                )
                pre_score = soft_mask * score
                final_score = (
                    pre_score * presence
                    if bool(self.use_presence_score)
                    else pre_score
                )
                actual_index = (
                    int(selected_indices[local_index].item())
                    if isinstance(selected_indices, torch.Tensor)
                    and local_index < selected_indices.numel()
                    else int(local_index)
                )
                metadata = dict(
                    record_index=int(record_idx),
                    view_id=str(record.get('view_id', 'unknown')),
                    crop_box=record.get('crop_box'),
                    query_index=query_index,
                    class_index=int(record.get('class_index', -1)),
                    query_word=str(record.get('query_word', '')),
                    object_query_index=actual_index,
                    raw_score=_finite(raw_scores[local_index].item()),
                    raw_presence_score=_finite(
                        presence_scores[local_index].item()),
                    prompt_presence=_finite(presence),
                )
                local_pre_scores.append(pre_score)
                local_final_scores.append(final_score)
                local_soft_masks.append(soft_mask)
                local_metadata.append(metadata)

            if local_pre_scores:
                local_pre_tensor = torch.stack(
                    local_pre_scores, dim=0)
                local_max, local_winner = local_pre_tensor.max(dim=0)
                prompt_sum[query_index] += local_max
                # Queries that never win the prompt-local max do not form the
                # baseline instance map.  They are counted for capture
                # completeness but not retained as dense atomic sources.
                winning_local = torch.unique(
                    local_winner[coverage]).detach().cpu().tolist()
                for local_index in winning_local:
                    source_id = len(source_scores)
                    metadata = dict(local_metadata[local_index])
                    winner_mask = (
                        local_winner == int(local_index)) & coverage
                    metadata.update(
                        source_id=source_id,
                        local_winner_pixels=int(
                            winner_mask.sum().item()),
                    )
                    source_scores.append(
                        local_final_scores[local_index]
                        * winner_mask.float())
                    source_binary_masks.append(
                        local_soft_masks[local_index]
                        >= float(self.query_topology_mask_threshold))
                    source_metadata.append(metadata)

        reconstructed = prompt_sum / prompt_count.clamp_min(1.0)
        if source_scores:
            source_scores = torch.stack(source_scores, dim=0)
            source_binary_masks = torch.stack(
                source_binary_masks, dim=0)
            for source_index, metadata in enumerate(source_metadata):
                query_index = int(metadata['query_index'])
                source_scores[source_index] = (
                    source_scores[source_index]
                    / prompt_count[query_index].clamp_min(1.0)
                )
        else:
            source_scores = torch.empty(
                (0, diag_h, diag_w),
                device=self.device,
                dtype=torch.float32,
            )
            source_binary_masks = torch.empty(
                (0, diag_h, diag_w),
                device=self.device,
                dtype=torch.bool,
            )
        return dict(
            reconstructed_query_instance=reconstructed,
            prompt_view_count=prompt_count,
            source_scores=source_scores,
            source_binary_masks=source_binary_masks,
            source_metadata=source_metadata,
            reported_candidate_count=int(reported_candidates),
            reported_kept_count=int(reported_kept),
            selected_kept_count=int(selected_kept),
            kept_selection_complete=bool(
                reported_kept == selected_kept),
        )

    @staticmethod
    def _qtd_topology_from_scores(scores, region, threshold):
        if scores is None or scores.numel() == 0:
            inside = None
        else:
            inside = scores[:, region].float().clamp_min(0.0)
        return QueryTopologyDiagnosticMixin._qtd_topology_from_inside(
            inside, region, threshold)

    @staticmethod
    def _qtd_topology_from_inside(inside, region, threshold):
        if inside is None or inside.numel() == 0 or not region.any():
            return dict(
                supported_fraction=0.0,
                dominant_source_coverage=0.0,
                effective_source_count=0.0,
                switch_boundary_density=0.0,
                mean_support_count=0.0,
                consensus_fraction=0.0,
                overlapping_source_count=0,
            )
        inside = inside.float().clamp_min(0.0)
        maxima, winners = inside.max(dim=0)
        supported = maxima >= float(threshold)
        supported_count = int(supported.sum().item())
        area = int(region.sum().item())
        if supported_count:
            histogram = torch.bincount(
                winners[supported],
                minlength=int(inside.shape[0]),
            ).float()
            dominant = float(
                histogram.max().item() / max(1, supported_count))
        else:
            dominant = 0.0
        masses = inside.sum(dim=1)
        positive = masses > 0
        if positive.any():
            probabilities = masses[positive] / masses[positive].sum()
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum()
            effective = float(torch.exp(entropy).item())
        else:
            effective = 0.0

        full_winners = torch.full_like(
            region, -1, dtype=torch.long)
        full_supported = torch.zeros_like(region, dtype=torch.bool)
        full_winners[region] = winners
        full_supported[region] = supported
        horizontal = (
            region[:, :-1] & region[:, 1:]
            & full_supported[:, :-1] & full_supported[:, 1:]
        )
        vertical = (
            region[:-1, :] & region[1:, :]
            & full_supported[:-1, :] & full_supported[1:, :]
        )
        boundary_pairs = int(horizontal.sum().item() + vertical.sum().item())
        switches = int(
            ((full_winners[:, :-1] != full_winners[:, 1:])
             & horizontal).sum().item()
            + ((full_winners[:-1, :] != full_winners[1:, :])
               & vertical).sum().item()
        )
        support_count = (inside >= float(threshold)).sum(dim=0).float()
        return dict(
            supported_fraction=_safe_div(supported_count, area) or 0.0,
            dominant_source_coverage=_finite(dominant),
            effective_source_count=_finite(effective),
            switch_boundary_density=(
                _safe_div(switches, boundary_pairs) or 0.0),
            mean_support_count=_finite(
                support_count.mean().item()) if area else 0.0,
            consensus_fraction=_finite(
                (support_count >= 2).float().mean().item()) if area else 0.0,
            overlapping_source_count=int(
                (masses >= float(threshold)).sum().item()),
        )

    @staticmethod
    def _qtd_roll_control(scores):
        if scores is None or scores.numel() == 0:
            return scores
        height, width = scores.shape[-2:]
        shifted = []
        for index, source in enumerate(scores):
            shift_y = ((index * 37 + 11) % max(1, height)) or 1
            shift_x = ((index * 53 + 17) % max(1, width)) or 1
            shifted.append(torch.roll(
                source,
                shifts=(shift_y, shift_x),
                dims=(-2, -1),
            ))
        return torch.stack(shifted, dim=0)

    @staticmethod
    def _qtd_component_overlapping(mask, region, min_pixels):
        components = QueryTopologyDiagnosticMixin._qtd_components(
            mask, min_pixels)
        if not components:
            return torch.zeros_like(region, dtype=torch.bool)
        selected = max(
            (component for _, component in components),
            key=lambda item: int((item & region).sum().item()),
        )
        if not (selected & region).any():
            return torch.zeros_like(region, dtype=torch.bool)
        return selected

    @staticmethod
    def _qtd_local_iou(mask, target, window):
        prediction = mask & window
        truth = target & window
        union = int((prediction | truth).sum().item())
        if union == 0:
            return 1.0
        return float((prediction & truth).sum().item()) / float(union)

    def _qtd_action_masks(
            self, region, class_scores, class_binary_masks, semantic_score):
        empty = torch.zeros_like(region, dtype=torch.bool)
        actions = dict(baseline=region)
        if class_scores is None or class_scores.numel() == 0:
            actions.update(
                dominant_query=empty,
                query_union=empty,
                query_consensus=empty,
            )
        else:
            binary = class_binary_masks
            overlaps = (binary & region.unsqueeze(0)).flatten(1).sum(dim=1)
            areas = binary.flatten(1).sum(dim=1).clamp_min(1)
            overlap_ratios = overlaps.float() / areas.float()
            best = int(overlaps.argmax().item())
            actions['dominant_query'] = self._qtd_component_overlapping(
                binary[best],
                region,
                self.query_topology_min_pixels,
            )
            eligible = (
                (overlaps > 0)
                & (overlap_ratios
                   >= float(self.query_topology_action_min_overlap))
            )
            if eligible.any():
                union = binary[eligible].any(dim=0)
                consensus = binary[eligible].sum(dim=0) >= 2
                actions['query_union'] = self._qtd_component_overlapping(
                    union, region, self.query_topology_min_pixels)
                actions['query_consensus'] = (
                    self._qtd_component_overlapping(
                        consensus,
                        region,
                        self.query_topology_min_pixels,
                    )
                )
            else:
                actions['query_union'] = empty
                actions['query_consensus'] = empty
        if semantic_score is None:
            actions['semantic_completion'] = empty
        else:
            semantic_mask = (
                semantic_score
                >= float(self.query_topology_semantic_threshold)
            )
            actions['semantic_completion'] = (
                self._qtd_component_overlapping(
                    semantic_mask,
                    region,
                    self.query_topology_min_pixels,
                )
            )
        return actions

    @staticmethod
    def _qtd_eval_window(actions, padding):
        combined = torch.zeros_like(
            next(iter(actions.values())), dtype=torch.bool)
        for mask in actions.values():
            combined |= mask
        coordinates = torch.nonzero(combined, as_tuple=False)
        window = torch.zeros_like(combined, dtype=torch.bool)
        if coordinates.numel() == 0:
            return window
        height, width = combined.shape
        y1 = max(0, int(coordinates[:, 0].min().item()) - padding)
        y2 = min(height, int(coordinates[:, 0].max().item()) + padding + 1)
        x1 = max(0, int(coordinates[:, 1].min().item()) - padding)
        x2 = min(width, int(coordinates[:, 1].max().item()) + padding + 1)
        window[y1:y2, x1:x2] = True
        return window

    @staticmethod
    def _qtd_formation_label(topology, instance_fraction):
        if instance_fraction < 0.10:
            return 'semantic_dominant'
        if topology['supported_fraction'] < 0.25:
            return 'weak_instance_support'
        if (
                topology['dominant_source_coverage'] >= 0.75
                and topology['effective_source_count'] <= 2.0):
            return 'coherent_single_source'
        if (
                topology['consensus_fraction'] >= 0.50
                and topology['switch_boundary_density'] <= 0.10):
            return 'overlap_consensus'
        if (
                topology['effective_source_count'] >= 3.0
                and topology['switch_boundary_density'] >= 0.20):
            return 'fragmented_mosaic'
        return 'mixed_multi_source'

    def _qtd_save_npz(self, image_path, arrays, source_metadata):
        if not bool(self.query_topology_save_npz):
            return None
        if self._query_topology_saved_images >= int(
                self.query_topology_max_saved_images):
            return None
        rank = int(os.environ.get('RANK', 0))
        digest = hashlib.sha1(
            str(image_path).encode('utf-8')).hexdigest()[:12]
        directory = os.path.join(
            str(self.query_topology_artifact_dir), f'rank{rank}')
        os.makedirs(directory, exist_ok=True)
        npz_path = os.path.join(directory, f'{digest}.npz')
        json_path = os.path.join(directory, f'{digest}.sources.json')
        np.savez_compressed(
            npz_path,
            **{
                key: value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor) else np.asarray(value)
                for key, value in arrays.items()
            },
        )
        with open(json_path, 'w') as handle:
            json.dump(
                dict(
                    schema_version=SCHEMA_VERSION,
                    image_path=str(image_path),
                    source_metadata=source_metadata,
                ),
                handle,
                indent=2,
                allow_nan=False,
            )
        self._query_topology_saved_images += 1
        return dict(npz=npz_path, sources=json_path)

    def _build_query_topology_diagnostic(
            self,
            base_logits,
            base_pred,
            query_logits,
            query_semantic_logits,
            query_instance_logits,
            components,
            data_sample,
            image_path,
    ):
        gt_sem_seg = getattr(data_sample, 'gt_sem_seg', None)
        if (
                components is None
                or gt_sem_seg is None
                or not hasattr(gt_sem_seg, 'data')):
            return None
        raw_candidates = components.get('raw_mask_candidates') or []
        height, width = base_logits.shape[-2:]
        diag_shape = self._qtd_shape(height, width)
        base_scores = self._interpolate_float32(
            base_logits.detach().float().unsqueeze(0),
            diag_shape,
        ).squeeze(0)
        base_pred_diag = F.interpolate(
            base_pred.float().view(1, 1, height, width),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        query_scores = self._interpolate_float32(
            query_logits.detach().float().unsqueeze(0),
            diag_shape,
        ).squeeze(0)
        query_semantic = self._interpolate_float32(
            query_semantic_logits.detach().float().unsqueeze(0),
            diag_shape,
        ).squeeze(0)
        query_instance = self._interpolate_float32(
            query_instance_logits.detach().float().unsqueeze(0),
            diag_shape,
        ).squeeze(0)
        gt_data = gt_sem_seg.data.squeeze().to(self.device).long()
        gt = F.interpolate(
            gt_data.float().view(1, 1, *gt_data.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid = gt != 255

        evidence = self._qtd_collect_sources(
            raw_candidates,
            (height, width),
            diag_shape,
            int(query_scores.shape[0]),
        )
        reconstruction_error = (
            evidence['reconstructed_query_instance'] - query_instance
        ).abs()
        query_to_class = [
            int(value) for value in self.query_idx.detach().cpu().tolist()
        ]
        class_semantic = self._aggregate_query_logits_to_classes(
            query_semantic)
        source_scores = evidence['source_scores']
        source_binary_masks = evidence['source_binary_masks']
        rolled_source_scores = self._qtd_roll_control(source_scores)
        source_metadata = evidence['source_metadata']
        source_by_class = defaultdict(list)
        for index, metadata in enumerate(source_metadata):
            source_by_class[int(metadata['class_index'])].append(index)

        region_rows = []
        region_label_map = torch.full(
            diag_shape, -1, device=self.device, dtype=torch.int32)
        winner_source_map = torch.full_like(region_label_map, -1)
        winning_prompt_map = torch.full_like(region_label_map, -1)
        head_map = torch.zeros_like(region_label_map, dtype=torch.uint8)
        support_count_map = torch.zeros_like(region_label_map, dtype=torch.int16)
        region_candidates = []
        for class_index in range(int(base_scores.shape[0])):
            if (
                    class_index == int(self.bg_idx)
                    and not bool(self.query_topology_include_background)):
                continue
            for area, region in self._qtd_components(
                    base_pred_diag == class_index,
                    self.query_topology_min_pixels):
                region_candidates.append((area, class_index, region))
        region_candidates.sort(key=lambda item: item[0], reverse=True)
        dropped_regions = max(
            0,
            len(region_candidates) - int(self.query_topology_max_regions),
        )
        region_candidates = region_candidates[
            :int(self.query_topology_max_regions)]

        for area, class_index, region in region_candidates:
            if not (region & valid).any():
                continue
            region_index = len(region_rows)
            prompt_indices = [
                index for index, value in enumerate(query_to_class)
                if value == class_index
            ]
            if not prompt_indices:
                continue
            prompt_tensor = torch.tensor(
                prompt_indices, device=self.device, dtype=torch.long)
            prompt_local = query_scores[prompt_tensor]
            prompt_winner_local = prompt_local.argmax(dim=0)
            prompt_winner = prompt_tensor[prompt_winner_local]
            winning_prompt_map[region] = prompt_winner[region].to(
                winning_prompt_map.dtype)
            semantic_winner = torch.gather(
                query_semantic,
                0,
                prompt_winner.unsqueeze(0),
            ).squeeze(0)
            instance_winner = torch.gather(
                query_instance,
                0,
                prompt_winner.unsqueeze(0),
            ).squeeze(0)
            instance_pixels = instance_winner >= semantic_winner
            instance_fraction = float(
                instance_pixels[region].float().mean().item())
            head_map[region & instance_pixels] = 1

            source_indices = source_by_class.get(class_index, [])
            if source_indices:
                index_tensor = torch.tensor(
                    source_indices, device=self.device, dtype=torch.long)
                class_source_scores = source_scores[index_tensor]
                class_binary_masks = source_binary_masks[index_tensor]
                region_prompt = prompt_winner[region]
                source_prompt = torch.tensor(
                    [
                        source_metadata[index]['query_index']
                        for index in source_indices
                    ],
                    device=self.device,
                    dtype=torch.long,
                )
                prompt_match = (
                    source_prompt[:, None] == region_prompt[None, :])
                filtered_inside = class_source_scores[:, region] * (
                    prompt_match.float())
                maxima, winners = filtered_inside.max(dim=0)
                supported = (
                    maxima
                    >= float(self.query_topology_support_threshold))
                global_winners = index_tensor[winners]
                current_sources = winner_source_map[region]
                current_sources[supported] = global_winners[
                    supported].to(current_sources.dtype)
                winner_source_map[region] = current_sources
                support_count_map[region] = (
                    filtered_inside
                    >= float(self.query_topology_support_threshold)
                ).sum(dim=0).to(support_count_map.dtype)
                class_rolled_scores = rolled_source_scores[index_tensor]
                control_inside = class_rolled_scores[:, region] * (
                    prompt_match.float())
            else:
                class_source_scores = source_scores[:0]
                class_binary_masks = source_binary_masks[:0]
                filtered_inside = source_scores.new_empty(
                    (0, int(region.sum().item())))
                control_inside = filtered_inside

            topology = self._qtd_topology_from_inside(
                filtered_inside,
                region,
                self.query_topology_support_threshold,
            )
            control = self._qtd_topology_from_inside(
                control_inside,
                region,
                self.query_topology_support_threshold,
            )
            actions = self._qtd_action_masks(
                region,
                class_source_scores,
                class_binary_masks,
                class_semantic[class_index],
            )
            window = self._qtd_eval_window(
                actions, int(self.query_topology_action_padding))
            window &= valid
            target = (gt == class_index) & valid
            action_ious = {
                name: self._qtd_local_iou(mask, target, window)
                for name, mask in actions.items()
            }
            baseline_iou = action_ious['baseline']
            alternative_names = [
                name for name in actions if name != 'baseline'
            ]
            best_action = max(
                alternative_names,
                key=lambda name: action_ious[name],
            )
            best_delta = action_ious[best_action] - baseline_iou

            region_scores = base_scores[:, region]
            top_values = torch.topk(
                region_scores, k=min(2, int(base_scores.shape[0])), dim=0
            ).values
            mean_margin = (
                float((top_values[0] - top_values[1]).mean().item())
                if top_values.shape[0] > 1 else 0.0
            )
            valid_region = region & valid
            purity = (
                float((gt[valid_region] == class_index).float().mean().item())
                if valid_region.any() else None
            )
            class_sources = [
                source_metadata[index]
                for index in source_indices
            ]
            formation = self._qtd_formation_label(
                topology, instance_fraction)
            row = dict(
                region_id=int(region_index),
                class_index=int(class_index),
                class_name=str(self.class_names[class_index]),
                is_background=bool(class_index == int(self.bg_idx)),
                area_pixels=int(area),
                area_fraction=_safe_div(area, int(valid.numel())),
                final_score_mean=_finite(
                    base_scores[class_index][region].mean().item()),
                final_score_max=_finite(
                    base_scores[class_index][region].max().item()),
                final_margin_mean=_finite(mean_margin),
                semantic_score_mean=_finite(
                    class_semantic[class_index][region].mean().item()),
                instance_score_mean=_finite(
                    instance_winner[region].mean().item()),
                instance_head_fraction=_finite(instance_fraction),
                gt_purity=_finite(purity),
                baseline_local_iou=_finite(baseline_iou),
                best_action=str(best_action),
                best_action_local_iou=_finite(
                    action_ious[best_action]),
                best_action_delta=_finite(best_delta),
                formation=str(formation),
                source_view_count=len(set(
                    metadata['view_id'] for metadata in class_sources)),
                source_prompt_count=len(set(
                    metadata['query_index'] for metadata in class_sources)),
                **topology,
                **{
                    f'control_roll_{key}': value
                    for key, value in control.items()
                },
                **{
                    f'action_{name}_local_iou': _finite(value)
                    for name, value in action_ious.items()
                },
                **{
                    f'action_{name}_changed_pixels': int(
                        (mask ^ region).sum().item())
                    for name, mask in actions.items()
                },
            )
            region_rows.append(row)
            region_label_map[region] = int(region_index)

        final_values = torch.topk(
            base_scores,
            k=min(2, int(base_scores.shape[0])),
            dim=0,
        ).values
        final_margin = (
            final_values[0] - final_values[1]
            if final_values.shape[0] > 1
            else torch.zeros_like(final_values[0])
        )
        artifact_paths = self._qtd_save_npz(
            image_path,
            dict(
                prediction=base_pred_diag.to(torch.int16),
                ground_truth=gt.to(torch.int16),
                valid_mask=valid.to(torch.uint8),
                region_id=region_label_map,
                winner_source_id=winner_source_map,
                winning_prompt_id=winning_prompt_map,
                winning_head=head_map,
                source_support_count=support_count_map,
                final_confidence=base_scores.max(dim=0).values,
                final_margin=final_margin,
                query_instance_reconstruction_error=(
                    reconstruction_error.max(dim=0).values),
            ),
            source_metadata,
        )
        formation_counts = Counter(
            row['formation'] for row in region_rows)
        raw_view_ids = set(
            str(record.get('view_id', 'unknown'))
            for record in raw_candidates
        )
        expected_raw_records = (
            len(raw_view_ids) * len(query_to_class)
            if raw_view_ids else len(query_to_class)
        )
        return dict(
            schema_version=SCHEMA_VERSION,
            dataset_name=getattr(self, 'seed_dataset_name', None),
            diagnostic_shape=list(diag_shape),
            class_names=list(self.class_names),
            query_to_class=query_to_class,
            prediction_sha1=hashlib.sha1(
                base_pred_diag.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            raw_record_count=len(raw_candidates),
            expected_raw_record_count=int(expected_raw_records),
            raw_record_complete=bool(
                len(raw_candidates) == expected_raw_records),
            reported_candidate_count=evidence[
                'reported_candidate_count'],
            reported_kept_count=evidence['reported_kept_count'],
            selected_kept_count=evidence['selected_kept_count'],
            kept_selection_complete=evidence[
                'kept_selection_complete'],
            reconstruction_mae=_finite(
                reconstruction_error.mean().item()),
            reconstruction_max_error=_finite(
                reconstruction_error.max().item()),
            region_count=len(region_rows),
            dropped_region_count=int(dropped_regions),
            formation_counts=dict(formation_counts),
            artifact_paths=artifact_paths,
            source_metadata=source_metadata,
            regions=region_rows,
        )

    def _write_query_topology_stats(self, record):
        if not self.dump_query_topology_stats:
            return
        if self._query_topology_stats_file is None:
            path = _ranked_jsonl_path(
                self.query_topology_stats_path
                or 'query_topology.jsonl')
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._query_topology_stats_file = open(path, 'a')
        self._query_topology_stats_file.write(
            json.dumps(record, allow_nan=False) + '\n')
        self._query_topology_stats_file.flush()
