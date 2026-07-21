"""Prediction-preserving feasibility diagnostic for SAM3 region proposals.

The diagnostic deliberately stops before RemoteCLIP feature extraction.  Its
job is to answer a more basic causal question: do SAM3-derived regions provide
enough coverage, purity, and top-2 correction opportunities to justify a
learned region verifier?
"""

import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _ranked_jsonl_path(path):
    rank = int(os.environ.get('RANK', 0))
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext or '.jsonl'}"


def _safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


class CandidateRegionQualityDiagnosticMixin:
    """Compare raw, grouped, semantic, and hybrid candidate-region pools."""

    def _uses_candidate_region_quality_diagnostic(self):
        return bool(self.dump_candidate_region_quality_stats)

    @staticmethod
    def _crq_parse_names(value):
        if isinstance(value, str):
            value = value.split(',')
        return [
            str(item).strip().lower()
            for item in (value or [])
            if str(item).strip()
        ]

    def _crq_shape(self, height, width):
        max_side = max(16, int(self.candidate_region_quality_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    @staticmethod
    def _crq_components(mask, min_pixels):
        array = mask.detach().bool().cpu().numpy().astype(np.uint8)
        count, labels = cv2.connectedComponents(array, connectivity=8)
        rows = []
        for label in range(1, count):
            component = torch.from_numpy(labels == label).to(mask.device)
            if int(component.sum().item()) >= int(min_pixels):
                rows.append(component)
        return rows

    @staticmethod
    def _crq_pair_overlap(first, second):
        intersection = int((first & second).sum().item())
        if intersection == 0:
            return 0.0, 0.0
        first_area = int(first.sum().item())
        second_area = int(second.sum().item())
        union = first_area + second_area - intersection
        return (
            float(intersection) / float(max(1, union)),
            float(intersection) / float(max(1, min(first_area, second_area))),
        )

    def _crq_group_instance_regions(self, soft_masks, binary_masks, metadata):
        if binary_masks is None or binary_masks.numel() == 0:
            return None, None, []
        count = int(binary_masks.shape[0])
        parent = list(range(count))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first, second):
            root_first, root_second = find(first), find(second)
            if root_first != root_second:
                parent[root_second] = root_first

        flat = binary_masks.flatten(1).float()
        intersections = flat @ flat.T
        areas = flat.sum(dim=1)
        unions = areas[:, None] + areas[None, :] - intersections
        ious = intersections / unions.clamp_min(1.0)
        containments = intersections / torch.minimum(
            areas[:, None], areas[None, :]).clamp_min(1.0)
        merge = torch.triu(
            (ious >= float(self.candidate_region_quality_group_iou))
            | (containments >= float(
                self.candidate_region_quality_group_containment)),
            diagonal=1,
        )
        for first, second in torch.nonzero(
                merge, as_tuple=False).detach().cpu().tolist():
            union(first, second)

        clusters = defaultdict(list)
        for index in range(count):
            clusters[find(index)].append(index)
        grouped_soft, grouped_binary, grouped_metadata = [], [], []
        for member_ids in clusters.values():
            member = torch.tensor(
                member_ids, device=binary_masks.device, dtype=torch.long)
            merged_soft = soft_masks[member].max(dim=0).values
            merged_binary = binary_masks[member].any(dim=0)
            grouped_soft.append(merged_soft)
            grouped_binary.append(merged_binary)
            prompt_classes = sorted(set(
                int(metadata[index].get('class_index', -1))
                for index in member_ids
                if int(metadata[index].get('class_index', -1)) >= 0
            ))
            grouped_metadata.append(dict(
                source='grouped_instance',
                member_count=len(member_ids),
                prompt_classes=prompt_classes,
                raw_score=max(
                    float(metadata[index].get('raw_score', 0.0))
                    for index in member_ids),
                raw_presence_score=max(
                    float(metadata[index].get(
                        'raw_presence_score', 0.0))
                    for index in member_ids),
                kept=any(bool(metadata[index].get('kept', False))
                         for index in member_ids),
            ))
        return (
            torch.stack(grouped_soft),
            torch.stack(grouped_binary),
            grouped_metadata,
        )

    def _crq_semantic_regions(self, semantic_scores):
        if semantic_scores is None or semantic_scores.numel() == 0:
            return None, None, []
        winner = semantic_scores.argmax(dim=0)
        soft_masks, binary_masks, metadata = [], [], []
        threshold = float(self.candidate_region_quality_semantic_threshold)
        max_per_class = max(
            1, int(self.candidate_region_quality_semantic_max_per_class))
        for class_idx in range(int(semantic_scores.shape[0])):
            foreground = (
                (winner == class_idx)
                & (semantic_scores[class_idx] >= threshold)
            )
            components = self._crq_components(
                foreground, self.candidate_region_quality_min_pixels)
            components.sort(
                key=lambda item: int(item.sum().item()), reverse=True)
            for component in components[:max_per_class]:
                soft_masks.append(semantic_scores[class_idx] * component)
                binary_masks.append(component)
                metadata.append(dict(
                    source='semantic_cc',
                    member_count=1,
                    prompt_classes=[class_idx],
                    raw_score=float(
                        semantic_scores[class_idx][component].mean().item()),
                    raw_presence_score=None,
                    kept=True,
                ))
        if not binary_masks:
            return None, None, []
        return torch.stack(soft_masks), torch.stack(binary_masks), metadata

    def _crq_hybrid_regions(self, grouped, semantic):
        candidates = []
        for source in (grouped, semantic):
            soft_masks, binary_masks, metadata = source
            if binary_masks is None:
                continue
            for index in range(int(binary_masks.shape[0])):
                candidates.append((
                    soft_masks[index], binary_masks[index], metadata[index]))
        if not candidates:
            return None, None, []
        candidate_masks = torch.stack([row[1] for row in candidates])
        flat = candidate_masks.flatten(1).float()
        areas = flat.sum(dim=1)
        intersections = flat @ flat.T
        unions = areas[:, None] + areas[None, :] - intersections
        pair_ious = intersections / unions.clamp_min(1.0)
        order = torch.argsort(areas, descending=True).detach().cpu().tolist()
        selected_ids = []
        dedup_iou = float(self.candidate_region_quality_hybrid_dedup_iou)
        for candidate_id in order:
            duplicate = (
                bool((pair_ious[candidate_id, selected_ids] >= dedup_iou)
                     .any().item())
                if selected_ids else False
            )
            if not duplicate:
                selected_ids.append(candidate_id)
            if len(selected_ids) >= int(
                    self.candidate_region_quality_max_regions):
                break
        selected = [candidates[index] for index in selected_ids]
        return (
            torch.stack([row[0] for row in selected]),
            torch.stack([row[1] for row in selected]),
            [dict(row[2], source='hybrid') for row in selected],
        )

    def _crq_gt_components(self, gt, valid):
        rows = []
        for class_idx in range(self.num_cls):
            for component in self._crq_components(
                    (gt == class_idx) & valid,
                    self.candidate_region_quality_gt_min_pixels):
                rows.append((class_idx, component))
        return rows

    def _crq_region_rows(
            self, source_name, binary_masks, base_scores, base_pred, gt,
            valid):
        totals = defaultdict(float)
        action_by_class = defaultdict(lambda: defaultdict(float))
        if binary_masks is None:
            return totals, action_by_class
        overlap = binary_masks.sum(dim=0)
        totals['regions'] = int(binary_masks.shape[0])
        totals['covered_valid_pixels'] = int(
            ((overlap > 0) & valid).sum().item())
        totals['valid_pixels'] = int(valid.sum().item())
        totals['overlap_valid_pixels'] = int(
            ((overlap > 1) & valid).sum().item())
        high_purity = float(self.candidate_region_quality_high_purity)
        for mask in binary_masks:
            region_valid = mask & valid
            valid_pixels = int(region_valid.sum().item())
            if valid_pixels == 0:
                continue
            labels = gt[region_valid]
            counts = torch.bincount(labels, minlength=self.num_cls)
            gt_class = int(counts.argmax().item())
            purity = float(counts[gt_class].item()) / float(valid_pixels)
            pooled = base_scores[:, region_valid].mean(dim=1)
            top_count = min(2, self.num_cls)
            top_values, top_indices = torch.topk(pooled, k=top_count)
            top1 = int(top_indices[0].item())
            top2 = int(top_indices[1].item()) if top_count > 1 else top1
            margin = (
                float((top_values[0] - top_values[1]).item())
                if top_count > 1 else 0.0)
            baseline_mode = int(torch.bincount(
                base_pred[region_valid], minlength=self.num_cls
            ).argmax().item())
            baseline_consistency = float(
                (base_pred[region_valid] == baseline_mode).float().mean().item())
            if purity >= high_purity and gt_class == top1:
                action = 'keep'
            elif purity >= high_purity and gt_class == top2:
                action = 'switch'
            else:
                action = 'abstain'
            area = int(mask.sum().item())
            totals['evaluated_regions'] += 1
            totals['region_pixels'] += area
            totals['purity_sum'] += purity
            totals['purity_pixel_sum'] += purity * area
            totals['baseline_consistency_sum'] += baseline_consistency
            totals['margin_sum'] += margin
            totals[f'action_{action}_regions'] += 1
            totals[f'action_{action}_pixels'] += area
            if purity >= high_purity:
                totals['high_purity_regions'] += 1
                totals['high_purity_pixels'] += area
                if top1 != gt_class:
                    totals['high_purity_top1_wrong_regions'] += 1
            if gt_class in (top1, top2):
                totals['top2_label_covered_regions'] += 1
            class_bucket = action_by_class[gt_class]
            class_bucket['regions'] += 1
            class_bucket['pixels'] += area
            class_bucket[f'action_{action}_regions'] += 1
            class_bucket[f'action_{action}_pixels'] += area
        return totals, action_by_class

    def _crq_component_rows(self, binary_masks, gt_components):
        totals = defaultdict(float)
        totals['gt_components'] = len(gt_components)
        totals['gt_component_pixels'] = sum(
            int(mask.sum().item()) for _, mask in gt_components)
        if binary_masks is None or not gt_components:
            return totals
        component_labels = torch.zeros_like(
            gt_components[0][1], dtype=torch.long)
        gt_area_values = []
        for component_idx, (_, component) in enumerate(gt_components, 1):
            component_labels[component] = component_idx
            gt_area_values.append(int(component.sum().item()))
        intersections = torch.zeros(
            (int(binary_masks.shape[0]), len(gt_components)),
            device=binary_masks.device,
            dtype=torch.float32,
        )
        for candidate_idx, candidate in enumerate(binary_masks):
            intersections[candidate_idx] = torch.bincount(
                component_labels[candidate],
                minlength=len(gt_components) + 1,
            )[1:].float()
        candidate_areas = binary_masks.flatten(1).sum(
            dim=1, keepdim=True).float()
        gt_areas = torch.tensor(
            gt_area_values,
            device=binary_masks.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        unions = candidate_areas + gt_areas - intersections
        ious = intersections / unions.clamp_min(1.0)
        coverages = intersections / gt_areas.clamp_min(1.0)
        best_ious = ious.max(dim=0).values
        best_coverages = coverages.max(dim=0).values
        fragments_per_gt = (
            coverages >= float(
                self.candidate_region_quality_fragment_coverage)
        ).sum(dim=0)
        for index in range(len(gt_components)):
            best_iou = float(best_ious[index].item())
            best_coverage = float(best_coverages[index].item())
            fragments = int(fragments_per_gt[index].item())
            totals['best_iou_sum'] += best_iou
            totals['best_coverage_sum'] += best_coverage
            for threshold in (0.25, 0.50):
                suffix = str(threshold).replace('.', '')
                if best_iou >= threshold:
                    totals[f'gt_recalled_iou{suffix}'] += 1
            for threshold in (0.50, 0.80):
                suffix = str(threshold).replace('.', '')
                if best_coverage >= threshold:
                    totals[f'gt_recalled_coverage{suffix}'] += 1
            totals['fragment_links'] += fragments
            if fragments > 1:
                totals['fragmented_gt_components'] += 1
        return totals

    def _build_candidate_region_quality_diagnostic(
            self, base_logits, base_pred, components, data_sample):
        if components is None:
            return None
        gt_sem_seg = getattr(data_sample, 'gt_sem_seg', None)
        if gt_sem_seg is None or not hasattr(gt_sem_seg, 'data'):
            return None
        height, width = base_logits.shape[-2:]
        diag_shape = self._crq_shape(height, width)
        base_scores = self._interpolate_float32(
            base_logits.detach().float().unsqueeze(0), diag_shape).squeeze(0)
        base_pred_diag = F.interpolate(
            base_pred.float().view(1, 1, height, width),
            size=diag_shape, mode='nearest').squeeze().long()
        gt_data = gt_sem_seg.data.squeeze().to(self.device).long()
        gt = F.interpolate(
            gt_data.float().view(1, 1, *gt_data.shape[-2:]),
            size=diag_shape, mode='nearest').squeeze().long()
        valid = gt != 255
        semantic = components.get('semantic_logits')
        semantic_scores = (
            self._interpolate_float32(
                semantic.detach().float().unsqueeze(0), diag_shape).squeeze(0)
            if isinstance(semantic, torch.Tensor) else None
        )

        raw_soft, raw_binary, raw_metadata = (
            self._region_readout_collect_regions(
                components.get('raw_mask_candidates') or [],
                (height, width), diag_shape))
        if raw_metadata:
            raw_metadata = [dict(row, source='raw_instance')
                            for row in raw_metadata]
        raw = (raw_soft, raw_binary, raw_metadata)
        grouped = self._crq_group_instance_regions(*raw)
        semantic_regions = self._crq_semantic_regions(semantic_scores)
        hybrid = self._crq_hybrid_regions(grouped, semantic_regions)
        pools = dict(
            raw_instance=raw,
            grouped_instance=grouped,
            semantic_cc=semantic_regions,
            hybrid=hybrid,
        )
        gt_components = self._crq_gt_components(gt, valid)
        source_rows, class_rows = [], []
        for source_name in self._crq_parse_names(
                self.candidate_region_quality_sources):
            if source_name not in pools:
                continue
            _, binary_masks, _ = pools[source_name]
            region, action_by_class = self._crq_region_rows(
                source_name, binary_masks, base_scores, base_pred_diag,
                gt, valid)
            component = self._crq_component_rows(
                binary_masks, gt_components)
            foreground_component = self._crq_component_rows(
                binary_masks,
                [row for row in gt_components if row[0] != int(self.bg_idx)],
            )
            row = dict(source=source_name)
            row.update(region)
            row.update(component)
            row.update({
                f'fg_{key}': value
                for key, value in foreground_component.items()
            })
            source_rows.append(row)
            for class_idx, values in action_by_class.items():
                class_rows.append(dict(
                    source=source_name,
                    class_index=int(class_idx),
                    class_name=str(self.class_names[int(class_idx)]),
                    is_background=bool(int(class_idx) == int(self.bg_idx)),
                    **values,
                ))
        return dict(
            dataset_name=getattr(self, 'seed_dataset_name', None),
            class_names=list(self.class_names),
            diagnostic_shape=list(diag_shape),
            high_purity=float(self.candidate_region_quality_high_purity),
            source_rows=source_rows,
            class_rows=class_rows,
        )

    def _write_candidate_region_quality_stats(self, record):
        if not self.dump_candidate_region_quality_stats:
            return
        if self._candidate_region_quality_stats_file is None:
            path = _ranked_jsonl_path(
                self.candidate_region_quality_stats_path
                or 'candidate_region_quality.jsonl')
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._candidate_region_quality_stats_file = open(path, 'a')
        self._candidate_region_quality_stats_file.write(
            json.dumps(record, allow_nan=False) + '\n')
        self._candidate_region_quality_stats_file.flush()
