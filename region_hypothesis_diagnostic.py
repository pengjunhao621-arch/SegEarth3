import json
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F


def _safe_div(numerator, denominator):
    if denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _ranked_jsonl_path(path):
    rank = int(os.environ.get('RANK', 0))
    root, ext = os.path.splitext(path)
    if not ext:
        ext = '.jsonl'
    return f'{root}.rank{rank}{ext}'


def _confusion_matrix(gt, pred, valid, class_count):
    encoded = gt[valid].long() * class_count + pred[valid].long()
    matrix = torch.bincount(
        encoded,
        minlength=class_count * class_count,
    ).view(class_count, class_count)
    return matrix.detach().cpu().tolist()


class RegionHypothesisDiagnosticMixin:
    """Prediction-preserving diagnostics for complete SAM3 region hypotheses."""

    def _uses_region_hypothesis_v2(self):
        return bool(self.dump_region_hypothesis_v2_stats)

    @staticmethod
    def _rh2_row_normalize(scores):
        finite = torch.isfinite(scores)
        clean = torch.nan_to_num(
            scores, nan=0.0, posinf=0.0, neginf=0.0)
        count = finite.sum(dim=1, keepdim=True).clamp_min(1)
        mean = clean.sum(dim=1, keepdim=True) / count
        centered = torch.where(
            finite, scores - mean, torch.zeros_like(scores))
        variance = centered.square().sum(dim=1, keepdim=True) / count
        normalized = centered / variance.sqrt().clamp_min(1e-6)
        single = finite.sum(dim=1, keepdim=True) == 1
        normalized = torch.where(
            single & finite,
            torch.ones_like(normalized),
            normalized,
        )
        normalized = torch.where(
            single & (~finite),
            torch.zeros_like(normalized),
            normalized,
        )
        normalized[~finite & (~single.expand_as(finite))] = -1e6
        return normalized

    def _rh2_class_role(self, class_idx):
        name = str(self.class_names[int(class_idx)]).lower()
        tokens = {
            'background': (
                'background', 'clutter', 'other', 'unknown'),
            'vegetation': (
                'vegetation', 'grass', 'tree', 'forest',
                'agricultural', 'cropland', 'crop'),
            'surface': (
                'road', 'pavement', 'impervious', 'bareland',
                'barren', 'soil', 'water', 'river', 'harbor',
                'track', 'field', 'court', 'pool'),
            'object': (
                'building', 'roof', 'house', 'wall', 'facade',
                'vehicle', 'car', 'ship', 'plane', 'helicopter',
                'bridge', 'tank', 'diamond', 'roundabout'),
        }
        for role, role_tokens in tokens.items():
            if any(token in name for token in role_tokens):
                return role
        return 'unknown'

    @staticmethod
    def _rh2_area_bin(area_ratio):
        if area_ratio < 0.01:
            return 'small'
        if area_ratio < 0.10:
            return 'medium'
        return 'large'

    @staticmethod
    def _rh2_shape_bin(fill_ratio, core_ratio):
        if fill_ratio >= 0.60 and core_ratio >= 0.45:
            return 'compact'
        if core_ratio <= 0.15:
            return 'thin_boundary'
        return 'irregular'

    def _rh2_region_geometry(
            self, soft_masks, binary_masks, metadata):
        region_count, height, width = binary_masks.shape
        overlap = binary_masks.sum(dim=0)
        kernel = max(3, int(self.region_hypothesis_v2_core_kernel))
        if kernel % 2 == 0:
            kernel += 1
        eroded = -F.max_pool2d(
            -binary_masks.float().unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).squeeze(1)
        eroded = eroded >= 0.999
        rows = []
        for region_idx, mask in enumerate(binary_masks):
            area = int(mask.sum().item())
            ys, xs = torch.nonzero(mask, as_tuple=True)
            if area > 0:
                box_area = (
                    int(ys.max().item() - ys.min().item() + 1)
                    * int(xs.max().item() - xs.min().item() + 1)
                )
                fill = float(area) / float(max(1, box_area))
                core = float(eroded[region_idx].sum().item()) / area
                overlap_mean = float(
                    overlap[mask].float().mean().item())
            else:
                fill = 0.0
                core = 0.0
                overlap_mean = 0.0
            raw_score = float(
                metadata[region_idx].get('raw_score', 0.0))
            raw_presence = float(
                metadata[region_idx].get(
                    'raw_presence_score', raw_score))
            if not 0.0 <= raw_score <= 1.0:
                raw_score = float(torch.sigmoid(
                    torch.tensor(raw_score)).item())
            if not 0.0 <= raw_presence <= 1.0:
                raw_presence = float(torch.sigmoid(
                    torch.tensor(raw_presence)).item())
            rows.append(dict(
                area_pixels=area,
                area_ratio=float(area) / float(height * width),
                fill_ratio=fill,
                core_ratio=core,
                boundary_ratio=1.0 - core,
                overlap_mean=overlap_mean,
                overlap_stability=1.0 / max(1.0, overlap_mean),
                raw_score=max(0.0, min(1.0, raw_score)),
                raw_presence=max(
                    0.0, min(1.0, raw_presence)),
                kept=float(bool(
                    metadata[region_idx].get('kept', False))),
            ))
        return rows, overlap

    @staticmethod
    def _rh2_pool_maps(class_maps, soft_masks):
        flat_maps = class_maps.flatten(1)
        flat_masks = soft_masks.flatten(1)
        area = flat_masks.sum(dim=1).clamp_min(1e-6)
        return (flat_masks @ flat_maps.T) / area.unsqueeze(1)

    def _rh2_formula_bank(
            self, query_maps, soft_masks, binary_masks, metadata,
            query_presence):
        query_mean = query_maps.flatten(1).mean(dim=1)
        query_std = query_maps.flatten(1).std(
            dim=1, unbiased=False).clamp_min(1e-6)
        query_z = (
            query_maps - query_mean[:, None, None]
        ) / query_std[:, None, None]
        mean, top, ring = self._region_readout_query_scores(
            query_maps, soft_masks, binary_masks)
        z_mean, z_top, z_ring = self._region_readout_query_scores(
            query_z, soft_masks, binary_masks)
        bank = {
            'inside_mean': self._region_readout_aggregate_queries(mean),
            'inside_top': self._region_readout_aggregate_queries(top),
            'contrast_mean': self._region_readout_aggregate_queries(
                mean - ring),
            'contrast_top': self._region_readout_aggregate_queries(
                top - ring),
            'zcontrast_mean': self._region_readout_aggregate_queries(
                z_mean - z_ring),
            'zcontrast_top': self._region_readout_aggregate_queries(
                z_top - z_ring),
        }
        origin = torch.full(
            (len(metadata), self.num_cls),
            float('-inf'),
            device=self.device,
            dtype=torch.float32,
        )
        for region_idx, row in enumerate(metadata):
            class_idx = int(row.get('class_index', -1))
            if 0 <= class_idx < self.num_cls:
                origin[region_idx, class_idx] = float(
                    row.get('raw_score', 0.0))
        bank['origin_prompt'] = origin
        presence_prior = self._region_readout_presence_prior(
            query_presence)
        bank['contrast_presence'] = (
            bank['contrast_top']
            + float(self.region_hypothesis_v2_presence_weight)
            * presence_prior.unsqueeze(0)
        )
        requested = self._region_readout_parse_names(
            self.region_hypothesis_v2_formulas,
            ','.join(bank),
        )
        return {
            name: self._rh2_row_normalize(bank[name])
            for name in requested
            if name in bank
        }

    @staticmethod
    def _rh2_formula_state(formula_bank):
        names = list(formula_bank)
        scores = torch.stack(
            [formula_bank[name] for name in names], dim=0)
        top_values, predictions = torch.topk(
            scores, k=min(2, scores.shape[-1]), dim=2)
        pred = predictions[:, :, 0]
        if scores.shape[-1] > 1:
            margin = top_values[:, :, 0] - top_values[:, :, 1]
        else:
            margin = top_values[:, :, 0]
        probabilities = torch.softmax(scores, dim=2)
        entropy = -(
            probabilities
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=2)
        entropy = entropy / max(
            1e-6, math.log(max(2, scores.shape[-1])))
        return names, scores, pred, margin, entropy

    def _rh2_select_formulas(
            self, selector_name, formula_names, formula_scores,
            formula_pred, formula_margin, metadata):
        formula_count, region_count = formula_pred.shape
        selected = torch.zeros(
            region_count, device=self.device, dtype=torch.long)
        majority_class = torch.zeros_like(selected)
        for region_idx in range(region_count):
            counts = torch.bincount(
                formula_pred[:, region_idx],
                minlength=self.num_cls,
            )
            majority = int(counts.argmax().item())
            majority_class[region_idx] = majority
            eligible = torch.ones(
                formula_count,
                device=self.device,
                dtype=torch.bool,
            )
            if selector_name == 'majority_margin':
                eligible = formula_pred[:, region_idx] == majority
            elif selector_name == 'origin_guarded':
                origin = int(
                    metadata[region_idx].get('class_index', -1))
                origin_match = (
                    formula_pred[:, region_idx] == origin)
                eligible = (
                    origin_match
                    if origin_match.any()
                    else formula_pred[:, region_idx] == majority
                )
            elif selector_name == 'role_routed':
                role = self._rh2_class_role(majority)
                preferred = {
                    'background': 'inside_mean',
                    'surface': 'inside_top',
                    'vegetation': 'zcontrast_top',
                    'object': 'contrast_top',
                    'unknown': 'contrast_top',
                }[role]
                if preferred in formula_names:
                    selected[region_idx] = formula_names.index(preferred)
                    continue
                eligible = formula_pred[:, region_idx] == majority
            elif selector_name != 'max_margin':
                raise ValueError(
                    f'Unknown region formula selector {selector_name!r}.')
            candidate_margin = formula_margin[:, region_idx].clone()
            candidate_margin[~eligible] = -1e6
            selected[region_idx] = candidate_margin.argmax()
        region_indices = torch.arange(
            region_count, device=self.device)
        selected_scores = formula_scores[
            selected, region_indices]
        selected_pred = selected_scores.argmax(dim=1)
        return selected, selected_scores, selected_pred, majority_class

    def _rh2_selector_quality(
            self, selected_formula, selected_scores, selected_pred,
            formula_pred, formula_margin, formula_entropy,
            head_scores, base_pred, binary_masks, metadata,
            geometry):
        region_count = selected_pred.numel()
        region_indices = torch.arange(
            region_count, device=self.device)
        selected_margin = formula_margin[
            selected_formula, region_indices]
        selected_entropy = formula_entropy[
            selected_formula, region_indices]
        agreement = (
            formula_pred == selected_pred.unsqueeze(0)
        ).float().mean(dim=0)
        head_pred = torch.stack(
            [scores.argmax(dim=1) for scores in head_scores],
            dim=0,
        )
        head_agreement = (
            head_pred == selected_pred.unsqueeze(0)
        ).float().mean(dim=0)
        raw_prompt_agreement = torch.tensor(
            [
                float(
                    int(metadata[idx].get('class_index', -1))
                    == int(selected_pred[idx].item()))
                for idx in range(region_count)
            ],
            device=self.device,
        )
        baseline_consistency = torch.zeros(
            region_count, device=self.device)
        baseline_topk = torch.zeros_like(baseline_consistency)
        pooled_final = head_scores[-1]
        final_topk = torch.topk(
            pooled_final,
            k=max(
                1,
                min(
                    int(self.region_hypothesis_v2_topk),
                    self.num_cls),
            ),
            dim=1,
        ).indices
        for region_idx, mask in enumerate(binary_masks):
            baseline_consistency[region_idx] = (
                (base_pred[mask] == selected_pred[region_idx])
                .float().mean()
                if mask.any() else 0.0
            )
            baseline_topk[region_idx] = float(
                (
                    final_topk[region_idx]
                    == selected_pred[region_idx]
                ).any().item()
            )
        margin_score = 1.0 - torch.exp(
            -selected_margin.clamp_min(0.0))
        entropy_score = 1.0 - selected_entropy.clamp(0.0, 1.0)
        objectness = torch.tensor(
            [row['raw_score'] for row in geometry],
            device=self.device)
        presence = torch.tensor(
            [row['raw_presence'] for row in geometry],
            device=self.device)
        fill = torch.tensor(
            [row['fill_ratio'] for row in geometry],
            device=self.device)
        core = torch.tensor(
            [row['core_ratio'] for row in geometry],
            device=self.device)
        overlap = torch.tensor(
            [row['overlap_stability'] for row in geometry],
            device=self.device)
        semantic_quality = torch.stack(
            [
                agreement,
                head_agreement,
                raw_prompt_agreement,
                baseline_topk,
                baseline_consistency,
                margin_score,
                entropy_score,
            ],
            dim=0,
        ).mean(dim=0)
        spatial_quality = torch.stack(
            [
                objectness.sqrt(),
                presence.sqrt(),
                fill.sqrt(),
                core.sqrt(),
                overlap,
            ],
            dim=0,
        ).mean(dim=0)
        quality = (
            0.70 * semantic_quality
            + 0.30 * spatial_quality
        ).clamp(0.0, 1.0)
        features = dict(
            quality=quality,
            semantic_quality=semantic_quality,
            spatial_quality=spatial_quality,
            formula_agreement=agreement,
            head_agreement=head_agreement,
            raw_prompt_agreement=raw_prompt_agreement,
            baseline_topk=baseline_topk,
            baseline_consistency=baseline_consistency,
            margin_score=margin_score,
            entropy_score=entropy_score,
            objectness=objectness,
            presence=presence,
            fill_ratio=fill,
            core_ratio=core,
            overlap_stability=overlap,
        )
        return quality, features

    def _rh2_project(
            self, projection_name, region_scores, soft_masks,
            binary_masks, metadata, geometry, active=None,
            quality=None):
        region_count, height, width = binary_masks.shape
        if active is None:
            active = torch.ones(
                region_count, device=self.device, dtype=torch.bool)
        if quality is None:
            quality = torch.ones(
                region_count, device=self.device)
        objectness = torch.tensor(
            [row['raw_score'] for row in geometry],
            device=self.device,
            dtype=torch.float32,
        )
        active_masks = binary_masks & active[:, None, None]
        coverage = active_masks.any(dim=0)
        overlap = active_masks.sum(dim=0)
        extra_gate = torch.ones(
            (height, width), device=self.device, dtype=torch.bool)
        quality_map = torch.zeros(
            (height, width), device=self.device)

        if projection_name == 'class_wise_max':
            class_maps = torch.full(
                (self.num_cls, height, width),
                -1e6,
                device=self.device,
            )
            for region_idx, mask in enumerate(active_masks):
                if not mask.any():
                    continue
                class_maps[:, mask] = torch.maximum(
                    class_maps[:, mask],
                    region_scores[region_idx].unsqueeze(1),
                )
                quality_map[mask] = torch.maximum(
                    quality_map[mask], quality[region_idx])
            return class_maps, coverage, extra_gate, quality_map

        weights = (
            soft_masks
            * objectness[:, None, None]
            * active[:, None, None].float()
        )
        if projection_name == 'soft_mask_mixture':
            flat_weights = weights.flatten(1)
            denominator = flat_weights.sum(dim=0).clamp_min(1e-6)
            class_maps = (
                region_scores.T @ flat_weights
            ).view(self.num_cls, height, width)
            class_maps = class_maps / denominator.view(1, height, width)
            quality_map = (
                quality.unsqueeze(0) @ flat_weights
            ).view(height, width) / denominator.view(height, width)
            class_maps[:, ~coverage] = -1e6
            return class_maps, coverage, extra_gate, quality_map

        selector = weights.masked_fill(~active_masks, -1.0)
        winner_value, winner_idx = selector.max(dim=0)
        coverage = winner_value >= 0.0
        flat_idx = winner_idx.flatten()
        class_maps = region_scores[flat_idx].view(
            height, width, self.num_cls).permute(2, 0, 1)
        quality_map = quality[flat_idx].view(height, width)
        class_maps[:, ~coverage] = -1e6
        quality_map[~coverage] = 0.0
        if projection_name == 'overlap_abstain':
            extra_gate &= (
                overlap
                <= int(self.region_hypothesis_v2_overlap_max))
        elif projection_name not in (
                'region_winner', 'baseline_compete'):
            raise ValueError(
                f'Unknown region projection {projection_name!r}.')
        return class_maps, coverage, extra_gate, quality_map

    def _rh2_action_from_projection(
            self, base_scores, base_pred, class_maps, coverage,
            extra_gate, projection_name):
        region_scores = self._region_readout_class_zscore(class_maps)
        region_scores = torch.nan_to_num(
            region_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        base_z = self._region_readout_class_zscore(base_scores)
        mixed = (
            base_z
            + float(self.region_hypothesis_v2_blend)
            * region_scores
        )
        region_values, region_pred = torch.topk(
            region_scores,
            k=min(2, self.num_cls),
            dim=0,
        )
        if self.num_cls > 1:
            region_margin = region_values[0] - region_values[1]
        else:
            region_margin = region_values[0]
        candidate = mixed.argmax(dim=0)
        gate = (
            coverage
            & extra_gate
            & (
                region_margin
                >= float(self.region_hypothesis_v2_min_margin)
            )
        )
        topk = torch.topk(
            base_scores,
            k=max(
                1,
                min(
                    int(self.region_hypothesis_v2_topk),
                    self.num_cls),
            ),
            dim=0,
        ).indices
        gate &= (
            topk == candidate.unsqueeze(0)).any(dim=0)
        if projection_name == 'baseline_compete':
            base_region_score = torch.gather(
                region_scores,
                0,
                base_pred.clamp(
                    0, self.num_cls - 1).unsqueeze(0),
            ).squeeze(0)
            candidate_region_score = torch.gather(
                region_scores,
                0,
                candidate.unsqueeze(0),
            ).squeeze(0)
            gate &= (
                candidate_region_score - base_region_score
                >= float(
                    self.region_hypothesis_v2_baseline_advantage)
            )
        pred = base_pred.clone()
        pred[gate] = candidate[gate]
        return pred, gate

    def _rh2_exact_action_row(
            self, action_name, pred_diag, base_pred_diag,
            base_pred_full, gt_full, valid_full):
        changed_diag = pred_diag != base_pred_diag
        candidate_full = F.interpolate(
            pred_diag.float().view(1, 1, *pred_diag.shape),
            size=base_pred_full.shape,
            mode='nearest',
        ).squeeze().long()
        changed_full = F.interpolate(
            changed_diag.float().view(
                1, 1, *changed_diag.shape),
            size=base_pred_full.shape,
            mode='nearest',
        ).squeeze().bool()
        pred_full = base_pred_full.clone()
        pred_full[changed_full] = candidate_full[changed_full]
        base_correct = (base_pred_full == gt_full) & valid_full
        pred_correct = (pred_full == gt_full) & valid_full
        changed = (pred_full != base_pred_full) & valid_full
        improved = changed & (~base_correct) & pred_correct
        harmed = changed & base_correct & (~pred_correct)
        wrong_to_wrong = (
            changed & (~base_correct) & (~pred_correct))
        return dict(
            action_name=action_name,
            confusion=_confusion_matrix(
                gt_full, pred_full, valid_full, self.num_cls),
            changed_pixels=int(changed.sum().item()),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            wrong_to_wrong_pixels=int(
                wrong_to_wrong.sum().item()),
            utility_precision=_safe_div(
                int(improved.sum().item()),
                int(improved.sum().item())
                + int(harmed.sum().item())),
        )

    def _rh2_region_targets(
            self, binary_masks, gt, valid):
        majority = torch.zeros(
            binary_masks.shape[0],
            device=self.device,
            dtype=torch.long,
        )
        purity = torch.zeros(
            binary_masks.shape[0], device=self.device)
        valid_pixels = torch.zeros_like(majority)
        for region_idx, mask in enumerate(binary_masks):
            mask = mask & valid
            pixels = int(mask.sum().item())
            valid_pixels[region_idx] = pixels
            if pixels == 0:
                continue
            counts = torch.bincount(
                gt[mask], minlength=self.num_cls)
            count, class_idx = counts.max(dim=0)
            majority[region_idx] = class_idx
            purity[region_idx] = count.float() / pixels
        return majority, purity, valid_pixels

    def _rh2_formula_rows(
            self, formula_names, formula_pred, formula_margin,
            formula_entropy, gt_majority, gt_purity, valid_pixels,
            metadata, geometry):
        groups = defaultdict(lambda: dict(
            regions=0,
            correct_regions=0,
            high_purity_regions=0,
            high_purity_correct_regions=0,
            origin_correct_regions=0,
            margin_sum=0.0,
            entropy_sum=0.0,
            purity_sum=0.0,
        ))
        high_purity = float(
            self.region_hypothesis_v2_high_purity)
        for formula_idx, formula_name in enumerate(formula_names):
            for region_idx in range(gt_majority.numel()):
                if int(valid_pixels[region_idx].item()) <= 0:
                    continue
                gt_class = int(gt_majority[region_idx].item())
                role = self._rh2_class_role(gt_class)
                area_bin = self._rh2_area_bin(
                    geometry[region_idx]['area_ratio'])
                shape_bin = self._rh2_shape_bin(
                    geometry[region_idx]['fill_ratio'],
                    geometry[region_idx]['core_ratio'],
                )
                key = (formula_name, role, area_bin, shape_bin)
                bucket = groups[key]
                correct = int(
                    formula_pred[
                        formula_idx, region_idx].item()
                    == gt_class)
                is_high = (
                    float(gt_purity[region_idx].item())
                    >= high_purity)
                origin_correct = int(
                    int(metadata[region_idx].get(
                        'class_index', -1)) == gt_class)
                bucket['regions'] += 1
                bucket['correct_regions'] += correct
                bucket['high_purity_regions'] += int(is_high)
                bucket['high_purity_correct_regions'] += (
                    int(is_high) * correct)
                bucket['origin_correct_regions'] += origin_correct
                bucket['margin_sum'] += float(
                    formula_margin[
                        formula_idx, region_idx].item())
                bucket['entropy_sum'] += float(
                    formula_entropy[
                        formula_idx, region_idx].item())
                bucket['purity_sum'] += float(
                    gt_purity[region_idx].item())
        return [
            dict(
                formula_name=key[0],
                gt_role=key[1],
                area_bin=key[2],
                shape_bin=key[3],
                **bucket,
            )
            for key, bucket in groups.items()
        ]

    def _rh2_selector_rows(
            self, selector_name, selected_formula, selected_pred,
            quality, formula_names, formula_oracle, gt_majority,
            gt_purity, valid_pixels, binary_masks, base_pred, gt,
            valid):
        rows = []
        correct = selected_pred == gt_majority
        formula_match = selected_formula == formula_oracle
        useful = torch.zeros_like(correct)
        for region_idx, mask in enumerate(binary_masks):
            region_valid = mask & valid
            if not region_valid.any():
                continue
            candidate = selected_pred[region_idx]
            base_correct = (
                base_pred[region_valid] == gt[region_valid])
            candidate_correct = (
                candidate == gt[region_valid])
            improved = int(
                ((~base_correct) & candidate_correct).sum().item())
            harmed = int(
                (base_correct & (~candidate_correct)).sum().item())
            useful[region_idx] = improved > harmed
        thresholds = self._region_readout_parse_floats(
            self.region_hypothesis_v2_selector_thresholds,
            '0.45,0.55,0.65,0.75',
        )
        for threshold in [None] + thresholds:
            selected = valid_pixels > 0
            if threshold is not None:
                selected &= quality >= float(threshold)
            rows.append(dict(
                selector_name=selector_name,
                threshold=(
                    '__all__' if threshold is None
                    else float(threshold)),
                regions=int(selected.sum().item()),
                correct_regions=int(
                    (selected & correct).sum().item()),
                formula_oracle_match_regions=int(
                    (selected & formula_match).sum().item()),
                useful_regions=int(
                    (selected & useful).sum().item()),
                high_purity_regions=int(
                    (
                        selected
                        & (
                            gt_purity
                            >= float(
                                self.region_hypothesis_v2_high_purity)
                        )
                    ).sum().item()),
                selected_formula_counts={
                    formula_names[formula_idx]: int(
                        (
                            selected
                            & (selected_formula == formula_idx)
                        ).sum().item())
                    for formula_idx in range(len(formula_names))
                },
            ))
        return rows, useful

    def _rh2_quality_bin_rows(
            self, selector_name, quality, features, selected_pred,
            gt_majority, useful, valid_pixels):
        bins = self._region_readout_parse_floats(
            self.region_hypothesis_v2_quality_bins,
            '0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01',
        )
        if len(bins) < 2:
            bins = [0.0, 1.01]
        rows = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            selected = (
                (valid_pixels > 0)
                & (quality >= lower)
                & (quality < upper)
            )
            count = int(selected.sum().item())
            if count == 0:
                continue
            row = dict(
                selector_name=selector_name,
                quality_lower=float(lower),
                quality_upper=float(upper),
                regions=count,
                correct_regions=int(
                    (
                        selected
                        & (selected_pred == gt_majority)
                    ).sum().item()),
                useful_regions=int(
                    (selected & useful).sum().item()),
            )
            for feature_name, values in features.items():
                row[f'{feature_name}_sum'] = float(
                    values[selected].sum().item())
            rows.append(row)
        return rows

    def _build_region_hypothesis_v2_diagnostic(
            self, base_logits, base_pred, query_semantic_logits,
            components, data_sample):
        height, width = base_logits.shape[-2:]
        diag_shape = self._region_readout_shape(height, width)
        base_scores = self._interpolate_float32(
            base_logits.detach().unsqueeze(0),
            diag_shape,
        ).squeeze(0)
        base_pred_diag = F.interpolate(
            base_pred.float().view(1, 1, height, width),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        query_maps = self._interpolate_float32(
            query_semantic_logits.detach().float().unsqueeze(0),
            diag_shape,
        ).squeeze(0)
        gt_sem_seg = getattr(data_sample, 'gt_sem_seg', None)
        gt_data = (
            gt_sem_seg.data.squeeze().to(self.device).long()
            if gt_sem_seg is not None
            and hasattr(gt_sem_seg, 'data')
            else None
        )
        if gt_data is None:
            return None
        gt_full = F.interpolate(
            gt_data.float().view(1, 1, *gt_data.shape[-2:]),
            size=(height, width),
            mode='nearest',
        ).squeeze().long()
        valid_full = gt_full != 255
        gt = F.interpolate(
            gt_data.float().view(1, 1, *gt_data.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid = gt != 255
        baseline_row = dict(
            action_name='baseline',
            confusion=_confusion_matrix(
                gt_full, base_pred, valid_full, self.num_cls),
            changed_pixels=0,
            improved_pixels=0,
            harmed_pixels=0,
            wrong_to_wrong_pixels=0,
            utility_precision=None,
        )
        soft_masks, binary_masks, metadata = (
            self._region_readout_collect_regions(
                components.get('raw_mask_candidates') or [],
                (height, width),
                diag_shape,
            )
        )
        if soft_masks is None:
            return dict(
                dataset_name=getattr(
                    self, 'seed_dataset_name', None),
                class_names=list(self.class_names),
                diagnostic_shape=list(diag_shape),
                region_count=0,
                formula_names=[],
                formula_rows=[],
                selector_rows=[],
                quality_bin_rows=[],
                action_rows=[baseline_row],
            )

        geometry, overlap = self._rh2_region_geometry(
            soft_masks, binary_masks, metadata)
        formula_bank = self._rh2_formula_bank(
            query_maps,
            soft_masks,
            binary_masks,
            metadata,
            components.get('presence_query_scores'),
        )
        (
            formula_names,
            formula_scores,
            formula_pred,
            formula_margin,
            formula_entropy,
        ) = self._rh2_formula_state(formula_bank)
        gt_majority, gt_purity, valid_pixels = (
            self._rh2_region_targets(
                binary_masks, gt, valid))
        gt_indices = gt_majority.view(1, -1, 1).expand(
            formula_scores.shape[0], -1, 1)
        gt_scores = torch.gather(
            formula_scores, 2, gt_indices).squeeze(2)
        other_scores = formula_scores.clone()
        other_scores.scatter_(2, gt_indices, -1e6)
        formula_oracle = (
            gt_scores - other_scores.max(dim=2).values
        ).argmax(dim=0)
        region_indices = torch.arange(
            binary_masks.shape[0], device=self.device)
        oracle_scores = formula_scores[
            formula_oracle, region_indices]
        oracle_pred = oracle_scores.argmax(dim=1)

        class_semantic = self._interpolate_float32(
            components['semantic_logits'].detach().float().unsqueeze(0),
            diag_shape,
        ).squeeze(0)
        class_instance = self._interpolate_float32(
            components['instance_logits'].detach().float().unsqueeze(0),
            diag_shape,
        ).squeeze(0)
        head_scores = [
            self._rh2_pool_maps(class_semantic, soft_masks),
            self._rh2_pool_maps(class_instance, soft_masks),
            self._rh2_pool_maps(base_scores, soft_masks),
        ]

        selector_names = self._region_readout_parse_names(
            self.region_hypothesis_v2_formula_selectors,
            'max_margin,majority_margin,origin_guarded,role_routed',
        )
        selector_states = {}
        selector_rows = []
        quality_bin_rows = []
        for selector_name in selector_names:
            (
                selected_formula,
                selected_scores,
                selected_pred,
                _,
            ) = self._rh2_select_formulas(
                selector_name,
                formula_names,
                formula_scores,
                formula_pred,
                formula_margin,
                metadata,
            )
            quality, features = self._rh2_selector_quality(
                selected_formula,
                selected_scores,
                selected_pred,
                formula_pred,
                formula_margin,
                formula_entropy,
                head_scores,
                base_pred_diag,
                binary_masks,
                metadata,
                geometry,
            )
            rows, useful = self._rh2_selector_rows(
                selector_name,
                selected_formula,
                selected_pred,
                quality,
                formula_names,
                formula_oracle,
                gt_majority,
                gt_purity,
                valid_pixels,
                binary_masks,
                base_pred_diag,
                gt,
                valid,
            )
            selector_rows.extend(rows)
            quality_bin_rows.extend(
                self._rh2_quality_bin_rows(
                    selector_name,
                    quality,
                    features,
                    selected_pred,
                    gt_majority,
                    useful,
                    valid_pixels,
                )
            )
            selector_states[selector_name] = dict(
                formula=selected_formula,
                scores=selected_scores,
                pred=selected_pred,
                quality=quality,
                useful=useful,
            )

        action_rows = [baseline_row]
        all_active = valid_pixels > 0
        for formula_name in formula_names:
            class_maps, coverage, extra_gate, _ = self._rh2_project(
                'region_winner',
                formula_bank[formula_name],
                soft_masks,
                binary_masks,
                metadata,
                geometry,
                active=all_active,
            )
            pred, _ = self._rh2_action_from_projection(
                base_scores,
                base_pred_diag,
                class_maps,
                coverage,
                extra_gate,
                'region_winner',
            )
            action_rows.append(self._rh2_exact_action_row(
                f'formula__{formula_name}',
                pred,
                base_pred_diag,
                base_pred,
                gt_full,
                valid_full,
            ))

        shared_selector = (
            'majority_margin'
            if 'majority_margin' in selector_states
            else selector_names[0]
        )
        shared = selector_states[shared_selector]
        projection_names = self._region_readout_parse_names(
            self.region_hypothesis_v2_projections,
            (
                'class_wise_max,region_winner,soft_mask_mixture,'
                'baseline_compete,overlap_abstain'
            ),
        )
        for projection_name in projection_names:
            class_maps, coverage, extra_gate, _ = self._rh2_project(
                projection_name,
                shared['scores'],
                soft_masks,
                binary_masks,
                metadata,
                geometry,
                active=all_active,
                quality=shared['quality'],
            )
            pred, _ = self._rh2_action_from_projection(
                base_scores,
                base_pred_diag,
                class_maps,
                coverage,
                extra_gate,
                projection_name,
            )
            action_rows.append(self._rh2_exact_action_row(
                f'projection__{projection_name}',
                pred,
                base_pred_diag,
                base_pred,
                gt_full,
                valid_full,
            ))

        class_maps, coverage, extra_gate, _ = self._rh2_project(
            'region_winner',
            oracle_scores,
            soft_masks,
            binary_masks,
            metadata,
            geometry,
            active=all_active,
        )
        formula_oracle_pred, _ = self._rh2_action_from_projection(
            base_scores,
            base_pred_diag,
            class_maps,
            coverage,
            extra_gate,
            'region_winner',
        )
        action_rows.append(self._rh2_exact_action_row(
            'oracle__formula',
            formula_oracle_pred,
            base_pred_diag,
            base_pred,
            gt_full,
            valid_full,
        ))

        oracle_useful = torch.zeros_like(all_active)
        for region_idx, mask in enumerate(binary_masks):
            region_valid = mask & valid
            if not region_valid.any():
                continue
            base_correct = (
                base_pred_diag[region_valid] == gt[region_valid])
            candidate_correct = (
                oracle_pred[region_idx] == gt[region_valid])
            improved = int(
                ((~base_correct) & candidate_correct).sum().item())
            harmed = int(
                (base_correct & (~candidate_correct)).sum().item())
            oracle_useful[region_idx] = improved > harmed
        class_maps, coverage, extra_gate, _ = self._rh2_project(
            'region_winner',
            oracle_scores,
            soft_masks,
            binary_masks,
            metadata,
            geometry,
            active=oracle_useful,
        )
        region_oracle_pred, _ = self._rh2_action_from_projection(
            base_scores,
            base_pred_diag,
            class_maps,
            coverage,
            extra_gate,
            'region_winner',
        )
        action_rows.append(self._rh2_exact_action_row(
            'oracle__region',
            region_oracle_pred,
            base_pred_diag,
            base_pred,
            gt_full,
            valid_full,
        ))

        projection_oracle_pred = base_pred_diag.clone()
        base_wrong = (base_pred_diag != gt) & valid
        for region_idx, mask in enumerate(binary_masks):
            if not oracle_useful[region_idx]:
                continue
            recover = (
                mask
                & base_wrong
                & (gt == oracle_pred[region_idx])
            )
            projection_oracle_pred[recover] = gt[recover]
        action_rows.append(self._rh2_exact_action_row(
            'oracle__projection',
            projection_oracle_pred,
            base_pred_diag,
            base_pred,
            gt_full,
            valid_full,
        ))

        selector_projection = str(
            self.region_hypothesis_v2_selector_projection
        ).strip().lower()
        thresholds = self._region_readout_parse_floats(
            self.region_hypothesis_v2_selector_thresholds,
            '0.45,0.55,0.65,0.75',
        )
        for selector_name, state in selector_states.items():
            for threshold in thresholds:
                active = (
                    all_active
                    & (state['quality'] >= float(threshold))
                )
                class_maps, coverage, extra_gate, _ = self._rh2_project(
                    selector_projection,
                    state['scores'],
                    soft_masks,
                    binary_masks,
                    metadata,
                    geometry,
                    active=active,
                    quality=state['quality'],
                )
                pred, _ = self._rh2_action_from_projection(
                    base_scores,
                    base_pred_diag,
                    class_maps,
                    coverage,
                    extra_gate,
                    selector_projection,
                )
                action_rows.append(self._rh2_exact_action_row(
                    (
                        f'selector__{selector_name}'
                        f'__t{float(threshold):g}'
                    ),
                    pred,
                    base_pred_diag,
                    base_pred,
                    gt_full,
                    valid_full,
                ))

        return dict(
            dataset_name=getattr(
                self, 'seed_dataset_name', None),
            class_names=list(self.class_names),
            diagnostic_shape=list(diag_shape),
            region_count=int(binary_masks.shape[0]),
            overlap_pixel_ratio=_safe_div(
                int((overlap > 1).sum().item()),
                int((overlap > 0).sum().item())),
            formula_names=formula_names,
            formula_rows=self._rh2_formula_rows(
                formula_names,
                formula_pred,
                formula_margin,
                formula_entropy,
                gt_majority,
                gt_purity,
                valid_pixels,
                metadata,
                geometry,
            ),
            selector_rows=selector_rows,
            quality_bin_rows=quality_bin_rows,
            action_rows=action_rows,
        )

    def _write_region_hypothesis_v2_stats(self, record):
        if not self.dump_region_hypothesis_v2_stats:
            return
        if self._region_hypothesis_v2_stats_file is None:
            path = (
                self.region_hypothesis_v2_stats_path
                or './work_dirs/evidence_stats/'
                   'region_hypothesis_v2.jsonl'
            )
            path = _ranked_jsonl_path(path)
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._region_hypothesis_v2_stats_file = open(
                path, 'a', buffering=1)
        self._region_hypothesis_v2_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')
