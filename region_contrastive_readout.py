import json
import math
import os

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


def _masked_sum(values, mask):
    if not mask.any():
        return 0.0
    return float(values[mask].sum().item())


class RegionContrastiveReadoutMixin:
    """Training-free SAM3 region/prompt readout and strict diagnostics."""

    def _uses_region_contrastive_readout(self):
        return (
            bool(self.use_region_contrastive_readout)
            or bool(self.dump_region_contrastive_readout_stats)
        )

    @staticmethod
    def _region_readout_parse_floats(value, default):
        if value is None:
            value = default
        if isinstance(value, str):
            items = value.split(',')
        elif isinstance(value, (list, tuple)):
            items = value
        else:
            items = [value]
        result = []
        for item in items:
            item = str(item).strip()
            if not item:
                continue
            number = float(item)
            if number not in result:
                result.append(number)
        return result

    @staticmethod
    def _region_readout_parse_names(value, default):
        if value is None:
            value = default
        if isinstance(value, str):
            items = value.split(',')
        elif isinstance(value, (list, tuple)):
            items = value
        else:
            items = [value]
        result = []
        for item in items:
            name = str(item).strip().lower()
            if name and name not in result:
                result.append(name)
        return result

    def _region_readout_shape(self, height, width):
        max_side = max(16, int(self.region_readout_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _region_readout_reconstruct_mask(
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
            diag_shape,
            device=self.device,
            dtype=torch.float32,
        )
        result[dy1:dy2, dx1:dx2] = local
        return result

    def _region_readout_collect_regions(
            self, candidates, full_shape, diag_shape):
        metadata = []
        per_prompt_limit = max(
            1, int(self.region_readout_masks_per_prompt))
        for record_idx, record in enumerate(candidates or []):
            raw_masks = record.get('raw_masks_lowres')
            raw_scores = record.get('raw_scores')
            presence_scores = record.get('raw_presence_scores')
            keep_mask = record.get('raw_keep_mask')
            if (
                    not isinstance(raw_masks, torch.Tensor)
                    or not isinstance(raw_scores, torch.Tensor)
                    or raw_masks.numel() == 0):
                continue
            count = min(
                int(raw_masks.shape[0]),
                int(raw_scores.numel()),
            )
            order = torch.argsort(
                raw_scores[:count].float(),
                descending=True,
            )[:per_prompt_limit]
            for local_idx in order.tolist():
                raw_score = float(raw_scores[local_idx].item())
                presence_score = (
                    float(presence_scores[local_idx].item())
                    if isinstance(presence_scores, torch.Tensor)
                    and local_idx < presence_scores.numel()
                    else raw_score
                )
                metadata.append(dict(
                    record_idx=record_idx,
                    local_idx=int(local_idx),
                    rank_score=raw_score,
                    raw_score=raw_score,
                    raw_presence_score=presence_score,
                    kept=bool(
                        keep_mask[local_idx].item())
                    if isinstance(keep_mask, torch.Tensor)
                    and local_idx < keep_mask.numel()
                    else False,
                    query_index=int(record.get('query_index', -1)),
                    class_index=int(record.get('class_index', -1)),
                    query_word=str(record.get('query_word', '')),
                    crop_box=record.get('crop_box'),
                ))

        metadata.sort(
            key=lambda row: (
                row['rank_score'],
                row['raw_presence_score'],
                row['kept'],
            ),
            reverse=True,
        )
        max_regions = max(1, int(self.region_readout_max_regions))
        mask_threshold = float(self.region_readout_mask_threshold)
        min_pixels = max(1, int(self.region_readout_min_pixels))
        dedup_iou = float(self.region_readout_dedup_iou)
        soft_masks = []
        binary_masks = []
        kept_metadata = []
        for row in metadata:
            record = candidates[row['record_idx']]
            raw_mask = record['raw_masks_lowres'][row['local_idx']]
            soft_mask = self._region_readout_reconstruct_mask(
                raw_mask,
                row['crop_box'],
                full_shape,
                diag_shape,
            )
            binary = soft_mask >= mask_threshold
            area = int(binary.sum().item())
            if area < min_pixels:
                continue
            duplicate = False
            if dedup_iou > 0:
                for previous in binary_masks:
                    intersection = int((binary & previous).sum().item())
                    if intersection == 0:
                        continue
                    union = int((binary | previous).sum().item())
                    if union > 0 and intersection / union >= dedup_iou:
                        duplicate = True
                        break
            if duplicate:
                continue
            row = dict(row)
            row['area_pixels'] = area
            soft_masks.append(soft_mask)
            binary_masks.append(binary)
            kept_metadata.append(row)
            if len(soft_masks) >= max_regions:
                break

        if not soft_masks:
            return None, None, []
        return (
            torch.stack(soft_masks, dim=0),
            torch.stack(binary_masks, dim=0),
            kept_metadata,
        )

    @staticmethod
    def _region_readout_shift_masks(masks):
        height, width = masks.shape[-2:]
        shift_y = max(1, height // 4)
        shift_x = max(1, width // 4)
        shifted = torch.roll(
            masks, shifts=(shift_y, shift_x), dims=(-2, -1))
        shifted[:, :shift_y, :] = False
        shifted[:, :, :shift_x] = False
        return shifted

    @staticmethod
    def _region_readout_rectangle_masks(masks):
        rectangles = torch.zeros_like(masks)
        height, width = masks.shape[-2:]
        for region_idx, mask in enumerate(masks):
            ys, xs = torch.nonzero(
                mask, as_tuple=True)
            if ys.numel() == 0:
                continue
            area = int(ys.numel())
            y1, y2 = int(ys.min().item()), int(ys.max().item()) + 1
            x1, x2 = int(xs.min().item()), int(xs.max().item()) + 1
            box_h = max(1, y2 - y1)
            box_w = max(1, x2 - x1)
            aspect = float(box_w) / float(box_h)
            rect_h = max(1, int(round(math.sqrt(area / max(aspect, 1e-6)))))
            rect_w = max(1, int(round(area / rect_h)))
            rect_h = min(rect_h, height)
            rect_w = min(rect_w, width)
            center_y = int(round(float(ys.float().mean().item())))
            center_x = int(round(float(xs.float().mean().item())))
            ry1 = max(0, min(height - rect_h, center_y - rect_h // 2))
            rx1 = max(0, min(width - rect_w, center_x - rect_w // 2))
            rectangles[
                region_idx,
                ry1:ry1 + rect_h,
                rx1:rx1 + rect_w,
            ] = True
        return rectangles

    def _region_readout_query_scores(
            self, query_maps, soft_masks, binary_masks):
        region_count = int(binary_masks.shape[0])
        query_count = int(query_maps.shape[0])
        flat_query = query_maps.flatten(1)
        flat_soft = soft_masks.flatten(1)
        soft_area = flat_soft.sum(dim=1).clamp_min(1e-6)
        inside_mean = (
            flat_soft @ flat_query.T) / soft_area.unsqueeze(1)

        top_fraction = min(
            1.0,
            max(1e-3, float(self.region_readout_top_fraction)),
        )
        inside_top = torch.zeros(
            (region_count, query_count),
            device=self.device,
            dtype=torch.float32,
        )
        for region_idx, mask in enumerate(binary_masks):
            values = query_maps[:, mask]
            if values.numel() == 0:
                continue
            top_count = max(
                1, int(math.ceil(values.shape[1] * top_fraction)))
            inside_top[region_idx] = torch.topk(
                values, k=top_count, dim=1).values.mean(dim=1)

        kernel = max(3, int(self.region_readout_ring_kernel))
        if kernel % 2 == 0:
            kernel += 1
        dilated = F.max_pool2d(
            binary_masks.float().unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).squeeze(1).bool()
        rings = dilated & (~binary_masks)
        flat_rings = rings.flatten(1).float()
        ring_area = flat_rings.sum(dim=1)
        ring_mean = (
            flat_rings @ flat_query.T
        ) / ring_area.clamp_min(1.0).unsqueeze(1)
        ring_mean = torch.where(
            (ring_area > 0).unsqueeze(1),
            ring_mean,
            inside_mean,
        )
        return inside_mean, inside_top, ring_mean

    def _region_readout_aggregate_queries(self, region_query_scores):
        class_scores = torch.full(
            (region_query_scores.shape[0], self.num_cls),
            float('-inf'),
            device=self.device,
            dtype=torch.float32,
        )
        for class_idx in range(self.num_cls):
            query_ids = torch.nonzero(
                self.query_idx == class_idx,
                as_tuple=False,
            ).flatten()
            if query_ids.numel() > 0:
                class_scores[:, class_idx] = region_query_scores[
                    :, query_ids].max(dim=1).values
        return class_scores

    def _region_readout_project(
            self, region_scores, masks, diag_shape):
        class_maps = torch.full(
            (self.num_cls, *diag_shape),
            float('-inf'),
            device=self.device,
            dtype=torch.float32,
        )
        coverage = masks.any(dim=0)
        for region_idx, mask in enumerate(masks):
            if not mask.any():
                continue
            current = class_maps[:, mask]
            candidate = region_scores[region_idx].unsqueeze(1)
            class_maps[:, mask] = torch.maximum(current, candidate)
        return class_maps, coverage

    def _region_readout_project_winner(
            self, region_scores, soft_masks, binary_masks, metadata):
        raw_quality = torch.tensor(
            [float(row.get('raw_score', 0.0)) for row in metadata],
            device=self.device,
            dtype=torch.float32,
        )
        probability_like = (
            (raw_quality >= 0.0) & (raw_quality <= 1.0))
        raw_quality = torch.where(
            probability_like,
            raw_quality,
            torch.sigmoid(raw_quality),
        ).clamp(0.0, 1.0)
        selector = (
            soft_masks
            * raw_quality[:, None, None]
        )
        selector = selector.masked_fill(
            ~binary_masks,
            -1.0,
        )
        winner_quality, winner_index = selector.max(dim=0)
        coverage = winner_quality >= 0.0
        height, width = coverage.shape
        flat_index = winner_index.flatten()
        class_maps = region_scores[flat_index].view(
            height, width, self.num_cls).permute(2, 0, 1)
        class_maps[:, ~coverage] = float('-inf')
        return class_maps, coverage

    def _region_readout_project_source(
            self, region_scores, soft_masks, binary_masks, metadata,
            diag_shape):
        projection = str(
            self.region_readout_projection or 'class_max'
        ).strip().lower()
        if projection == 'class_max':
            return self._region_readout_project(
                region_scores,
                binary_masks,
                diag_shape,
            )
        if projection == 'region_winner':
            return self._region_readout_project_winner(
                region_scores,
                soft_masks,
                binary_masks,
                metadata,
            )
        raise ValueError(
            "region_readout_projection must be 'class_max' or "
            f"'region_winner', but got {projection!r}")

    @staticmethod
    def _region_readout_class_zscore(scores):
        finite = torch.isfinite(scores)
        clean = torch.nan_to_num(
            scores, nan=0.0, posinf=0.0, neginf=0.0)
        count = finite.sum(dim=0).clamp_min(1)
        mean = clean.sum(dim=0) / count
        centered = torch.where(
            finite,
            scores - mean.unsqueeze(0),
            torch.zeros_like(scores),
        )
        variance = centered.square().sum(dim=0) / count
        normalized = centered / variance.sqrt().clamp_min(1e-6).unsqueeze(0)
        normalized[~finite] = float('-inf')
        return normalized

    def _region_readout_presence_prior(self, query_presence):
        if not isinstance(query_presence, torch.Tensor):
            return torch.zeros(
                self.num_cls, device=self.device, dtype=torch.float32)
        query_presence = query_presence.to(self.device).float().flatten()
        class_presence = torch.zeros(
            self.num_cls, device=self.device, dtype=torch.float32)
        for class_idx in range(self.num_cls):
            query_ids = torch.nonzero(
                self.query_idx == class_idx,
                as_tuple=False,
            ).flatten()
            if query_ids.numel() > 0:
                class_presence[class_idx] = query_presence[
                    query_ids].max()
        prior = class_presence.clamp(1e-4, 1.0).log()
        prior = prior - prior.mean()
        return prior.clamp(-2.0, 2.0)

    def _region_readout_prepare_route_candidate(
            self, base_logits, source_map, coverage, blend,
            source_maps, coverage_maps):
        height, width = base_logits.shape[-2:]
        region_z_diag = self._region_readout_class_zscore(
            source_map)
        region_clean_diag = torch.nan_to_num(
            region_z_diag,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        top_values_diag, top_indices_diag = torch.topk(
            region_clean_diag,
            k=min(2, self.num_cls),
            dim=0,
        )
        if self.num_cls > 1:
            margin_diag = top_values_diag[0] - top_values_diag[1]
        else:
            margin_diag = top_values_diag[0]
        candidate_diag = top_indices_diag[0]
        gate_diag = coverage & (
            margin_diag >= float(self.region_readout_min_margin))

        agreement_required = max(
            1, int(self.region_readout_formula_agreement))
        if agreement_required > 1:
            agreement = torch.zeros_like(
                candidate_diag, dtype=torch.int16)
            agreement_sources = self._region_readout_parse_names(
                self.region_readout_agreement_sources,
                (
                    'true_inside_mean,true_inside_top,'
                    'true_contrast_top,true_zcontrast_top'
                ),
            )
            for source_name in agreement_sources:
                compare = source_maps.get(source_name)
                compare_coverage = coverage_maps.get(source_name)
                if compare is None or compare_coverage is None:
                    continue
                compare_pred = torch.nan_to_num(
                    self._region_readout_class_zscore(compare),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).argmax(dim=0)
                agreement += (
                    compare_coverage
                    & (compare_pred == candidate_diag)
                ).to(torch.int16)
            gate_diag &= agreement >= agreement_required

        region_full = self._interpolate_float32(
            region_clean_diag.unsqueeze(0),
            (height, width),
        ).squeeze(0)
        gate_full = F.interpolate(
            gate_diag.float().view(1, 1, *gate_diag.shape),
            size=(height, width),
            mode='nearest',
        ).squeeze().bool()
        base_topk = torch.topk(
            base_logits,
            k=max(
                1,
                min(int(self.region_readout_apply_topk), self.num_cls),
            ),
            dim=0,
        ).indices
        candidate_pred = region_full.argmax(dim=0)
        gate_full &= (
            base_topk == candidate_pred.unsqueeze(0)).any(dim=0)
        base_scale = base_logits.std(
            dim=0, unbiased=False).clamp_min(1e-4)
        candidate_logits = (
            base_logits
            + float(blend)
            * region_full
            * base_scale.unsqueeze(0)
        )
        candidate_pred = candidate_logits.argmax(dim=0)
        return candidate_logits, candidate_pred, gate_full

    def _region_readout_route_masks(self, base_logits, base_pred):
        raw_score, raw_pred = base_logits.max(dim=0)
        semantic_background = raw_pred == int(self.bg_idx)
        threshold_reject = (
            (base_pred == int(self.bg_idx))
            & (raw_pred != int(self.bg_idx))
            & (raw_score < float(self.prob_thd))
        )
        foreground = base_pred != int(self.bg_idx)
        return dict(
            foreground=foreground,
            threshold_reject=threshold_reject,
            semantic_background=semantic_background,
            raw_pred=raw_pred,
            raw_score=raw_score,
        )

    def _region_readout_action_signature(self):
        variant = str(
            self.region_readout_route_variant or 'legacy'
        ).strip().lower()
        if variant == 'threshold_disabled':
            return variant
        return (
            f'{variant}'
            f'__proj-{str(self.region_readout_projection).lower()}'
            f'__fg{float(self.region_readout_foreground_blend):g}'
            f'__r{float(self.region_readout_reject_blend):g}'
            f'__b{float(self.region_readout_background_blend):g}'
            f'__a{int(self.region_readout_formula_agreement)}'
        )

    def _region_readout_apply_route_variant(
            self, base_logits, base_pred, source_maps, coverage_maps):
        variant = str(
            self.region_readout_route_variant or 'legacy'
        ).strip().lower()
        if variant == 'threshold_disabled':
            return (
                base_logits,
                base_logits.argmax(dim=0),
                dict(),
            )
        if variant == 'legacy':
            source_name = str(
                self.region_readout_apply_source).strip().lower()
            logits = self._region_readout_apply_full_resolution(
                base_logits,
                source_maps.get(source_name),
                coverage_maps.get(source_name),
            )
            return logits, self._threshold_with_reject_recovery(
                logits, None), dict()

        definitions = {
            'foreground_contrast': dict(
                foreground='true_contrast_top'),
            'foreground_zcontrast': dict(
                foreground='true_zcontrast_top'),
            'reject_inside_mean': dict(
                threshold_reject='true_inside_mean'),
            'reject_inside_top': dict(
                threshold_reject='true_inside_top'),
            'foreground_reject_contrast_mean': dict(
                foreground='true_contrast_top',
                threshold_reject='true_inside_mean'),
            'foreground_reject_zcontrast_mean': dict(
                foreground='true_zcontrast_top',
                threshold_reject='true_inside_mean'),
            'three_route_contrast_mean': dict(
                foreground='true_contrast_top',
                threshold_reject='true_inside_mean',
                semantic_background='true_inside_mean'),
            'three_route_zcontrast_mean': dict(
                foreground='true_zcontrast_top',
                threshold_reject='true_inside_mean',
                semantic_background='true_inside_mean'),
        }
        route_sources = definitions.get(variant)
        if route_sources is None:
            raise ValueError(
                'Unsupported region_readout_route_variant '
                f'{variant!r}.')

        blend_by_route = {
            'foreground': float(
                self.region_readout_foreground_blend),
            'threshold_reject': float(
                self.region_readout_reject_blend),
            'semantic_background': float(
                self.region_readout_background_blend),
        }
        route_masks = self._region_readout_route_masks(
            base_logits, base_pred)
        output_logits = base_logits.clone()
        output_pred = base_pred.clone()
        route_info = {}
        cache = {}
        for route_name, source_name in route_sources.items():
            source_map = source_maps.get(source_name)
            coverage = coverage_maps.get(source_name)
            if source_map is None or coverage is None:
                continue
            blend = blend_by_route[route_name]
            cache_key = (source_name, blend)
            if cache_key not in cache:
                cache[cache_key] = (
                    self._region_readout_prepare_route_candidate(
                        base_logits,
                        source_map,
                        coverage,
                        blend,
                        source_maps,
                        coverage_maps,
                    )
                )
            candidate_logits, candidate_pred, candidate_gate = (
                cache[cache_key])
            eligible = route_masks[route_name]
            accepted = eligible & candidate_gate
            if route_name in ('foreground', 'threshold_reject'):
                accepted &= candidate_pred != int(self.bg_idx)
            output_logits[:, accepted] = candidate_logits[:, accepted]
            output_pred[accepted] = candidate_pred[accepted]
            route_info[route_name] = dict(
                eligible=eligible,
                accepted=accepted,
                source_name=source_name,
                blend=blend,
            )
        return output_logits, output_pred, route_info

    def _region_readout_route_rows(
            self, route_info, pred, base_pred, gt, valid):
        rows = []
        base_correct = (base_pred == gt) & valid
        pred_correct = (pred == gt) & valid
        changed = (pred != base_pred) & valid
        for route_name, info in route_info.items():
            eligible = info['eligible'] & valid
            accepted = info['accepted'] & valid
            improved = accepted & (~base_correct) & pred_correct
            harmed = accepted & base_correct & (~pred_correct)
            wrong_to_wrong = (
                accepted & (~base_correct) & (~pred_correct))
            rows.append(dict(
                route_name=route_name,
                source_name=info['source_name'],
                blend=float(info['blend']),
                eligible_pixels=int(eligible.sum().item()),
                accepted_pixels=int(accepted.sum().item()),
                accepted_ratio=_safe_div(
                    int(accepted.sum().item()),
                    int(eligible.sum().item())),
                gt_foreground_pixels=int(
                    (eligible & (gt != int(self.bg_idx))).sum().item()),
                gt_background_pixels=int(
                    (eligible & (gt == int(self.bg_idx))).sum().item()),
                baseline_wrong_pixels=int(
                    (eligible & (~base_correct)).sum().item()),
                changed_pixels=int(
                    (changed & eligible).sum().item()),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                wrong_to_wrong_pixels=int(
                    wrong_to_wrong.sum().item()),
                utility_precision=_safe_div(
                    int(improved.sum().item()),
                    int(improved.sum().item())
                    + int(harmed.sum().item())),
            ))
        return rows

    def _region_readout_mix_prediction(
            self, base_scores, base_pred, region_scores, coverage,
            blend, gated):
        base_z = self._region_readout_class_zscore(base_scores)
        region_z = self._region_readout_class_zscore(region_scores)
        region_clean = torch.nan_to_num(
            region_z, nan=0.0, posinf=0.0, neginf=0.0)
        top_values, top_indices = torch.topk(
            region_clean, k=min(2, self.num_cls), dim=0)
        region_pred = top_indices[0]
        if self.num_cls > 1:
            region_margin = top_values[0] - top_values[1]
        else:
            region_margin = top_values[0]
        gate = coverage & (
            region_margin >= float(self.region_readout_min_margin))
        if gated:
            topk = max(
                1,
                min(int(self.region_readout_apply_topk), self.num_cls),
            )
            base_topk = torch.topk(
                base_scores, k=topk, dim=0).indices
            gate &= (
                base_topk == region_pred.unsqueeze(0)).any(dim=0)
        mixed = base_z.clone()
        mixed[:, gate] = (
            base_z[:, gate]
            + float(blend) * region_clean[:, gate]
        )
        pred = base_pred.clone()
        pred[gate] = mixed[:, gate].argmax(dim=0)
        return pred, gate, region_pred, region_margin, region_clean

    def _region_readout_source_stats(
            self, source_name, scores, coverage, base_pred, gt, valid,
            topk):
        clean = torch.nan_to_num(
            scores, nan=-1e6, posinf=1e6, neginf=-1e6)
        values, indices = torch.topk(
            clean, k=min(topk, self.num_cls), dim=0)
        source_pred = indices[0]
        source_valid = coverage & valid
        base_wrong = (base_pred != gt) & valid
        gt_idx = gt.clamp(0, self.num_cls - 1)
        gt_topk = (
            indices == gt_idx.unsqueeze(0)).any(dim=0)
        gt_score = torch.gather(
            clean, 0, gt_idx.unsqueeze(0)).squeeze(0)
        base_score = torch.gather(
            clean,
            0,
            base_pred.clamp(0, self.num_cls - 1).unsqueeze(0),
        ).squeeze(0)
        source_correct = (source_pred == gt) & source_valid
        wrong_covered = base_wrong & source_valid
        return dict(
            source_name=source_name,
            covered_pixels=int(source_valid.sum().item()),
            coverage_ratio=_safe_div(
                int(source_valid.sum().item()),
                int(valid.sum().item())),
            baseline_wrong_covered_pixels=int(
                wrong_covered.sum().item()),
            baseline_wrong_coverage=_safe_div(
                int(wrong_covered.sum().item()),
                int(base_wrong.sum().item())),
            wrong_top1_recovers_gt_pixels=int(
                (wrong_covered & source_correct).sum().item()),
            wrong_top1_recovers_gt_ratio=_safe_div(
                int((wrong_covered & source_correct).sum().item()),
                int(wrong_covered.sum().item())),
            wrong_topk_contains_gt_pixels=int(
                (wrong_covered & gt_topk).sum().item()),
            wrong_topk_contains_gt_ratio=_safe_div(
                int((wrong_covered & gt_topk).sum().item()),
                int(wrong_covered.sum().item())),
            wrong_gt_beats_base_pixels=int(
                (wrong_covered & (gt_score > base_score)).sum().item()),
            wrong_gt_beats_base_ratio=_safe_div(
                int((wrong_covered & (gt_score > base_score)).sum().item()),
                int(wrong_covered.sum().item())),
            wrong_gt_vs_base_margin_sum=_masked_sum(
                gt_score - base_score, wrong_covered),
        )

    def _region_readout_action_row(
            self, action_name, pred, base_pred, gt, valid):
        base_correct = (base_pred == gt) & valid
        selected_correct = (pred == gt) & valid
        changed = (pred != base_pred) & valid
        improved = changed & (~base_correct) & selected_correct
        harmed = changed & base_correct & (~selected_correct)
        wrong_to_wrong = (
            changed & (~base_correct) & (~selected_correct))
        return dict(
            action_name=action_name,
            confusion=_confusion_matrix(
                gt, pred, valid, self.num_cls),
            changed_pixels=int(changed.sum().item()),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            wrong_to_wrong_pixels=int(
                wrong_to_wrong.sum().item()),
            net_correct_pixels=(
                int(improved.sum().item())
                - int(harmed.sum().item())),
            utility_precision=_safe_div(
                int(improved.sum().item()),
                int(improved.sum().item())
                + int(harmed.sum().item())),
        )

    def _region_readout_class_rows(
            self, action_name, pred, base_pred, gt, valid):
        rows = []
        for class_idx, class_name in enumerate(self.class_names):
            gt_mask = (gt == class_idx) & valid
            pred_mask = (pred == class_idx) & valid
            base_mask = (base_pred == class_idx) & valid
            intersection = int((gt_mask & pred_mask).sum().item())
            union = int((gt_mask | pred_mask).sum().item())
            changed = (pred != base_pred) & valid
            rows.append(dict(
                action_name=action_name,
                class_index=class_idx,
                class_name=class_name,
                gt_pixels=int(gt_mask.sum().item()),
                pred_pixels=int(pred_mask.sum().item()),
                baseline_pred_pixels=int(base_mask.sum().item()),
                intersection_pixels=intersection,
                union_pixels=union,
                changed_pixels=int(
                    (changed & (gt_mask | pred_mask | base_mask)).sum().item()),
                improved_pixels=int(
                    (
                        changed
                        & gt_mask
                        & pred_mask
                        & (~base_mask)
                    ).sum().item()),
                harmed_pixels=int(
                    (
                        changed
                        & gt_mask
                        & base_mask
                        & (~pred_mask)
                    ).sum().item()),
            ))
        return rows

    def _region_readout_pair_rows(
            self, action_name, pred, base_pred, gt, valid):
        rows = []
        base_wrong = (base_pred != gt) & valid
        for gt_class in range(self.num_cls):
            for pred_class in range(self.num_cls):
                if gt_class == pred_class:
                    continue
                pair = (
                    base_wrong
                    & (gt == gt_class)
                    & (base_pred == pred_class)
                )
                pixels = int(pair.sum().item())
                if pixels < int(
                        self.region_readout_min_pair_pixels):
                    continue
                changed = pair & (pred != base_pred)
                corrected = pair & (pred == gt_class)
                rows.append(dict(
                    action_name=action_name,
                    gt_class_index=gt_class,
                    gt_class_name=self.class_names[gt_class],
                    base_pred_class_index=pred_class,
                    base_pred_class_name=self.class_names[pred_class],
                    pixels=pixels,
                    changed_pixels=int(changed.sum().item()),
                    corrected_pixels=int(corrected.sum().item()),
                    correction_ratio=_safe_div(
                        int(corrected.sum().item()), pixels),
                    correction_precision=_safe_div(
                        int(corrected.sum().item()),
                        int(changed.sum().item())),
                ))
        return rows

    def _region_readout_region_rows(
            self, score_by_name, masks, metadata, gt, valid):
        rows = []
        high_purity = float(
            self.region_readout_high_purity_threshold)
        for source_name, region_scores in score_by_name.items():
            region_count = 0
            valid_region_count = 0
            correct_count = 0
            high_purity_count = 0
            high_purity_correct = 0
            origin_correct = 0
            area_sum = 0
            purity_sum = 0.0
            margin_sum = 0.0
            class_buckets = {}
            top_values, top_indices = torch.topk(
                region_scores, k=min(2, self.num_cls), dim=1)
            for region_idx, mask in enumerate(masks):
                region_count += 1
                mask_valid = mask & valid
                pixels = int(mask_valid.sum().item())
                if pixels == 0:
                    continue
                valid_region_count += 1
                labels = gt[mask_valid]
                counts = torch.bincount(
                    labels, minlength=self.num_cls)
                majority_count, majority_class = counts.max(dim=0)
                majority_class = int(majority_class.item())
                purity = float(majority_count.item()) / pixels
                pred_class = int(top_indices[region_idx, 0].item())
                margin = float(
                    top_values[region_idx, 0].item()
                    - top_values[region_idx, 1].item()
                ) if self.num_cls > 1 else float(
                    top_values[region_idx, 0].item())
                if not math.isfinite(margin):
                    margin = 0.0
                correct = pred_class == majority_class
                is_high = purity >= high_purity
                origin = int(metadata[region_idx]['class_index'])
                correct_count += int(correct)
                high_purity_count += int(is_high)
                high_purity_correct += int(is_high and correct)
                origin_correct += int(origin == majority_class)
                area_sum += pixels
                purity_sum += purity
                margin_sum += margin
                bucket = class_buckets.setdefault(
                    majority_class,
                    dict(
                        regions=0,
                        correct_regions=0,
                        high_purity_regions=0,
                        high_purity_correct_regions=0,
                        area_pixels=0,
                        purity_sum=0.0,
                        margin_sum=0.0,
                    ),
                )
                bucket['regions'] += 1
                bucket['correct_regions'] += int(correct)
                bucket['high_purity_regions'] += int(is_high)
                bucket['high_purity_correct_regions'] += int(
                    is_high and correct)
                bucket['area_pixels'] += pixels
                bucket['purity_sum'] += purity
                bucket['margin_sum'] += margin
            rows.append(dict(
                source_name=source_name,
                class_index=None,
                class_name='__all__',
                regions=region_count,
                valid_regions=valid_region_count,
                correct_regions=correct_count,
                high_purity_regions=high_purity_count,
                high_purity_correct_regions=high_purity_correct,
                origin_correct_regions=origin_correct,
                area_pixels=area_sum,
                purity_sum=purity_sum,
                margin_sum=margin_sum,
            ))
            for class_idx, bucket in class_buckets.items():
                rows.append(dict(
                    source_name=source_name,
                    class_index=class_idx,
                    class_name=self.class_names[class_idx],
                    valid_regions=bucket['regions'],
                    correct_regions=bucket['correct_regions'],
                    high_purity_regions=(
                        bucket['high_purity_regions']),
                    high_purity_correct_regions=(
                        bucket['high_purity_correct_regions']),
                    area_pixels=bucket['area_pixels'],
                    purity_sum=bucket['purity_sum'],
                    margin_sum=bucket['margin_sum'],
                ))
        return rows

    def _build_region_contrastive_readout(
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
        query_mean = query_maps.flatten(1).mean(dim=1)
        query_std = query_maps.flatten(1).std(
            dim=1, unbiased=False).clamp_min(1e-6)
        query_z = (
            query_maps - query_mean[:, None, None]
        ) / query_std[:, None, None]

        soft_masks, binary_masks, metadata = (
            self._region_readout_collect_regions(
                components.get('raw_mask_candidates') or [],
                (height, width),
                diag_shape,
            )
        )
        empty_context = dict(
            dataset_name=getattr(self, 'seed_dataset_name', None),
            class_names=list(self.class_names),
            diagnostic_shape=list(diag_shape),
            region_count=0,
            baseline_confusion=None,
            source_rows=[],
            action_rows=[],
            class_rows=[],
            pair_rows=[],
            region_rows=[],
            pixel_oracle_confusion=None,
        )
        gt_data = None
        if data_sample is not None and hasattr(
                data_sample, 'gt_sem_seg'):
            gt_sem_seg = data_sample.gt_sem_seg
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(
                    self.device).long()
        gt = None
        valid = None
        gt_full = None
        valid_full = None
        if gt_data is not None:
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
            baseline_confusion = _confusion_matrix(
                gt, base_pred_diag, valid, self.num_cls)
            empty_context.update(
                baseline_confusion=baseline_confusion,
                action_rows=[
                    self._region_readout_action_row(
                        'baseline',
                        base_pred_diag,
                        base_pred_diag,
                        gt,
                        valid,
                    )
                ],
                class_rows=self._region_readout_class_rows(
                    'baseline',
                    base_pred_diag,
                    base_pred_diag,
                    gt,
                    valid,
                ),
                pixel_oracle_confusion=baseline_confusion,
            )
        if soft_masks is None:
            (
                applied_logits,
                applied_pred,
                route_info,
            ) = self._region_readout_apply_route_variant(
                base_logits,
                base_pred,
                {},
                {},
            )
            if gt_full is not None:
                empty_context.update(
                    exact_action_name=(
                        self._region_readout_action_signature()),
                    exact_baseline_confusion=_confusion_matrix(
                        gt_full,
                        base_pred,
                        valid_full,
                        self.num_cls,
                    ),
                    exact_applied_confusion=_confusion_matrix(
                        gt_full,
                        applied_pred,
                        valid_full,
                        self.num_cls,
                    ),
                    exact_threshold_disabled_confusion=(
                        _confusion_matrix(
                            gt_full,
                            base_logits.argmax(dim=0),
                            valid_full,
                            self.num_cls,
                        )
                    ),
                    route_rows=self._region_readout_route_rows(
                        route_info,
                        applied_pred,
                        base_pred,
                        gt_full,
                        valid_full,
                    ),
                )
            return applied_logits, applied_pred, empty_context

        rect_masks = self._region_readout_rectangle_masks(binary_masks)
        shift_masks = self._region_readout_shift_masks(binary_masks)
        rect_soft = rect_masks.float()
        shift_soft = shift_masks.float()

        true_mean, true_top, true_ring = (
            self._region_readout_query_scores(
                query_maps, soft_masks, binary_masks))
        true_z_mean, true_z_top, true_z_ring = (
            self._region_readout_query_scores(
                query_z, soft_masks, binary_masks))
        _, rect_z_top, rect_z_ring = (
            self._region_readout_query_scores(
                query_z, rect_soft, rect_masks))
        _, shift_z_top, shift_z_ring = (
            self._region_readout_query_scores(
                query_z, shift_soft, shift_masks))

        region_score_by_name = {
            'true_inside_mean': self._region_readout_aggregate_queries(
                true_mean),
            'true_inside_top': self._region_readout_aggregate_queries(
                true_top),
            'true_contrast_top': self._region_readout_aggregate_queries(
                true_top - true_ring),
            'true_zcontrast_mean': self._region_readout_aggregate_queries(
                true_z_mean - true_z_ring),
            'true_zcontrast_top': self._region_readout_aggregate_queries(
                true_z_top - true_z_ring),
            'rect_zcontrast_top': self._region_readout_aggregate_queries(
                rect_z_top - rect_z_ring),
            'shift_zcontrast_top': self._region_readout_aggregate_queries(
                shift_z_top - shift_z_ring),
        }
        presence_prior = self._region_readout_presence_prior(
            components.get('presence_query_scores'))
        region_score_by_name['true_zcontrast_top_presence'] = (
            region_score_by_name['true_zcontrast_top']
            + float(self.region_readout_presence_weight)
            * presence_prior.unsqueeze(0)
        )
        region_score_by_name['classperm_zcontrast_top'] = torch.roll(
            region_score_by_name['true_zcontrast_top'],
            shifts=1,
            dims=1,
        )
        origin_raw = torch.full_like(
            region_score_by_name['true_zcontrast_top'],
            float('-inf'),
        )
        for region_idx, row in enumerate(metadata):
            class_idx = row['class_index']
            if 0 <= class_idx < self.num_cls:
                origin_raw[region_idx, class_idx] = row['raw_score']
        region_score_by_name['origin_raw'] = origin_raw

        source_maps = {}
        coverage_maps = {}
        true_sources = {
            'true_inside_mean',
            'true_inside_top',
            'true_contrast_top',
            'true_zcontrast_mean',
            'true_zcontrast_top',
            'true_zcontrast_top_presence',
            'classperm_zcontrast_top',
            'origin_raw',
        }
        for source_name, region_scores in region_score_by_name.items():
            if source_name.startswith('rect_'):
                geometry_soft = rect_soft
                geometry_binary = rect_masks
            elif source_name.startswith('shift_'):
                geometry_soft = shift_soft
                geometry_binary = shift_masks
            else:
                geometry_soft = soft_masks
                geometry_binary = binary_masks
            source_map, coverage = self._region_readout_project_source(
                region_scores,
                geometry_soft,
                geometry_binary,
                metadata,
                diag_shape,
            )
            source_maps[source_name] = source_map
            coverage_maps[source_name] = coverage

        class_semantic = self._aggregate_query_logits_to_classes(
            query_maps)
        class_semantic_z = self._aggregate_query_logits_to_classes(
            query_z)
        source_maps['pixel_semantic'] = class_semantic
        source_maps['pixel_semantic_z'] = class_semantic_z
        coverage_maps['pixel_semantic'] = torch.ones(
            diag_shape, device=self.device, dtype=torch.bool)
        coverage_maps['pixel_semantic_z'] = coverage_maps[
            'pixel_semantic']

        (
            applied_logits,
            applied_pred,
            route_info,
        ) = self._region_readout_apply_route_variant(
            base_logits,
            base_pred,
            source_maps,
            coverage_maps,
        )
        if gt_data is None:
            return applied_logits, applied_pred, empty_context

        topk = max(
            1, min(int(self.region_readout_topk), self.num_cls))
        source_rows = [
            self._region_readout_source_stats(
                source_name,
                source_map,
                coverage_maps[source_name],
                base_pred_diag,
                gt,
                valid,
                topk,
            )
            for source_name, source_map in source_maps.items()
        ]

        action_rows = [
            self._region_readout_action_row(
                'baseline', base_pred_diag, base_pred_diag, gt, valid)
        ]
        class_rows = self._region_readout_class_rows(
            'baseline', base_pred_diag, base_pred_diag, gt, valid)
        pair_rows = []
        mixed_predictions = {'baseline': base_pred_diag}
        blend_values = self._region_readout_parse_floats(
            self.region_readout_blends, '0.25,0.50,1.00')
        mix_sources = self._region_readout_parse_names(
            self.region_readout_mix_sources,
            (
                'true_zcontrast_top,true_zcontrast_top_presence,'
                'rect_zcontrast_top,shift_zcontrast_top,'
                'classperm_zcontrast_top,pixel_semantic'
            ),
        )
        for source_name in mix_sources:
            if source_name not in source_maps:
                continue
            for blend in blend_values:
                for gated in (False, True):
                    gate_name = 'topk_gate' if gated else 'ungated'
                    action_name = (
                        f'{source_name}__b{blend:g}__{gate_name}')
                    pred, _, _, _, _ = (
                        self._region_readout_mix_prediction(
                            base_scores,
                            base_pred_diag,
                            source_maps[source_name],
                            coverage_maps[source_name],
                            blend,
                            gated,
                        )
                    )
                    mixed_predictions[action_name] = pred
                    action_rows.append(
                        self._region_readout_action_row(
                            action_name,
                            pred,
                            base_pred_diag,
                            gt,
                            valid,
                        )
                    )
                    class_rows.extend(
                        self._region_readout_class_rows(
                            action_name,
                            pred,
                            base_pred_diag,
                            gt,
                            valid,
                        )
                    )
                    pair_rows.extend(
                        self._region_readout_pair_rows(
                            action_name,
                            pred,
                            base_pred_diag,
                            gt,
                            valid,
                        )
                    )

        oracle_pred = base_pred_diag.clone()
        for pred in mixed_predictions.values():
            recover = (
                valid
                & (oracle_pred != gt)
                & (pred == gt)
            )
            oracle_pred[recover] = gt[recover]

        region_rows = self._region_readout_region_rows(
            {
                name: scores
                for name, scores in region_score_by_name.items()
                if name in true_sources
                or name.startswith('rect_')
                or name.startswith('shift_')
            },
            binary_masks,
            metadata,
            gt,
            valid,
        )
        context = dict(
            dataset_name=getattr(self, 'seed_dataset_name', None),
            class_names=list(self.class_names),
            diagnostic_shape=list(diag_shape),
            region_count=len(metadata),
            mask_threshold=float(
                self.region_readout_mask_threshold),
            ring_kernel=int(self.region_readout_ring_kernel),
            top_fraction=float(self.region_readout_top_fraction),
            presence_weight=float(
                self.region_readout_presence_weight),
            baseline_confusion=_confusion_matrix(
                gt, base_pred_diag, valid, self.num_cls),
            source_rows=source_rows,
            action_rows=action_rows,
            class_rows=class_rows,
            pair_rows=pair_rows,
            region_rows=region_rows,
            pixel_oracle_confusion=_confusion_matrix(
                gt, oracle_pred, valid, self.num_cls),
            exact_action_name=(
                self._region_readout_action_signature()),
            exact_baseline_confusion=_confusion_matrix(
                gt_full,
                base_pred,
                valid_full,
                self.num_cls,
            ),
            exact_applied_confusion=_confusion_matrix(
                gt_full,
                applied_pred,
                valid_full,
                self.num_cls,
            ),
            exact_threshold_disabled_confusion=_confusion_matrix(
                gt_full,
                base_logits.argmax(dim=0),
                valid_full,
                self.num_cls,
            ),
            route_rows=self._region_readout_route_rows(
                route_info,
                applied_pred,
                base_pred,
                gt_full,
                valid_full,
            ),
        )
        return applied_logits, applied_pred, context

    def _region_readout_apply_full_resolution(
            self, base_logits, source_map, coverage):
        if (
                not self.use_region_contrastive_readout
                or source_map is None
                or coverage is None
                or not coverage.any()):
            return base_logits
        height, width = base_logits.shape[-2:]
        region_z = self._region_readout_class_zscore(
            source_map)
        region_z = self._interpolate_float32(
            torch.nan_to_num(
                region_z,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).unsqueeze(0),
            (height, width),
        ).squeeze(0)
        coverage_full = F.interpolate(
            coverage.float().view(1, 1, *coverage.shape),
            size=(height, width),
            mode='nearest',
        ).squeeze().bool()
        top_values, top_indices = torch.topk(
            region_z, k=min(2, self.num_cls), dim=0)
        if self.num_cls > 1:
            region_margin = top_values[0] - top_values[1]
        else:
            region_margin = top_values[0]
        gate = coverage_full & (
            region_margin >= float(self.region_readout_min_margin))
        topk = max(
            1,
            min(int(self.region_readout_apply_topk), self.num_cls),
        )
        base_topk = torch.topk(
            base_logits, k=topk, dim=0).indices
        gate &= (
            base_topk == top_indices[0].unsqueeze(0)).any(dim=0)
        base_scale = base_logits.std(
            dim=0, unbiased=False).clamp_min(1e-4)
        residual = (
            float(self.region_readout_apply_blend)
            * region_z
            * base_scale.unsqueeze(0)
        )
        output = base_logits.clone()
        output[:, gate] = (
            base_logits[:, gate] + residual[:, gate])
        return output

    def _write_region_contrastive_readout_stats(self, record):
        if not self.dump_region_contrastive_readout_stats:
            return
        if self._region_contrastive_readout_stats_file is None:
            path = (
                self.region_contrastive_readout_stats_path
                or './work_dirs/evidence_stats/'
                   'region_contrastive_readout.jsonl'
            )
            path = _ranked_jsonl_path(path)
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._region_contrastive_readout_stats_file = open(
                path, 'a', buffering=1)
        self._region_contrastive_readout_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')
