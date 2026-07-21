import json
import os

import torch
import torch.nn.functional as F


def _cs_safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _cs_ranked_jsonl_path(path):
    rank = int(os.environ.get('RANK', 0))
    root, ext = os.path.splitext(path)
    if not ext:
        ext = '.jsonl'
    return f'{root}_rank{rank}{ext}'


def _cs_confusion_matrix(gt, pred, valid, class_count):
    encoded = (
        gt[valid].long().clamp(min=0, max=class_count - 1) * class_count
        + pred[valid].long().clamp(min=0, max=class_count - 1)
    )
    matrix = torch.bincount(
        encoded,
        minlength=class_count * class_count,
    ).view(class_count, class_count)
    return matrix.detach().cpu().tolist()


def _cs_class_tokens(name):
    tokens = set()
    normalized = str(name or '').lower().replace('/', ',').replace('-', ' ')
    for item in normalized.split(','):
        item = item.strip()
        if not item:
            continue
        tokens.add(item)
        tokens.update(part for part in item.split() if part)
    return tokens


def _cs_remote_sensing_role(name):
    tokens = _cs_class_tokens(name)
    if tokens & {'background', 'other', 'clutter', 'void', 'unknown'}:
        return 'catch_all'
    if tokens & {'tree', 'forest', 'wood', 'canopy'}:
        return 'woody_vegetation'
    if tokens & {
            'grass', 'vegetation', 'low', 'crop', 'cropland',
            'agricultural', 'agriculture', 'farmland', 'field'}:
        return 'low_vegetation'
    if tokens & {'water', 'river', 'lake', 'sea', 'pond'}:
        return 'water'
    if tokens & {
            'road', 'pavement', 'impervious', 'surface',
            'sidewalk', 'parking'}:
        return 'impervious_surface'
    if tokens & {
            'building', 'roof', 'house', 'facade', 'wall',
            'construction'}:
        return 'built_object'
    if tokens & {'car', 'vehicle', 'truck', 'ship', 'airplane', 'plane'}:
        return 'vehicle'
    if tokens & {'bareland', 'barren', 'soil', 'sand', 'bare'}:
        return 'bareland'
    return 'other_landcover'


def _cs_zscore(values):
    values = values.float()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


class ConceptSpecificityDiagnosticMixin:
    """Mechanism-level diagnostics for remote-sensing concept hallucination.

    The diagnostic asks whether a high-response class is supported by
    scene-specific evidence, or only by scene-common remote-sensing texture.
    GT labels are used only to evaluate ranking and oracle budget effects.
    """

    def _uses_concept_specificity_diagnostic(self):
        return bool(
            self.dump_concept_specificity_stats
            or self.use_concept_specificity_pruning)

    def _concept_specificity_shape(self, height, width):
        max_side = max(16, int(self.concept_specificity_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    @staticmethod
    def _cs_resize_maps(maps, shape):
        if maps is None:
            return None
        return F.interpolate(
            maps.detach().float().unsqueeze(0),
            size=shape,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

    def _concept_specificity_score_threshold(self):
        if self.concept_specificity_score_thd is not None:
            return float(self.concept_specificity_score_thd)
        return float(min(max(float(self.prob_thd) * 0.5, 0.03), 0.10))

    def _query_ids_for_class(self, class_idx):
        query_idx = self.query_idx.detach().cpu()
        return [
            idx for idx, value in enumerate(query_idx.tolist())
            if int(value) == int(class_idx)
        ]

    def _prompt_stability(self, query_maps, query_ids, threshold):
        if query_maps is None or len(query_ids) <= 1:
            return 1.0, len(query_ids)
        masks = query_maps[query_ids] >= threshold
        pair_ious = []
        for i in range(len(query_ids)):
            for j in range(i + 1, len(query_ids)):
                intersection = (masks[i] & masks[j]).sum().item()
                union = (masks[i] | masks[j]).sum().item()
                pair_ious.append(_cs_safe_div(intersection, union))
        if not pair_ious:
            return 1.0, len(query_ids)
        return float(sum(pair_ious) / len(pair_ious)), len(query_ids)

    def _class_specificity_features(
            self, class_idx, class_map, semantic_map, instance_map,
            common_map, query_maps, threshold):
        flat = class_map.flatten()
        common_flat = common_map.flatten()
        p50 = torch.quantile(flat, 0.50)
        p90 = torch.quantile(flat, 0.90)
        p95 = torch.quantile(flat, 0.95)
        std = flat.std(unbiased=False).clamp_min(1e-6)
        support = class_map >= threshold
        support_area = support.float().mean()
        top_mask = class_map >= p90
        if not top_mask.any():
            top_mask = class_map >= p95
        top_mean = class_map[top_mask].mean() if top_mask.any() else flat.mean()
        common_on_top = (
            common_map[top_mask].mean() if top_mask.any()
            else common_flat.mean())
        common_p90 = torch.quantile(common_flat, 0.90)
        common_overlap = (
            (common_map[top_mask] >= common_p90).float().mean()
            if top_mask.any() else torch.tensor(1.0, device=class_map.device))
        scene_contrast = (top_mean - common_on_top) / std

        positive = class_map.clamp(min=0.0).flatten()
        total_mass = positive.sum().clamp_min(1e-6)
        top_count = max(1, int(round(positive.numel() * 0.05)))
        top_mass = torch.topk(positive, k=top_count).values.sum()
        mass_concentration = top_mass / total_mass
        peakiness = (p95 - p50) / p95.abs().clamp_min(1e-6)

        head_iou = 0.0
        agreement_area = 0.0
        if semantic_map is not None and instance_map is not None:
            sem_support = semantic_map >= threshold
            inst_support = instance_map >= threshold
            intersection = (sem_support & inst_support).sum().item()
            union = (sem_support | inst_support).sum().item()
            head_iou = _cs_safe_div(intersection, union)
            agreement_area = float(
                (sem_support & inst_support).float().mean().item())

        prompt_stability, prompt_count = self._prompt_stability(
            query_maps,
            self._query_ids_for_class(class_idx),
            threshold,
        )
        return dict(
            mean=float(flat.mean().item()),
            std=float(std.item()),
            p50=float(p50.item()),
            p90=float(p90.item()),
            p95=float(p95.item()),
            max=float(flat.max().item()),
            support_area=float(support_area.item()),
            scene_contrast=float(scene_contrast.item()),
            common_overlap=float(common_overlap.item()),
            mass_concentration=float(mass_concentration.item()),
            peakiness=float(peakiness.item()),
            head_iou=float(head_iou),
            agreement_area=float(agreement_area),
            prompt_stability=float(prompt_stability),
            prompt_count=int(prompt_count),
        )

    def _concept_specificity_rankers(self, feature_rows):
        names = [
            'presence_score',
            'scene_contrast',
            'mass_concentration',
            'peakiness',
            'head_iou',
            'prompt_stability',
        ]
        values = {
            name: torch.tensor(
                [float(row[name]) for row in feature_rows],
                device=self.device,
                dtype=torch.float32,
            )
            for name in names
        }
        combo = (
            _cs_zscore(values['scene_contrast'])
            + _cs_zscore(values['mass_concentration'])
            + _cs_zscore(values['peakiness'])
            + _cs_zscore(values['head_iou'])
            + _cs_zscore(values['prompt_stability'])
            - _cs_zscore(torch.tensor(
                [float(row['common_overlap']) for row in feature_rows],
                device=self.device,
                dtype=torch.float32,
            ))
        )
        rankers = {
            'presence_score': values['presence_score'],
            'scene_contrast': values['scene_contrast'],
            'region_specificity': (
                values['mass_concentration'] + values['peakiness']),
            'head_prompt_specificity': (
                values['head_iou'] + values['prompt_stability']),
            'specificity_combo': combo,
            'presence_specificity_combo': (
                combo + _cs_zscore(values['presence_score'])),
        }
        return rankers

    def _active_mask_from_scores(self, scores, active_count):
        active_count = max(1, min(int(active_count), int(scores.numel())))
        active = torch.zeros_like(scores, dtype=torch.bool)
        active[torch.topk(scores, k=active_count).indices] = True
        if self.concept_specificity_keep_bg and 0 <= int(self.bg_idx) < scores.numel():
            active[int(self.bg_idx)] = True
        return active

    def _apply_concept_specificity_pruning_from_mask(self, logits, active_mask):
        if active_mask is None or active_mask.numel() != logits.shape[0]:
            return logits
        pruned = logits.clone()
        inactive = ~active_mask.to(device=logits.device, dtype=torch.bool)
        if inactive.any():
            pruned[inactive] = float(self.concept_specificity_suppress_value)
        return pruned

    def _apply_concept_specificity_pruning_from_context(
            self, logits, context, ranker_name):
        ranker_name = str(ranker_name or 'specificity_combo').lower()
        for row in context.get('ranker_results', []):
            if str(row.get('ranker')).lower() == ranker_name:
                scores = torch.tensor(
                    row.get('scores', []),
                    device=logits.device,
                    dtype=torch.float32,
                )
                if scores.numel() != logits.shape[0]:
                    return logits
                active_count = int(
                    context.get('prune_active_count', scores.numel()))
                active_mask = self._active_mask_from_scores(
                    scores,
                    active_count,
                )
                return self._apply_concept_specificity_pruning_from_mask(
                    logits, active_mask)
        return logits

    def _build_concept_specificity_context(
            self, base_logits, base_pred, query_logits, query_semantic_logits,
            components, data_sample):
        gt = data_sample.gt_sem_seg.data
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        gt = gt.to(base_logits.device)
        valid = gt != 255
        class_count = int(self.num_cls)
        valid_pixels = int(valid.sum().item())
        threshold = self._concept_specificity_score_threshold()
        height, width = base_logits.shape[-2:]
        diag_shape = self._concept_specificity_shape(height, width)
        base_small = self._cs_resize_maps(base_logits, diag_shape)
        query_small = self._cs_resize_maps(query_semantic_logits, diag_shape)
        if query_small is None:
            query_small = self._cs_resize_maps(query_logits, diag_shape)
        semantic_small = None
        instance_small = None
        if components is not None:
            semantic_small = self._cs_resize_maps(
                components.get('semantic_logits'), diag_shape)
            instance_small = self._cs_resize_maps(
                components.get('instance_logits'), diag_shape)
        presence_scores = (
            components.get('presence_scores') if components is not None
            else None)

        catch_all = torch.tensor(
            [
                _cs_remote_sensing_role(name) == 'catch_all'
                for name in self.class_names
            ],
            device=base_logits.device,
            dtype=torch.bool,
        )
        non_catch = ~catch_all
        if non_catch.any():
            common_map = base_small[non_catch].mean(dim=0)
        else:
            common_map = base_small.mean(dim=0)

        gt_valid = gt[valid].long().clamp(min=0, max=class_count - 1)
        gt_counts = torch.bincount(gt_valid, minlength=class_count)
        gt_present = gt_counts > 0
        pred_valid = base_pred[valid].long().clamp(min=0, max=class_count - 1)
        pred_counts = torch.bincount(pred_valid, minlength=class_count)
        pred_present = pred_counts > 0
        if self.concept_specificity_keep_bg and 0 <= int(self.bg_idx) < class_count:
            bg_idx = int(self.bg_idx)
            gt_present_with_bg = gt_present.clone()
            gt_present_with_bg[bg_idx] = True
            pred_present_with_bg = pred_present.clone()
            pred_present_with_bg[bg_idx] = True
        else:
            gt_present_with_bg = gt_present
            pred_present_with_bg = pred_present
        budget = int(gt_present_with_bg.sum().item())
        if self.concept_specificity_prune_active_count is not None:
            prune_active_count = int(self.concept_specificity_prune_active_count)
        else:
            prune_active_count = int(pred_present_with_bg.sum().item())
        prune_active_count = max(1, min(prune_active_count, class_count))

        class_rows = []
        for class_idx, class_name in enumerate(self.class_names):
            semantic_map = (
                semantic_small[class_idx] if semantic_small is not None
                else None)
            instance_map = (
                instance_small[class_idx] if instance_small is not None
                else None)
            features = self._class_specificity_features(
                class_idx,
                base_small[class_idx],
                semantic_map,
                instance_map,
                common_map,
                query_small,
                threshold,
            )
            features.update(dict(
                class_index=int(class_idx),
                class_name=class_name,
                role=_cs_remote_sensing_role(class_name),
                presence_score=float(
                    presence_scores[class_idx].detach().float().item())
                if isinstance(presence_scores, torch.Tensor)
                and class_idx < presence_scores.numel() else 0.0,
                gt_present=bool(gt_present[class_idx].item()),
                gt_pixels=int(gt_counts[class_idx].item()),
                pred_present=bool(pred_present[class_idx].item()),
                pred_pixels=int(pred_counts[class_idx].item()),
            ))
            class_rows.append(features)

        baseline_confusion = _cs_confusion_matrix(
            gt, base_pred, valid, class_count)
        ranker_rows = []
        rankers = self._concept_specificity_rankers(class_rows)
        for ranker_name, scores in rankers.items():
            active_mask = self._active_mask_from_scores(scores, budget)
            prune_active_mask = self._active_mask_from_scores(
                scores,
                prune_active_count,
            )
            pruned_logits = self._apply_concept_specificity_pruning_from_mask(
                base_logits, active_mask)
            pred = self._threshold_with_reject_recovery(
                pruned_logits, components)
            no_gt_pruned_logits = (
                self._apply_concept_specificity_pruning_from_mask(
                    base_logits,
                    prune_active_mask,
                )
            )
            no_gt_pred = self._threshold_with_reject_recovery(
                no_gt_pruned_logits, components)
            changed = valid & (pred != base_pred)
            no_gt_changed = valid & (no_gt_pred != base_pred)
            base_correct = base_pred == gt
            new_correct = pred == gt
            no_gt_correct = no_gt_pred == gt
            active_gt = int((active_mask & gt_present).sum().item())
            no_gt_active_gt = int(
                (prune_active_mask & gt_present).sum().item())
            ranker_rows.append(dict(
                ranker=ranker_name,
                budget=int(budget),
                active_count=int(active_mask.sum().item()),
                gt_present_count=int(gt_present.sum().item()),
                active_gt_count=active_gt,
                active_recall=_cs_safe_div(
                    active_gt, int(gt_present.sum().item())),
                active_precision=_cs_safe_div(
                    active_gt, int(active_mask.sum().item())),
                inactive_gt_classes=[
                    self.class_names[idx]
                    for idx in range(class_count)
                    if bool(gt_present[idx].item())
                    and not bool(active_mask[idx].item())
                ],
                changed_pixels=int(changed.sum().item()),
                improved_pixels=int(
                    (changed & ~base_correct & new_correct).sum().item()),
                harmed_pixels=int(
                    (changed & base_correct & ~new_correct).sum().item()),
                wrong_to_wrong_pixels=int(
                    (changed & ~base_correct & ~new_correct).sum().item()),
                confusion=_cs_confusion_matrix(
                    gt, pred, valid, class_count),
                scores=[
                    float(value)
                    for value in scores.detach().cpu().tolist()
                ],
                active_mask=[
                    bool(value)
                    for value in active_mask.detach().cpu().tolist()
                ],
                no_gt_prune_active_count=int(prune_active_mask.sum().item()),
                no_gt_prune_active_gt_count=no_gt_active_gt,
                no_gt_prune_active_recall=_cs_safe_div(
                    no_gt_active_gt, int(gt_present.sum().item())),
                no_gt_prune_active_precision=_cs_safe_div(
                    no_gt_active_gt,
                    int(prune_active_mask.sum().item())),
                no_gt_prune_inactive_gt_classes=[
                    self.class_names[idx]
                    for idx in range(class_count)
                    if bool(gt_present[idx].item())
                    and not bool(prune_active_mask[idx].item())
                ],
                no_gt_prune_changed_pixels=int(
                    no_gt_changed.sum().item()),
                no_gt_prune_improved_pixels=int(
                    (
                        no_gt_changed
                        & ~base_correct
                        & no_gt_correct
                    ).sum().item()),
                no_gt_prune_harmed_pixels=int(
                    (
                        no_gt_changed
                        & base_correct
                        & ~no_gt_correct
                    ).sum().item()),
                no_gt_prune_wrong_to_wrong_pixels=int(
                    (
                        no_gt_changed
                        & ~base_correct
                        & ~no_gt_correct
                    ).sum().item()),
                no_gt_prune_confusion=_cs_confusion_matrix(
                    gt, no_gt_pred, valid, class_count),
                no_gt_prune_active_mask=[
                    bool(value)
                    for value in prune_active_mask.detach().cpu().tolist()
                ],
            ))

        return dict(
            class_names=list(self.class_names),
            score_threshold=float(threshold),
            diagnostic_shape=list(diag_shape),
            valid_pixels=valid_pixels,
            baseline_confusion=baseline_confusion,
            gt_present_count=int(gt_present.sum().item()),
            gt_budget_with_bg=int(budget),
            prune_active_count=int(prune_active_count),
            class_features=class_rows,
            ranker_results=ranker_rows,
        )

    def _write_concept_specificity_stats(self, record):
        if not self.dump_concept_specificity_stats:
            return
        if self._concept_specificity_stats_file is None:
            path = (
                self.concept_specificity_stats_path
                or './work_dirs/evidence_stats/'
                   'concept_specificity_stats.jsonl'
            )
            path = _cs_ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._concept_specificity_stats_file = open(
                path, 'a', buffering=1)
        self._concept_specificity_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')
