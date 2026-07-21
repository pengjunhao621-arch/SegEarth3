import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def _safe_div(num, den):
    if den is None or den == 0:
        return None
    return float(num) / float(den)


def _ranked_jsonl_path(path):
    rank = os.environ.get('RANK')
    if rank is None:
        return path
    base, ext = os.path.splitext(path)
    if ext == '':
        ext = '.jsonl'
    return f'{base}_rank{rank}{ext}'


def _masked_mean(value, mask):
    if value is None or mask is None or not bool(mask.any()):
        return None
    return float(value[mask].detach().float().mean().item())


def _masked_sum(value, mask):
    if value is None or mask is None or not bool(mask.any()):
        return 0.0
    return float(value[mask].detach().float().sum().item())


def _class_name_tokens(name):
    tokens = set()
    normalized = str(name or '').lower().replace('/', ',').replace('-', ' ')
    for item in normalized.split(','):
        item = item.strip()
        if item:
            tokens.add(item)
            for part in item.split():
                if part:
                    tokens.add(part)
    return tokens


def _remote_sensing_class_role(name):
    tokens = _class_name_tokens(name)
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


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


class Sam3GeometryRequeryDiagnosticMixin:
    """Prediction-preserving diagnostic for SAM3 geometric prompts.

    The diagnostic asks whether baseline top-k conflicts become separable when
    SAM3 is re-queried with its official positive/negative box prompt API. Pair
    and box selection uses only baseline-visible evidence. Ground truth is used
    only to score whether the re-query would have helped or harmed.
    """

    def _uses_sam3_geometry_requery_diagnostic(self):
        return _as_bool(getattr(
            self, 'dump_sam3_geometry_requery_stats', False))

    def _write_sam3_geometry_requery_stats(self, record):
        if not self.dump_sam3_geometry_requery_stats:
            return
        if self._sam3_geometry_requery_stats_file is None:
            path = (self.sam3_geometry_requery_stats_path
                    or './work_dirs/evidence_stats/sam3_geometry_requery.jsonl')
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._sam3_geometry_requery_stats_file = open(
                path, 'a', buffering=1)
        self._sam3_geometry_requery_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _sam3_geometry_parse_name_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text:
            return []
        return [item.strip() for item in text.split(',') if item.strip()]

    def _sam3_geometry_prompt_for_class(self, class_idx):
        class_idx = int(class_idx)
        prompt_mode = str(getattr(
            self, 'sam3_geometry_requery_prompt_mode', 'best_query'))
        if prompt_mode == 'class_name':
            return str(self.class_names[class_idx])
        query_idx = getattr(self, 'query_idx', None)
        query_words = getattr(self, 'query_words', None)
        if query_idx is not None and query_words is not None:
            for word, idx in zip(query_words, query_idx):
                idx = int(idx.item()) if hasattr(idx, 'item') else int(idx)
                if idx == class_idx:
                    return str(word)
        return str(self.class_names[class_idx])

    def _sam3_geometry_resize_image(self, image):
        max_side = int(getattr(self, 'sam3_geometry_requery_max_side', 1024))
        if max_side <= 0:
            return image, 1.0
        width, height = image.size
        long_side = max(width, height)
        if long_side <= max_side:
            return image, 1.0
        scale = float(max_side) / float(long_side)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return image.resize(new_size), scale

    def _sam3_geometry_get_gt(self, data_sample):
        if data_sample is None or not hasattr(data_sample, 'gt_sem_seg'):
            return None
        gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
        if not hasattr(gt_sem_seg, 'data'):
            return None
        return gt_sem_seg.data.squeeze().to(self.device)

    def _sam3_geometry_top_fraction_mask(self, score, mask, fraction=0.25):
        selected = torch.zeros_like(mask, dtype=torch.bool)
        if mask is None or not bool(mask.any()):
            return selected
        values = score[mask].detach().float()
        if values.numel() == 0:
            return selected
        min_pixels = int(getattr(
            self, 'sam3_geometry_requery_min_region_pixels', 32))
        k = max(min_pixels, int(math.ceil(values.numel() * float(fraction))))
        k = min(k, values.numel())
        threshold = torch.topk(values, k).values[-1]
        return mask & (score >= threshold)

    def _sam3_geometry_box_from_mask(self, mask, logit_hw):
        if mask is None or not bool(mask.any()):
            return None
        min_box = int(getattr(self, 'sam3_geometry_requery_min_box_size', 4))
        context_scale = float(getattr(
            self, 'sam3_geometry_requery_context_scale', 1.0))
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            return None
        h, w = int(logit_hw[0]), int(logit_hw[1])
        y0 = float(coords[:, 0].min().item())
        y1 = float(coords[:, 0].max().item()) + 1.0
        x0 = float(coords[:, 1].min().item())
        x1 = float(coords[:, 1].max().item()) + 1.0
        bw = max(float(min_box), x1 - x0)
        bh = max(float(min_box), y1 - y0)
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        bw *= context_scale
        bh *= context_scale
        x0 = max(0.0, cx - bw * 0.5)
        x1 = min(float(w), cx + bw * 0.5)
        y0 = max(0.0, cy - bh * 0.5)
        y1 = min(float(h), cy + bh * 0.5)
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        return [
            max(0.0, min(1.0, cx / float(max(1, w)))),
            max(0.0, min(1.0, cy / float(max(1, h)))),
            max(1.0 / float(max(1, w)), min(1.0, bw / float(max(1, w)))),
            max(1.0 / float(max(1, h)), min(1.0, bh / float(max(1, h)))),
        ]

    def _sam3_geometry_pair_candidates(self, base_logits, base_pred, valid_mask):
        topk = int(getattr(self, 'sam3_geometry_requery_topk', 3))
        topk = max(2, min(topk, int(base_logits.shape[0])))
        min_pair_pixels = int(getattr(
            self, 'sam3_geometry_requery_min_pair_pixels', 64))
        include_bg = _as_bool(getattr(
            self, 'sam3_geometry_requery_include_bg', True))
        max_pairs = int(getattr(self, 'sam3_geometry_requery_max_pairs', 4))

        _, top_idx = torch.topk(base_logits.detach().float(), k=topk, dim=0)
        candidates = []
        seen = set()
        for pred_cls in range(int(base_logits.shape[0])):
            if not include_bg and pred_cls == int(self.bg_idx):
                continue
            pred_mask = valid_mask & (base_pred == pred_cls)
            if int(pred_mask.sum().item()) < min_pair_pixels:
                continue
            target_values = torch.unique(top_idx[:, pred_mask]).detach().cpu()
            for target_cls in target_values.tolist():
                target_cls = int(target_cls)
                if target_cls == pred_cls:
                    continue
                if not include_bg and target_cls == int(self.bg_idx):
                    continue
                support = pred_mask & (top_idx == target_cls).any(dim=0)
                support_pixels = int(support.sum().item())
                if support_pixels < min_pair_pixels:
                    continue
                key = (pred_cls, target_cls)
                if key in seen:
                    continue
                seen.add(key)
                pred_score = base_logits[pred_cls]
                target_score = base_logits[target_cls]
                margin = pred_score - target_score
                candidates.append(dict(
                    pred_cls=pred_cls,
                    target_cls=target_cls,
                    support_mask=support,
                    support_pixels=support_pixels,
                    mean_pred_score=_masked_mean(pred_score, support),
                    mean_target_score=_masked_mean(target_score, support),
                    mean_base_gap=_masked_mean(margin, support),
                ))
        candidates.sort(
            key=lambda item: (
                item['support_pixels'],
                0.0 if item['mean_base_gap'] is None
                else float(item['mean_base_gap'])),
            reverse=True)
        if max_pairs > 0:
            candidates = candidates[:max_pairs]
        return candidates

    def _sam3_geometry_region_masks(self, base_logits, pair):
        pair_mask = pair['support_mask']
        target_cls = int(pair['target_cls'])
        pred_cls = int(pair['pred_cls'])
        target_score = base_logits[target_cls]
        pred_score = base_logits[pred_cls]
        score_thd = getattr(self, 'sam3_geometry_requery_score_thd', None)
        try:
            score_thd = None if score_thd is None else float(score_thd)
        except (TypeError, ValueError):
            score_thd = None
        if score_thd is None or score_thd < 0:
            score_thd = max(float(getattr(self, 'prob_thd', 0.0)) * 0.5, 0.03)
        min_pixels = int(getattr(
            self, 'sam3_geometry_requery_min_region_pixels', 32))

        target_region = pair_mask & (target_score >= score_thd)
        if int(target_region.sum().item()) < min_pixels:
            target_region = self._sam3_geometry_top_fraction_mask(
                target_score, pair_mask, fraction=0.25)

        pred_region = pair_mask & (pred_score >= score_thd)
        if int(pred_region.sum().item()) < min_pixels:
            pred_region = self._sam3_geometry_top_fraction_mask(
                pred_score, pair_mask, fraction=0.25)
        return target_region, pred_region

    def _sam3_geometry_run_prompt(self, image, prompt, boxes, out_hw):
        resized_image, _ = self._sam3_geometry_resize_image(image)
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        autocast_ctx = (
            torch.autocast(device_type=device_type, dtype=torch.bfloat16)
            if self.device.type == 'cuda' else nullcontext())
        with torch.no_grad(), autocast_ctx:
            state = self.processor.set_image(resized_image)
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt=prompt, state=state)
            for box, label in boxes:
                state = self.processor.add_geometric_prompt(
                    box=box, label=bool(label), state=state)
            map_h, map_w = resized_image.size[1], resized_image.size[0]
            variant, semantic, instance = self._compute_prompt_variant_logit(
                state, map_h, map_w)
        if variant.shape[-2:] != tuple(out_hw):
            variant = F.interpolate(
                variant.detach().float().view(1, 1, *variant.shape[-2:]),
                size=tuple(out_hw),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).squeeze(0)
        semantic_score = None
        instance_score = None
        if semantic is not None:
            if semantic.shape[-2:] != tuple(out_hw):
                semantic = F.interpolate(
                    semantic.detach().float().view(1, 1, *semantic.shape[-2:]),
                    size=tuple(out_hw),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            semantic_score = semantic
        if instance is not None:
            if instance.shape[-2:] != tuple(out_hw):
                instance = F.interpolate(
                    instance.detach().float().view(1, 1, *instance.shape[-2:]),
                    size=tuple(out_hw),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            instance_score = instance
        return dict(
            score=variant.detach().float(),
            semantic_score=semantic_score.detach().float()
            if semantic_score is not None else None,
            instance_score=instance_score.detach().float()
            if instance_score is not None else None,
            presence_score=float(state.get('presence_score', 0.0))
            if not isinstance(state.get('presence_score'), torch.Tensor)
            else float(state.get('presence_score').detach().float().mean().item()),
            num_masks=int(state.get('masks_logits').shape[0])
            if 'masks_logits' in state else 0,
        )

    def _sam3_geometry_eval_map(self, score_map, base_logits, pair, gt, valid_mask):
        pred_cls = int(pair['pred_cls'])
        target_cls = int(pair['target_cls'])
        pair_mask = pair['support_mask'] & valid_mask
        target_gt = pair_mask & (gt == target_cls)
        pred_gt = pair_mask & (gt == pred_cls)
        other_gt = pair_mask & (gt != target_cls) & (gt != pred_cls)
        base_target = base_logits[target_cls]
        base_pred = base_logits[pred_cls]
        gain = score_map - base_target
        requery_margin = score_map - base_pred
        return dict(
            target_gt_pixels=int(target_gt.sum().item()),
            pred_gt_pixels=int(pred_gt.sum().item()),
            other_gt_pixels=int(other_gt.sum().item()),
            target_gt_ratio=_safe_div(
                int(target_gt.sum().item()), int(pair_mask.sum().item())),
            pred_gt_ratio=_safe_div(
                int(pred_gt.sum().item()), int(pair_mask.sum().item())),
            mean_base_target_on_target_gt=_masked_mean(
                base_target, target_gt),
            mean_base_pred_on_target_gt=_masked_mean(base_pred, target_gt),
            mean_base_gap_on_target_gt=_masked_mean(
                base_pred - base_target, target_gt),
            mean_requery_score_on_target_gt=_masked_mean(score_map, target_gt),
            mean_requery_gain_on_target_gt=_masked_mean(gain, target_gt),
            requery_beats_base_pred_on_target_gt_pixels=int(
                ((score_map > base_pred) & target_gt).sum().item()),
            requery_beats_base_pred_on_target_gt_ratio=_safe_div(
                int(((score_map > base_pred) & target_gt).sum().item()),
                int(target_gt.sum().item())),
            mean_requery_margin_on_target_gt=_masked_mean(
                requery_margin, target_gt),
            mean_requery_score_on_pred_gt=_masked_mean(score_map, pred_gt),
            mean_requery_gain_on_pred_gt=_masked_mean(gain, pred_gt),
            harmful_requery_beats_base_pred_on_pred_gt_pixels=int(
                ((score_map > base_pred) & pred_gt).sum().item()),
            harmful_requery_beats_base_pred_on_pred_gt_ratio=_safe_div(
                int(((score_map > base_pred) & pred_gt).sum().item()),
                int(pred_gt.sum().item())),
            mean_requery_margin_on_pred_gt=_masked_mean(
                requery_margin, pred_gt),
            pair_mean_base_target=_masked_mean(base_target, pair_mask),
            pair_mean_base_pred=_masked_mean(base_pred, pair_mask),
            pair_mean_requery_score=_masked_mean(score_map, pair_mask),
            pair_mean_requery_gain=_masked_mean(gain, pair_mask),
            pair_requery_beats_base_pred_pixels=int(
                ((score_map > base_pred) & pair_mask).sum().item()),
            pair_requery_beats_base_pred_ratio=_safe_div(
                int(((score_map > base_pred) & pair_mask).sum().item()),
                int(pair_mask.sum().item())),
            pair_requery_support_prob_thd_pixels=int(
                ((score_map >= float(getattr(self, 'prob_thd', 0.0)))
                 & pair_mask).sum().item()),
            pair_requery_support_prob_thd_ratio=_safe_div(
                int(((score_map >= float(getattr(self, 'prob_thd', 0.0)))
                     & pair_mask).sum().item()),
                int(pair_mask.sum().item())),
        )

    def _build_sam3_geometry_requery_stats(
            self, base_logits, base_pred, components, data_sample, image):
        gt = self._sam3_geometry_get_gt(data_sample)
        if gt is None:
            return dict(valid_pixels=0, reason='missing_gt', pair_stats=[])
        if base_logits.shape[-2:] != gt.shape[-2:]:
            base_logits = F.interpolate(
                base_logits.unsqueeze(0),
                size=gt.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        if base_pred.shape[-2:] != gt.shape[-2:]:
            base_pred = F.interpolate(
                base_pred.float().view(1, 1, *base_pred.shape[-2:]),
                size=gt.shape[-2:],
                mode='nearest',
            ).squeeze(0).squeeze(0).long()

        valid_mask = gt != 255
        valid_pixels = int(valid_mask.sum().item())
        if valid_pixels == 0:
            return dict(valid_pixels=0, reason='empty_valid_mask',
                        pair_stats=[])

        pair_candidates = self._sam3_geometry_pair_candidates(
            base_logits, base_pred, valid_mask)
        if not pair_candidates:
            return dict(valid_pixels=valid_pixels, reason='no_pair_candidate',
                        pair_stats=[])

        box_modes = self._sam3_geometry_parse_name_list(getattr(
            self,
            'sam3_geometry_requery_box_modes',
            'target_pos,pred_neg,target_pos_pred_neg'))
        if not box_modes:
            box_modes = ['target_pos', 'pred_neg', 'target_pos_pred_neg']

        h, w = gt.shape[-2:]
        pair_stats = []
        empty_cache = _as_bool(getattr(
            self, 'sam3_geometry_requery_empty_cache', True))
        for pair in pair_candidates:
            target_cls = int(pair['target_cls'])
            pred_cls = int(pair['pred_cls'])
            target_region, pred_region = self._sam3_geometry_region_masks(
                base_logits, pair)
            target_box = self._sam3_geometry_box_from_mask(
                target_region, (h, w))
            pred_box = self._sam3_geometry_box_from_mask(pred_region, (h, w))
            if target_box is None or pred_box is None:
                continue
            target_prompt = self._sam3_geometry_prompt_for_class(target_cls)
            pred_prompt = self._sam3_geometry_prompt_for_class(pred_cls)
            pair_record = dict(
                pred_class_index=pred_cls,
                pred_class_name=str(self.class_names[pred_cls]),
                pred_role=_remote_sensing_class_role(
                    self.class_names[pred_cls]),
                target_class_index=target_cls,
                target_class_name=str(self.class_names[target_cls]),
                target_role=_remote_sensing_class_role(
                    self.class_names[target_cls]),
                support_pixels=int(pair['support_pixels']),
                mean_pred_score=pair.get('mean_pred_score'),
                mean_target_score=pair.get('mean_target_score'),
                mean_base_gap=pair.get('mean_base_gap'),
                target_seed_pixels=int(target_region.sum().item()),
                pred_seed_pixels=int(pred_region.sum().item()),
                target_box=target_box,
                pred_box=pred_box,
                target_prompt=target_prompt,
                pred_prompt=pred_prompt,
                mode_stats=[],
            )
            for mode in box_modes:
                if mode == 'target_pos':
                    prompt = target_prompt
                    boxes = [(target_box, True)]
                    queried_class = target_cls
                elif mode == 'pred_neg':
                    prompt = target_prompt
                    boxes = [(pred_box, False)]
                    queried_class = target_cls
                elif mode == 'target_pos_pred_neg':
                    prompt = target_prompt
                    boxes = [(target_box, True), (pred_box, False)]
                    queried_class = target_cls
                elif mode == 'pred_pos_control':
                    prompt = pred_prompt
                    boxes = [(pred_box, True)]
                    queried_class = pred_cls
                elif mode == 'pred_pos_target_neg':
                    prompt = pred_prompt
                    boxes = [(pred_box, True), (target_box, False)]
                    queried_class = pred_cls
                else:
                    continue
                try:
                    output = self._sam3_geometry_run_prompt(
                        image, prompt, boxes, (h, w))
                except RuntimeError as exc:
                    pair_record['mode_stats'].append(dict(
                        mode=mode,
                        prompt=prompt,
                        queried_class_index=int(queried_class),
                        queried_class_name=str(self.class_names[queried_class]),
                        error=repr(exc),
                    ))
                    if self.device.type == 'cuda' and empty_cache:
                        torch.cuda.empty_cache()
                    continue
                eval_stats = self._sam3_geometry_eval_map(
                    output['score'], base_logits, pair, gt, valid_mask)
                mode_record = dict(
                    mode=mode,
                    prompt=prompt,
                    queried_class_index=int(queried_class),
                    queried_class_name=str(self.class_names[queried_class]),
                    presence_score=output['presence_score'],
                    num_masks=output['num_masks'],
                )
                mode_record.update(eval_stats)
                pair_record['mode_stats'].append(mode_record)
                if self.device.type == 'cuda' and empty_cache:
                    torch.cuda.empty_cache()
            pair_stats.append(pair_record)

        return dict(
            valid_pixels=valid_pixels,
            topk=int(getattr(self, 'sam3_geometry_requery_topk', 3)),
            max_pairs=int(getattr(self, 'sam3_geometry_requery_max_pairs', 4)),
            min_pair_pixels=int(getattr(
                self, 'sam3_geometry_requery_min_pair_pixels', 64)),
            max_side=int(getattr(
                self, 'sam3_geometry_requery_max_side', 1024)),
            box_modes=box_modes,
            prompt_mode=str(getattr(
                self, 'sam3_geometry_requery_prompt_mode', 'best_query')),
            pair_stats=pair_stats,
        )
