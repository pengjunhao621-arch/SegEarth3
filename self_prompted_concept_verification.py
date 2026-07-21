import json
import math
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from sam3_geometry_requery_diagnostic import (
    _as_bool,
    _masked_mean,
    _ranked_jsonl_path,
    _remote_sensing_class_role,
    _safe_div,
)


class SelfPromptedConceptVerificationMixin:
    """Prediction-preserving SAM3 self-prompted verification diagnostic.

    The diagnostic treats a baseline top-k conflict region as a region
    hypothesis, then re-queries SAM3 with the same positive geometry prompt for
    the baseline top1 concept and a competing top-k concept. The goal is not to
    change predictions here, but to ask whether geometry-conditioned re-query
    produces a model-visible validity signal for choosing between concepts.
    """

    def _uses_self_prompted_concept_verification(self):
        return _as_bool(getattr(
            self, 'dump_self_prompted_concept_verification_stats', False))

    def _write_self_prompted_concept_verification_stats(self, record):
        if not self.dump_self_prompted_concept_verification_stats:
            return
        if self._self_prompted_concept_verification_stats_file is None:
            path = (
                self.self_prompted_concept_verification_stats_path
                or './work_dirs/evidence_stats/'
                   'self_prompted_concept_verification.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._self_prompted_concept_verification_stats_file = open(
                path, 'a', buffering=1)
        self._self_prompted_concept_verification_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _spcv_score_threshold(self):
        value = getattr(
            self, 'self_prompted_concept_verification_score_thd', None)
        try:
            value = None if value is None else float(value)
        except (TypeError, ValueError):
            value = None
        if value is None or value < 0:
            value = max(float(getattr(self, 'prob_thd', 0.0)) * 0.5, 0.03)
        return value

    def _spcv_pair_candidates(self, base_logits, base_pred, valid_mask):
        topk = int(getattr(
            self, 'self_prompted_concept_verification_topk', 3))
        topk = max(2, min(topk, int(base_logits.shape[0])))
        max_pairs = int(getattr(
            self, 'self_prompted_concept_verification_max_pairs', 3))
        min_pair_pixels = int(getattr(
            self, 'self_prompted_concept_verification_min_pair_pixels', 96))
        include_bg = _as_bool(getattr(
            self, 'self_prompted_concept_verification_include_bg', True))
        try:
            max_base_gap = float(getattr(
                self,
                'self_prompted_concept_verification_max_base_gap',
                1.0))
        except (TypeError, ValueError):
            max_base_gap = 1.0
        try:
            min_candidate_score = float(getattr(
                self,
                'self_prompted_concept_verification_min_candidate_score',
                -1.0))
        except (TypeError, ValueError):
            min_candidate_score = -1.0

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
                base_gap = base_logits[pred_cls] - base_logits[target_cls]
                if max_base_gap >= 0:
                    support = support & (base_gap <= max_base_gap)
                if min_candidate_score >= 0:
                    support = support & (
                        base_logits[target_cls] >= min_candidate_score)
                support_pixels = int(support.sum().item())
                if support_pixels < min_pair_pixels:
                    continue
                key = (pred_cls, target_cls)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(dict(
                    pred_cls=pred_cls,
                    target_cls=target_cls,
                    support_mask=support,
                    support_pixels=support_pixels,
                    mean_pred_score=_masked_mean(
                        base_logits[pred_cls], support),
                    mean_target_score=_masked_mean(
                        base_logits[target_cls], support),
                    mean_base_gap=_masked_mean(base_gap, support),
                ))
        candidates.sort(
            key=lambda item: (
                item['support_pixels'],
                0.0 if item['mean_base_gap'] is None
                else -float(item['mean_base_gap']),
            ),
            reverse=True)
        if max_pairs > 0:
            candidates = candidates[:max_pairs]
        return candidates

    def _spcv_region_mask(self, base_logits, pair):
        support = pair['support_mask']
        target_cls = int(pair['target_cls'])
        pred_cls = int(pair['pred_cls'])
        target_score = base_logits[target_cls]
        pred_score = base_logits[pred_cls]
        score_thd = self._spcv_score_threshold()
        region = support & (
            (target_score >= score_thd) | (pred_score >= score_thd))
        min_pixels = int(getattr(
            self, 'self_prompted_concept_verification_min_region_pixels', 32))
        if int(region.sum().item()) >= min_pixels:
            return region
        fraction = float(getattr(
            self,
            'self_prompted_concept_verification_region_fraction',
            0.35))
        joint_score = torch.maximum(target_score, pred_score)
        selected = torch.zeros_like(support, dtype=torch.bool)
        if not bool(support.any()):
            return selected
        values = joint_score[support].detach().float()
        if values.numel() == 0:
            return selected
        k = max(min_pixels, int(math.ceil(values.numel() * fraction)))
        k = min(k, values.numel())
        threshold = torch.topk(values, k).values[-1]
        return support & (joint_score >= threshold)

    def _spcv_box_from_mask(self, mask, logit_hw):
        if mask is None or not bool(mask.any()):
            return None
        min_box = int(getattr(
            self, 'self_prompted_concept_verification_min_box_size', 4))
        context_scale = float(getattr(
            self, 'self_prompted_concept_verification_context_scale', 1.05))
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.numel() == 0:
            return None
        h, w = int(logit_hw[0]), int(logit_hw[1])
        y0 = float(coords[:, 0].min().item())
        y1 = float(coords[:, 0].max().item()) + 1.0
        x0 = float(coords[:, 1].min().item())
        x1 = float(coords[:, 1].max().item()) + 1.0
        bw = max(float(min_box), x1 - x0) * context_scale
        bh = max(float(min_box), y1 - y0) * context_scale
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
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

    def _spcv_resize_score(self, value, out_hw):
        if value is None:
            return None
        value = value.detach().float()
        if value.shape[-2:] == tuple(out_hw):
            return value
        return F.interpolate(
            value.view(1, 1, *value.shape[-2:]),
            size=tuple(out_hw),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)

    def _spcv_prompt_for_class(self, class_idx):
        class_idx = int(class_idx)
        prompt_mode = str(getattr(
            self,
            'self_prompted_concept_verification_prompt_mode',
            'best_query'))
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

    def _spcv_resize_image(self, image):
        max_side = int(getattr(
            self, 'self_prompted_concept_verification_max_side', 1024))
        if max_side <= 0:
            return image
        width, height = image.size
        long_side = max(width, height)
        if long_side <= max_side:
            return image
        scale = float(max_side) / float(long_side)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return image.resize(new_size)

    def _spcv_run_region_prompt(self, image, class_idx, region_box, out_hw):
        prompt = self._spcv_prompt_for_class(class_idx)
        resized_image = self._spcv_resize_image(image)
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        autocast_ctx = (
            torch.autocast(device_type=device_type, dtype=torch.bfloat16)
            if self.device.type == 'cuda' else nullcontext())
        with torch.no_grad(), autocast_ctx:
            state = self.processor.set_image(resized_image)
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt=prompt, state=state)
            state = self.processor.add_geometric_prompt(
                box=region_box, label=True, state=state)
            map_h, map_w = resized_image.size[1], resized_image.size[0]
            variant, semantic, instance = self._compute_prompt_variant_logit(
                state, map_h, map_w)
        variant = self._spcv_resize_score(variant, out_hw)
        semantic = self._spcv_resize_score(semantic, out_hw)
        instance = self._spcv_resize_score(instance, out_hw)
        presence = state.get('presence_score', 0.0)
        if isinstance(presence, torch.Tensor):
            presence = float(presence.detach().float().mean().item())
        output = dict(
            score=variant.detach().float(),
            semantic_score=(
                semantic.detach().float() if semantic is not None else None),
            instance_score=(
                instance.detach().float() if instance is not None else None),
            presence_score=float(presence),
            num_masks=int(state.get('masks_logits').shape[0])
            if 'masks_logits' in state else 0,
        )
        return prompt, output

    def _spcv_binary_iou(self, a, b, mask):
        if a is None or b is None or mask is None or not bool(mask.any()):
            return None
        a = a & mask
        b = b & mask
        inter = int((a & b).sum().item())
        union = int((a | b).sum().item())
        return _safe_div(inter, union)

    def _spcv_eval_pair(
            self, pred_output, target_output, base_logits, pair, gt,
            valid_mask, region_mask):
        pred_cls = int(pair['pred_cls'])
        target_cls = int(pair['target_cls'])
        pair_mask = pair['support_mask'] & valid_mask
        target_gt = pair_mask & (gt == target_cls)
        pred_gt = pair_mask & (gt == pred_cls)
        other_gt = pair_mask & (gt != target_cls) & (gt != pred_cls)

        base_pred = base_logits[pred_cls]
        base_target = base_logits[target_cls]
        pred_requery = pred_output['score']
        target_requery = target_output['score']
        base_margin = base_target - base_pred
        requery_margin = target_requery - pred_requery
        target_gain = target_requery - base_target
        pred_gain = pred_requery - base_pred
        margin_gain = requery_margin - base_margin

        score_thd = self._spcv_score_threshold()
        base_target_binary = base_target >= score_thd
        target_requery_binary = target_requery >= score_thd
        base_pred_binary = base_pred >= score_thd
        pred_requery_binary = pred_requery >= score_thd
        target_beats_pred = target_requery > pred_requery

        target_gt_pixels = int(target_gt.sum().item())
        pred_gt_pixels = int(pred_gt.sum().item())
        other_gt_pixels = int(other_gt.sum().item())
        pair_pixels = int(pair_mask.sum().item())
        target_help_pixels = int((target_beats_pred & target_gt).sum().item())
        pred_harm_pixels = int((target_beats_pred & pred_gt).sum().item())
        other_switch_pixels = int((target_beats_pred & other_gt).sum().item())

        return dict(
            pair_pixels=pair_pixels,
            region_pixels=int(region_mask.sum().item()),
            target_gt_pixels=target_gt_pixels,
            pred_gt_pixels=pred_gt_pixels,
            other_gt_pixels=other_gt_pixels,
            target_gt_ratio=_safe_div(target_gt_pixels, pair_pixels),
            pred_gt_ratio=_safe_div(pred_gt_pixels, pair_pixels),
            target_direction_is_helpful=bool(target_gt_pixels > pred_gt_pixels),
            switch_oracle_net_pixels=target_gt_pixels - pred_gt_pixels,
            mean_base_target_minus_pred=_masked_mean(base_margin, pair_mask),
            mean_requery_target_minus_pred=(
                _masked_mean(requery_margin, pair_mask)),
            mean_requery_margin_gain=_masked_mean(margin_gain, pair_mask),
            mean_target_gain=_masked_mean(target_gain, pair_mask),
            mean_pred_gain=_masked_mean(pred_gain, pair_mask),
            mean_target_gain_minus_pred_gain=(
                _masked_mean(target_gain - pred_gain, pair_mask)),
            mean_target_requery_score=(
                _masked_mean(target_requery, pair_mask)),
            mean_pred_requery_score=_masked_mean(pred_requery, pair_mask),
            mean_base_target_score=_masked_mean(base_target, pair_mask),
            mean_base_pred_score=_masked_mean(base_pred, pair_mask),
            target_requery_support_ratio=_safe_div(
                int((target_requery_binary & pair_mask).sum().item()),
                pair_pixels),
            pred_requery_support_ratio=_safe_div(
                int((pred_requery_binary & pair_mask).sum().item()),
                pair_pixels),
            target_support_gain_ratio=_safe_div(
                int((target_requery_binary & ~base_target_binary
                     & pair_mask).sum().item()),
                pair_pixels),
            pred_support_gain_ratio=_safe_div(
                int((pred_requery_binary & ~base_pred_binary
                     & pair_mask).sum().item()),
                pair_pixels),
            target_original_requery_iou=self._spcv_binary_iou(
                base_target_binary, target_requery_binary, pair_mask),
            pred_original_requery_iou=self._spcv_binary_iou(
                base_pred_binary, pred_requery_binary, pair_mask),
            target_requery_beats_pred_pixels=int(
                (target_beats_pred & pair_mask).sum().item()),
            target_requery_beats_pred_ratio=_safe_div(
                int((target_beats_pred & pair_mask).sum().item()),
                pair_pixels),
            target_help_pixels=target_help_pixels,
            pred_harm_pixels=pred_harm_pixels,
            other_switch_pixels=other_switch_pixels,
            target_help_ratio=_safe_div(target_help_pixels, target_gt_pixels),
            pred_harm_ratio=_safe_div(pred_harm_pixels, pred_gt_pixels),
            mean_requery_margin_on_target_gt=(
                _masked_mean(requery_margin, target_gt)),
            mean_requery_margin_on_pred_gt=(
                _masked_mean(requery_margin, pred_gt)),
            mean_margin_gain_on_target_gt=(
                _masked_mean(margin_gain, target_gt)),
            mean_margin_gain_on_pred_gt=(
                _masked_mean(margin_gain, pred_gt)),
            mean_base_margin_on_target_gt=(
                _masked_mean(base_margin, target_gt)),
            mean_base_margin_on_pred_gt=(
                _masked_mean(base_margin, pred_gt)),
        )

    def _build_self_prompted_concept_verification_stats(
            self, base_logits, base_pred, data_sample, image):
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

        pair_candidates = self._spcv_pair_candidates(
            base_logits, base_pred, valid_mask)
        if not pair_candidates:
            return dict(valid_pixels=valid_pixels, reason='no_pair_candidate',
                        pair_stats=[])

        h, w = gt.shape[-2:]
        pair_stats = []
        empty_cache = _as_bool(getattr(
            self, 'self_prompted_concept_verification_empty_cache', True))
        for pair in pair_candidates:
            pred_cls = int(pair['pred_cls'])
            target_cls = int(pair['target_cls'])
            region_mask = self._spcv_region_mask(base_logits, pair)
            region_box = self._spcv_box_from_mask(region_mask, (h, w))
            if region_box is None:
                continue
            try:
                pred_prompt, pred_output = self._spcv_run_region_prompt(
                    image, pred_cls, region_box, (h, w))
                target_prompt, target_output = self._spcv_run_region_prompt(
                    image, target_cls, region_box, (h, w))
            except RuntimeError as exc:
                pair_stats.append(dict(
                    pred_class_index=pred_cls,
                    pred_class_name=str(self.class_names[pred_cls]),
                    target_class_index=target_cls,
                    target_class_name=str(self.class_names[target_cls]),
                    support_pixels=int(pair['support_pixels']),
                    region_box=region_box,
                    error=repr(exc),
                ))
                if self.device.type == 'cuda' and empty_cache:
                    torch.cuda.empty_cache()
                continue

            eval_stats = self._spcv_eval_pair(
                pred_output,
                target_output,
                base_logits,
                pair,
                gt,
                valid_mask,
                region_mask,
            )
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
                region_box=region_box,
                pred_prompt=pred_prompt,
                target_prompt=target_prompt,
                pred_presence_score=pred_output.get('presence_score'),
                target_presence_score=target_output.get('presence_score'),
                pred_num_masks=pred_output.get('num_masks'),
                target_num_masks=target_output.get('num_masks'),
                pred_semantic_mean=_masked_mean(
                    pred_output.get('semantic_score'), pair['support_mask']),
                target_semantic_mean=_masked_mean(
                    target_output.get('semantic_score'), pair['support_mask']),
                pred_instance_mean=_masked_mean(
                    pred_output.get('instance_score'), pair['support_mask']),
                target_instance_mean=_masked_mean(
                    target_output.get('instance_score'), pair['support_mask']),
            )
            pair_record.update(eval_stats)
            pair_stats.append(pair_record)
            if self.device.type == 'cuda' and empty_cache:
                torch.cuda.empty_cache()

        return dict(
            valid_pixels=valid_pixels,
            topk=int(getattr(
                self, 'self_prompted_concept_verification_topk', 3)),
            max_pairs=int(getattr(
                self, 'self_prompted_concept_verification_max_pairs', 3)),
            min_pair_pixels=int(getattr(
                self,
                'self_prompted_concept_verification_min_pair_pixels',
                96)),
            max_base_gap=float(getattr(
                self,
                'self_prompted_concept_verification_max_base_gap',
                1.0)),
            max_side=int(getattr(
                self, 'self_prompted_concept_verification_max_side', 1024)),
            score_thd=float(self._spcv_score_threshold()),
            pair_stats=pair_stats,
        )
