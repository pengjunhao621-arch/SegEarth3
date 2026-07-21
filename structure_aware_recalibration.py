import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from ontology_readout_oracle import _remote_sensing_role
from sam3_geometry_requery_diagnostic import _ranked_jsonl_path, _safe_div


def _parse_csv(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).replace('|', ',').split(',')
    result = []
    for item in items:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


class StructureAwareRecalibrationMixin:
    """Role-validity atlas and same-forward context recalibration.

    The diagnostic asks whether SAM3's candidate concept evidence is present
    but selected unreliably under remote-sensing domain shift. The optional
    method is deliberately small: it uses only the current SAM3 forward pass,
    smooths class evidence inside local image context, and conservatively
    boosts candidates whose contextual support is stronger than the fixed
    prompt-wise baseline readout.
    """

    def _uses_structure_aware_recalibration(self):
        return bool(
            getattr(self, 'use_structure_aware_recalibration', False)
            or getattr(self, 'dump_structure_aware_recalibration_stats', False)
        )

    def _write_structure_aware_recalibration_stats(self, record):
        if not getattr(self, 'dump_structure_aware_recalibration_stats', False):
            return
        if self._structure_aware_recalibration_stats_file is None:
            path = (
                getattr(self, 'structure_aware_recalibration_stats_path', None)
                or './work_dirs/evidence_stats/'
                   'structure_aware_recalibration.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._structure_aware_recalibration_stats_file = open(
                path, 'a', buffering=1)
        self._structure_aware_recalibration_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _sar_class_roles(self):
        return [_remote_sensing_role(name) for name in self.class_names]

    def _sar_role_mask(self, device):
        roles = self._sar_class_roles()
        include = {
            item.lower() for item in _parse_csv(
                getattr(self, 'structure_recalibration_include_roles', 'all'))
        }
        exclude = {
            item.lower() for item in _parse_csv(
                getattr(self, 'structure_recalibration_exclude_roles',
                        'catch_all'))
        }
        allow = []
        for class_idx, role in enumerate(roles):
            role_l = str(role).lower()
            class_name_l = str(self.class_names[class_idx]).lower()
            ok = (not include or 'all' in include or role_l in include
                  or class_name_l in include)
            if role_l in exclude or class_name_l in exclude:
                ok = False
            if (bool(getattr(self, 'structure_recalibration_protect_bg', True))
                    and class_idx == int(self.bg_idx)):
                ok = False
            allow.append(ok)
        return torch.tensor(allow, dtype=torch.bool, device=device)

    def _sar_get_gt(self, data_sample, device):
        if data_sample is None or not hasattr(data_sample, 'gt_sem_seg'):
            return None
        gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
        if not hasattr(gt_sem_seg, 'data'):
            return None
        gt = gt_sem_seg.data.squeeze().to(device)
        if gt.ndim != 2:
            return None
        return gt.long()

    def _sar_source_logits(self, base_logits, components, variant):
        variant = str(variant or 'final_context').lower()
        sources = []
        if 'semantic' in variant and components is not None:
            semantic = components.get('semantic_logits')
            if isinstance(semantic, torch.Tensor):
                sources.append(semantic.detach().float())
        if 'instance' in variant and components is not None:
            instance = components.get('instance_logits')
            if isinstance(instance, torch.Tensor):
                sources.append(instance.detach().float())
        if 'final' in variant or not sources:
            sources.insert(0, base_logits.detach().float())
        if not sources:
            return base_logits.detach().float()
        stacked = torch.stack(sources, dim=0)
        if 'max' in variant:
            return stacked.max(dim=0)[0]
        return stacked.mean(dim=0)

    def _sar_local_context(self, source_logits):
        kernel = max(
            1, int(getattr(self, 'structure_recalibration_kernel', 9)))
        if kernel % 2 == 0:
            kernel += 1
        padding = kernel // 2
        source = source_logits.detach().float().unsqueeze(0)
        return F.avg_pool2d(
            source,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        ).squeeze(0)

    def _sar_pred_data(self, logits, topk=3):
        clean = torch.nan_to_num(
            logits.detach().float(),
            nan=-1e6,
            posinf=1e6,
            neginf=-1e6,
        )
        k = max(1, min(int(topk), int(clean.shape[0])))
        values, indices = torch.topk(clean, k=k, dim=0)
        if k > 1:
            margin = values[0] - values[1]
        else:
            margin = torch.zeros_like(values[0])
        return dict(
            pred=indices[0],
            topk_idx=indices,
            top1_score=values[0],
            margin=margin,
        )

    def _build_structure_aware_recalibration_logits(
            self, base_logits, base_pred, components):
        source_logits = self._sar_source_logits(
            base_logits,
            components,
            getattr(self, 'structure_recalibration_variant',
                    'final_context'),
        )
        context_logits = self._sar_local_context(source_logits)
        alpha = float(getattr(self, 'structure_recalibration_alpha', 0.25))
        min_gain = float(
            getattr(self, 'structure_recalibration_min_context_gain', 0.01))
        max_base_margin = float(
            getattr(self, 'structure_recalibration_max_base_margin', 1.0))
        min_context_score = float(
            getattr(self, 'structure_recalibration_min_context_score', 0.0))

        enhanced = base_logits.detach().float().clone()
        gain = context_logits - enhanced
        base_data = self._sar_pred_data(base_logits, topk=2)
        role_allow = self._sar_role_mask(base_logits.device).view(-1, 1, 1)
        class_not_allowed = ~role_allow
        gain[class_not_allowed.expand_as(gain)] = 0.0
        gain = gain.clamp_min(0.0)
        gain[gain < min_gain] = 0.0
        gain[context_logits < min_context_score] = 0.0
        if max_base_margin >= 0:
            gain[:, base_data['margin'] > max_base_margin] = 0.0

        # Avoid turning confident background/reject pixels into foreground
        # unless the caller explicitly asks for it.
        if bool(getattr(self, 'structure_recalibration_protect_bg', True)):
            gain[:, base_pred == int(self.bg_idx)] = 0.0

        enhanced = enhanced + alpha * gain
        context = dict(
            source_variant=str(
                getattr(self, 'structure_recalibration_variant',
                        'final_context')),
            kernel=int(getattr(self, 'structure_recalibration_kernel', 9)),
            alpha=alpha,
            min_context_gain=min_gain,
            min_context_score=min_context_score,
            max_base_margin=max_base_margin,
            protect_bg=bool(
                getattr(self, 'structure_recalibration_protect_bg', True)),
            include_roles=str(
                getattr(self, 'structure_recalibration_include_roles', 'all')),
            exclude_roles=str(
                getattr(self, 'structure_recalibration_exclude_roles',
                        'catch_all')),
            allowed_classes=int(role_allow.sum().item()),
            total_positive_gain_pixels=int((gain > 0).any(dim=0).sum().item()),
            mean_positive_gain=None if int((gain > 0).sum().item()) == 0 else
            float(gain[gain > 0].mean().item()),
        )
        return enhanced, context

    def _sar_class_metric_rows(self, pred, gt, valid, prefix):
        rows = []
        roles = self._sar_class_roles()
        for class_idx in range(int(self.num_cls)):
            gt_c = (gt == class_idx) & valid
            pred_c = (pred == class_idx) & valid
            tp = int((gt_c & pred_c).sum().item())
            gt_pixels = int(gt_c.sum().item())
            pred_pixels = int(pred_c.sum().item())
            fp = max(0, pred_pixels - tp)
            fn = max(0, gt_pixels - tp)
            rows.append(dict(
                metric_prefix=prefix,
                class_index=int(class_idx),
                class_name=self.class_names[class_idx],
                role=roles[class_idx],
                gt_pixels=gt_pixels,
                pred_pixels=pred_pixels,
                tp_pixels=tp,
                fp_pixels=fp,
                fn_pixels=fn,
                iou=_safe_div(tp, tp + fp + fn),
                precision=_safe_div(tp, pred_pixels),
                recall=_safe_div(tp, gt_pixels),
                pred_gt_area_ratio=_safe_div(pred_pixels, gt_pixels),
            ))
        return rows

    def _sar_readout_rows(
            self, name, logits, gt, valid, base_wrong, high_conf_wrong, topk):
        if logits is None or not isinstance(logits, torch.Tensor):
            return []
        if logits.shape[-2:] != gt.shape[-2:]:
            logits = F.interpolate(
                logits.detach().float().unsqueeze(0),
                size=gt.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        data = self._sar_pred_data(logits, topk=topk)
        gt_clamped = gt.clamp(min=0, max=int(self.num_cls) - 1)
        gt_topk = (data['topk_idx'] == gt_clamped.unsqueeze(0)).any(dim=0)
        roles = self._sar_class_roles()
        buckets = defaultdict(lambda: defaultdict(int))
        for class_idx, role in enumerate(roles):
            role_mask = valid & (gt == class_idx)
            wrong_mask = role_mask & base_wrong
            high_wrong_mask = role_mask & high_conf_wrong
            bucket = buckets[role]
            bucket['gt_pixels'] += int(role_mask.sum().item())
            bucket['base_wrong_pixels'] += int(wrong_mask.sum().item())
            bucket['high_conf_wrong_pixels'] += int(
                high_wrong_mask.sum().item())
            bucket['top1_correct_pixels'] += int(
                (role_mask & (data['pred'] == class_idx)).sum().item())
            bucket['topk_contains_gt_pixels'] += int(
                (role_mask & gt_topk).sum().item())
            bucket['wrong_top1_recovers_gt_pixels'] += int(
                (wrong_mask & (data['pred'] == class_idx)).sum().item())
            bucket['wrong_topk_contains_gt_pixels'] += int(
                (wrong_mask & gt_topk).sum().item())
        rows = []
        for role, bucket in buckets.items():
            rows.append(dict(
                readout_name=name,
                role=role,
                gt_pixels=int(bucket['gt_pixels']),
                base_wrong_pixels=int(bucket['base_wrong_pixels']),
                high_conf_wrong_pixels=int(bucket['high_conf_wrong_pixels']),
                top1_correct_pixels=int(bucket['top1_correct_pixels']),
                topk_contains_gt_pixels=int(
                    bucket['topk_contains_gt_pixels']),
                wrong_top1_recovers_gt_pixels=int(
                    bucket['wrong_top1_recovers_gt_pixels']),
                wrong_topk_contains_gt_pixels=int(
                    bucket['wrong_topk_contains_gt_pixels']),
                high_conf_wrong_ratio=_safe_div(
                    bucket['high_conf_wrong_pixels'], bucket['gt_pixels']),
                top1_accuracy=_safe_div(
                    bucket['top1_correct_pixels'], bucket['gt_pixels']),
                topk_contains_gt_ratio=_safe_div(
                    bucket['topk_contains_gt_pixels'], bucket['gt_pixels']),
                wrong_top1_recovers_gt_ratio=_safe_div(
                    bucket['wrong_top1_recovers_gt_pixels'],
                    bucket['base_wrong_pixels']),
                wrong_topk_contains_gt_ratio=_safe_div(
                    bucket['wrong_topk_contains_gt_pixels'],
                    bucket['base_wrong_pixels']),
            ))
        return rows

    def _build_structure_aware_recalibration_stats(
            self, base_logits, base_pred, recal_logits, recal_pred,
            components, data_sample, context):
        gt = self._sar_get_gt(data_sample, base_logits.device)
        if gt is None:
            return None
        if gt.shape[-2:] != base_pred.shape[-2:]:
            gt = F.interpolate(
                gt.float().view(1, 1, *gt.shape[-2:]),
                size=base_pred.shape[-2:],
                mode='nearest',
            ).squeeze().long()
        valid = (gt >= 0) & (gt < int(self.num_cls)) & (gt != 255)
        valid_pixels = int(valid.sum().item())
        if valid_pixels == 0:
            return dict(valid_pixels=0)

        base_data = self._sar_pred_data(
            base_logits,
            topk=max(2, int(getattr(self, 'structure_validity_topk', 3))),
        )
        base_correct = (base_pred == gt) & valid
        recal_correct = (recal_pred == gt) & valid
        base_wrong = (~base_correct) & valid
        high_conf_thd = float(
            getattr(self, 'structure_validity_high_conf_margin', 0.20))
        high_conf_wrong = base_wrong & (base_data['margin'] >= high_conf_thd)
        changed = (base_pred != recal_pred) & valid
        improved = changed & (~base_correct) & recal_correct
        harmed = changed & base_correct & (~recal_correct)

        semantic_logits = None
        instance_logits = None
        if components is not None:
            semantic_logits = components.get('semantic_logits')
            instance_logits = components.get('instance_logits')
        source_logits = self._sar_source_logits(
            base_logits,
            components,
            getattr(self, 'structure_recalibration_variant',
                    'final_context'))
        context_logits = self._sar_local_context(source_logits)
        topk = max(1, min(
            int(getattr(self, 'structure_validity_topk', 3)),
            int(self.num_cls)))
        readout_rows = []
        for name, logits in (
                ('final', base_logits),
                ('semantic', semantic_logits),
                ('instance', instance_logits),
                ('context', context_logits),
                ('recalibrated', recal_logits),
        ):
            readout_rows.extend(
                self._sar_readout_rows(
                    name, logits, gt, valid, base_wrong,
                    high_conf_wrong, topk))

        class_rows = (
            self._sar_class_metric_rows(base_pred, gt, valid, 'base')
            + self._sar_class_metric_rows(recal_pred, gt, valid, 'recal')
        )

        pair_rows = []
        min_pair_pixels = int(
            getattr(self, 'structure_validity_min_pair_pixels', 32))
        roles = self._sar_class_roles()
        for gt_class in range(int(self.num_cls)):
            gt_wrong = base_wrong & (gt == gt_class)
            if int(gt_wrong.sum().item()) == 0:
                continue
            for pred_class in range(int(self.num_cls)):
                if pred_class == gt_class:
                    continue
                pair_mask = gt_wrong & (base_pred == pred_class)
                pixels = int(pair_mask.sum().item())
                if pixels < min_pair_pixels:
                    continue
                pair_rows.append(dict(
                    gt_class_index=int(gt_class),
                    gt_class_name=self.class_names[gt_class],
                    gt_role=roles[gt_class],
                    base_pred_class_index=int(pred_class),
                    base_pred_class_name=self.class_names[pred_class],
                    pred_role=roles[pred_class],
                    pixels=pixels,
                    high_conf_wrong_pixels=int(
                        (pair_mask & high_conf_wrong).sum().item()),
                    recal_recovers_gt_pixels=int(
                        (pair_mask & (recal_pred == gt_class)).sum().item()),
                    recal_recovers_gt_ratio=_safe_div(
                        int((pair_mask & (recal_pred == gt_class)).sum().item()),
                        pixels),
                    recal_harms_to_other_pixels=int(
                        (pair_mask & (recal_pred != gt_class)
                         & (recal_pred != pred_class)).sum().item()),
                ))

        return dict(
            valid_pixels=valid_pixels,
            base_correct_pixels=int(base_correct.sum().item()),
            base_wrong_pixels=int(base_wrong.sum().item()),
            high_conf_wrong_pixels=int(high_conf_wrong.sum().item()),
            high_conf_wrong_ratio=_safe_div(
                int(high_conf_wrong.sum().item()), int(base_wrong.sum().item())),
            changed_pixels=int(changed.sum().item()),
            changed_ratio=_safe_div(int(changed.sum().item()), valid_pixels),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            net_improved_pixels=(
                int(improved.sum().item()) - int(harmed.sum().item())),
            correction_precision=_safe_div(
                int(improved.sum().item()), int(changed.sum().item())),
            harmful_rate=_safe_div(
                int(harmed.sum().item()), int(changed.sum().item())),
            context=context or {},
            readout_role_stats=readout_rows,
            class_stats=class_rows,
            pair_stats=pair_rows,
        )
