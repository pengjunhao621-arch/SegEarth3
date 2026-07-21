import json
import os

import torch
import torch.nn.functional as F

from sam3_geometry_requery_diagnostic import (
    _as_bool,
    _masked_mean,
    _ranked_jsonl_path,
    _remote_sensing_class_role,
    _safe_div,
)


def _name_tokens(name):
    tokens = set()
    normalized = str(name or '').lower().replace('/', ',').replace('-', ' ')
    for item in normalized.split(','):
        item = item.strip()
        if not item:
            continue
        tokens.add(item)
        for part in item.split():
            if part:
                tokens.add(part)
    return tokens


def _ontology_prompt_variants(class_name):
    name = str(class_name).strip()
    tokens = _name_tokens(name)
    prompts = [
        name,
        f'remote sensing {name}',
        f'aerial image {name}',
        f'satellite image {name}',
        f'{name} land cover',
        f'{name} region',
    ]
    if tokens & {'roof', 'rooftop'}:
        prompts.extend([
            'building roof',
            'roof surface',
            'horizontal building roof',
            'top of building in aerial image',
        ])
    if tokens & {'facade', 'wall'}:
        prompts.extend([
            'building facade',
            'building wall',
            'vertical facade',
            'side of building',
        ])
    if tokens & {'tree', 'canopy'}:
        prompts.extend([
            'tree canopy',
            'tree crown',
            'woody vegetation',
            'trees in aerial image',
        ])
    if tokens & {'grass', 'vegetation'}:
        prompts.extend([
            'low vegetation',
            'grassland',
            'lawn',
            'green vegetation',
        ])
    if tokens & {'forest', 'woodland'}:
        prompts.extend([
            'forest canopy',
            'dense trees',
            'woodland',
            'forested area',
        ])
    if tokens & {'agricultural', 'agriculture', 'farmland', 'field'}:
        prompts.extend([
            'agricultural field',
            'cropland',
            'farmland',
            'cultivated land',
        ])
    if tokens & {'road', 'street'}:
        prompts.extend([
            'paved road',
            'road surface',
            'asphalt road',
            'street in aerial image',
        ])
    if tokens & {'pavement', 'sidewalk'}:
        prompts.extend([
            'paved surface',
            'pavement area',
            'sidewalk',
            'impervious pavement',
        ])
    if tokens & {'building', 'house'}:
        prompts.extend([
            'building footprint',
            'building rooftop',
            'man-made building',
            'building area',
        ])
    if tokens & {'background', 'other', 'clutter'}:
        prompts.extend([
            'background',
            'other land cover',
            'clutter region',
            'unlabeled background',
        ])
    if tokens & {'water', 'river', 'lake'}:
        prompts.extend([
            'water surface',
            'river or lake',
            'open water',
            'water body',
        ])
    if tokens & {'car', 'vehicle', 'truck'}:
        prompts.extend([
            'vehicle',
            'car',
            'small vehicle',
            'vehicle in aerial image',
        ])
    deduped = []
    seen = set()
    for prompt in prompts:
        prompt = str(prompt).strip()
        key = prompt.lower()
        if not prompt or key in seen:
            continue
        seen.add(key)
        deduped.append(prompt)
    return deduped


class EvidenceEnhancementMixin:
    """Optional SAM3 prompt-level evidence enhancement.

    This module enhances the evidence before final dense decision, rather than
    deciding whether to switch a finished prediction to another class. The
    prompt variants are generated from class names only; GT is used only for
    diagnostics.
    """

    def _uses_evidence_enhancement(self):
        return (
            _as_bool(getattr(self, 'use_evidence_enhancement', False))
            or _as_bool(getattr(self, 'dump_evidence_enhancement_stats', False))
        )

    def _write_evidence_enhancement_stats(self, record):
        if not self.dump_evidence_enhancement_stats:
            return
        if self._evidence_enhancement_stats_file is None:
            path = (
                self.evidence_enhancement_stats_path
                or './work_dirs/evidence_stats/evidence_enhancement.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._evidence_enhancement_stats_file = open(
                path, 'a', buffering=1)
        self._evidence_enhancement_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _ee_parse_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text:
            return []
        sep = '|' if '|' in text else ','
        return [item.strip() for item in text.split(sep) if item.strip()]

    def _ee_class_indices(self):
        spec = str(getattr(self, 'evidence_enhancement_classes', 'all'))
        spec = spec.strip()
        if not spec or spec.lower() == 'all':
            return list(range(self.num_cls))
        if spec.lower() == 'auto':
            dataset = (getattr(self, 'seed_dataset_name', '') or '').lower()
            tokens = []
            if dataset == 'vdd':
                tokens = ['roof', 'facade', 'background']
            elif dataset == 'vaihingen':
                tokens = ['grass', 'tree', 'clutter', 'road']
            elif dataset == 'potsdam':
                tokens = ['tree', 'grass', 'clutter']
            elif dataset == 'udd5':
                tokens = ['road', 'vegetation', 'building', 'background']
            elif dataset == 'openearthmap':
                tokens = ['pavement', 'building', 'tree', 'grass']
            elif dataset == 'loveda':
                tokens = ['forest', 'agricultural', 'background', 'water']
            selected = []
            for token in tokens:
                token_set = _name_tokens(token)
                for idx, class_name in enumerate(self.class_names):
                    class_tokens = _name_tokens(class_name)
                    if token in str(class_name).lower() or token_set & class_tokens:
                        if idx not in selected:
                            selected.append(idx)
            return selected

        selected = []
        for item in self._ee_parse_list(spec):
            if item.isdigit():
                idx = int(item)
                if 0 <= idx < self.num_cls and idx not in selected:
                    selected.append(idx)
                continue
            item_tokens = _name_tokens(item)
            for idx, class_name in enumerate(self.class_names):
                class_tokens = _name_tokens(class_name)
                if (
                        item.lower() == str(class_name).lower()
                        or item.lower() in str(class_name).lower()
                        or bool(item_tokens & class_tokens)):
                    if idx not in selected:
                        selected.append(idx)
        return selected

    def _ee_prompts_for_class(self, class_idx):
        class_name = str(self.class_names[int(class_idx)])
        prompts = []
        if _as_bool(getattr(
                self, 'evidence_enhancement_use_ontology_prompts', True)):
            prompts.extend(_ontology_prompt_variants(class_name))
        templates = self._ee_parse_list(getattr(
            self,
            'evidence_enhancement_templates',
            '{class}|remote sensing {class}|aerial image {class}|'
            'satellite image {class}|{class} land cover|{class} region'))
        for template in templates:
            prompts.append(template.replace('{class}', class_name))
        deduped = []
        seen = set()
        max_prompts = int(getattr(
            self, 'evidence_enhancement_max_prompts', 4))
        for prompt in prompts:
            prompt = str(prompt).strip()
            key = prompt.lower()
            if not prompt or key in seen:
                continue
            seen.add(key)
            deduped.append(prompt)
            if max_prompts > 0 and len(deduped) >= max_prompts:
                break
        return deduped

    def _ee_prompt_score(self, inference_state, h, w):
        final_map, semantic_map, instance_map = self._compute_prompt_variant_logit(
            inference_state, h, w)
        source = str(getattr(
            self, 'evidence_enhancement_source', 'semantic')).lower()
        if source == 'final':
            score = final_map
        elif source == 'instance':
            score = instance_map
        elif source == 'semantic':
            score = semantic_map
        else:
            raise ValueError(
                "evidence_enhancement_source must be one of "
                "'semantic', 'instance', or 'final', "
                f'but got {source!r}')
        if score is None:
            score = torch.zeros((h, w), device=self.device)

        presence_mode = str(getattr(
            self, 'evidence_enhancement_presence_mode', 'prompt')).lower()
        presence = inference_state.get('presence_score', None)
        if isinstance(presence, torch.Tensor):
            presence_value = presence.detach().float()
        elif presence is None:
            presence_value = torch.tensor(1.0, device=self.device)
        else:
            presence_value = torch.tensor(float(presence), device=self.device)
        if source != 'final':
            if presence_mode in ('prompt', 'linear'):
                score = score * presence_value
            elif presence_mode in ('sqrt', 'sqrt_prompt'):
                score = score * torch.sqrt(torch.clamp(presence_value, min=0.0))
            elif presence_mode in ('none', 'off', 'false'):
                pass
            else:
                raise ValueError(
                    'Unsupported evidence_enhancement_presence_mode: '
                    f'{presence_mode!r}')
        return (
            score.detach().float(),
            semantic_map.detach().float() if semantic_map is not None else None,
            instance_map.detach().float() if instance_map is not None else None,
            float(presence_value.detach().float().mean().item()),
        )

    def _ee_reduce_prompt_scores(self, maps):
        if not maps:
            return None
        stacked = torch.stack(maps, dim=0)
        reduce = str(getattr(
            self, 'evidence_enhancement_reduce', 'max')).lower()
        if reduce == 'max':
            return stacked.max(dim=0)[0]
        if reduce == 'mean':
            return stacked.mean(dim=0)
        if reduce == 'top2_mean':
            k = min(2, stacked.shape[0])
            return torch.topk(stacked, k=k, dim=0).values.mean(dim=0)
        raise ValueError(
            "evidence_enhancement_reduce must be 'max', 'mean', or "
            f"'top2_mean', but got {reduce!r}")

    def _ee_confusion(self, gt, pred, valid):
        matrix = torch.zeros(
            (self.num_cls, self.num_cls),
            dtype=torch.long,
            device='cpu')
        gt_cpu = gt[valid].detach().long().cpu()
        pred_cpu = pred[valid].detach().long().cpu()
        for gt_idx, pred_idx in zip(gt_cpu.tolist(), pred_cpu.tolist()):
            if 0 <= gt_idx < self.num_cls and 0 <= pred_idx < self.num_cls:
                matrix[gt_idx, pred_idx] += 1
        return matrix.tolist()

    def _ee_class_rows(self, gt, base_pred, enhanced_pred, valid,
                       base_logits, enhanced_logits, selected_classes):
        rows = []
        selected = set(int(idx) for idx in selected_classes)
        for class_idx, class_name in enumerate(self.class_names):
            gt_mask = valid & (gt == class_idx)
            pred_mask = valid & (base_pred == class_idx)
            enhanced_mask = valid & (enhanced_pred == class_idx)
            gt_pixels = int(gt_mask.sum().item())
            base_inter = int((gt_mask & pred_mask).sum().item())
            enhanced_inter = int((gt_mask & enhanced_mask).sum().item())
            base_union = int((gt_mask | pred_mask).sum().item())
            enhanced_union = int((gt_mask | enhanced_mask).sum().item())
            base_iou = _safe_div(base_inter, base_union)
            enhanced_iou = _safe_div(enhanced_inter, enhanced_union)
            iou_delta = (
                None if base_iou is None or enhanced_iou is None
                else enhanced_iou - base_iou)
            rows.append(dict(
                class_index=class_idx,
                class_name=str(class_name),
                role=_remote_sensing_class_role(class_name),
                enhanced=class_idx in selected,
                gt_pixels=gt_pixels,
                base_iou=base_iou,
                enhanced_iou=enhanced_iou,
                iou_delta=iou_delta,
                base_recall=_safe_div(base_inter, gt_pixels),
                enhanced_recall=_safe_div(enhanced_inter, gt_pixels),
                base_pred_pixels=int(pred_mask.sum().item()),
                enhanced_pred_pixels=int(enhanced_mask.sum().item()),
                mean_base_logit_on_gt=_masked_mean(
                    base_logits[class_idx], gt_mask),
                mean_enhanced_logit_on_gt=_masked_mean(
                    enhanced_logits[class_idx], gt_mask),
                mean_logit_gain_on_gt=_masked_mean(
                    enhanced_logits[class_idx] - base_logits[class_idx],
                    gt_mask),
            ))
        return rows

    def _build_evidence_enhancement_logits(
            self, base_logits, base_pred, image, data_sample=None,
            components=None):
        w, h = image.size
        selected_classes = self._ee_class_indices()
        enhanced_logits = base_logits.clone()
        class_prompt_rows = []
        combine = str(getattr(
            self, 'evidence_enhancement_combine', 'max')).lower()
        alpha = float(getattr(self, 'evidence_enhancement_alpha', 1.0))

        with torch.no_grad(), torch.autocast(
                device_type='cuda' if self.device.type == 'cuda' else 'cpu',
                dtype=torch.bfloat16,
                enabled=self.device.type == 'cuda'):
            state = self.processor.set_image(image)
            for class_idx in selected_classes:
                prompts = self._ee_prompts_for_class(class_idx)
                if not prompts:
                    continue
                prompt_scores = []
                prompt_rows = []
                for order, prompt in enumerate(prompts):
                    self.processor.reset_all_prompts(state)
                    state = self.processor.set_text_prompt(
                        state=state,
                        prompt=prompt,
                    )
                    score, semantic, instance, presence = self._ee_prompt_score(
                        state, h, w)
                    prompt_scores.append(score)
                    prompt_rows.append(dict(
                        class_index=int(class_idx),
                        class_name=str(self.class_names[int(class_idx)]),
                        role=_remote_sensing_class_role(
                            self.class_names[int(class_idx)]),
                        prompt_order=int(order),
                        prompt=prompt,
                        presence_score=presence,
                        score_mean=float(score.mean().item()),
                        score_max=float(score.max().item()),
                        semantic_mean=(
                            None if semantic is None
                            else float(semantic.mean().item())),
                        instance_mean=(
                            None if instance is None
                            else float(instance.mean().item())),
                    ))
                reduced = self._ee_reduce_prompt_scores(prompt_scores)
                if reduced is None:
                    continue
                if reduced.shape[-2:] != base_logits.shape[-2:]:
                    reduced = F.interpolate(
                        reduced.view(1, 1, *reduced.shape[-2:]),
                        size=base_logits.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                old = enhanced_logits[int(class_idx)]
                if combine == 'max':
                    new = torch.maximum(old, reduced)
                elif combine == 'replace':
                    new = reduced
                elif combine == 'residual':
                    new = old + alpha * torch.clamp(reduced - old, min=0.0)
                else:
                    raise ValueError(
                        "evidence_enhancement_combine must be 'max', "
                        f"'replace', or 'residual', but got {combine!r}")
                enhanced_logits[int(class_idx)] = new
                gain = new - base_logits[int(class_idx)]
                for row in prompt_rows:
                    row['selected_prompt_count'] = len(prompts)
                    row['reduced_score_mean'] = float(reduced.mean().item())
                    row['reduced_score_max'] = float(reduced.max().item())
                    row['applied_gain_mean'] = float(gain.mean().item())
                    row['applied_gain_max'] = float(gain.max().item())
                class_prompt_rows.extend(prompt_rows)

        context = dict(
            dataset_name=getattr(self, 'seed_dataset_name', None),
            selected_classes=[int(idx) for idx in selected_classes],
            selected_class_names=[
                str(self.class_names[int(idx)]) for idx in selected_classes],
            source=str(getattr(
                self, 'evidence_enhancement_source', 'semantic')),
            presence_mode=str(getattr(
                self, 'evidence_enhancement_presence_mode', 'prompt')),
            reduce=str(getattr(self, 'evidence_enhancement_reduce', 'max')),
            combine=str(getattr(self, 'evidence_enhancement_combine', 'max')),
            max_prompts=int(getattr(
                self, 'evidence_enhancement_max_prompts', 4)),
            class_prompt_rows=class_prompt_rows,
        )

        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt = data_sample.gt_sem_seg.data.squeeze().to(self.device)
            if gt.shape[-2:] != enhanced_logits.shape[-2:]:
                gt = F.interpolate(
                    gt.float().view(1, 1, *gt.shape[-2:]),
                    size=enhanced_logits.shape[-2:],
                    mode='nearest',
                ).squeeze().long()
            valid = gt != 255
            enhanced_pred = self._threshold_with_reject_recovery(
                enhanced_logits, components)
            context.update(dict(
                valid_pixels=int(valid.sum().item()),
                baseline_confusion=self._ee_confusion(gt, base_pred, valid),
                enhanced_confusion=self._ee_confusion(
                    gt, enhanced_pred, valid),
                class_rows=self._ee_class_rows(
                    gt,
                    base_pred,
                    enhanced_pred,
                    valid,
                    base_logits,
                    enhanced_logits,
                    selected_classes,
                ),
            ))
        return enhanced_logits, context
