import json
import os

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
    if not ext:
        ext = '.jsonl'
    return f'{base}_rank{rank}{ext}'


def _parse_name_list(value):
    if value is None:
        return []
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


def _remote_sensing_role(name):
    tokens = _class_name_tokens(name)
    if tokens & {'background', 'other', 'clutter', 'void', 'unknown'}:
        return 'catch_all'
    if tokens & {'car', 'vehicle', 'truck', 'ship', 'airplane', 'plane'}:
        return 'vehicle'
    if tokens & {
            'building', 'roof', 'house', 'facade', 'wall',
            'construction'}:
        return 'built_object'
    if tokens & {
            'road', 'pavement', 'impervious', 'surface',
            'sidewalk', 'parking'}:
        return 'impervious_surface'
    if tokens & {'tree', 'forest', 'wood', 'canopy'}:
        return 'woody_vegetation'
    if tokens & {
            'grass', 'vegetation', 'low', 'crop', 'cropland',
            'agricultural', 'agriculture', 'farmland', 'field'}:
        return 'low_vegetation'
    if tokens & {'water', 'river', 'lake', 'sea', 'pond'}:
        return 'water'
    if tokens & {'bareland', 'barren', 'soil', 'sand', 'bare'}:
        return 'bareland'
    return 'other_landcover'


class OntologyReadoutOracleMixin:
    """Prediction-preserving readout-vs-ontology oracle diagnostic."""

    def _uses_ontology_readout_oracle(self):
        return bool(getattr(self, 'dump_ontology_readout_oracle_stats', False))

    def _ontology_readout_source_names(self):
        names = _parse_name_list(
            getattr(
                self,
                'ontology_readout_oracle_sources',
                'semantic,instance,raw_mask,presence_gated,pe_layer0',
            ))
        return names or [
            'semantic',
            'instance',
            'raw_mask',
            'presence_gated',
            'pe_layer0',
        ]

    def _ontology_readout_internal_sources(self):
        sources = []
        for name in self._ontology_readout_source_names():
            if name in ('raw_mask', 'raw_object'):
                mapped = 'raw_object'
            elif name in ('raw_presence', 'raw_mask_presence'):
                mapped = 'raw_presence'
            elif name == 'pe_layer0':
                mapped = 'pe_layer_0'
            elif name.startswith('pe_layer_'):
                mapped = name
            elif name in (
                    'fusion_no_presence', 'semantic_presence',
                    'instance_presence', 'encoder_level_0',
                    'encoder_level_1', 'encoder_level_2'):
                mapped = name
            else:
                mapped = None
            if mapped and mapped not in sources:
                sources.append(mapped)
        return sources

    def _ontology_readout_diag_shape(self, height, width):
        max_side = max(
            1, int(getattr(self, 'ontology_readout_oracle_max_side', 256)))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _ontology_readout_resize_map(self, score_map, target_shape):
        return self._resize_internal_source_map(score_map, target_shape)

    def _ontology_readout_score_maps(
            self, base_logits, components, target_shape):
        maps = {}
        requested = self._ontology_readout_source_names()

        if 'presence_gated' in requested or 'final' in requested:
            resized = self._ontology_readout_resize_map(
                base_logits, target_shape)
            if resized is not None:
                if 'presence_gated' in requested:
                    maps['presence_gated'] = resized
                if 'final' in requested:
                    maps['final'] = resized

        if components is not None:
            if 'semantic' in requested:
                resized = self._ontology_readout_resize_map(
                    components.get('semantic_logits'), target_shape)
                if resized is not None:
                    maps['semantic'] = resized
            if 'instance' in requested:
                resized = self._ontology_readout_resize_map(
                    components.get('instance_logits'), target_shape)
                if resized is not None:
                    maps['instance'] = resized

            internal_maps = components.get('internal_source_maps') or {}
            aliases = {
                'raw_mask': 'raw_object',
                'raw_object': 'raw_object',
                'raw_presence': 'raw_presence',
                'raw_mask_presence': 'raw_presence',
                'pe_layer0': 'pe_layer_0',
            }
            for readout_name in requested:
                internal_name = aliases.get(readout_name, readout_name)
                if internal_name not in internal_maps:
                    continue
                resized = self._ontology_readout_resize_map(
                    internal_maps.get(internal_name), target_shape)
                if resized is not None:
                    maps[readout_name] = resized

        return {name: maps[name] for name in requested if name in maps}

    def _ontology_readout_pred_data(self, scores, topk):
        finite = torch.isfinite(scores)
        valid = finite.sum(dim=0) >= 1
        clean = torch.nan_to_num(
            scores.detach().float(),
            nan=-1e6,
            posinf=1e6,
            neginf=-1e6,
        )
        k = max(1, min(int(topk), int(clean.shape[0])))
        values, indices = torch.topk(clean, k=k, dim=0)
        if k == 1:
            margin = torch.zeros_like(values[0])
        else:
            margin = values[0] - values[1]
        return dict(
            pred=indices[0],
            topk_idx=indices,
            top1_score=values[0],
            margin=margin,
            valid=valid,
            finite=finite,
        )

    def _ontology_readout_class_rows(
            self, readout_name, pred, gt, valid_mask, source_valid,
            base_correct, base_wrong):
        rows = []
        for class_idx in range(int(self.num_cls)):
            gt_class = (gt == class_idx) & valid_mask
            pred_class = (pred == class_idx) & source_valid & valid_mask
            tp_mask = gt_class & pred_class
            tp = int(tp_mask.sum().item())
            gt_pixels = int(gt_class.sum().item())
            pred_pixels = int(pred_class.sum().item())
            fp = max(0, pred_pixels - tp)
            fn = max(0, gt_pixels - tp)
            class_wrong = gt_class & base_wrong
            wrong_recovered = int((class_wrong & pred_class).sum().item())
            changed = pred_class & (pred != gt) & valid_mask
            rows.append(dict(
                readout_name=readout_name,
                class_index=int(class_idx),
                class_name=self.class_names[class_idx],
                role=_remote_sensing_role(self.class_names[class_idx]),
                gt_pixels=gt_pixels,
                pred_pixels=pred_pixels,
                tp_pixels=tp,
                fp_pixels=fp,
                fn_pixels=fn,
                class_iou=_safe_div(tp, tp + fp + fn),
                class_precision=_safe_div(tp, pred_pixels),
                class_recall=_safe_div(tp, gt_pixels),
                baseline_correct_pixels=int((gt_class & base_correct).sum().item()),
                baseline_wrong_pixels=int(class_wrong.sum().item()),
                wrong_recovered_pixels=wrong_recovered,
                wrong_recovered_ratio=_safe_div(
                    wrong_recovered, int(class_wrong.sum().item())),
                changed_pred_pixels=int(changed.sum().item()),
            ))
        return rows

    def _ontology_readout_summary_row(
            self, readout_name, pred, gt, valid_mask, source_valid,
            base_correct, base_wrong, class_rows):
        source_valid = source_valid & valid_mask
        correct = (pred == gt) & source_valid
        changed = (pred != gt) & source_valid
        improved = source_valid & base_wrong & correct
        harmed = source_valid & base_correct & (~correct)
        ious = [
            row['class_iou'] for row in class_rows
            if row.get('gt_pixels', 0) > 0 and row.get('class_iou') is not None
        ]
        recalls = [
            row['class_recall'] for row in class_rows
            if row.get('gt_pixels', 0) > 0
            and row.get('class_recall') is not None
        ]
        return dict(
            readout_name=readout_name,
            valid_source_pixels=int(source_valid.sum().item()),
            valid_source_ratio=_safe_div(
                int(source_valid.sum().item()), int(valid_mask.sum().item())),
            correct_pixels=int(correct.sum().item()),
            accuracy=_safe_div(
                int(correct.sum().item()), int(source_valid.sum().item())),
            mIoU=None if not ious else sum(ious) / len(ious),
            mAcc=None if not recalls else sum(recalls) / len(recalls),
            baseline_wrong_pixels=int(base_wrong.sum().item()),
            wrong_recovered_pixels=int(improved.sum().item()),
            wrong_recovered_ratio=_safe_div(
                int(improved.sum().item()), int(base_wrong.sum().item())),
            changed_pixels=int(changed.sum().item()),
            improved_pixels=int(improved.sum().item()),
            harmed_pixels=int(harmed.sum().item()),
            net_improved_pixels=(
                int(improved.sum().item()) - int(harmed.sum().item())),
        )

    def _build_ontology_readout_oracle_stats(
            self, base_logits, base_pred, data_sample, components):
        if components is None or int(self.num_cls) <= 0:
            return None
        gt_data = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt_sem_seg = getattr(data_sample, 'gt_sem_seg')
            if hasattr(gt_sem_seg, 'data'):
                gt_data = gt_sem_seg.data.squeeze().to(self.device)
        if gt_data is None:
            return None

        diag_shape = self._ontology_readout_diag_shape(
            int(base_logits.shape[-2]), int(base_logits.shape[-1]))
        gt_diag = F.interpolate(
            gt_data.float().view(1, 1, *gt_data.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        base_pred_diag = F.interpolate(
            base_pred.float().view(1, 1, *base_pred.shape[-2:]),
            size=diag_shape,
            mode='nearest',
        ).squeeze().long()
        valid_mask = (gt_diag >= 0) & (gt_diag < int(self.num_cls)) & (
            gt_diag != 255)
        valid_pixels = int(valid_mask.sum().item())
        if valid_pixels == 0:
            return dict(
                dataset_name=getattr(self, 'seed_dataset_name', None),
                diagnostic_shape=list(diag_shape),
                valid_pixels=0,
                readout_stats=[],
                class_stats=[],
                pair_stats=[],
                pixel_oracle_stats={},
            )

        base_correct = (base_pred_diag == gt_diag) & valid_mask
        base_wrong = (~base_correct) & valid_mask
        topk = max(
            1,
            min(
                int(getattr(self, 'ontology_readout_oracle_topk', 3)),
                int(self.num_cls),
            ),
        )

        score_maps = self._ontology_readout_score_maps(
            base_logits, components, diag_shape)
        readout_data = {}
        readout_stats = []
        class_stats = []
        pair_stats = []

        baseline_valid = valid_mask
        baseline_class_rows = self._ontology_readout_class_rows(
            'baseline_thresholded',
            base_pred_diag,
            gt_diag,
            valid_mask,
            baseline_valid,
            base_correct,
            base_wrong,
        )
        class_stats.extend(baseline_class_rows)
        readout_stats.append(
            self._ontology_readout_summary_row(
                'baseline_thresholded',
                base_pred_diag,
                gt_diag,
                valid_mask,
                baseline_valid,
                base_correct,
                base_wrong,
                baseline_class_rows,
            ))

        gt_clamped = gt_diag.clamp(min=0, max=int(self.num_cls) - 1)
        for readout_name, scores in score_maps.items():
            data = self._ontology_readout_pred_data(scores, topk)
            readout_data[readout_name] = data
            source_valid = data['valid'] & valid_mask
            rows = self._ontology_readout_class_rows(
                readout_name,
                data['pred'],
                gt_diag,
                valid_mask,
                source_valid,
                base_correct,
                base_wrong,
            )
            class_stats.extend(rows)
            readout_stats.append(
                self._ontology_readout_summary_row(
                    readout_name,
                    data['pred'],
                    gt_diag,
                    valid_mask,
                    source_valid,
                    base_correct,
                    base_wrong,
                    rows,
                ))

            gt_topk = (
                data['topk_idx'] == gt_clamped.unsqueeze(0)).any(dim=0)
            gt_available = torch.gather(
                data['finite'].float(),
                0,
                gt_clamped.unsqueeze(0),
            ).squeeze(0).bool()
            for gt_class in range(int(self.num_cls)):
                gt_wrong = base_wrong & (gt_diag == gt_class)
                for pred_class in range(int(self.num_cls)):
                    if pred_class == gt_class:
                        continue
                    pair_mask = gt_wrong & (base_pred_diag == pred_class)
                    pair_pixels = int(pair_mask.sum().item())
                    if pair_pixels < int(
                            getattr(
                                self,
                                'ontology_readout_oracle_min_pair_pixels',
                                16,
                            )):
                        continue
                    source_correct = (
                        data['pred'] == gt_class) & source_valid
                    pair_stats.append(dict(
                        readout_name=readout_name,
                        gt_class_index=int(gt_class),
                        gt_class_name=self.class_names[gt_class],
                        gt_role=_remote_sensing_role(
                            self.class_names[gt_class]),
                        base_pred_class_index=int(pred_class),
                        base_pred_class_name=self.class_names[pred_class],
                        pred_role=_remote_sensing_role(
                            self.class_names[pred_class]),
                        pixels=pair_pixels,
                        source_top1_recovers_gt_pixels=int(
                            (pair_mask & source_correct).sum().item()),
                        source_top1_recovers_gt_ratio=_safe_div(
                            int((pair_mask & source_correct).sum().item()),
                            pair_pixels),
                        source_topk_contains_gt_pixels=int(
                            (pair_mask & gt_topk & gt_available).sum().item()),
                        source_topk_contains_gt_ratio=_safe_div(
                            int((pair_mask & gt_topk & gt_available).sum().item()),
                            pair_pixels),
                    ))

        correct_stack = []
        valid_stack = []
        readout_names = []
        for readout_name, data in readout_data.items():
            readout_names.append(readout_name)
            source_valid = data['valid'] & valid_mask
            valid_stack.append(source_valid)
            correct_stack.append((data['pred'] == gt_diag) & source_valid)
        if correct_stack:
            correct_stack = torch.stack(correct_stack, dim=0)
            valid_stack = torch.stack(valid_stack, dim=0)
            any_correct = correct_stack.any(dim=0) & valid_mask
            wrong_any = any_correct & base_wrong
            per_readout_wrong = [
                int((correct_stack[idx] & base_wrong).sum().item())
                for idx in range(len(readout_names))
            ]
            pixel_oracle_stats = dict(
                candidate_readouts=readout_names,
                baseline_wrong_pixels=int(base_wrong.sum().item()),
                any_readout_top1_correct_pixels=int(any_correct.sum().item()),
                baseline_wrong_any_readout_recovers_pixels=int(
                    wrong_any.sum().item()),
                baseline_wrong_any_readout_recovers_ratio=_safe_div(
                    int(wrong_any.sum().item()), int(base_wrong.sum().item())),
                per_readout_wrong_recovered=dict(
                    zip(readout_names, per_readout_wrong)),
            )
        else:
            pixel_oracle_stats = dict(
                candidate_readouts=[],
                baseline_wrong_pixels=int(base_wrong.sum().item()),
            )

        return dict(
            dataset_name=getattr(self, 'seed_dataset_name', None),
            diagnostic_shape=list(diag_shape),
            valid_pixels=valid_pixels,
            baseline_correct_pixels=int(base_correct.sum().item()),
            baseline_wrong_pixels=int(base_wrong.sum().item()),
            requested_readouts=self._ontology_readout_source_names(),
            available_readouts=list(score_maps.keys()),
            topk=topk,
            readout_stats=readout_stats,
            class_stats=class_stats,
            pair_stats=pair_stats,
            pixel_oracle_stats=pixel_oracle_stats,
        )

    def _write_ontology_readout_oracle_stats(self, record):
        if not self._uses_ontology_readout_oracle():
            return
        if self._ontology_readout_oracle_stats_file is None:
            path = (
                getattr(self, 'ontology_readout_oracle_stats_path', None)
                or './work_dirs/evidence_stats/ontology_readout_oracle.jsonl'
            )
            path = _ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._ontology_readout_oracle_stats_file = open(
                path, 'a', buffering=1)
        self._ontology_readout_oracle_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')
