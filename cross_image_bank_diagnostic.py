import json
import os

import torch
import torch.nn.functional as F


def _cib_safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _cib_ranked_jsonl_path(path):
    rank = int(os.environ.get('RANK', 0))
    root, ext = os.path.splitext(path)
    if not ext:
        ext = '.jsonl'
    return f'{root}_rank{rank}{ext}'


def _cib_resize_label(label, shape):
    return F.interpolate(
        label.float().view(1, 1, *label.shape),
        size=shape,
        mode='nearest',
    ).squeeze().long()


def _cib_resize_bool(mask, shape):
    return F.interpolate(
        mask.float().view(1, 1, *mask.shape),
        size=shape,
        mode='nearest',
    ).squeeze().bool()


class CrossImageBankDiagnosticMixin:
    """Dump sufficient statistics for cross-image SAM3 concept banks.

    The online pass never builds a GT-based bank. It only records no-GT clean
    seed prototypes and GT-sliced error feature means for offline diagnostics.
    """

    def _uses_cross_image_bank_diagnostic(self):
        return bool(self.dump_cross_image_bank_stats)

    def _uses_cross_image_bank_recomposition(self):
        return bool(
            getattr(self, 'use_cross_image_bank_recomposition', False)
            or getattr(
                self,
                'dump_cross_image_bank_recomposition_stats',
                False))

    def _uses_cross_image_bank_features(self):
        return (
            self._uses_cross_image_bank_diagnostic()
            or self._uses_cross_image_bank_recomposition())

    def _get_cross_image_bank_spaces(self):
        spaces = self.cross_image_bank_spaces
        if spaces is None:
            return ['vision', 'pe_layer_0']
        if isinstance(spaces, str):
            spaces = [item.strip() for item in spaces.split(',')]
        elif isinstance(spaces, (list, tuple)):
            spaces = [str(item).strip() for item in spaces]
        else:
            spaces = [str(spaces).strip()]
        normalized = []
        for space in spaces:
            if not space:
                continue
            if space == 'pe_all':
                for item in [
                        'pe_layer_0', 'pe_layer_1',
                        'pe_layer_2', 'pe_layer_3']:
                    if item not in normalized:
                        normalized.append(item)
                continue
            is_pe = (
                space.startswith('pe_layer_')
                and space[len('pe_layer_'):].isdigit())
            if space != 'vision' and not is_pe:
                raise ValueError(
                    "cross_image_bank_spaces supports 'vision', 'pe_all', "
                    "and 'pe_layer_{idx}', "
                    f'but got {space!r}')
            if space not in normalized:
                normalized.append(space)
        return normalized or ['vision', 'pe_layer_0']

    def _cross_image_bank_uses_pe(self):
        if not self._uses_cross_image_bank_features():
            return False
        if self._uses_cross_image_bank_recomposition():
            apply_space = str(
                getattr(self, 'cross_image_bank_apply_space', '') or '')
            if apply_space.startswith('pe_layer_'):
                return True
        try:
            spaces = self._get_cross_image_bank_spaces()
        except ValueError:
            return False
        return any(space.startswith('pe_layer_') for space in spaces)

    def _load_cross_image_bank(self):
        if hasattr(self, '_cross_image_bank_cache'):
            return self._cross_image_bank_cache
        path = getattr(self, 'cross_image_bank_file', None)
        if not path:
            self._cross_image_bank_cache = None
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'cross_image_bank_file does not exist: {path}')
        with open(path, 'r') as f:
            self._cross_image_bank_cache = json.load(f)
        return self._cross_image_bank_cache

    def _get_cross_image_bank_entry(self):
        bank = self._load_cross_image_bank()
        if not bank:
            return None
        dataset = (
            getattr(self, 'cross_image_bank_dataset_name', None)
            or getattr(self, 'seed_dataset_name', None)
            or 'unknown')
        space = str(getattr(
            self, 'cross_image_bank_apply_space', 'pe_layer_0'))
        variant = str(getattr(
            self, 'cross_image_bank_apply_variant', 'image_centered'))
        banks = bank.get('banks') or {}
        dataset_bank = banks.get(dataset)
        if dataset_bank is None and len(banks) == 1:
            dataset_bank = next(iter(banks.values()))
        if dataset_bank is None:
            raise KeyError(
                f'Dataset {dataset!r} not found in cross-image bank file.')
        space_bank = dataset_bank.get(space)
        if space_bank is None:
            raise KeyError(
                f'Space {space!r} not found in cross-image bank file.')
        variant_bank = space_bank.get(variant)
        if variant_bank is None:
            raise KeyError(
                f'Variant {variant!r} not found in cross-image bank file.')
        return variant_bank

    def _cib_vector_to_list(self, vector):
        decimals = int(self.cross_image_bank_vector_decimals)
        vector = torch.nan_to_num(vector.detach().float(), nan=0.0)
        return [round(float(value), decimals)
                for value in vector.detach().cpu().tolist()]

    def _cib_resize_image_for_features(self, image):
        max_side = int(self.cross_image_bank_feature_max_side)
        if max_side <= 0:
            return image
        width, height = image.size
        scale = min(1.0, float(max_side) / float(max(width, height)))
        if scale >= 1.0:
            return image
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return image.resize(new_size)

    def _extract_cross_image_bank_features(self, image):
        resized = self._cib_resize_image_for_features(image)
        with torch.no_grad(), torch.autocast(device_type='cuda',
                                             dtype=torch.bfloat16):
            state = self.processor.set_image(resized)
        features = {}
        if isinstance(state.get('vision_features'), torch.Tensor):
            features['vision'] = state['vision_features'].detach()
        if self._cross_image_bank_uses_pe():
            backbone_out = state.get('backbone_out') or {}
            backbone_fpn = backbone_out.get('backbone_fpn')
            if isinstance(backbone_fpn, (list, tuple)):
                for idx, feat in enumerate(backbone_fpn):
                    if isinstance(feat, torch.Tensor):
                        features[f'pe_layer_{idx}'] = feat.detach()
        return features

    def _get_cross_image_bank_feature(self, components, space,
                                      image=None, reencoded=None):
        feature = self._get_feature_map_for_similarity_space(components, space)
        if feature is None and reencoded is not None:
            feature = reencoded.get(space)
        if feature is None:
            return None
        if feature.ndim == 4:
            feature = feature.squeeze(0)
        if feature.ndim != 3:
            return None
        return torch.nan_to_num(feature.detach().float(), nan=0.0)

    def _transform_cross_image_bank_feature(self, feature, base_logits,
                                            base_pred, components, variant):
        if variant == 'raw':
            return feature, dict(center='raw')

        channel_count, feat_h, feat_w = feature.shape
        feature_2d = feature.view(channel_count, -1)
        image_mean = feature_2d.mean(dim=1)
        center = image_mean
        center_name = 'image_mean'

        if variant in ('seed_centered', 'class_balanced_centered'):
            seed_context = self._build_seed_rule_context(
                base_logits, base_pred, components)
            seed_mask = self._select_seed_rule(
                seed_context,
                getattr(
                    self,
                    'cross_image_bank_apply_seed_rule',
                    self.cross_image_bank_seed_rule))
            if seed_mask is not None and seed_mask.any():
                seed_low = _cib_resize_bool(seed_mask, (feat_h, feat_w))
                if seed_low.any():
                    if variant == 'seed_centered':
                        center = feature[:, seed_low].mean(dim=1)
                        center_name = 'seed_mean'
                    else:
                        seed_class = seed_context['final_top1_idx']
                        seed_class_low = _cib_resize_label(
                            seed_class.clamp(
                                min=0, max=self.num_cls - 1),
                            (feat_h, feat_w),
                        )
                        class_protos = []
                        for class_idx in range(self.num_cls):
                            class_seed = (
                                seed_low
                                & (seed_class_low == int(class_idx)))
                            if int(class_seed.sum().item()) < int(
                                    getattr(
                                        self,
                                        'cross_image_bank_apply_min_seed_pixels',
                                        self.cross_image_bank_min_seed_pixels)):
                                continue
                            class_protos.append(
                                feature[:, class_seed].mean(dim=1))
                        if class_protos:
                            center = torch.stack(class_protos, dim=0).mean(
                                dim=0)
                            center_name = 'class_balanced_seed_mean'
        elif variant != 'image_centered':
            raise ValueError(
                'cross_image_bank_apply_variant must be one of raw, '
                'image_centered, seed_centered, class_balanced_centered, '
                f'but got {variant!r}')

        return (
            feature - center.view(-1, 1, 1),
            dict(center=center_name))

    def _bank_vectors_for_current_image(self, bank_entry, image_id):
        class_rows = bank_entry.get('classes') or []
        if not class_rows:
            return None, None
        first_proto = class_rows[0].get('prototype') or []
        if not first_proto:
            return None, None
        dim = len(first_proto)
        vectors = torch.full(
            (self.num_cls, dim),
            float('nan'),
            device=self.device,
            dtype=torch.float32,
        )
        weights = torch.zeros(
            self.num_cls,
            device=self.device,
            dtype=torch.float32,
        )
        image_contribs = bank_entry.get('image_contribs') or {}
        current_contrib = image_contribs.get(str(image_id), {})
        min_weight = float(getattr(
            self, 'cross_image_bank_apply_min_class_weight', 1.0))
        exclude_current = bool(getattr(
            self, 'cross_image_bank_apply_exclude_current_image', True))

        for row in class_rows:
            class_idx = int(row.get('class_index'))
            if class_idx < 0 or class_idx >= self.num_cls:
                continue
            weight = float(row.get('weight', 0.0) or 0.0)
            total_sum = torch.tensor(
                row.get('sum') or [],
                device=self.device,
                dtype=torch.float32,
            )
            if total_sum.numel() != dim:
                proto = torch.tensor(
                    row.get('prototype') or [],
                    device=self.device,
                    dtype=torch.float32,
                )
                total_sum = proto * weight
            if exclude_current:
                contrib = current_contrib.get(str(class_idx))
                if contrib:
                    contrib_sum = torch.tensor(
                        contrib.get('sum') or [],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    if contrib_sum.numel() == dim:
                        total_sum = total_sum - contrib_sum
                        weight -= float(contrib.get('weight', 0.0) or 0.0)
            if weight < min_weight:
                continue
            vectors[class_idx] = total_sum / max(weight, 1e-12)
            weights[class_idx] = weight
        return vectors, weights

    def _feature_bank_affinity(self, feature, bank_vectors):
        channel_count, feat_h, feat_w = feature.shape
        if bank_vectors is None or bank_vectors.shape[1] != channel_count:
            return None
        feature_flat = feature.view(channel_count, -1)
        feature_norm = F.normalize(feature_flat, dim=0, eps=1e-6)
        bank = torch.nan_to_num(bank_vectors, nan=0.0)
        valid = torch.isfinite(bank_vectors).all(dim=1)
        bank_norm = F.normalize(bank, dim=1, eps=1e-6)
        affinity = torch.matmul(bank_norm, feature_norm).view(
            self.num_cls, feat_h, feat_w)
        affinity[~valid] = float('nan')
        return affinity

    def _build_cross_image_bank_recomposition_logits(
            self, base_logits, base_pred, components, image=None,
            image_id=None, data_sample=None):
        bank_entry = self._get_cross_image_bank_entry()
        if bank_entry is None:
            return base_logits, dict(reason='missing_bank')

        space = str(getattr(
            self, 'cross_image_bank_apply_space', 'pe_layer_0'))
        variant = str(getattr(
            self, 'cross_image_bank_apply_variant', 'image_centered'))
        reencoded = None
        feature = self._get_cross_image_bank_feature(components, space)
        if feature is None and image is not None and bool(
                self.cross_image_bank_reencode_missing_features):
            reencoded = self._extract_cross_image_bank_features(image)
            feature = self._get_cross_image_bank_feature(
                components, space, image=image, reencoded=reencoded)
        if feature is None:
            return base_logits, dict(reason='missing_feature', space=space)

        feature, center_info = self._transform_cross_image_bank_feature(
            feature, base_logits, base_pred, components, variant)
        bank_vectors, bank_weights = self._bank_vectors_for_current_image(
            bank_entry, image_id)
        affinity = self._feature_bank_affinity(feature, bank_vectors)
        if affinity is None:
            return base_logits, dict(reason='bad_affinity', space=space)
        if affinity.shape[-2:] != base_logits.shape[-2:]:
            affinity = self._interpolate_float32(
                affinity.unsqueeze(0),
                base_logits.shape[-2:],
            ).squeeze(0)

        topk = max(2, int(getattr(
            self, 'cross_image_bank_apply_topk', 3)))
        topk = min(topk, self.num_cls)
        top_vals, top_idx = torch.topk(base_logits, k=topk, dim=0)
        top1_idx = top_idx[0]
        top1_score = top_vals[0]
        top1_aff = torch.gather(
            affinity,
            0,
            top1_idx.clamp(min=0, max=self.num_cls - 1).unsqueeze(0),
        ).squeeze(0)

        best_margin = torch.full_like(top1_score, float('-inf'))
        best_candidate = torch.full_like(top1_idx, -1)
        best_rank = torch.zeros_like(top1_idx)
        max_base_gap = float(getattr(
            self, 'cross_image_bank_apply_max_base_gap', 0.60))
        min_bank_margin = float(getattr(
            self, 'cross_image_bank_apply_min_bank_margin', 0.10))
        min_candidate_score = float(getattr(
            self, 'cross_image_bank_apply_min_candidate_score', -1.0))
        min_candidate_affinity = float(getattr(
            self, 'cross_image_bank_apply_min_candidate_affinity', -1.0))
        protect_bg = bool(getattr(
            self, 'cross_image_bank_apply_protect_bg', True))
        exclude_bg_candidate = bool(getattr(
            self, 'cross_image_bank_apply_exclude_bg_candidate', True))
        require_base_non_bg = bool(getattr(
            self, 'cross_image_bank_apply_require_base_non_bg', True))
        base_non_bg = base_pred != int(self.bg_idx)
        top1_non_bg = top1_idx != int(self.bg_idx)

        for rank in range(1, topk):
            cand_idx = top_idx[rank]
            cand_score = top_vals[rank]
            cand_aff = torch.gather(
                affinity,
                0,
                cand_idx.clamp(min=0, max=self.num_cls - 1).unsqueeze(0),
            ).squeeze(0)
            bank_margin = cand_aff - top1_aff
            base_gap = top1_score - cand_score
            gate = torch.isfinite(bank_margin)
            gate &= bank_margin >= min_bank_margin
            gate &= base_gap <= max_base_gap
            gate &= cand_score >= min_candidate_score
            gate &= cand_aff >= min_candidate_affinity
            if require_base_non_bg:
                gate &= base_non_bg
            if protect_bg:
                gate &= top1_non_bg
            if exclude_bg_candidate:
                gate &= cand_idx != int(self.bg_idx)
            gate &= bank_margin > best_margin
            best_margin = torch.where(gate, bank_margin, best_margin)
            best_candidate = torch.where(gate, cand_idx, best_candidate)
            best_rank = torch.where(
                gate,
                torch.full_like(best_rank, int(rank + 1)),
                best_rank)

        change_mask = best_candidate >= 0
        new_logits = base_logits.clone()
        boost = float(getattr(
            self, 'cross_image_bank_apply_logit_boost', 1e-4))
        if change_mask.any():
            flat_logits = new_logits.view(self.num_cls, -1)
            flat_mask = change_mask.view(-1)
            flat_positions = torch.nonzero(
                flat_mask, as_tuple=False).squeeze(1)
            flat_candidates = best_candidate.view(-1)[flat_positions]
            target_values = top1_score.view(-1)[flat_positions] + boost
            current_values = flat_logits[flat_candidates, flat_positions]
            flat_logits[flat_candidates, flat_positions] = torch.maximum(
                current_values,
                target_values,
            )

        context = dict(
            reason='ok',
            space=space,
            variant=variant,
            center=center_info.get('center'),
            topk=topk,
            min_bank_margin=min_bank_margin,
            max_base_gap=max_base_gap,
            changed_pixels=int(change_mask.sum().item()),
            changed_ratio=_cib_safe_div(
                int(change_mask.sum().item()),
                int(change_mask.numel())),
        )

        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt = data_sample.gt_sem_seg.data
            if gt.ndim == 3:
                gt = gt.squeeze(0)
            gt = gt.to(base_logits.device)
            valid = gt != 255
            new_pred = self._threshold_with_reject_recovery(
                new_logits, components)
            changed = valid & (base_pred != new_pred)
            base_correct = base_pred == gt
            new_correct = new_pred == gt
            context.update(dict(
                valid_pixels=int(valid.sum().item()),
                final_changed_pixels=int(changed.sum().item()),
                improved_pixels=int(
                    (changed & ~base_correct & new_correct).sum().item()),
                harmed_pixels=int(
                    (changed & base_correct & ~new_correct).sum().item()),
                wrong_to_wrong_pixels=int(
                    (changed & ~base_correct & ~new_correct).sum().item()),
            ))
            pair_rows = []
            for old_class in range(self.num_cls):
                old_mask = changed & (base_pred == int(old_class))
                if not old_mask.any():
                    continue
                for new_class in range(self.num_cls):
                    pair_mask = old_mask & (new_pred == int(new_class))
                    pixels = int(pair_mask.sum().item())
                    if pixels <= 0:
                        continue
                    pair_rows.append(dict(
                        from_class_index=int(old_class),
                        from_class_name=self.class_names[old_class],
                        to_class_index=int(new_class),
                        to_class_name=self.class_names[new_class],
                        pixels=pixels,
                        improved_pixels=int(
                            (pair_mask & ~base_correct & new_correct).sum(
                            ).item()),
                        harmed_pixels=int(
                            (pair_mask & base_correct & ~new_correct).sum(
                            ).item()),
                        wrong_to_wrong_pixels=int(
                            (pair_mask & ~base_correct & ~new_correct).sum(
                            ).item()),
                    ))
            pair_rows.sort(key=lambda row: row['pixels'], reverse=True)
            context['changed_pairs'] = pair_rows[:50]
        return new_logits, context

    def _write_cross_image_bank_recomposition_stats(self, record):
        if not getattr(self, 'dump_cross_image_bank_recomposition_stats',
                       False):
            return
        if self._cross_image_bank_recomposition_stats_file is None:
            path = (
                self.cross_image_bank_recomposition_stats_path
                or './work_dirs/evidence_stats/'
                'cross_image_bank_recomposition_stats.jsonl'
            )
            path = _cib_ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._cross_image_bank_recomposition_stats_file = open(
                path, 'a', buffering=1)
        self._cross_image_bank_recomposition_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _build_cross_image_bank_stats(
            self, base_logits, base_pred, data_sample, components, image=None):
        if data_sample is None or not hasattr(data_sample, 'gt_sem_seg'):
            return None
        gt = data_sample.gt_sem_seg.data
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        gt = gt.to(base_logits.device)
        valid = gt != 255
        if int(valid.sum().item()) == 0:
            return None

        seed_context = self._build_seed_rule_context(
            base_logits, base_pred, components)
        seed_mask = self._select_seed_rule(
            seed_context, self.cross_image_bank_seed_rule)
        if seed_mask is None:
            return None
        seed_mask = seed_mask & valid
        seed_class = seed_context['final_top1_idx']
        base_wrong = valid & (base_pred != gt)
        final_pred_score = torch.gather(
            base_logits,
            0,
            base_pred.clamp(min=0, max=self.num_cls - 1).unsqueeze(0),
        ).squeeze(0)
        final_gt_score = torch.gather(
            base_logits,
            0,
            gt.clamp(min=0, max=self.num_cls - 1).unsqueeze(0),
        ).squeeze(0)

        reencoded = None
        spaces = self._get_cross_image_bank_spaces()
        if image is not None and bool(
                self.cross_image_bank_reencode_missing_features):
            missing = [
                space for space in spaces
                if self._get_feature_map_for_similarity_space(
                    components, space) is None
            ]
            if missing:
                reencoded = self._extract_cross_image_bank_features(image)

        space_rows = []
        for space in spaces:
            feature = self._get_cross_image_bank_feature(
                components, space, image=image, reencoded=reencoded)
            if feature is None:
                continue
            channel_count, feat_h, feat_w = feature.shape
            shape = (feat_h, feat_w)
            valid_low = _cib_resize_bool(valid, shape)
            gt_low = _cib_resize_label(gt.clamp(min=0), shape)
            pred_low = _cib_resize_label(
                base_pred.clamp(min=0, max=self.num_cls - 1), shape)
            seed_mask_low = _cib_resize_bool(seed_mask, shape) & valid_low
            seed_class_low = _cib_resize_label(
                seed_class.clamp(min=0, max=self.num_cls - 1), shape)
            pred_score_low = self._interpolate_float32(
                final_pred_score.detach().unsqueeze(0).unsqueeze(0),
                shape,
            ).squeeze()
            gt_score_low = self._interpolate_float32(
                final_gt_score.detach().unsqueeze(0).unsqueeze(0),
                shape,
            ).squeeze()

            valid_features = feature[:, valid_low]
            if valid_features.numel() == 0:
                continue
            image_mean = valid_features.mean(dim=1)

            class_rows = []
            class_proto_vectors = []
            for class_idx in range(self.num_cls):
                class_seed = (
                    seed_mask_low
                    & (seed_class_low == int(class_idx))
                    & valid_low)
                seed_pixels = int(class_seed.sum().item())
                if seed_pixels < int(self.cross_image_bank_min_seed_pixels):
                    continue
                proto = feature[:, class_seed].mean(dim=1)
                class_proto_vectors.append(proto)
                correct = int(((gt_low == class_idx) & class_seed).sum().item())
                class_rows.append(dict(
                    class_index=int(class_idx),
                    class_name=self.class_names[class_idx],
                    seed_pixels=seed_pixels,
                    seed_purity=_cib_safe_div(correct, seed_pixels),
                    prototype=self._cib_vector_to_list(proto),
                ))

            if class_proto_vectors:
                seed_feature_mask = seed_mask_low & valid_low
                seed_mean = feature[:, seed_feature_mask].mean(dim=1)
                class_balanced_mean = torch.stack(
                    class_proto_vectors, dim=0).mean(dim=0)
            else:
                seed_mean = image_mean
                class_balanced_mean = image_mean

            pair_rows = []
            for gt_class in range(self.num_cls):
                gt_wrong = (
                    valid_low
                    & (gt_low == int(gt_class))
                    & (pred_low != int(gt_class)))
                if not gt_wrong.any():
                    continue
                for pred_class in range(self.num_cls):
                    if pred_class == gt_class:
                        continue
                    pair_mask = gt_wrong & (pred_low == int(pred_class))
                    pixels = int(pair_mask.sum().item())
                    if pixels < int(self.cross_image_bank_min_pair_pixels):
                        continue
                    pair_feature = feature[:, pair_mask].mean(dim=1)
                    gap = pred_score_low[pair_mask] - gt_score_low[pair_mask]
                    pair_rows.append(dict(
                        gt_class_index=int(gt_class),
                        gt_class_name=self.class_names[gt_class],
                        pred_class_index=int(pred_class),
                        pred_class_name=self.class_names[pred_class],
                        pixels=pixels,
                        mean_final_pred_gt_margin=float(
                            gap.detach().float().mean().item()),
                        pair_feature=self._cib_vector_to_list(pair_feature),
                    ))
            pair_rows.sort(key=lambda item: item['pixels'], reverse=True)

            space_rows.append(dict(
                space=space,
                feature_shape=[int(channel_count), int(feat_h), int(feat_w)],
                image_mean=self._cib_vector_to_list(image_mean),
                seed_mean=self._cib_vector_to_list(seed_mean),
                class_balanced_seed_mean=(
                    self._cib_vector_to_list(class_balanced_mean)),
                class_prototypes=class_rows,
                pair_features=pair_rows,
            ))

        return dict(
            dataset_name=self.seed_dataset_name,
            class_names=list(self.class_names),
            valid_pixels=int(valid.sum().item()),
            baseline_wrong_pixels=int(base_wrong.sum().item()),
            seed_rule=str(self.cross_image_bank_seed_rule),
            min_seed_pixels=int(self.cross_image_bank_min_seed_pixels),
            min_pair_pixels=int(self.cross_image_bank_min_pair_pixels),
            spaces=space_rows,
        )

    def _write_cross_image_bank_stats(self, record):
        if not self.dump_cross_image_bank_stats:
            return
        if self._cross_image_bank_stats_file is None:
            path = (
                self.cross_image_bank_stats_path
                or './work_dirs/evidence_stats/cross_image_bank_stats.jsonl'
            )
            path = _cib_ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._cross_image_bank_stats_file = open(
                path, 'a', buffering=1)
        self._cross_image_bank_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')
