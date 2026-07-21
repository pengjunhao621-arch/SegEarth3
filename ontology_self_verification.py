import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F


def _osv_safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _osv_ranked_jsonl_path(path):
    rank = os.environ.get('RANK')
    if rank is None:
        return path
    root, ext = os.path.splitext(path)
    if not ext:
        ext = '.jsonl'
    return f'{root}_rank{rank}{ext}'


def _osv_gather(map_tensor, index_map):
    return torch.gather(
        map_tensor,
        0,
        index_map.clamp(min=0, max=map_tensor.shape[0] - 1).unsqueeze(0),
    ).squeeze(0)


class OntologySelfVerificationMixin:
    """SAM3-native same-image concept self-verification.

    This mixin is intentionally conservative. It does not introduce external
    models or GT-dependent inference rules. It mines high-confidence same-image
    seeds, builds class prototypes in SAM3-visible spaces, and only lets a
    top-k candidate override the current top1 when the prototype evidence
    favors the candidate by a configurable margin.
    """

    def _uses_ontology_self_verification(self):
        return bool(
            getattr(self, 'use_ontology_self_verification', False)
            or getattr(self, 'dump_ontology_self_verification_stats', False))

    def _get_ontology_self_verification_spaces(self):
        spaces = getattr(
            self, 'ontology_self_verification_spaces', 'evidence,vision')
        if spaces is None:
            return ['evidence', 'vision']
        if isinstance(spaces, str):
            raw_items = [item.strip() for item in spaces.split(',')]
        elif isinstance(spaces, (list, tuple)):
            raw_items = [str(item).strip() for item in spaces]
        else:
            raw_items = [str(spaces).strip()]
        normalized = []
        for item in raw_items:
            if not item:
                continue
            item = item.lower()
            if item == 'pe_all':
                expanded = [
                    'pe_layer_0',
                    'pe_layer_1',
                    'pe_layer_2',
                    'pe_layer_3',
                ]
                for space in expanded:
                    if space not in normalized:
                        normalized.append(space)
                continue
            is_pe = (
                item.startswith('pe_layer_')
                and item[len('pe_layer_'):].isdigit())
            if item not in ('evidence', 'vision') and not is_pe:
                raise ValueError(
                    "ontology_self_verification_spaces supports 'evidence', "
                    "'vision', 'pe_all', and 'pe_layer_{idx}', "
                    f'but got {item!r}')
            if item not in normalized:
                normalized.append(item)
        return normalized or ['evidence', 'vision']

    def _ontology_self_verification_wants_pe(self):
        if not self._uses_ontology_self_verification():
            return False
        try:
            spaces = self._get_ontology_self_verification_spaces()
        except ValueError:
            return False
        return any(space.startswith('pe_layer_') for space in spaces)

    def _osv_role_mask(self, roles, include_roles, exclude_roles):
        include = set(self._parse_name_list(include_roles))
        exclude = set(self._parse_name_list(exclude_roles))
        values = []
        for role in roles:
            role = str(role or 'other_landcover').lower()
            allowed = True
            if include:
                allowed = role in include
            if exclude and role in exclude:
                allowed = False
            values.append(allowed)
        return torch.tensor(values, device=self.device, dtype=torch.bool)

    def _osv_standardize_similarity(self, similarity):
        finite = torch.isfinite(similarity)
        clean = torch.nan_to_num(similarity.detach().float(), nan=0.0)
        flat = clean.flatten(1)
        finite_flat = finite.flatten(1)
        count = finite_flat.float().sum(dim=1).clamp_min(1.0)
        mean = (flat * finite_flat.float()).sum(dim=1) / count
        centered = (flat - mean[:, None]) * finite_flat.float()
        var = centered.square().sum(dim=1) / count
        std = var.sqrt().clamp_min(1e-6)
        standardized = (clean - mean[:, None, None]) / std[:, None, None]
        standardized[~finite] = float('nan')
        return standardized

    def _osv_build_similarity(self, base_logits, base_pred, components,
                              seed_mask, seed_class):
        spaces = self._get_ontology_self_verification_spaces()
        sim_sum = torch.zeros_like(base_logits.detach().float())
        sim_count = torch.zeros_like(base_logits.detach().float())
        has_seed = torch.zeros(
            self.num_cls,
            device=base_logits.device,
            dtype=torch.bool,
        )
        used_spaces = []
        space_rows = []

        for space in spaces:
            if space == 'evidence':
                similarity, space_has_seed = self._build_seed_similarity_maps(
                    base_logits,
                    components,
                    seed_mask,
                    seed_class,
                )
            else:
                similarity, space_has_seed = (
                    self._build_feature_seed_similarity_maps(
                        space,
                        base_logits,
                        components,
                        seed_mask,
                        seed_class,
                    ))
            if similarity is None or space_has_seed is None:
                space_rows.append(dict(space=space, used=False))
                continue
            standardized = self._osv_standardize_similarity(similarity)
            finite = torch.isfinite(standardized)
            sim_sum += torch.nan_to_num(standardized, nan=0.0)
            sim_count += finite.float()
            has_seed |= space_has_seed.to(has_seed.device).bool()
            used_spaces.append(space)
            space_rows.append(dict(
                space=space,
                used=True,
                seeded_classes=int(space_has_seed.sum().item()),
            ))

        if not used_spaces:
            return None, has_seed, used_spaces, space_rows
        combined = sim_sum / sim_count.clamp_min(1.0)
        combined[sim_count <= 0] = float('nan')
        return combined, has_seed, used_spaces, space_rows

    def _osv_scope_gate(self, scope, base_pred, top1_idx, candidate_idx):
        scope = str(scope or 'non_bg_to_non_bg').lower()
        bg = int(self.bg_idx)
        if scope in ('non_bg', 'non_bg_to_non_bg', 'foreground'):
            return (
                (base_pred != bg)
                & (top1_idx != bg)
                & (candidate_idx != bg)
            )
        if scope in ('non_bg_candidate', 'all_non_bg_candidate'):
            return candidate_idx != bg
        if scope in ('background_recovery', 'reject_recovery'):
            return (base_pred == bg) & (candidate_idx != bg)
        if scope in ('all', 'any'):
            return torch.ones_like(base_pred, dtype=torch.bool)
        raise ValueError(
            'ontology_self_verification_apply_scope must be one of '
            'non_bg_to_non_bg, all_non_bg_candidate, background_recovery, '
            f'or all, got {scope!r}')

    def _build_ontology_self_verification_logits(
            self, base_logits, base_pred, components, data_sample=None):
        if components is None:
            return base_logits, dict(reason='missing_components')
        if self.num_cls < 2:
            return base_logits, dict(reason='single_class')

        seed_context = self._build_seed_rule_context(
            base_logits, base_pred, components)
        seed_mask = self._select_seed_rule(
            seed_context,
            getattr(
                self,
                'ontology_self_verification_seed_rule',
                'final_score_margin_any_head_agree_local_core',
            ),
        )
        if seed_context is None or seed_mask is None:
            return base_logits, dict(reason='missing_seed_context')

        roles = self._active_concept_roles()
        seed_class = seed_context['final_top1_idx'].detach().long()
        candidate_role_allowed = self._osv_role_mask(
            roles,
            getattr(self, 'ontology_self_verification_include_roles', ''),
            getattr(
                self,
                'ontology_self_verification_exclude_roles',
                'catch_all',
            ),
        )
        seed_role_allowed = self._osv_role_mask(
            roles,
            getattr(self, 'ontology_self_verification_seed_include_roles', ''),
            getattr(
                self,
                'ontology_self_verification_seed_exclude_roles',
                'catch_all',
            ),
        )
        seed_allowed_map = _osv_gather(
            seed_role_allowed.float().view(self.num_cls, 1, 1).expand_as(
                base_logits),
            seed_class,
        ).bool()
        seed_mask = seed_mask.detach().bool() & seed_allowed_map
        if not seed_mask.any():
            return base_logits, dict(reason='no_allowed_seed')

        similarity, has_seed, used_spaces, space_rows = (
            self._osv_build_similarity(
                base_logits,
                base_pred,
                components,
                seed_mask,
                seed_class,
            ))
        if similarity is None:
            return base_logits, dict(
                reason='missing_similarity',
                spaces=space_rows,
            )

        topk = max(2, int(getattr(
            self, 'ontology_self_verification_topk', 3)))
        topk = min(topk, int(self.num_cls))
        top_vals, top_idx = torch.topk(base_logits, k=topk, dim=0)
        top1_idx = top_idx[0]
        top1_score = top_vals[0]
        top1_sim = _osv_gather(similarity, top1_idx)
        top1_has_seed = _osv_gather(
            has_seed.float().view(self.num_cls, 1, 1).expand_as(base_logits),
            top1_idx,
        ).bool()

        min_margin = float(getattr(
            self, 'ontology_self_verification_min_similarity_margin', 0.35))
        max_base_gap = float(getattr(
            self, 'ontology_self_verification_max_base_gap', 0.35))
        min_candidate_score = getattr(
            self, 'ontology_self_verification_min_candidate_score', None)
        if min_candidate_score is None:
            min_candidate_score = -1.0
        min_candidate_score = float(min_candidate_score)
        require_top1_seed = bool(getattr(
            self, 'ontology_self_verification_require_top1_seed', True))
        apply_scope = getattr(
            self,
            'ontology_self_verification_apply_scope',
            'non_bg_to_non_bg',
        )

        best_margin = torch.full_like(top1_score, float('-inf'))
        best_candidate = torch.full_like(top1_idx, -1)
        best_rank = torch.zeros_like(top1_idx)

        candidate_allowed_volume = candidate_role_allowed.float().view(
            self.num_cls, 1, 1).expand_as(base_logits)
        has_seed_volume = has_seed.float().view(
            self.num_cls, 1, 1).expand_as(base_logits)

        for rank in range(1, topk):
            candidate_idx = top_idx[rank]
            candidate_score = top_vals[rank]
            candidate_sim = _osv_gather(similarity, candidate_idx)
            candidate_has_seed = _osv_gather(
                has_seed_volume,
                candidate_idx,
            ).bool()
            candidate_role_ok = _osv_gather(
                candidate_allowed_volume,
                candidate_idx,
            ).bool()
            similarity_margin = candidate_sim - top1_sim
            base_gap = top1_score - candidate_score
            gate = torch.isfinite(similarity_margin)
            gate &= candidate_has_seed
            if require_top1_seed:
                gate &= top1_has_seed
            gate &= candidate_role_ok
            gate &= similarity_margin >= min_margin
            gate &= base_gap <= max_base_gap
            gate &= candidate_score >= min_candidate_score
            gate &= self._osv_scope_gate(
                apply_scope,
                base_pred,
                top1_idx,
                candidate_idx,
            )
            gate &= similarity_margin > best_margin
            best_margin = torch.where(gate, similarity_margin, best_margin)
            best_candidate = torch.where(gate, candidate_idx, best_candidate)
            best_rank = torch.where(
                gate,
                torch.full_like(best_rank, int(rank + 1)),
                best_rank,
            )

        change_mask = best_candidate >= 0
        new_logits = base_logits.clone()
        if change_mask.any():
            flat_logits = new_logits.view(self.num_cls, -1)
            flat_mask = change_mask.view(-1)
            flat_positions = torch.nonzero(
                flat_mask, as_tuple=False).squeeze(1)
            flat_candidates = best_candidate.view(-1)[flat_positions]
            target_values = top1_score.view(-1)[flat_positions] + float(
                getattr(
                    self,
                    'ontology_self_verification_logit_boost',
                    1e-4,
                ))
            current_values = flat_logits[flat_candidates, flat_positions]
            flat_logits[flat_candidates, flat_positions] = torch.maximum(
                current_values,
                target_values,
            )

        context = self._build_ontology_self_verification_context(
            base_logits=base_logits,
            base_pred=base_pred,
            new_logits=new_logits,
            components=components,
            data_sample=data_sample,
            seed_mask=seed_mask,
            seed_class=seed_class,
            similarity=similarity,
            has_seed=has_seed,
            top_idx=top_idx,
            top_vals=top_vals,
            top1_sim=top1_sim,
            best_candidate=best_candidate,
            best_rank=best_rank,
            best_margin=best_margin,
            change_mask=change_mask,
            used_spaces=used_spaces,
            space_rows=space_rows,
        )
        return new_logits, context

    def _build_ontology_self_verification_context(
            self, base_logits, base_pred, new_logits, components, data_sample,
            seed_mask, seed_class, similarity, has_seed, top_idx, top_vals,
            top1_sim, best_candidate, best_rank, best_margin, change_mask,
            used_spaces, space_rows):
        roles = self._active_concept_roles()
        new_pred = self._threshold_with_reject_recovery(new_logits, components)
        valid = torch.ones_like(base_pred, dtype=torch.bool)
        gt = None
        if data_sample is not None and hasattr(data_sample, 'gt_sem_seg'):
            gt = data_sample.gt_sem_seg.data
            if gt.ndim == 3:
                gt = gt.squeeze(0)
            gt = gt.to(base_logits.device).long()
            valid = gt != 255

        base_confusion = None
        new_confusion = None
        class_rows = []
        role_bucket = defaultdict(lambda: defaultdict(int))
        changed_pair_bucket = defaultdict(lambda: defaultdict(int))

        if gt is not None:
            gt_clamped = gt.clamp(min=0, max=self.num_cls - 1)
            base_confusion = self._candidate_residual_miou_confusion(
                gt_clamped, base_pred.clamp(min=0, max=self.num_cls - 1),
                valid)
            new_confusion = self._candidate_residual_miou_confusion(
                gt_clamped, new_pred.clamp(min=0, max=self.num_cls - 1),
                valid)

            base_wrong = valid & (base_pred != gt_clamped)
            changed = valid & (base_pred != new_pred)
            base_correct = base_pred == gt_clamped
            new_correct = new_pred == gt_clamped
            gt_in_topk = (top_idx == gt_clamped.unsqueeze(0)).any(dim=0)
            gt_has_seed = _osv_gather(
                has_seed.float().view(self.num_cls, 1, 1).expand_as(
                    base_logits),
                gt_clamped,
            ).bool()
            gt_similarity = _osv_gather(similarity, gt_clamped)
            gt_score = _osv_gather(base_logits, gt_clamped)
            gt_rank_gate = (
                base_wrong
                & gt_in_topk
                & gt_has_seed
                & torch.isfinite(gt_similarity)
            )
            gt_similarity_favors = (
                gt_rank_gate
                & (
                    gt_similarity - top1_sim
                    >= float(getattr(
                        self,
                        'ontology_self_verification_min_similarity_margin',
                        0.35,
                    ))
                )
                & (
                    top_vals[0] - gt_score
                    <= float(getattr(
                        self,
                        'ontology_self_verification_max_base_gap',
                        0.35,
                    ))
                )
            )

            for class_idx in range(self.num_cls):
                class_valid = valid & (gt_clamped == int(class_idx))
                class_seed = seed_mask & (seed_class == int(class_idx))
                seed_pixels = int(class_seed.sum().item())
                seed_correct = int(
                    (class_seed & valid & (gt_clamped == int(class_idx))).sum(
                    ).item())
                wrong_gt = base_wrong & (gt_clamped == int(class_idx))
                changed_gt = changed & (gt_clamped == int(class_idx))
                improved = changed_gt & ~base_correct & new_correct
                harmed = changed & (base_pred == int(class_idx)) & (
                    base_correct & ~new_correct)
                class_rows.append(dict(
                    class_index=int(class_idx),
                    class_name=self.class_names[class_idx],
                    role=roles[class_idx],
                    valid_pixels=int(class_valid.sum().item()),
                    seed_pixels=seed_pixels,
                    seed_correct_pixels=seed_correct,
                    seed_purity=_osv_safe_div(seed_correct, seed_pixels),
                    wrong_gt_pixels=int(wrong_gt.sum().item()),
                    wrong_gt_topk_contains_pixels=int(
                        (wrong_gt & gt_in_topk).sum().item()),
                    wrong_gt_seed_available_pixels=int(
                        (wrong_gt & gt_has_seed).sum().item()),
                    wrong_gt_similarity_favors_pixels=int(
                        (wrong_gt & gt_similarity_favors).sum().item()),
                    selected_gt_pixels=int(changed_gt.sum().item()),
                    improved_pixels=int(improved.sum().item()),
                    harmed_from_class_pixels=int(harmed.sum().item()),
                ))

            for role in sorted(set(roles)):
                role_mask = torch.zeros_like(valid)
                for idx, item in enumerate(roles):
                    if item == role:
                        role_mask |= gt_clamped == int(idx)
                mask = valid & role_mask
                if not mask.any():
                    continue
                role_bucket[role]['valid_pixels'] += int(mask.sum().item())
                role_bucket[role]['base_wrong_pixels'] += int(
                    (mask & base_wrong).sum().item())
                role_bucket[role]['gt_topk_pixels'] += int(
                    (mask & base_wrong & gt_in_topk).sum().item())
                role_bucket[role]['gt_seed_available_pixels'] += int(
                    (mask & base_wrong & gt_has_seed).sum().item())
                role_bucket[role]['gt_similarity_favors_pixels'] += int(
                    (mask & gt_similarity_favors).sum().item())
                role_bucket[role]['changed_pixels'] += int(
                    (mask & changed).sum().item())
                role_bucket[role]['improved_pixels'] += int(
                    (mask & changed & ~base_correct & new_correct).sum(
                    ).item())
                role_bucket[role]['harmed_pixels'] += int(
                    (mask & changed & base_correct & ~new_correct).sum(
                    ).item())

            for old_class in range(self.num_cls):
                old_mask = changed & (base_pred == int(old_class))
                if not old_mask.any():
                    continue
                for new_class in range(self.num_cls):
                    pair_mask = old_mask & (new_pred == int(new_class))
                    pixels = int(pair_mask.sum().item())
                    if pixels <= 0:
                        continue
                    key = (old_class, new_class)
                    changed_pair_bucket[key]['pixels'] += pixels
                    changed_pair_bucket[key]['improved_pixels'] += int(
                        (pair_mask & ~base_correct & new_correct).sum().item())
                    changed_pair_bucket[key]['harmed_pixels'] += int(
                        (pair_mask & base_correct & ~new_correct).sum().item())
                    changed_pair_bucket[key]['wrong_to_wrong_pixels'] += int(
                        (pair_mask & ~base_correct & ~new_correct).sum().item())

        seed_pixels_total = int(seed_mask.sum().item())
        changed_pixels = int(change_mask.sum().item())
        role_rows = [
            dict(role=role, **dict(values))
            for role, values in sorted(role_bucket.items())
        ]
        pair_rows = []
        for (old_class, new_class), values in changed_pair_bucket.items():
            row = dict(values)
            row.update(dict(
                from_class_index=int(old_class),
                from_class_name=self.class_names[old_class],
                from_role=roles[old_class],
                to_class_index=int(new_class),
                to_class_name=self.class_names[new_class],
                to_role=roles[new_class],
            ))
            pair_rows.append(row)
        pair_rows.sort(key=lambda row: row.get('pixels', 0), reverse=True)

        return dict(
            reason='ok',
            class_names=list(self.class_names),
            roles=list(roles),
            prob_thd=float(self.prob_thd),
            bg_idx=int(self.bg_idx),
            spaces=space_rows,
            used_spaces=list(used_spaces),
            seed_rule=str(getattr(
                self,
                'ontology_self_verification_seed_rule',
                'final_score_margin_any_head_agree_local_core',
            )),
            apply_scope=str(getattr(
                self,
                'ontology_self_verification_apply_scope',
                'non_bg_to_non_bg',
            )),
            topk=int(getattr(
                self, 'ontology_self_verification_topk', 3)),
            min_similarity_margin=float(getattr(
                self,
                'ontology_self_verification_min_similarity_margin',
                0.35,
            )),
            max_base_gap=float(getattr(
                self, 'ontology_self_verification_max_base_gap', 0.35)),
            require_top1_seed=bool(getattr(
                self, 'ontology_self_verification_require_top1_seed', True)),
            seed_pixels=seed_pixels_total,
            seeded_classes=int(has_seed.sum().item()),
            selected_pixels=changed_pixels,
            selected_ratio=_osv_safe_div(changed_pixels, int(base_pred.numel())),
            base_confusion=base_confusion,
            new_confusion=new_confusion,
            class_stats=class_rows,
            role_stats=role_rows,
            changed_pairs=pair_rows[:80],
        )

    def _write_ontology_self_verification_stats(self, record):
        if not getattr(self, 'dump_ontology_self_verification_stats', False):
            return
        if self._ontology_self_verification_stats_file is None:
            path = (
                self.ontology_self_verification_stats_path
                or './work_dirs/evidence_stats/'
                   'ontology_self_verification.jsonl'
            )
            path = _osv_ranked_jsonl_path(path)
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._ontology_self_verification_stats_file = open(
                path, 'a', buffering=1)
        self._ontology_self_verification_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')
