"""Class-role alignment diagnostics for frozen SAM3 OVSS inference."""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from role_functional_text_definitions import (
    CLASS_ROLE_ALIGNMENT_PROTOCOL,
    CLASS_ROLE_ALIGNMENT_PE_LAYER,
    CLASS_ROLE_ALIGNMENT_SCHEMA_VERSION,
    CRA_COCO_WEIGHTS,
    CRA_FCRA_TEMPERATURES,
    CRA_VARIANT_NAMES,
)


ROLE_NAMES = ('presence', 'semantic', 'instance')


def _safe_mean(value, mask):
    return (float(value[mask].mean().item()) if mask.any() else None)


class ClassRoleAlignmentMixin:
    """CoCo-style class calibration plus factorized class-role evidence."""

    @staticmethod
    def _cra_variant_names():
        return CRA_VARIANT_NAMES

    @staticmethod
    def _cra_sign_gate(delta, positive_gate, negative_gate):
        return delta.clamp_min(0.0) * positive_gate + (
            delta.clamp_max(0.0) * negative_gate)

    @staticmethod
    def _cra_pool_embedding(embeddings):
        value = torch.stack([item.float() for item in embeddings]).mean(dim=0)
        return F.normalize(value, dim=0, eps=1e-6)

    @staticmethod
    def _cra_topk_pool(value, fraction=0.10):
        flat = value.flatten()
        count = max(1, int(round(float(flat.numel()) * float(fraction))))
        return flat.topk(min(count, int(flat.numel()))).values.mean()

    def _cra_instance_from_raw(
            self, package, presence, admission_presence, output_shape):
        return self._rpt_head_role_instance_from_raw(
            package, presence, output_shape,
            admission_presence=admission_presence,
            amplitude_presence=presence)[0]

    def _cra_query_region_gates(
            self, evidence, packages, admission_presence, temperature):
        """Read class-role evidence inside admitted query regions vs rings."""
        height, width = evidence.shape[-2:]
        masks = []
        for package in packages:
            if package is None:
                continue
            count = min(
                int(package['raw_masks'].shape[0]),
                int(package['raw_scores'].numel()))
            if count == 0:
                continue
            keep = (
                package['raw_scores'][:count].float()
                * float(torch.as_tensor(admission_presence).float().item())
                > float(self.processor.confidence_threshold))
            if keep.any():
                raw = package['raw_masks'][:count][keep].to(
                    self.device).float()
                masks.append(F.interpolate(
                    raw.unsqueeze(1), size=(height, width),
                    mode='bilinear', align_corners=False
                ).squeeze(1).sigmoid())
        fallback_positive = torch.sigmoid(evidence / float(temperature))
        fallback_negative = torch.sigmoid(-evidence / float(temperature))
        if not masks:
            return fallback_positive, fallback_negative, dict(
                admitted_queries=0, region_score_mean=None,
                region_score_std=None)

        probabilities = torch.cat(masks, dim=0)
        supports = probabilities >= 0.5
        expanded = F.max_pool2d(
            supports.float().unsqueeze(1), 3, stride=1,
            padding=1).squeeze(1).bool()
        rings = expanded & ~supports
        scores = []
        for support, ring in zip(supports, rings):
            inside = evidence[support].mean() if support.any() else evidence.mean()
            outside = evidence[ring].mean() if ring.any() else evidence.mean()
            scores.append(inside - outside)
        scores = torch.stack(scores)
        positive_weights = torch.sigmoid(scores / float(temperature))
        negative_weights = torch.sigmoid(-scores / float(temperature))
        denominator = probabilities.sum(dim=0)
        positive = (
            probabilities * positive_weights[:, None, None]).sum(dim=0)
        negative = (
            probabilities * negative_weights[:, None, None]).sum(dim=0)
        covered = denominator > 1e-4
        positive = torch.where(
            covered, positive / denominator.clamp_min(1e-4),
            fallback_positive)
        negative = torch.where(
            covered, negative / denominator.clamp_min(1e-4),
            fallback_negative)
        return positive, negative, dict(
            admitted_queries=int(probabilities.shape[0]),
            region_score_mean=float(scores.mean().item()),
            region_score_std=float(scores.float().std(
                unbiased=False).item()),
        )

    def _cra_build_variants(
            self, class_inputs, pe_feature, output_shape,
            default_alpha, default_clip):
        if len(class_inputs) != int(self.num_cls):
            raise RuntimeError('Class-role inputs do not cover all classes.')
        feature = F.normalize(
            pe_feature.squeeze(0).float(), dim=0, eps=1e-6)
        feature_size = tuple(feature.shape[-2:])
        anchor_embeddings = torch.stack([
            item['anchor_embedding'].to(self.device).float()
            for item in class_inputs])
        role_embeddings = torch.stack([
            torch.stack([
                item[f'{role}_embedding'].to(self.device).float()
                for role in ROLE_NAMES])
            for item in class_inputs])
        anchor_embeddings = F.normalize(anchor_embeddings, dim=-1, eps=1e-6)
        role_embeddings = F.normalize(role_embeddings, dim=-1, eps=1e-6)
        anchor_similarity = torch.einsum(
            'cd,dhw->chw', anchor_embeddings, feature)
        role_similarity = torch.einsum(
            'crd,dhw->crhw', role_embeddings, feature)
        evidence_d = role_similarity - anchor_similarity[:, None]
        evidence_gamma = evidence_d - evidence_d.mean(dim=0, keepdim=True)
        log_prior = F.log_softmax(anchor_similarity, dim=0)
        prior = log_prior.exp()
        prior_entropy = -(prior * log_prior).sum(dim=0)
        prior_top2 = anchor_similarity.topk(
            min(2, int(anchor_similarity.shape[0])), dim=0).values
        prior_margin = (
            prior_top2[0] - prior_top2[1]
            if int(prior_top2.shape[0]) > 1
            else prior_top2[0])
        log_prior_full = F.interpolate(
            log_prior.unsqueeze(0), size=output_shape,
            mode='bilinear', align_corners=False).squeeze(0).cpu()

        official = torch.stack([
            item['official_final'].float() for item in class_inputs])
        best_maps = []
        full_states = []
        for item in class_inputs:
            anchor_presence = torch.as_tensor(
                item['anchor_presence']).float().reshape(())
            presence = anchor_presence
            if item['presence_package'] is not None:
                presence, _ = self._rft_bounded(
                    anchor_presence,
                    item['presence_package']['presence'],
                    default_alpha, default_clip)
            semantic = item['anchor_semantic'].float()
            if item['semantic_package'] is not None:
                semantic, _ = self._rft_bounded(
                    semantic, item['semantic_package']['semantic'],
                    default_alpha, default_clip)
            admission = (
                anchor_presence
                if item['selected_admission'] == 'anchor_admission'
                else presence)
            base_instance = self._cra_instance_from_raw(
                item['anchor_package'], presence, admission, output_shape)
            instance = base_instance
            if item['instance_package'] is not None:
                candidate_instance = self._cra_instance_from_raw(
                    item['instance_package'], presence, admission,
                    output_shape)
                instance, _ = self._rft_bounded(
                    base_instance, candidate_instance,
                    default_alpha, default_clip)
            role_final = self._rft_role_final(
                semantic, instance, presence, output_shape)
            final = (
                item['official_final'].float() + role_final.float()
                - item['role_anchor_final'].float()).clamp(0.0, 1.0)
            best_maps.append(final)
            full_states.append(dict(
                presence=presence, semantic=semantic,
                admission=admission, instance=instance,
                base_instance=base_instance))
        best_full = torch.stack(best_maps)

        variants = OrderedDict(
            cra_official=official, cra_best_full=best_full)
        eps = 1e-4
        for base_name, base in (
                ('official', official), ('best', best_full)):
            for weight_name, weight in CRA_COCO_WEIGHTS:
                calibrated = torch.sigmoid(
                    torch.logit(base.clamp(eps, 1.0 - eps))
                    + float(weight) * log_prior_full)
                variants[f'cra_{base_name}_coco_{weight_name}'] = calibrated

        class_rows = []
        action_deltas = []
        gate_rows = []
        for evidence_name, evidence_source in (
                ('d', evidence_d), ('gamma', evidence_gamma)):
            for temperature_name, temperature in CRA_FCRA_TEMPERATURES:
                maps = []
                for class_index, (item, full) in enumerate(zip(
                        class_inputs, full_states)):
                    p_evidence = evidence_source[class_index, 0]
                    p_support = self._cra_topk_pool(p_evidence)
                    p_positive = torch.sigmoid(
                        p_support / float(temperature)).cpu()
                    p_negative = torch.sigmoid(
                        -p_support / float(temperature)).cpu()
                    anchor_presence = torch.as_tensor(
                        item['anchor_presence']).float().reshape(())
                    presence_delta = full['presence'] - anchor_presence
                    presence = (
                        anchor_presence + self._cra_sign_gate(
                            presence_delta, p_positive, p_negative)
                    ).clamp(0.0, 1.0)

                    s_evidence = F.interpolate(
                        evidence_source[class_index, 1][None, None],
                        size=output_shape, mode='bilinear',
                        align_corners=False).squeeze().cpu()
                    s_positive_gate = torch.sigmoid(
                        s_evidence / float(temperature))
                    s_negative_gate = torch.sigmoid(
                        -s_evidence / float(temperature))
                    semantic_delta = (
                        full['semantic'] - item['anchor_semantic'].float())
                    semantic = (
                        item['anchor_semantic'].float()
                        + self._cra_sign_gate(
                            semantic_delta,
                            s_positive_gate, s_negative_gate)
                    ).clamp(0.0, 1.0)

                    base_instance = self._cra_instance_from_raw(
                        item['anchor_package'], presence,
                        full['admission'], output_shape)
                    instance_delta = torch.zeros_like(base_instance)
                    region_stats = dict(
                        admitted_queries=0, region_score_mean=None,
                        region_score_std=None)
                    if item['instance_package'] is not None:
                        candidate_instance = self._cra_instance_from_raw(
                            item['instance_package'], presence,
                            full['admission'], output_shape)
                        full_instance, _ = self._rft_bounded(
                            base_instance, candidate_instance,
                            default_alpha, default_clip)
                        instance_delta = full_instance - base_instance
                        positive_gate, negative_gate, region_stats = (
                            self._cra_query_region_gates(
                                evidence_source[class_index, 2],
                                (item['anchor_package'],
                                 item['instance_package']),
                                full['admission'], temperature))
                        positive_gate = F.interpolate(
                            positive_gate[None, None], size=output_shape,
                            mode='bilinear', align_corners=False).squeeze().cpu()
                        negative_gate = F.interpolate(
                            negative_gate[None, None], size=output_shape,
                            mode='bilinear', align_corners=False).squeeze().cpu()
                        instance_delta = self._cra_sign_gate(
                            instance_delta, positive_gate, negative_gate)
                    instance = (base_instance + instance_delta).clamp(0.0, 1.0)
                    role_final = self._rft_role_final(
                        semantic, instance, presence, output_shape)
                    maps.append((
                        item['official_final'].float() + role_final.float()
                        - item['role_anchor_final'].float()
                    ).clamp(0.0, 1.0))
                    gate_rows.append(dict(
                        class_index=int(class_index),
                        class_name=item['class_name'],
                        evidence=evidence_name,
                        temperature=float(temperature),
                        presence_support=float(p_support.item()),
                        presence_positive_gate=float(p_positive.item()),
                        presence_negative_gate=float(p_negative.item()),
                        semantic_evidence_mean=float(s_evidence.mean().item()),
                        semantic_evidence_std=float(s_evidence.std(
                            unbiased=False).item()),
                        semantic_positive_gate_mean=float(
                            s_positive_gate.mean().item()),
                        semantic_negative_gate_mean=float(
                            s_negative_gate.mean().item()),
                        **region_stats))
                variants[
                    f'cra_best_fcra_{evidence_name}_{temperature_name}'
                ] = torch.stack(maps)

        for item, full in zip(class_inputs, full_states):
            anchor_presence = torch.as_tensor(
                item['anchor_presence']).float().reshape(())
            presence_delta = (full['presence'] - anchor_presence).expand(
                *feature_size)
            semantic_delta = F.interpolate(
                (full['semantic'] - item['anchor_semantic'].float())[
                    None, None],
                size=feature_size, mode='bilinear',
                align_corners=False).squeeze()
            instance_delta = F.interpolate(
                (full['instance'] - full['base_instance'])[
                    None, None],
                size=feature_size, mode='bilinear',
                align_corners=False).squeeze()
            action_deltas.append(torch.stack([
                presence_delta, semantic_delta, instance_delta]))
        action_deltas = torch.stack(action_deltas).float().cpu()

        for class_index, item in enumerate(class_inputs):
            class_rows.append(dict(
                class_index=int(class_index),
                class_name=item['class_name'],
                selected_slots=list(item['selected_slots']),
                official_prompts=list(item['official_prompts']),
                selected_prompts=dict(item['selected_prompts']),
                selected_admission=item['selected_admission'],
                anchor_similarity_mean=float(
                    anchor_similarity[class_index].mean().item()),
                d_presence_mean=float(
                    evidence_d[class_index, 0].mean().item()),
                d_semantic_mean=float(
                    evidence_d[class_index, 1].mean().item()),
                d_instance_mean=float(
                    evidence_d[class_index, 2].mean().item()),
                gamma_presence_mean=float(
                    evidence_gamma[class_index, 0].mean().item()),
                gamma_semantic_mean=float(
                    evidence_gamma[class_index, 1].mean().item()),
                gamma_instance_mean=float(
                    evidence_gamma[class_index, 2].mean().item()),
            ))

        if tuple(variants) != CRA_VARIANT_NAMES:
            raise RuntimeError('Class-role alignment variant order drifted.')
        diagnosis = dict(
            pe_feature_size=list(feature_size),
            prior_entropy_mean=float(prior_entropy.mean().item()),
            prior_margin_mean=float(prior_margin.mean().item()),
            class_rows=class_rows,
            gate_rows=gate_rows,
            _evidence_d=evidence_d.detach().float().cpu(),
            _evidence_gamma=evidence_gamma.detach().float().cpu(),
            _action_deltas=action_deltas,
        )
        return variants, diagnosis

    def _cra_enrich_actions(self, diagnosis, crop_gt):
        diagnosis = dict(diagnosis)
        evidence_d = diagnosis.pop('_evidence_d')
        evidence_gamma = diagnosis.pop('_evidence_gamma')
        deltas = diagnosis.pop('_action_deltas')
        target_size = tuple(deltas.shape[-2:])
        gt = F.interpolate(
            crop_gt.float()[None, None], size=target_size,
            mode='nearest').squeeze().long()
        valid = F.interpolate(
            (crop_gt != 255).float()[None, None], size=target_size,
            mode='nearest').squeeze().bool()
        rows = []
        for class_index, class_name in enumerate(self.class_names):
            target = valid & (gt == class_index)
            gt_present = bool(target.any().item())
            for role_index, role in enumerate(ROLE_NAMES):
                delta = deltas[class_index, role_index]
                d_map = evidence_d[class_index, role_index]
                gamma_map = evidence_gamma[class_index, role_index]
                if role == 'presence':
                    value = float(delta.flatten()[0].item())
                    active = abs(value) > 1e-7
                    helpful = active and (
                        (value > 0 and gt_present)
                        or (value < 0 and not gt_present))
                    rows.append(dict(
                        class_index=int(class_index), class_name=class_name,
                        role=role, gt_present=gt_present,
                        action_unit='image_class',
                        active_pixels=int(active),
                        helpful_pixels=int(helpful),
                        harmful_pixels=int(active and not helpful),
                        d_helpful_mean=(
                            float(self._cra_topk_pool(d_map).item())
                            if helpful else None),
                        d_harmful_mean=(
                            float(self._cra_topk_pool(d_map).item())
                            if active and not helpful else None),
                        gamma_helpful_mean=(
                            float(self._cra_topk_pool(gamma_map).item())
                            if helpful else None),
                        gamma_harmful_mean=(
                            float(self._cra_topk_pool(gamma_map).item())
                            if active and not helpful else None),
                        residual_abs_mean=abs(value),
                    ))
                    continue
                active = valid & (delta.abs() > 1e-7)
                helpful = active & (((delta > 0) & target)
                                    | ((delta < 0) & ~target))
                harmful = active & (((delta > 0) & ~target)
                                    | ((delta < 0) & target))
                rows.append(dict(
                    class_index=int(class_index), class_name=class_name,
                    role=role, gt_present=gt_present,
                    action_unit='pixel',
                    active_pixels=int(active.sum().item()),
                    helpful_pixels=int(helpful.sum().item()),
                    harmful_pixels=int(harmful.sum().item()),
                    d_helpful_mean=_safe_mean(d_map, helpful),
                    d_harmful_mean=_safe_mean(d_map, harmful),
                    gamma_helpful_mean=_safe_mean(gamma_map, helpful),
                    gamma_harmful_mean=_safe_mean(gamma_map, harmful),
                    residual_abs_mean=(
                        float(delta[active].abs().mean().item())
                        if active.any() else 0.0),
                ))
        diagnosis['action_rows'] = rows
        return diagnosis

    def _cra_record(self, variant_logits, gt, valid):
        if variant_logits is None:
            return None
        if tuple(variant_logits) != CRA_VARIANT_NAMES:
            raise RuntimeError('Recorded class-role variants drifted.')
        logits = {
            name: value.detach().float().cpu()
            for name, value in variant_logits.items()}
        predictions = {
            name: self._rpt_threshold(value).to(torch.int16)
            for name, value in logits.items()}
        rows = OrderedDict()
        for name in CRA_VARIANT_NAMES:
            reference_name = (
                'cra_official' if name.startswith('cra_official_')
                else 'cra_best_full')
            if name in ('cra_official', 'cra_best_full'):
                reference_name = name
            prediction = predictions[name]
            reference = predictions[reference_name]
            changed = valid & (prediction != reference)
            improved = changed & (prediction == gt) & (reference != gt)
            harmed = changed & (prediction != gt) & (reference == gt)
            top2 = logits[name].topk(
                min(2, int(logits[name].shape[0])), dim=0).values
            margin = (top2[0] - top2[1]
                      if int(top2.shape[0]) > 1 else top2[0])
            rows[name] = dict(
                reference_variant=reference_name,
                confusion=self._rpt_confusion(prediction, gt, valid),
                changed_pixels=int(changed.sum().item()),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                wrong_to_wrong_pixels=int((
                    changed & (prediction != gt) & (reference != gt)
                ).sum().item()),
                help_minus_harm=(int(improved.sum().item())
                                 - int(harmed.sum().item())),
                mean_top1_margin=float(margin[valid].mean().item()),
            )
        return dict(
            protocol=CLASS_ROLE_ALIGNMENT_PROTOCOL,
            schema_version=CLASS_ROLE_ALIGNMENT_SCHEMA_VERSION,
            selected_combination=list(
                self._rpt_role_selection['_best_overall_slots']),
            selected_admission=self._rpt_role_selection[
                'best_overall']['admission'],
            pe_layer=CLASS_ROLE_ALIGNMENT_PE_LAYER,
            text_embedding_source=(
                'valid_token_mean_post_transformer_pre_resizer'),
            variants=rows)
