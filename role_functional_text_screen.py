"""Active frozen SAM3 role-functional text screening experiment."""

import contextlib
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from role_functional_text_definitions import (
    DEFAULT_SETTING,
    PROTOCOL,
    RESIDUAL_SETTINGS,
    ROLE_FIELDS,
    SCHEMA_VERSION,
    VARIANT_NAMES,
    combo_variant_name,
    load_role_functional_text_bank,
    reference_variant,
    sensitivity_variant_name,
)
from sam3.model.data_misc import interpolate as sam3_interpolate


def _safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _ranked_jsonl_path(path):
    stem, suffix = os.path.splitext(path)
    return f'{stem}.rank{int(os.environ.get("RANK", 0))}{suffix}'


def _cuda_memory_snapshot(device):
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        return dict(device=str(device), available=False)
    scale = 1024.0 * 1024.0
    return dict(
        device=str(device),
        available=True,
        allocated_mb=float(torch.cuda.memory_allocated(device) / scale),
        reserved_mb=float(torch.cuda.memory_reserved(device) / scale),
        peak_allocated_mb=float(
            torch.cuda.max_memory_allocated(device) / scale),
        peak_reserved_mb=float(
            torch.cuda.max_memory_reserved(device) / scale),
    )


class RoleFunctionalTextScreenMixin:
    """Keep native SAM3 grounding, but audit different text per output role."""

    def _rpt_initialize(
            self,
            use_role_prompt_tta=False,
            dump_role_prompt_tta_stats=False,
            role_prompt_tta_protocol=PROTOCOL,
            role_prompt_tta_dataset_name=None,
            role_prompt_tta_prompt_bank=None,
            role_prompt_tta_stats_path=None,
            role_prompt_tta_artifact_dir=None,
            role_prompt_tta_primary_variant='baseline',
            role_prompt_tta_strict_integrity=True,
            role_prompt_tta_integrity_tolerance=1e-5,
            role_prompt_tta_save_npz=False,
            role_prompt_tta_artifact_max_side=128,
            role_prompt_tta_max_saved_images=8,
            **kwargs):
        self.use_role_prompt_tta = bool(use_role_prompt_tta)
        self.dump_role_prompt_tta_stats = bool(dump_role_prompt_tta_stats)
        self.role_prompt_tta_protocol = str(role_prompt_tta_protocol)
        self.role_prompt_tta_dataset_name = role_prompt_tta_dataset_name
        self.role_prompt_tta_prompt_bank = role_prompt_tta_prompt_bank
        self.role_prompt_tta_stats_path = role_prompt_tta_stats_path
        self.role_prompt_tta_artifact_dir = role_prompt_tta_artifact_dir
        self.role_prompt_tta_primary_variant = str(
            role_prompt_tta_primary_variant)
        self.role_prompt_tta_strict_integrity = bool(
            role_prompt_tta_strict_integrity)
        self.role_prompt_tta_integrity_tolerance = float(
            role_prompt_tta_integrity_tolerance)
        self.role_prompt_tta_save_npz = bool(role_prompt_tta_save_npz)
        self.role_prompt_tta_artifact_max_side = int(
            role_prompt_tta_artifact_max_side)
        self.role_prompt_tta_max_saved_images = int(
            role_prompt_tta_max_saved_images)
        self._role_prompt_tta_stats_file = None
        self._role_prompt_tta_saved_images = 0
        self._rpt_text_cache = None
        self._rpt_native_parity_checked = False
        self._rpt_native_parity_max_abs = None
        self._rpt_prompt_bank = None

        if not self._uses_role_prompt_tta():
            return
        if self.role_prompt_tta_protocol != PROTOCOL:
            raise ValueError(
                f'Only {PROTOCOL!r} is retained; got '
                f'{self.role_prompt_tta_protocol!r}.')
        if self.role_prompt_tta_primary_variant != 'baseline':
            raise ValueError(
                'The diagnostic must return the protected baseline.')
        if not role_prompt_tta_prompt_bank:
            raise ValueError('role_prompt_tta_prompt_bank is required.')
        official_prompts = [
            [
                self.query_words[query_index]
                for query_index, class_index
                in enumerate(self.query_idx.detach().cpu().tolist())
                if int(class_index) == class_idx
            ]
            for class_idx in range(int(self.num_cls))
        ]
        self._rpt_prompt_bank = load_role_functional_text_bank(
            role_prompt_tta_prompt_bank,
            self.class_names,
            official_prompts,
        )
        for parameter in self.processor.model.parameters():
            parameter.requires_grad_(False)

    def _uses_role_prompt_tta(self):
        return bool(
            getattr(self, 'use_role_prompt_tta', False)
            or getattr(self, 'dump_role_prompt_tta_stats', False))

    @staticmethod
    def _rpt_uses_class_space_variants():
        return True

    @staticmethod
    def _rpt_variant_names():
        return VARIANT_NAMES

    def _rpt_autocast_context(self):
        return (
            torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            if self.device.type == 'cuda'
            else contextlib.nullcontext())

    def _rpt_prepare_text_cache(self):
        if self._rpt_text_cache is not None:
            return self._rpt_text_cache
        bank_prompts = [
            prompt
            for item in self._rpt_prompt_bank['classes']
            for prompt in item['descriptions']
        ]
        prompts = list(dict.fromkeys(list(self.query_words) + bank_prompts))
        index = {prompt: idx for idx, prompt in enumerate(prompts)}
        with torch.no_grad(), self._rpt_autocast_context():
            encoded = self.processor.model.backbone.forward_text(
                prompts, device=self.device)
            language_features = encoded['language_features'].detach().clone()
            language_mask = encoded['language_mask'].detach().clone()
            # Independent encoding preserves native single-concept grounding.
            for prompt in prompts:
                single = self.processor.model.backbone.forward_text(
                    [prompt], device=self.device)
                prompt_index = index[prompt]
                language_features[:, prompt_index:prompt_index + 1] = (
                    single['language_features'])
                language_mask[prompt_index:prompt_index + 1] = (
                    single['language_mask'])
        self._rpt_text_cache = dict(
            prompts=prompts,
            index=index,
            language_features=language_features,
            language_mask=language_mask,
        )
        return self._rpt_text_cache

    def _rpt_set_cached_prompt(self, state, prompt):
        cache = self._rpt_prepare_text_cache()
        prompt_index = int(cache['index'][prompt])
        self.processor.reset_all_prompts(state)
        state['backbone_out']['language_features'] = (
            cache['language_features'][:, prompt_index:prompt_index + 1])
        state['backbone_out']['language_mask'] = (
            cache['language_mask'][prompt_index:prompt_index + 1])
        state['geometric_prompt'] = self.processor.model._get_dummy_prompt()
        return self.processor._forward_grounding(state)

    def _rpt_prompt_components(self, state, output_shape):
        height, width = output_shape
        semantic = state['semantic_mask_logits'].squeeze().float()
        if semantic.shape != output_shape:
            semantic = sam3_interpolate(
                semantic.reshape(1, 1, *semantic.shape),
                size=output_shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze().float()
        instance = torch.zeros(
            (height, width), device=self.device, dtype=torch.float32)
        masks = state.get('masks_logits')
        if self.use_transformer_decoder and isinstance(masks, torch.Tensor):
            for instance_index in range(int(masks.shape[0])):
                mask = masks[instance_index].squeeze()
                if mask.shape != output_shape:
                    mask = sam3_interpolate(
                        mask.reshape(1, 1, *mask.shape),
                        size=output_shape,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze()
                score = self._get_instance_score(state, instance_index)
                instance = torch.maximum(
                    instance, mask.float() * score.float())
        final = torch.maximum(semantic, instance)
        presence = state['presence_score'].detach().float().mean()
        if self.use_presence_score:
            final = final * presence
        raw_keep = state.get('raw_keep_mask', torch.zeros(0))
        return dict(
            final=final,
            semantic=semantic,
            instance=instance,
            presence=presence,
            raw_candidate_count=int(raw_keep.numel()),
            kept_candidate_count=int(raw_keep.bool().sum().item()),
        )

    def _rpt_head_role_raw_package(self, state, output_shape):
        native = self._rpt_prompt_components(state, output_shape)
        raw_masks = state.get('raw_masks_logits_lowres')
        raw_scores = state.get('raw_object_score')
        if not isinstance(raw_masks, torch.Tensor):
            raw_masks = torch.empty((0, 1, 1), dtype=torch.float32)
        if not isinstance(raw_scores, torch.Tensor):
            raw_scores = torch.empty((0,), dtype=torch.float32)
        return dict(
            semantic=native['semantic'].detach().float().cpu(),
            native_instance=native['instance'].detach().float().cpu(),
            native_final=native['final'].detach().float().cpu(),
            presence=native['presence'].detach().cpu(),
            raw_masks=raw_masks.detach().cpu(),
            raw_scores=raw_scores.detach().cpu(),
            native_kept_count=int(native['kept_candidate_count']),
            raw_candidate_count=int(native['raw_candidate_count']),
        )

    def _rpt_head_role_instance_from_raw(
            self, package, presence, output_shape):
        height, width = output_shape
        if not self.use_transformer_decoder:
            return torch.zeros((height, width), dtype=torch.float32), 0
        count = min(
            int(package['raw_masks'].shape[0]),
            int(package['raw_scores'].numel()))
        if count == 0:
            return torch.zeros((height, width), dtype=torch.float32), 0
        raw_scores = package['raw_scores'][:count].to(self.device)
        presence = torch.as_tensor(
            presence, device=self.device, dtype=raw_scores.dtype)
        candidate_scores = raw_scores * presence
        keep = candidate_scores > float(self.processor.confidence_threshold)
        kept_count = int(keep.sum().item())
        if kept_count == 0:
            return torch.zeros((height, width), dtype=torch.float32), 0
        masks = package['raw_masks'][:count].to(self.device)[keep]
        with torch.no_grad(), self._rpt_autocast_context():
            masks = sam3_interpolate(
                masks.unsqueeze(1),
                size=output_shape,
                mode='bilinear',
                align_corners=False,
            ).sigmoid().squeeze(1)
            amplitudes = (
                raw_scores[keep]
                if self.instance_score_type == 'raw'
                else candidate_scores[keep])
            instance = (
                masks.float() * amplitudes.float()[:, None, None]
            ).max(dim=0)[0]
        return instance.detach().float().cpu(), kept_count

    def _rpt_head_role_compose(
            self, semantic_package, instance_package, presence_package,
            output_shape):
        semantic = (
            semantic_package['semantic']
            if self.use_sem_seg
            else torch.zeros(output_shape, dtype=torch.float32))
        presence = torch.as_tensor(
            presence_package['presence']).float().reshape(())
        instance, kept_count = self._rpt_head_role_instance_from_raw(
            instance_package, presence, output_shape)
        final = torch.maximum(semantic.float(), instance.float())
        if self.use_presence_score:
            final = final * float(presence.item())
        return final, instance, kept_count

    @staticmethod
    def _rft_bounded(anchor, candidate, alpha, clip):
        anchor = torch.as_tensor(anchor).float()
        candidate = torch.as_tensor(candidate).float()
        delta = candidate - anchor
        value = (
            anchor + float(alpha) * delta.clamp(-float(clip), float(clip))
        ).clamp(0.0, 1.0)
        return value, delta

    @staticmethod
    def _rft_merge_anchor_packages(packages):
        masks = [item['raw_masks'] for item in packages
                 if item['raw_masks'].numel()]
        scores = [item['raw_scores'] for item in packages
                  if item['raw_scores'].numel()]
        return dict(
            semantic=torch.stack([
                item['semantic'].float() for item in packages
            ]).max(dim=0)[0],
            presence=torch.stack([
                torch.as_tensor(item['presence']).float().reshape(())
                for item in packages
            ]).max(),
            raw_masks=(torch.cat(masks, dim=0) if masks
                       else torch.empty((0, 1, 1), dtype=torch.float32)),
            raw_scores=(torch.cat(scores, dim=0) if scores
                        else torch.empty((0,), dtype=torch.float32)),
            native_instance=torch.stack([
                item['native_instance'].float() for item in packages
            ]).max(dim=0)[0],
            native_final=torch.stack([
                item['native_final'].float() for item in packages
            ]).max(dim=0)[0],
            native_kept_count=sum(
                int(item['native_kept_count']) for item in packages),
            raw_candidate_count=sum(
                int(item['raw_candidate_count']) for item in packages),
        )

    def _rft_role_final(self, semantic, instance, presence, output_shape):
        if not self.use_sem_seg:
            semantic = torch.zeros(output_shape, dtype=torch.float32)
        final = torch.maximum(semantic.float(), instance.float())
        if self.use_presence_score:
            final = final * float(torch.as_tensor(presence).float().item())
        return final

    def _rpt_infer_single_view(
            self, image, return_stats=False, return_components=False,
            view_id=None, crop_box=None):
        return self._rpt_role_functional_text_infer_single_view(
            image,
            return_stats=return_stats,
            return_components=return_components,
            view_id=view_id,
            crop_box=crop_box,
        )

    def _rpt_role_functional_text_infer_single_view(
            self, image, return_stats=False, return_components=False,
            view_id=None, crop_box=None):
        """Ground all frozen texts once and compose role-space variants."""
        width, height = image.size
        output_shape = (height, width)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        cache = self._rpt_prepare_text_cache()
        with torch.no_grad(), self._rpt_autocast_context():
            state = self.processor.set_image(image)

        packages = {}
        parity_errors = []

        def ground(prompt, check_native=False):
            if prompt in packages:
                return packages[prompt]
            native = None
            with torch.no_grad(), self._rpt_autocast_context():
                if check_native and not self._rpt_native_parity_checked:
                    self.processor.reset_all_prompts(state)
                    self.processor.set_text_prompt(prompt, state)
                    native = self._rpt_head_role_raw_package(
                        state, output_shape)
                self._rpt_set_cached_prompt(state, prompt)
                package = self._rpt_head_role_raw_package(
                    state, output_shape)
            if native is not None:
                for key in ('semantic', 'native_instance', 'native_final',
                            'presence'):
                    parity_errors.append(float((
                        torch.as_tensor(package[key]).float()
                        - torch.as_tensor(native[key]).float()
                    ).abs().max().item()))
            packages[prompt] = package
            return package

        baseline_packages = [
            ground(prompt, check_native=True) for prompt in self.query_words]
        if not self._rpt_native_parity_checked:
            self._rpt_native_parity_max_abs = max(parity_errors or [0.0])
            self._rpt_native_parity_checked = True
            if (self.role_prompt_tta_strict_integrity
                    and self._rpt_native_parity_max_abs
                    > self.role_prompt_tta_integrity_tolerance):
                raise RuntimeError(
                    'Cached prompt path differs from native SAM3: '
                    f'max_abs={self._rpt_native_parity_max_abs}.')

        baseline_query = {
            'final': torch.stack([
                item['native_final'] for item in baseline_packages]),
            'semantic': torch.stack([
                item['semantic'] for item in baseline_packages]),
            'instance': torch.stack([
                item['native_instance'] for item in baseline_packages]),
        }
        baseline_class = self._aggregate_query_logits_to_classes(
            baseline_query['final']).detach().float().cpu()
        variants = OrderedDict(
            (name, torch.empty(
                (self.num_cls, height, width), dtype=torch.float32,
                device='cpu'))
            for name in VARIANT_NAMES)
        variants['baseline'].copy_(baseline_class)
        settings = {
            name: (float(alpha), float(clip))
            for name, alpha, clip in RESIDUAL_SETTINGS
        }
        default_alpha, default_clip = settings[DEFAULT_SETTING]
        candidate_rows = []
        class_rows = []
        recomposition_errors = []
        query_indices = self.query_idx.detach().long().cpu()

        for class_index, item in enumerate(self._rpt_prompt_bank['classes']):
            official_indices = torch.nonzero(
                query_indices == class_index, as_tuple=False
            ).flatten().tolist()
            official_packages = [
                baseline_packages[index] for index in official_indices]
            anchor = self._rft_merge_anchor_packages(official_packages)
            anchor_presence = anchor['presence'].float()
            anchor_semantic = anchor['semantic'].float()
            instance_cache = {}

            def instance_for(package, presence):
                key = (id(package), float(torch.as_tensor(
                    presence).float().item()))
                if key not in instance_cache:
                    instance_cache[key] = (
                        self._rpt_head_role_instance_from_raw(
                            package, presence, output_shape))
                return instance_cache[key]

            anchor_instance, anchor_kept = instance_for(
                anchor, anchor_presence)
            role_anchor_final = self._rft_role_final(
                anchor_semantic, anchor_instance, anchor_presence,
                output_shape)
            official_final = baseline_class[class_index]
            variants['semantic_head_anchor'][class_index].copy_(
                anchor_semantic)
            variants['instance_head_anchor'][class_index].copy_(
                anchor_instance)

            role_packages = {}
            for role, field, _ in ROLE_FIELDS:
                role_packages[role] = [None]
                for slot, prompt in enumerate(item[field], 1):
                    package = ground(prompt)
                    role_packages[role].append(package)
                    candidate_instance, candidate_kept = instance_for(
                        package, anchor_presence)
                    rebuilt, _, _ = self._rpt_head_role_compose(
                        package, package, package, output_shape)
                    recomposition_error = float((
                        rebuilt - package['native_final']
                    ).abs().max().item())
                    recomposition_errors.append(recomposition_error)
                    prompt_index = int(cache['index'][prompt])
                    candidate_rows.append(dict(
                        class_index=int(class_index),
                        class_name=item['name'],
                        role=role,
                        slot=int(slot),
                        candidate_prompt=prompt,
                        token_count=int((
                            ~cache['language_mask'][prompt_index].bool()
                        ).sum().item()),
                        anchor_presence=float(anchor_presence.item()),
                        candidate_presence=float(torch.as_tensor(
                            package['presence']).float().item()),
                        presence_delta=float((torch.as_tensor(
                            package['presence']).float()
                            - anchor_presence).item()),
                        anchor_semantic_mean=float(
                            anchor_semantic.mean().item()),
                        candidate_semantic_mean=float(
                            package['semantic'].float().mean().item()),
                        semantic_abs_delta_mean=float((
                            package['semantic'].float() - anchor_semantic
                        ).abs().mean().item()),
                        anchor_instance_mean=float(
                            anchor_instance.mean().item()),
                        candidate_instance_mean=float(
                            candidate_instance.mean().item()),
                        instance_abs_delta_mean=float((
                            candidate_instance - anchor_instance
                        ).abs().mean().item()),
                        anchor_kept_count=int(anchor_kept),
                        candidate_kept_count_at_anchor_presence=int(
                            candidate_kept),
                        raw_candidate_count=int(
                            package['raw_candidate_count']),
                        raw_object_score_mean=(
                            float(package['raw_scores'].float().mean().item())
                            if package['raw_scores'].numel() else 0.0),
                        native_recomposition_max_abs=recomposition_error,
                    ))
                    if role == 'semantic':
                        value, _ = self._rft_bounded(
                            anchor_semantic, package['semantic'],
                            default_alpha, default_clip)
                        variants[f'semantic_head_s{slot}'][
                            class_index].copy_(value)
                    elif role == 'instance':
                        value, _ = self._rft_bounded(
                            anchor_instance, candidate_instance,
                            default_alpha, default_clip)
                        variants[f'instance_head_i{slot}'][
                            class_index].copy_(value)

            def compose(presence_slot=0, semantic_slot=0, instance_slot=0,
                        setting=DEFAULT_SETTING, shared_package=None):
                alpha, clip = settings[setting]
                presence_package = shared_package or (
                    role_packages['presence'][presence_slot]
                    if presence_slot else None)
                semantic_package = shared_package or (
                    role_packages['semantic'][semantic_slot]
                    if semantic_slot else None)
                instance_package = shared_package or (
                    role_packages['instance'][instance_slot]
                    if instance_slot else None)
                presence = anchor_presence
                if presence_package is not None:
                    presence, _ = self._rft_bounded(
                        anchor_presence, presence_package['presence'],
                        alpha, clip)
                semantic = anchor_semantic
                if semantic_package is not None:
                    semantic, _ = self._rft_bounded(
                        anchor_semantic, semantic_package['semantic'],
                        alpha, clip)
                base_instance, _ = instance_for(anchor, presence)
                instance = base_instance
                if instance_package is not None:
                    candidate_instance, _ = instance_for(
                        instance_package, presence)
                    instance, _ = self._rft_bounded(
                        base_instance, candidate_instance, alpha, clip)
                role_final = self._rft_role_final(
                    semantic, instance, presence, output_shape)
                return (
                    official_final.float()
                    + role_final.float() - role_anchor_final.float()
                ).clamp(0.0, 1.0)

            for presence_slot in range(3):
                for semantic_slot in range(4):
                    for instance_slot in range(3):
                        name = combo_variant_name(
                            presence_slot, semantic_slot, instance_slot)
                        variants[name][class_index].copy_(compose(
                            presence_slot, semantic_slot, instance_slot))
            for role, _, count in ROLE_FIELDS:
                for slot in range(1, count + 1):
                    for setting, _, _ in RESIDUAL_SETTINGS:
                        if setting == DEFAULT_SETTING:
                            continue
                        value = compose(
                            **{f'{role}_slot': slot, 'setting': setting})
                        variants[sensitivity_variant_name(
                            role, slot, setting)][class_index].copy_(value)
            for slot in range(1, 4):
                variants[f'shared_semantic_s{slot}'][class_index].copy_(
                    compose(shared_package=role_packages['semantic'][slot]))
            class_rows.append(dict(
                class_index=int(class_index),
                class_name=item['name'],
                official_prompts=list(item['official_prompts']),
                official_prompt_count=len(official_packages),
                anchor_presence=float(anchor_presence.item()),
                anchor_semantic_mean=float(anchor_semantic.mean().item()),
                anchor_instance_mean=float(anchor_instance.mean().item()),
                anchor_kept_count=int(anchor_kept),
                role_anchor_vs_official_abs_mean=float((
                    role_anchor_final - official_final
                ).abs().mean().item()),
                role_anchor_vs_official_max_abs=float((
                    role_anchor_final - official_final
                ).abs().max().item()),
            ))

        identity_error = float((
            variants['combo_p0_s0_i0'] - variants['baseline']
        ).abs().max().item())
        recomposition_error = max(recomposition_errors or [0.0])
        if tuple(variants) != VARIANT_NAMES:
            raise RuntimeError('Role-functional variant order drifted.')
        if (self.role_prompt_tta_strict_integrity
                and max(identity_error, recomposition_error)
                > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                'Role-functional integrity failed: '
                f'identity={identity_error}, '
                f'native_recomposition={recomposition_error}.')

        view_stats = dict(
            schema_version=SCHEMA_VERSION,
            protocol=PROTOCOL,
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            adaptation_unit=('sam3_crop' if crop_box is not None
                             else 'full_image'),
            baseline_reconstruction_max_abs=float(
                self._rpt_native_parity_max_abs or 0.0),
            no_update_identity_max_abs=identity_error,
            native_recomposition_max_abs=recomposition_error,
            official_query_count=int(self.num_queries),
            canonical_class_count=int(self.num_cls),
            diagnostic_variant_count=len(variants),
            diagnostic_cpu_bytes=int(sum(
                value.numel() * value.element_size()
                for value in variants.values())),
            default_residual_setting=DEFAULT_SETTING,
            residual_settings=[
                dict(name=name, alpha=alpha, clip=clip)
                for name, alpha, clip in RESIDUAL_SETTINGS],
            candidate_rows=candidate_rows,
            class_rows=class_rows,
            cuda_memory=dict(main=_cuda_memory_snapshot(self.device)),
        )
        primary = baseline_query['final'].to(self.device)
        components = dict(
            semantic_logits=baseline_query['semantic'].to(self.device),
            instance_logits=baseline_query['instance'].to(self.device),
            role_prompt_variant_class_logits=variants,
            role_prompt_view_stats=[view_stats],
        )
        stats = dict(
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            role_prompt_tta=view_stats,
        )
        if return_stats and return_components:
            return primary, stats, components
        if return_stats:
            return primary, stats
        if return_components:
            return primary, components
        return primary

    def _rpt_threshold(self, logits):
        prediction = logits.argmax(dim=0)
        prediction[logits.max(dim=0)[0] < self.prob_thd] = self.bg_idx
        return prediction

    def _rpt_confusion(self, prediction, gt, valid):
        count = int(self.num_cls)
        encoded = gt[valid].long() * count + prediction[valid].long()
        matrix = torch.bincount(
            encoded, minlength=count * count).view(count, count)
        intersection = matrix.diag().float()
        gt_area = matrix.sum(dim=1).float()
        pred_area = matrix.sum(dim=0).float()
        union = gt_area + pred_area - intersection
        iou = intersection / union.clamp_min(1.0)
        valid_class = union > 0
        miou = (iou[valid_class].mean() if valid_class.any()
                else torch.tensor(0.0))
        return dict(
            matrix=matrix.cpu().tolist(),
            intersection=intersection.cpu().tolist(),
            union=union.cpu().tolist(),
            iou=iou.cpu().tolist(),
            miou=float(miou.item() * 100.0),
            aacc=float(intersection.sum().item()
                       / gt_area.sum().clamp_min(1.0).item() * 100.0),
        )

    def _rpt_enrich_role_functional_stats_with_gt(self, view_stats, gt):
        enriched = []
        for source in view_stats:
            view = dict(source)
            crop_box = view.get('crop_box')
            if crop_box is None:
                crop_gt = gt
            else:
                x1, y1, x2, y2 = [int(value) for value in crop_box]
                crop_gt = gt[y1:y2, x1:x2]
            valid = crop_gt != 255
            gt_presence = [
                bool((valid & (crop_gt == class_index)).any().item())
                for class_index in range(self.num_cls)
            ]
            rows = []
            for source_row in view.get('candidate_rows', []):
                row = dict(source_row)
                row['gt_present'] = gt_presence[int(row['class_index'])]
                rows.append(row)
            view['candidate_rows'] = rows
            view['gt_class_presence'] = gt_presence
            enriched.append(view)
        return enriched

    def _rpt_record_image(
            self, variant_logits, view_stats, data_sample, image_path):
        if not self.dump_role_prompt_tta_stats:
            return
        if data_sample is None or not hasattr(data_sample, 'gt_sem_seg'):
            return
        gt = data_sample.gt_sem_seg.data.squeeze().detach().long().cpu()
        valid = gt != 255
        if not valid.any():
            return
        if tuple(variant_logits) != VARIANT_NAMES:
            raise RuntimeError('Recorded role-functional variants drifted.')
        predictions = {
            name: self._rpt_threshold(
                logits.detach().float().cpu()).to(torch.int16)
            for name, logits in variant_logits.items()
        }
        variant_stats = {}
        for name in VARIANT_NAMES:
            prediction = predictions[name]
            reference_name = reference_variant(name)
            reference = predictions[reference_name]
            changed = valid & (prediction != reference)
            improved = changed & (prediction == gt) & (reference != gt)
            harmed = changed & (prediction != gt) & (reference == gt)
            wrong_to_wrong = changed & (prediction != gt) & (reference != gt)
            variant_stats[name] = dict(
                reference_variant=reference_name,
                confusion=self._rpt_confusion(prediction, gt, valid),
                changed_pixels=int(changed.sum().item()),
                changed_ratio=_safe_div(
                    int(changed.sum().item()), int(valid.sum().item())),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                wrong_to_wrong_pixels=int(wrong_to_wrong.sum().item()),
                help_minus_harm=(int(improved.sum().item())
                                 - int(harmed.sum().item())),
                change_precision=_safe_div(
                    int(improved.sum().item()), int(changed.sum().item())),
            )
        record = dict(
            schema_version=SCHEMA_VERSION,
            rank=int(os.environ.get('RANK', 0)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            dataset_name=self.role_prompt_tta_dataset_name,
            img_path=image_path,
            class_names=list(self.class_names),
            valid_pixels=int(valid.sum().item()),
            prob_thd=float(self.prob_thd),
            confidence_threshold=float(self.confidence_threshold),
            primary_variant=self.role_prompt_tta_primary_variant,
            prompt_bank=os.path.abspath(self.role_prompt_tta_prompt_bank),
            prompt_count=int(self._rpt_prompt_bank['_prompt_count']),
            settings=dict(
                protocol=PROTOCOL,
                default_residual_setting=DEFAULT_SETTING,
                residual_settings=[
                    dict(name=name, alpha=alpha, clip=clip)
                    for name, alpha, clip in RESIDUAL_SETTINGS],
            ),
            variants=variant_stats,
            views=self._rpt_enrich_role_functional_stats_with_gt(
                view_stats, gt),
        )
        self._rpt_write_stats(record)
        self._rpt_save_artifact(image_path, predictions, gt, valid)

    def _rpt_write_stats(self, record):
        if self._role_prompt_tta_stats_file is None:
            path = self.role_prompt_tta_stats_path or (
                './work_dirs/role_functional_text_screen/screen.jsonl')
            path = _ranked_jsonl_path(path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self._role_prompt_tta_stats_file = open(
                path, 'a', buffering=1, encoding='utf-8')
        self._role_prompt_tta_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _rpt_save_artifact(self, image_path, predictions, gt, valid):
        if (not self.role_prompt_tta_save_npz
                or self._role_prompt_tta_saved_images
                >= self.role_prompt_tta_max_saved_images):
            return
        directory = self.role_prompt_tta_artifact_dir or (
            './work_dirs/role_functional_text_screen/artifacts')
        rank_dir = os.path.join(
            directory, f'rank{int(os.environ.get("RANK", 0))}')
        os.makedirs(rank_dir, exist_ok=True)
        height, width = gt.shape[-2:]
        scale = min(
            1.0, self.role_prompt_tta_artifact_max_side / max(height, width))
        target = (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

        def resize_label(value):
            return F.interpolate(
                value.float()[None, None], size=target,
                mode='nearest').squeeze().long().cpu().numpy()

        arrays = {
            'gt': resize_label(gt),
            'valid': resize_label(valid.long()).astype(np.uint8),
        }
        for name, prediction in predictions.items():
            arrays[f'pred_{name}'] = resize_label(prediction)
        stem = os.path.splitext(os.path.basename(image_path or 'image'))[0]
        np.savez_compressed(
            os.path.join(
                rank_dir,
                f'{self._role_prompt_tta_saved_images:04d}_{stem}.npz'),
            **arrays,
        )
        self._role_prompt_tta_saved_images += 1
