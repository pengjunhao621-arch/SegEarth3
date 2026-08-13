"""Active frozen SAM3 role-functional text screening experiment."""

import contextlib
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from role_functional_text_definitions import (
    COMPLETION_PROTOCOL,
    COMPLETION_SCHEMA_VERSION,
    DEFAULT_SETTING,
    PI_MECHANISM_MAP_NAMES,
    PI_PROTOCOL,
    PI_SCHEMA_VERSION,
    PI_VARIANT_NAMES,
    PE_FEATURE_NAMES,
    PE_LAYER_IDS,
    PE_PROTOCOL,
    PE_SCHEMA_VERSION,
    PE_VARIANT_NAMES,
    PROTOCOL,
    RESIDUAL_SETTINGS,
    ROLE_FIELDS,
    SCHEMA_VERSION,
    VARIANT_NAMES,
    combo_variant_name,
    completion_variant_name,
    load_role_functional_text_bank,
    load_role_text_selection_registry,
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
            role_prompt_tta_pi_diagnosis=False,
            role_prompt_tta_pi_presence_slot=0,
            role_prompt_tta_pi_instance_slot=0,
            role_prompt_tta_completion_diagnosis=False,
            role_prompt_tta_pe_diagnosis=False,
            role_prompt_tta_selection_registry=None,
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
        self.role_prompt_tta_pi_diagnosis = bool(
            role_prompt_tta_pi_diagnosis)
        self.role_prompt_tta_pi_presence_slot = int(
            role_prompt_tta_pi_presence_slot)
        self.role_prompt_tta_pi_instance_slot = int(
            role_prompt_tta_pi_instance_slot)
        self.role_prompt_tta_completion_diagnosis = bool(
            role_prompt_tta_completion_diagnosis)
        self.role_prompt_tta_pe_diagnosis = bool(
            role_prompt_tta_pe_diagnosis)
        self.role_prompt_tta_selection_registry = (
            role_prompt_tta_selection_registry)
        self._role_prompt_tta_stats_file = None
        self._role_prompt_tta_saved_images = 0
        self._rpt_text_cache = None
        self._rpt_native_parity_checked = False
        self._rpt_native_parity_max_abs = None
        self._rpt_prompt_bank = None
        self._rpt_role_selection = None

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
        if self.role_prompt_tta_pi_diagnosis:
            if self.role_prompt_tta_pi_presence_slot not in (1, 2):
                raise ValueError(
                    'PI diagnosis presence slot must be 1 or 2.')
            if self.role_prompt_tta_pi_instance_slot not in (1, 2):
                raise ValueError(
                    'PI diagnosis instance slot must be 1 or 2.')
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
        if (self.role_prompt_tta_completion_diagnosis
                and '_completion_combinations' not in self._rpt_prompt_bank):
            raise ValueError(
                'Completion diagnosis requires prompt-bank completion metadata.')
        if self.role_prompt_tta_pe_diagnosis:
            if not role_prompt_tta_selection_registry:
                raise ValueError(
                    'PE diagnosis requires role_prompt_tta_selection_registry.')
            self._rpt_role_selection = load_role_text_selection_registry(
                role_prompt_tta_selection_registry,
                self.role_prompt_tta_dataset_name)
            if os.path.normpath(
                    self._rpt_role_selection['prompt_bank']) != os.path.normpath(
                        role_prompt_tta_prompt_bank):
                raise ValueError(
                    'PE selection registry and configured prompt bank differ.')
        for parameter in self.processor.model.parameters():
            parameter.requires_grad_(False)

    def _uses_role_prompt_tta(self):
        return bool(
            getattr(self, 'use_role_prompt_tta', False)
            or getattr(self, 'dump_role_prompt_tta_stats', False)
            or getattr(self, 'role_prompt_tta_pi_diagnosis', False)
            or getattr(self, 'role_prompt_tta_completion_diagnosis', False)
            or getattr(self, 'role_prompt_tta_pe_diagnosis', False))

    @staticmethod
    def _rpt_uses_class_space_variants():
        return True

    def _rpt_uses_pi_diagnosis(self):
        return bool(getattr(self, 'role_prompt_tta_pi_diagnosis', False))

    def _rpt_uses_completion_diagnosis(self):
        return bool(getattr(
            self, 'role_prompt_tta_completion_diagnosis', False))

    def _rpt_uses_pe_diagnosis(self):
        return bool(getattr(self, 'role_prompt_tta_pe_diagnosis', False))

    @staticmethod
    def _rpt_pe_variant_names():
        return PE_VARIANT_NAMES

    def _rpt_completion_variant_names(self):
        if not self._rpt_uses_completion_diagnosis():
            return ()
        return tuple(
            completion_variant_name(*slots)
            for slots in self._rpt_prompt_bank['_completion_combinations'])

    @staticmethod
    def _rpt_variant_names():
        return VARIANT_NAMES

    @staticmethod
    def _rpt_pi_variant_names():
        return PI_VARIANT_NAMES

    @staticmethod
    def _rpt_pi_mechanism_map_names():
        return PI_MECHANISM_MAP_NAMES

    def _rpt_autocast_context(self):
        return (
            torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            if self.device.type == 'cuda'
            else contextlib.nullcontext())

    def _rpt_set_image_with_pe_features(self, image):
        """Encode one image and capture four true PE/ViT block outputs."""
        if not self._rpt_uses_pe_diagnosis():
            with torch.no_grad(), self._rpt_autocast_context():
                return self.processor.set_image(image), None
        trunk = self.processor.model.backbone.vision_backbone.trunk
        captured = {}
        handles = []
        for layer_id in PE_LAYER_IDS:
            def capture(_module, _inputs, output, layer_id=layer_id):
                captured[layer_id] = output.detach()
            handles.append(trunk.blocks[layer_id].register_forward_hook(capture))
        try:
            with torch.no_grad(), self._rpt_autocast_context():
                state = self.processor.set_image(image)
        finally:
            for handle in handles:
                handle.remove()
        if tuple(sorted(captured)) != PE_LAYER_IDS:
            raise RuntimeError(
                f'PE hooks captured {tuple(sorted(captured))}, '
                f'expected {PE_LAYER_IDS}.')

        features = OrderedDict()
        for layer_id in PE_LAYER_IDS:
            value = captured[layer_id]
            if value.ndim == 4 and value.shape[-1] > value.shape[1]:
                value = value.permute(0, 3, 1, 2)
            if value.ndim != 4:
                raise RuntimeError(
                    f'PE block {layer_id} returned shape {tuple(value.shape)}.')
            features[f'block{layer_id:02d}'] = F.normalize(
                value.float(), dim=1, eps=1e-6)
        final = state['backbone_out']['vision_features']
        features['fpn_final'] = F.normalize(
            final.float(), dim=1, eps=1e-6)
        rgb = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
        rgb = rgb.unsqueeze(0).to(self.device) / 255.0
        features['rgb'] = F.normalize(F.interpolate(
            rgb, size=features['block07'].shape[-2:], mode='bilinear',
            align_corners=False), dim=1, eps=1e-6)
        return state, features

    @staticmethod
    def _rft_shift(value, dy, dx):
        height, width = value.shape[-2:]
        padded = F.pad(value, (1, 1, 1, 1), mode='replicate')
        return padded[..., 1 + dy:1 + dy + height,
                      1 + dx:1 + dx + width]

    def _rft_spatial_refine(self, delta, feature, output_shape):
        """One local feature-affinity pass over a bounded semantic residual."""
        feature = feature.float()
        size = feature.shape[-2:]
        value = F.interpolate(
            delta[None, None].to(self.device).float(), size=size,
            mode='bilinear', align_corners=False)
        weighted = value.clone()
        denominator = torch.ones_like(value)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            neighbour = self._rft_shift(feature, dy, dx)
            similarity = (feature * neighbour).sum(dim=1, keepdim=True)
            weight = torch.exp((similarity.clamp(-1.0, 1.0) - 1.0) / 0.10)
            weighted = weighted + weight * self._rft_shift(value, dy, dx)
            denominator = denominator + weight
        refined = weighted / denominator.clamp_min(1e-6)
        return F.interpolate(
            refined, size=output_shape, mode='bilinear',
            align_corners=False).squeeze().detach().float().cpu()

    def _rft_query_region_rows(
            self, package, features, class_index, class_name, slot,
            admission_presence):
        """Read PE region evidence without changing query admission/output."""
        count = min(
            int(package['raw_masks'].shape[0]),
            int(package['raw_scores'].numel()))
        if count == 0:
            return []
        raw_scores = package['raw_scores'][:count].float()
        keep = raw_scores * float(admission_presence) > float(
            self.processor.confidence_threshold)
        kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
        if not kept_indices.numel():
            return []
        raw_masks = package['raw_masks'][:count][keep].to(self.device).float()
        rows = [dict(
            class_index=int(class_index), class_name=class_name,
            instance_slot=int(slot), query_index=int(query_index),
            object_score=float(raw_scores[query_index].item()),
            admission_presence=float(admission_presence),
            admission_score=float(
                raw_scores[query_index].item() * float(admission_presence)),
        ) for query_index in kept_indices.tolist()]
        count = len(rows)
        common_support = None
        for feature_name, feature in features.items():
            feature = feature.float()
            size = feature.shape[-2:]
            masks = F.interpolate(
                raw_masks.unsqueeze(1), size=size, mode='bilinear',
                align_corners=False).sigmoid().squeeze(1)
            flat_feature = feature[0].flatten(1)
            flat_masks = masks.flatten(1)
            mass = flat_masks.sum(dim=1).clamp_min(1e-6)
            prototypes = F.normalize(
                flat_masks.matmul(flat_feature.t()) / mass[:, None],
                dim=1, eps=1e-6)
            similarity = prototypes.matmul(flat_feature).reshape(
                count, *size)
            inside = (similarity * masks).flatten(1).sum(dim=1) / mass
            ring = (F.max_pool2d(
                masks.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
                    - masks).clamp_min(0.0)
            ring_mass = ring.flatten(1).sum(dim=1).clamp_min(1e-6)
            ring_score = (similarity * ring).flatten(1).sum(dim=1) / ring_mass

            boundary_sum = torch.zeros(count, device=self.device)
            boundary_mass = torch.zeros(count, device=self.device)
            for dy, dx in ((1, 0), (0, 1)):
                shifted_masks = self._rft_shift(masks, dy, dx)
                shifted_feature = self._rft_shift(feature, dy, dx)
                mask_edge = (masks - shifted_masks).abs()
                visual_edge = (1.0 - (
                    feature * shifted_feature).sum(dim=1)).clamp_min(0.0)
                boundary_sum += (mask_edge * visual_edge).flatten(1).sum(1)
                boundary_mass += mask_edge.flatten(1).sum(1)
            agreement = boundary_sum / boundary_mass.clamp_min(1e-6)
            if common_support is None:
                common_support = (masks >= 0.5).to(torch.uint8).cpu()
            for index, row in enumerate(rows):
                row[f'{feature_name}_inside_coherence'] = float(
                    inside[index].item())
                row[f'{feature_name}_ring_separation'] = float(
                    (inside[index] - ring_score[index]).item())
                row[f'{feature_name}_boundary_agreement'] = float(
                    agreement[index].item())
        for index, row in enumerate(rows):
            row['_support'] = common_support[index]
        return rows

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
            self, package, presence, output_shape,
            admission_presence=None, amplitude_presence=None,
            return_admission=False):
        height, width = output_shape
        if not self.use_transformer_decoder:
            empty = torch.zeros((height, width), dtype=torch.float32)
            if return_admission:
                return empty, 0, dict(
                    keep=torch.zeros(0, dtype=torch.bool),
                    margin=torch.zeros(0, dtype=torch.float32))
            return empty, 0
        count = min(
            int(package['raw_masks'].shape[0]),
            int(package['raw_scores'].numel()))
        if count == 0:
            empty = torch.zeros((height, width), dtype=torch.float32)
            if return_admission:
                return empty, 0, dict(
                    keep=torch.zeros(0, dtype=torch.bool),
                    margin=torch.zeros(0, dtype=torch.float32))
            return empty, 0
        raw_scores = package['raw_scores'][:count].to(self.device)
        presence = torch.as_tensor(
            presence, device=self.device, dtype=raw_scores.dtype)
        admission_presence = torch.as_tensor(
            presence if admission_presence is None else admission_presence,
            device=self.device, dtype=raw_scores.dtype)
        amplitude_presence = torch.as_tensor(
            presence if amplitude_presence is None else amplitude_presence,
            device=self.device, dtype=raw_scores.dtype)
        admission_scores = raw_scores * admission_presence
        margin = admission_scores.float() - float(
            self.processor.confidence_threshold)
        keep = admission_scores > float(self.processor.confidence_threshold)
        kept_count = int(keep.sum().item())
        if kept_count == 0:
            empty = torch.zeros((height, width), dtype=torch.float32)
            if return_admission:
                return empty, 0, dict(
                    keep=keep.detach().bool().cpu(),
                    margin=margin.detach().float().cpu())
            return empty, 0
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
                else (raw_scores * amplitude_presence)[keep])
            instance = (
                masks.float() * amplitudes.float()[:, None, None]
            ).max(dim=0)[0]
        result = instance.detach().float().cpu()
        if return_admission:
            return result, kept_count, dict(
                keep=keep.detach().bool().cpu(),
                margin=margin.detach().float().cpu())
        return result, kept_count

    def _rft_query_support_from_raw(self, package, selection, output_shape):
        """Union support of a selected raw-query subset for diagnosis."""
        height, width = output_shape
        count = min(
            int(package['raw_masks'].shape[0]),
            int(torch.as_tensor(selection).numel()))
        if count == 0:
            return torch.zeros((height, width), dtype=torch.float32)
        selection = torch.as_tensor(selection).bool().flatten()[:count]
        if not selection.any():
            return torch.zeros((height, width), dtype=torch.float32)
        masks = package['raw_masks'][:count][selection].to(
            self.device).float()
        with torch.no_grad():
            masks = sam3_interpolate(
                masks.unsqueeze(1), size=output_shape, mode='bilinear',
                align_corners=False).sigmoid().squeeze(1)
        return masks.max(dim=0)[0].detach().float().cpu()

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

    def _rft_pi_compatibility_replays(
            self, anchor, presence_package, instance_package, output_shape,
            alpha, clip):
        """Replay selected P/I residuals at native Presence intervention sites.

        These maps are counterfactual diagnostics. They preserve the shared
        SAM3 grounding outputs and change only where the bounded Presence
        residual participates in query admission, instance amplitude and the
        final branch fusion.
        """
        semantic = anchor['semantic'].float()
        p0 = torch.as_tensor(anchor['presence']).float().reshape(())
        p1, _ = self._rft_bounded(
            p0, presence_package['presence'], alpha, clip)

        def instance(package, admission, amplitude, details=False):
            return self._rpt_head_role_instance_from_raw(
                package, amplitude, output_shape,
                admission_presence=admission,
                amplitude_presence=amplitude,
                return_admission=details)

        i00, _, anchor_a0 = instance(anchor, p0, p0, True)
        i10, _ = instance(anchor, p1, p1)
        candidate_i0, _, candidate_a0 = instance(
            instance_package, p0, p0, True)
        candidate_i1, _, candidate_a1 = instance(
            instance_package, p1, p1, True)
        i01, _ = self._rft_bounded(
            i00, candidate_i0, alpha, clip)
        i11, _ = self._rft_bounded(
            i10, candidate_i1, alpha, clip)

        anchor_admission_frozen, _ = instance(anchor, p0, p1)
        candidate_admission_frozen, _ = instance(
            instance_package, p0, p1)
        i11_freeze_admission, _ = self._rft_bounded(
            anchor_admission_frozen, candidate_admission_frozen,
            alpha, clip)

        anchor_amplitude_frozen, _ = instance(anchor, p1, p0)
        candidate_amplitude_frozen, _ = instance(
            instance_package, p1, p0)
        i11_freeze_amplitude, _ = self._rft_bounded(
            anchor_amplitude_frozen, candidate_amplitude_frozen,
            alpha, clip)

        def native_final(presence, instance_map):
            return self._rft_role_final(
                semantic, instance_map, presence, output_shape)

        def branch_once(presence, instance_map):
            if not self.use_presence_score:
                return torch.maximum(semantic, instance_map)
            return torch.maximum(
                semantic * float(presence.item()), instance_map)

        f00 = native_final(p0, i00)
        f10 = native_final(p1, i10)
        f01 = native_final(p0, i01)
        f11 = native_final(p1, i11)
        i_only_semantic_wins = semantic >= i01
        held_winner = torch.where(i_only_semantic_wins, semantic, i11)
        if self.use_presence_score:
            held_winner = held_winner * float(p1.item())

        role_finals = OrderedDict((
            ('pi_native_p0_i0', f00),
            ('pi_native_p1_i0', f10),
            ('pi_native_p0_i1', f01),
            ('pi_native_p1_i1', f11),
            ('pi_freeze_admission_p1_i1', native_final(
                p1, i11_freeze_admission)),
            ('pi_freeze_amplitude_p1_i1', native_final(
                p1, i11_freeze_amplitude)),
            ('pi_hold_i_only_winner_p1_i1', held_winner),
            ('pi_branch_once_p0_i0', branch_once(p0, i00)),
            ('pi_branch_once_p1_i0', branch_once(p1, i10)),
            ('pi_branch_once_p0_i1', branch_once(p0, i01)),
            ('pi_branch_once_p1_i1', branch_once(p1, i11)),
        ))
        if tuple(role_finals) != PI_VARIANT_NAMES:
            raise RuntimeError('PI replay variant order drifted.')

        keep0 = candidate_a0['keep']
        keep1 = candidate_a1['keep']
        added = keep1 & ~keep0
        removed = keep0 & ~keep1
        margin0 = candidate_a0['margin']
        margin1 = candidate_a1['margin']
        query_row = dict(
            p0=float(p0.item()),
            p1=float(p1.item()),
            presence_delta=float((p1 - p0).item()),
            candidate_query_count=int(keep0.numel()),
            admitted_p0=int(keep0.sum().item()),
            admitted_p1=int(keep1.sum().item()),
            admission_added=int(added.sum().item()),
            admission_removed=int(removed.sum().item()),
            admission_unchanged=int((keep0 == keep1).sum().item()),
            min_abs_margin_p0=(float(margin0.abs().min().item())
                               if margin0.numel() else None),
            min_abs_margin_p1=(float(margin1.abs().min().item())
                               if margin1.numel() else None),
            anchor_admitted_p0=int(anchor_a0['keep'].sum().item()),
        )
        mechanism_maps = OrderedDict((
            ('semantic_anchor', semantic),
            ('instance_i_only', i01),
            ('instance_pi', i11),
            ('admission_added_support', self._rft_query_support_from_raw(
                instance_package, added, output_shape)),
            ('admission_removed_support', self._rft_query_support_from_raw(
                instance_package, removed, output_shape)),
        ))
        if tuple(mechanism_maps) != PI_MECHANISM_MAP_NAMES:
            raise RuntimeError('PI mechanism map order drifted.')
        return role_finals, mechanism_maps, query_row

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
        state, pe_features = self._rpt_set_image_with_pe_features(image)

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
        pi_variants = None
        pi_mechanism_maps = None
        pi_query_rows = []
        if self._rpt_uses_pi_diagnosis():
            pi_variants = OrderedDict(
                (name, torch.empty(
                    (self.num_cls, height, width), dtype=torch.float32,
                    device='cpu'))
                for name in PI_VARIANT_NAMES)
            pi_mechanism_maps = OrderedDict(
                (name, torch.empty(
                    (self.num_cls, height, width), dtype=torch.float32,
                    device='cpu'))
                for name in PI_MECHANISM_MAP_NAMES)
        completion_variants = None
        if self._rpt_uses_completion_diagnosis():
            completion_variants = OrderedDict(
                (name, torch.empty(
                    (self.num_cls, height, width), dtype=torch.float32,
                    device='cpu'))
                for name in self._rpt_completion_variant_names())
        pe_variants = None
        pe_query_rows = []
        if self._rpt_uses_pe_diagnosis():
            pe_variants = OrderedDict(
                (name, torch.empty(
                    (self.num_cls, height, width), dtype=torch.float32,
                    device='cpu'))
                for name in PE_VARIANT_NAMES)
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

            def instance_for(package, presence, admission_presence=None):
                amplitude_value = float(torch.as_tensor(
                    presence).float().item())
                admission_value = float(torch.as_tensor(
                    presence if admission_presence is None
                    else admission_presence).float().item())
                key = (id(package), amplitude_value, admission_value)
                if key not in instance_cache:
                    instance_cache[key] = (
                        self._rpt_head_role_instance_from_raw(
                            package, presence, output_shape,
                            admission_presence=admission_presence,
                            amplitude_presence=presence))
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

            if self._rpt_uses_pe_diagnosis():
                selected_slots = self._rpt_role_selection[
                    '_best_overall_slots']
                p_slot, s_slot, i_slot = selected_slots
                diagnostic_i_slot = self._rpt_role_selection[
                    '_instance_diagnostic_slot']
                presence_package = (
                    role_packages['presence'][p_slot] if p_slot else None)
                semantic_package = (
                    role_packages['semantic'][s_slot] if s_slot else None)
                instance_package = (
                    role_packages['instance'][i_slot] if i_slot else None)
                presence = anchor_presence
                if presence_package is not None:
                    presence, _ = self._rft_bounded(
                        anchor_presence, presence_package['presence'],
                        default_alpha, default_clip)
                semantic_delta = torch.zeros_like(anchor_semantic)
                if semantic_package is not None:
                    _, semantic_delta = self._rft_bounded(
                        anchor_semantic, semantic_package['semantic'],
                        default_alpha, default_clip)
                    semantic_delta = semantic_delta.clamp(
                        -default_clip, default_clip)
                semantic_sources = OrderedDict(unrefined=semantic_delta)
                for feature_name in PE_FEATURE_NAMES:
                    semantic_sources[feature_name] = self._rft_spatial_refine(
                        semantic_delta, pe_features[feature_name], output_shape)

                def selected_instance(anchor_admission):
                    admission = anchor_presence if anchor_admission else presence
                    base, _ = instance_for(anchor, presence, admission)
                    if instance_package is None:
                        return base
                    candidate, _ = instance_for(
                        instance_package, presence, admission)
                    value, _ = self._rft_bounded(
                        base, candidate, default_alpha, default_clip)
                    return value

                for source_name, residual in semantic_sources.items():
                    semantic = (
                        anchor_semantic
                        + default_alpha * residual).clamp(0.0, 1.0)
                    for admission_name, use_anchor_admission in (
                            ('native', False), ('anchor_admission', True)):
                        instance = selected_instance(use_anchor_admission)
                        role_final = self._rft_role_final(
                            semantic, instance, presence, output_shape)
                        name = f'pe_sem_{source_name}_{admission_name}'
                        pe_variants[name][class_index].copy_((
                            official_final.float() + role_final.float()
                            - role_anchor_final.float()).clamp(0.0, 1.0))

                diagnostic_package = role_packages['instance'][
                    diagnostic_i_slot]
                pe_query_rows.extend(self._rft_query_region_rows(
                    diagnostic_package, pe_features, class_index,
                    item['name'], diagnostic_i_slot, anchor_presence))

            if self._rpt_uses_pi_diagnosis():
                pi_role_finals, pi_maps, pi_query_row = (
                    self._rft_pi_compatibility_replays(
                        anchor,
                        role_packages['presence'][
                            self.role_prompt_tta_pi_presence_slot],
                        role_packages['instance'][
                            self.role_prompt_tta_pi_instance_slot],
                        output_shape, default_alpha, default_clip))
                for name, role_final in pi_role_finals.items():
                    pi_variants[name][class_index].copy_((
                        official_final.float() + role_final.float()
                        - role_anchor_final.float()).clamp(0.0, 1.0))
                for name, value in pi_maps.items():
                    pi_mechanism_maps[name][class_index].copy_(value)
                pi_query_rows.append(dict(
                    class_index=int(class_index),
                    class_name=item['name'],
                    presence_slot=int(
                        self.role_prompt_tta_pi_presence_slot),
                    instance_slot=int(
                        self.role_prompt_tta_pi_instance_slot),
                    presence_prompt=item['presence_candidates'][
                        self.role_prompt_tta_pi_presence_slot - 1],
                    instance_prompt=item['instance_candidates'][
                        self.role_prompt_tta_pi_instance_slot - 1],
                    **pi_query_row,
                ))

            def compose(presence_slot=0, semantic_slot=0, instance_slot=0,
                        setting=DEFAULT_SETTING, shared_package=None,
                        anchor_admission=False):
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
                admission_presence = (
                    anchor_presence if anchor_admission else presence)
                base_instance, _ = instance_for(
                    anchor, presence, admission_presence)
                instance = base_instance
                if instance_package is not None:
                    candidate_instance, _ = instance_for(
                        instance_package, presence, admission_presence)
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
            if completion_variants is not None:
                for slots in self._rpt_prompt_bank[
                        '_completion_combinations']:
                    name = completion_variant_name(*slots)
                    completion_variants[name][class_index].copy_(compose(
                        *slots, anchor_admission=True))
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
        pi_identity_error = 0.0
        pi_replay_match_error = 0.0
        if pi_variants is not None:
            pi_identity_error = float((
                pi_variants['pi_native_p0_i0'] - variants['baseline']
            ).abs().max().item())
            p_slot = self.role_prompt_tta_pi_presence_slot
            i_slot = self.role_prompt_tta_pi_instance_slot
            replay_pairs = (
                ('pi_native_p0_i0', combo_variant_name(0, 0, 0)),
                ('pi_native_p1_i0', combo_variant_name(p_slot, 0, 0)),
                ('pi_native_p0_i1', combo_variant_name(0, 0, i_slot)),
                ('pi_native_p1_i1', combo_variant_name(p_slot, 0, i_slot)),
            )
            pi_replay_match_error = max(float((
                pi_variants[pi_name] - variants[combo_name]
            ).abs().max().item()) for pi_name, combo_name in replay_pairs)
        recomposition_error = max(recomposition_errors or [0.0])
        completion_noop_error = 0.0
        if completion_variants is not None:
            completion_noop_error = max([
                float((completion_variants[completion_variant_name(*slots)]
                       - variants[combo_variant_name(*slots)]
                       ).abs().max().item())
                for slots in self._rpt_prompt_bank['_completion_combinations']
                if slots[0] == 0
            ] or [0.0])
        if tuple(variants) != VARIANT_NAMES:
            raise RuntimeError('Role-functional variant order drifted.')
        if (completion_variants is not None
                and tuple(completion_variants)
                != self._rpt_completion_variant_names()):
            raise RuntimeError('Completion variant order drifted.')
        if pe_variants is not None and tuple(pe_variants) != PE_VARIANT_NAMES:
            raise RuntimeError('PE variant order drifted.')
        if (self.role_prompt_tta_strict_integrity
                and max(identity_error, recomposition_error,
                        pi_identity_error, pi_replay_match_error,
                        completion_noop_error)
                > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                'Role-functional integrity failed: '
                f'identity={identity_error}, '
                f'pi_identity={pi_identity_error}, '
                f'pi_replay_match={pi_replay_match_error}, '
                f'completion_noop={completion_noop_error}, '
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
            pi_no_update_identity_max_abs=pi_identity_error,
            pi_replay_match_max_abs=pi_replay_match_error,
            completion_noop_max_abs=completion_noop_error,
            native_recomposition_max_abs=recomposition_error,
            official_query_count=int(self.num_queries),
            canonical_class_count=int(self.num_cls),
            diagnostic_variant_count=(
                len(variants) + len(completion_variants or {})),
            diagnostic_cpu_bytes=int(sum(
                value.numel() * value.element_size()
                for value in variants.values()) + sum(
                    value.numel() * value.element_size()
                    for value in (completion_variants or {}).values())),
            default_residual_setting=DEFAULT_SETTING,
            residual_settings=[
                dict(name=name, alpha=alpha, clip=clip)
                for name, alpha, clip in RESIDUAL_SETTINGS],
            candidate_rows=candidate_rows,
            class_rows=class_rows,
            pi_diagnosis=(
                dict(
                    schema_version=PI_SCHEMA_VERSION,
                    protocol=PI_PROTOCOL,
                    presence_slot=int(
                        self.role_prompt_tta_pi_presence_slot),
                    instance_slot=int(
                        self.role_prompt_tta_pi_instance_slot),
                    query_rows=pi_query_rows,
                ) if pi_variants is not None else None),
            completion_diagnosis=(
                dict(
                    schema_version=COMPLETION_SCHEMA_VERSION,
                    protocol=COMPLETION_PROTOCOL,
                    selected_combination=list(
                        self._rpt_prompt_bank['_completion_selected']),
                    anchor_admission_combinations=[
                        list(value) for value in self._rpt_prompt_bank[
                            '_completion_combinations']],
                    target_combinations=[
                        list(value) for value in self._rpt_prompt_bank[
                            '_completion_targets']],
                ) if completion_variants is not None else None),
            pe_diagnosis=(
                dict(
                    schema_version=PE_SCHEMA_VERSION,
                    protocol=PE_PROTOCOL,
                    pe_layers=list(PE_LAYER_IDS),
                    selected_combination=list(
                        self._rpt_role_selection['_best_overall_slots']),
                    selected_admission=self._rpt_role_selection[
                        'best_overall']['admission'],
                    best_all_nonzero_combination=list(
                        self._rpt_role_selection[
                            '_best_all_nonzero_slots']),
                    instance_diagnostic_slot=int(
                        self._rpt_role_selection[
                            '_instance_diagnostic_slot']),
                    query_rows=pe_query_rows,
                ) if pe_variants is not None else None),
            cuda_memory=dict(main=_cuda_memory_snapshot(self.device)),
        )
        primary = baseline_query['final'].to(self.device)
        components = dict(
            semantic_logits=baseline_query['semantic'].to(self.device),
            instance_logits=baseline_query['instance'].to(self.device),
            role_prompt_variant_class_logits=variants,
            role_prompt_view_stats=[view_stats],
        )
        if pi_variants is not None:
            components['role_prompt_pi_variant_class_logits'] = pi_variants
            components['role_prompt_pi_mechanism_class_maps'] = (
                pi_mechanism_maps)
        if completion_variants is not None:
            components['role_prompt_completion_variant_class_logits'] = (
                completion_variants)
        if pe_variants is not None:
            components['role_prompt_pe_variant_class_logits'] = pe_variants
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
            pe = view.get('pe_diagnosis')
            if pe is not None:
                query_rows = []
                for source_row in pe.get('query_rows', []):
                    row = dict(source_row)
                    support = row.pop('_support').bool()[None, None]
                    support = F.interpolate(
                        support.float(), size=crop_gt.shape[-2:],
                        mode='nearest').squeeze().bool()
                    class_index = int(row['class_index'])
                    target = valid & (crop_gt == class_index)
                    intersection = int((support & target).sum().item())
                    support_pixels = int((support & valid).sum().item())
                    target_pixels = int(target.sum().item())
                    union = support_pixels + target_pixels - intersection
                    row.update(
                        gt_present=bool(target_pixels),
                        support_pixels=support_pixels,
                        gt_pixels=target_pixels,
                        gt_intersection_pixels=intersection,
                        gt_false_positive_pixels=(
                            support_pixels - intersection),
                        gt_precision=_safe_div(
                            intersection, support_pixels),
                        gt_recall=_safe_div(intersection, target_pixels),
                        gt_iou=_safe_div(intersection, union),
                    )
                    query_rows.append(row)
                pe = dict(pe)
                pe['query_rows'] = query_rows
                view['pe_diagnosis'] = pe
            enriched.append(view)
        return enriched

    def _rft_pe_record(self, pe_variant_logits, gt, valid):
        if pe_variant_logits is None:
            return None
        if tuple(pe_variant_logits) != PE_VARIANT_NAMES:
            raise RuntimeError('Recorded PE variants drifted.')
        rows = OrderedDict()
        gt_boundary = torch.zeros_like(valid)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted_gt = self._rft_shift(gt[None, None].float(), dy, dx)
            shifted_valid = self._rft_shift(
                valid[None, None].float(), dy, dx).bool()
            gt_boundary |= (
                valid & shifted_valid.squeeze()
                & (gt != shifted_gt.squeeze().long()))
        gt_boundary = F.max_pool2d(
            gt_boundary.float()[None, None], 3, stride=1,
            padding=1).squeeze().bool() & valid
        for name in PE_VARIANT_NAMES:
            admission = (
                'anchor_admission' if name.endswith('_anchor_admission')
                else 'native')
            reference_name = f'pe_sem_unrefined_{admission}'
            reference = self._rpt_threshold(
                pe_variant_logits[reference_name].detach().float().cpu()
            ).to(torch.int16)
            prediction = self._rpt_threshold(
                pe_variant_logits[name].detach().float().cpu()
            ).to(torch.int16)
            changed = valid & (prediction != reference)
            improved = changed & (prediction == gt) & (reference != gt)
            harmed = changed & (prediction != gt) & (reference == gt)
            rows[name] = dict(
                reference_variant=reference_name,
                confusion=self._rpt_confusion(prediction, gt, valid),
                changed_pixels=int(changed.sum().item()),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                changed_boundary_pixels=int((changed & gt_boundary).sum()),
                improved_boundary_pixels=int((improved & gt_boundary).sum()),
                harmed_boundary_pixels=int((harmed & gt_boundary).sum()),
                help_minus_harm=(int(improved.sum().item())
                                 - int(harmed.sum().item())),
            )
        return dict(
            schema_version=PE_SCHEMA_VERSION,
            protocol=PE_PROTOCOL,
            selected_combination=list(
                self._rpt_role_selection['_best_overall_slots']),
            variants=rows,
        )

    def _rft_pi_record(
            self, pi_variant_logits, mechanism_maps, gt, valid):
        if pi_variant_logits is None or mechanism_maps is None:
            return None
        if tuple(pi_variant_logits) != PI_VARIANT_NAMES:
            raise RuntimeError('Recorded PI replay variants drifted.')
        if tuple(mechanism_maps) != PI_MECHANISM_MAP_NAMES:
            raise RuntimeError('Recorded PI mechanism maps drifted.')

        logits = {
            name: value.detach().float().cpu()
            for name, value in pi_variant_logits.items()
        }
        predictions = {
            name: self._rpt_threshold(value).to(torch.int16)
            for name, value in logits.items()
        }
        reference = predictions['pi_native_p0_i0']
        variant_stats = {}
        for name in PI_VARIANT_NAMES:
            prediction = predictions[name]
            changed = valid & (prediction != reference)
            improved = changed & (prediction == gt) & (reference != gt)
            harmed = changed & (prediction != gt) & (reference == gt)
            variant_stats[name] = dict(
                confusion=self._rpt_confusion(prediction, gt, valid),
                changed_pixels=int(changed.sum().item()),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                help_minus_harm=(int(improved.sum().item())
                                 - int(harmed.sum().item())),
            )

        semantic = mechanism_maps['semantic_anchor'].detach().float().cpu()
        instance_i_only = (
            mechanism_maps['instance_i_only'].detach().float().cpu())
        instance_pi = mechanism_maps['instance_pi'].detach().float().cpu()
        winner_i_only = semantic >= instance_i_only
        winner_pi = semantic >= instance_pi
        s_to_i = winner_i_only & ~winner_pi
        i_to_s = ~winner_i_only & winner_pi
        any_head_switch = (s_to_i | i_to_s).any(dim=0) & valid

        pred_i_only = predictions['pi_native_p0_i1']
        pred_full = predictions['pi_native_p1_i1']
        class_flip = valid & (pred_i_only != pred_full)
        improved = class_flip & (pred_full == gt) & (pred_i_only != gt)
        harmed = class_flip & (pred_full != gt) & (pred_i_only == gt)
        wrong_to_wrong = (
            class_flip & (pred_full != gt) & (pred_i_only != gt))

        def class_margin(value):
            count = int(value.shape[0])
            safe_gt = gt.clamp(0, count - 1)
            gt_score = value.gather(
                0, safe_gt.unsqueeze(0)).squeeze(0)
            other = value.clone()
            other.scatter_(0, safe_gt.unsqueeze(0), float('-inf'))
            return gt_score - other.max(dim=0)[0]

        margin_i_only = class_margin(logits['pi_native_p0_i1'])
        margin_full = class_margin(logits['pi_native_p1_i1'])
        margin_delta = margin_full - margin_i_only

        def subset_mean(value, mask):
            return (float(value[mask].mean().item())
                    if mask.any() else None)

        class_rows = []
        added_support = mechanism_maps[
            'admission_added_support'].detach().float().cpu() >= 0.5
        removed_support = mechanism_maps[
            'admission_removed_support'].detach().float().cpu() >= 0.5
        for class_index, class_name in enumerate(self.class_names):
            gt_class = valid & (gt == class_index)

            def support_stats(support):
                support = support[class_index] & valid
                overlap = support & gt_class
                return dict(
                    pixels=int(support.sum().item()),
                    gt_pixels=int(overlap.sum().item()),
                    precision=_safe_div(
                        int(overlap.sum().item()),
                        int(support.sum().item())),
                    gt_recall=_safe_div(
                        int(overlap.sum().item()),
                        int(gt_class.sum().item())),
                )

            added = support_stats(added_support)
            removed = support_stats(removed_support)
            class_rows.append(dict(
                class_index=int(class_index),
                class_name=class_name,
                s_to_i_pixels=int((s_to_i[class_index] & valid).sum().item()),
                s_to_i_gt_pixels=int((
                    s_to_i[class_index] & gt_class).sum().item()),
                i_to_s_pixels=int((i_to_s[class_index] & valid).sum().item()),
                i_to_s_gt_pixels=int((
                    i_to_s[class_index] & gt_class).sum().item()),
                added_support_pixels=added['pixels'],
                added_support_gt_pixels=added['gt_pixels'],
                added_support_precision=added['precision'],
                added_support_gt_recall=added['gt_recall'],
                removed_support_pixels=removed['pixels'],
                removed_support_gt_pixels=removed['gt_pixels'],
                removed_support_precision=removed['precision'],
                removed_support_gt_recall=removed['gt_recall'],
            ))

        return dict(
            schema_version=PI_SCHEMA_VERSION,
            protocol=PI_PROTOCOL,
            presence_slot=int(self.role_prompt_tta_pi_presence_slot),
            instance_slot=int(self.role_prompt_tta_pi_instance_slot),
            variants=variant_stats,
            mechanism=dict(
                class_flip_pixels=int(class_flip.sum().item()),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                wrong_to_wrong_pixels=int(wrong_to_wrong.sum().item()),
                any_head_switch_pixels=int(any_head_switch.sum().item()),
                class_flip_with_head_switch=int((
                    class_flip & any_head_switch).sum().item()),
                class_flip_without_head_switch=int((
                    class_flip & ~any_head_switch).sum().item()),
                improved_with_head_switch=int((
                    improved & any_head_switch).sum().item()),
                improved_without_head_switch=int((
                    improved & ~any_head_switch).sum().item()),
                harmed_with_head_switch=int((
                    harmed & any_head_switch).sum().item()),
                harmed_without_head_switch=int((
                    harmed & ~any_head_switch).sum().item()),
                mean_gt_margin_delta=subset_mean(margin_delta, valid),
                corrected_gt_margin_delta=subset_mean(
                    margin_delta, improved),
                harmed_gt_margin_delta=subset_mean(margin_delta, harmed),
                class_rows=class_rows,
            ),
        )

    def _rft_completion_record(
            self, completion_variant_logits, variant_logits, gt, valid):
        if completion_variant_logits is None:
            return None
        expected = self._rpt_completion_variant_names()
        if tuple(completion_variant_logits) != expected:
            raise RuntimeError('Recorded completion variants drifted.')
        rows = OrderedDict()
        for slots, name in zip(
                self._rpt_prompt_bank['_completion_combinations'], expected):
            native_name = combo_variant_name(*slots)
            native_prediction = self._rpt_threshold(
                variant_logits[native_name].detach().float().cpu()
            ).to(torch.int16)
            prediction = self._rpt_threshold(
                completion_variant_logits[name].detach().float().cpu()
            ).to(torch.int16)
            changed = valid & (prediction != native_prediction)
            improved = changed & (prediction == gt) & (native_prediction != gt)
            harmed = changed & (prediction != gt) & (native_prediction == gt)
            rows[name] = dict(
                slots=list(slots),
                native_variant=native_name,
                native_confusion=self._rpt_confusion(
                    native_prediction, gt, valid),
                anchor_admission_confusion=self._rpt_confusion(
                    prediction, gt, valid),
                changed_pixels=int(changed.sum().item()),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                help_minus_harm=(int(improved.sum().item())
                                 - int(harmed.sum().item())),
            )
        return dict(
            schema_version=COMPLETION_SCHEMA_VERSION,
            protocol=COMPLETION_PROTOCOL,
            selected_combination=list(
                self._rpt_prompt_bank['_completion_selected']),
            target_combinations=[
                list(value) for value in self._rpt_prompt_bank[
                    '_completion_targets']],
            variants=rows,
        )

    def _rpt_record_image(
            self, variant_logits, view_stats, data_sample, image_path,
            pi_variant_logits=None, pi_mechanism_maps=None,
            completion_variant_logits=None, pe_variant_logits=None):
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
            pi_role_compatibility=self._rft_pi_record(
                pi_variant_logits, pi_mechanism_maps, gt, valid),
            role_text_completion=self._rft_completion_record(
                completion_variant_logits, variant_logits, gt, valid),
            pe_role_evidence=self._rft_pe_record(
                pe_variant_logits, gt, valid),
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
