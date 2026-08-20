"""Frozen Fine/Context fusion screen for role-functional SAM3 grounding.

The experiment deliberately uses only the current evaluation image (or its
verified adjacent tiles).  It does not consume support images, exemplars,
retrieval banks, extra labels, or trainable parameters.
"""

import math
import os
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
from PIL import Image

from role_functional_text_definitions import (
    ROLE_MULTIMODAL_FUSION_ANCHOR_METHODS,
    ROLE_MULTIMODAL_FUSION_BLEND,
    ROLE_MULTIMODAL_FUSION_PROTOCOL,
    ROLE_MULTIMODAL_FUSION_ROLE_SCOPES,
    ROLE_MULTIMODAL_FUSION_SCHEMA_VERSION,
    ROLE_MULTIMODAL_FUSION_TEXT_METHODS,
    ROLE_MULTIMODAL_FUSION_VARIANT_NAMES,
    ROLE_MULTIMODAL_FUSION_WINDOW,
)


RMF_PROTOCOL = ROLE_MULTIMODAL_FUSION_PROTOCOL
RMF_SCHEMA_VERSION = ROLE_MULTIMODAL_FUSION_SCHEMA_VERSION
RMF_BLEND = ROLE_MULTIMODAL_FUSION_BLEND
RMF_WINDOW = ROLE_MULTIMODAL_FUSION_WINDOW
RMF_SCOPES = ROLE_MULTIMODAL_FUSION_ROLE_SCOPES
RMF_ANCHOR_METHODS = ROLE_MULTIMODAL_FUSION_ANCHOR_METHODS
RMF_TEXT_METHODS = ROLE_MULTIMODAL_FUSION_TEXT_METHODS
RMF_VARIANTS = ROLE_MULTIMODAL_FUSION_VARIANT_NAMES


class RoleMultimodalFusionMixin:
    """Evaluate V0--V2 and V4--V8 without external reference data."""

    @staticmethod
    def _rmf_shapes(value):
        return tuple(
            (int(row[0]), int(row[1]))
            for row in torch.as_tensor(value).detach().cpu().tolist())

    @staticmethod
    def _rmf_split_memory(memory, spatial_shapes):
        if not isinstance(memory, torch.Tensor) or memory.ndim != 3:
            raise ValueError('SAM3 encoder memory must be [tokens,batch,dim].')
        values, start = [], 0
        for height, width in spatial_shapes:
            stop = start + int(height) * int(width)
            values.append(memory[start:stop])
            start = stop
        if start != int(memory.shape[0]):
            raise ValueError(
                'Encoder spatial shapes do not cover the complete memory.')
        return values

    @classmethod
    def _rmf_align_memory(
            cls, memory, spatial_shapes, roi, image_size,
            target_spatial_shapes):
        """Crop the Context target ROI and align it to Fine memory grids."""
        source_levels = cls._rmf_split_memory(memory, spatial_shapes)
        if len(source_levels) != len(target_spatial_shapes):
            raise ValueError('Fine and Context encoder levels do not match.')
        image_width, image_height = image_size
        x1, y1, x2, y2 = roi
        aligned = []
        for source, (source_h, source_w), (target_h, target_w) in zip(
                source_levels, spatial_shapes, target_spatial_shapes):
            left = max(0, min(
                int(math.floor(float(x1) * source_w / image_width)),
                source_w - 1))
            top = max(0, min(
                int(math.floor(float(y1) * source_h / image_height)),
                source_h - 1))
            right = max(left + 1, min(
                int(math.ceil(float(x2) * source_w / image_width)),
                source_w))
            bottom = max(top + 1, min(
                int(math.ceil(float(y2) * source_h / image_height)),
                source_h))
            feature = source.permute(1, 2, 0).reshape(
                source.shape[1], source.shape[2], source_h, source_w)
            feature = F.interpolate(
                feature[..., top:bottom, left:right].float(),
                size=(target_h, target_w), mode='bilinear',
                align_corners=False).to(source.dtype)
            aligned.append(feature.flatten(2).permute(2, 0, 1))
        return torch.cat(aligned, dim=0)

    @classmethod
    def _rmf_local_attention(
            cls, query, key, value, spatial_shapes, window=RMF_WINDOW):
        """Fixed local attention used only as a frozen readout operator."""
        if int(window) % 2 != 1:
            raise ValueError('Local attention window must be odd.')
        outputs, entropies = [], []
        query_levels = cls._rmf_split_memory(query, spatial_shapes)
        key_levels = cls._rmf_split_memory(key, spatial_shapes)
        value_levels = cls._rmf_split_memory(value, spatial_shapes)
        padding = int(window) // 2
        for q_level, k_level, v_level, (height, width) in zip(
                query_levels, key_levels, value_levels, spatial_shapes):
            batch, channels = q_level.shape[1], q_level.shape[2]
            q = q_level.permute(1, 2, 0).reshape(
                batch, channels, height, width).float()
            k = k_level.permute(1, 2, 0).reshape(
                batch, channels, height, width).float()
            v = v_level.permute(1, 2, 0).reshape(
                batch, channels, height, width).float()
            q = F.normalize(q, dim=1, eps=1e-6).flatten(2).unsqueeze(2)
            k_windows = F.unfold(
                F.normalize(k, dim=1, eps=1e-6),
                kernel_size=window, padding=padding).reshape(
                    batch, channels, window * window, height * width)
            logits = (q * k_windows).sum(dim=1) * math.sqrt(channels)
            weights = torch.softmax(logits, dim=1)
            v_windows = F.unfold(
                v, kernel_size=window, padding=padding).reshape(
                    batch, channels, window * window, height * width)
            output = (weights.unsqueeze(1) * v_windows).sum(dim=2)
            outputs.append(output.permute(2, 0, 1).to(query.dtype))
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
            entropies.append(float(
                (entropy / math.log(window * window)).mean().item()))
        return torch.cat(outputs, dim=0), sum(entropies) / len(entropies)

    @staticmethod
    def _rmf_blend(anchor, candidate, weight=RMF_BLEND):
        return anchor + float(weight) * (candidate - anchor)

    @classmethod
    def _rmf_operator(cls, name, fine, context, spatial_shapes):
        """Return one fixed candidate memory and mechanism diagnostics."""
        if name == 'v4_transport':
            fine_from_context, entropy_fc = cls._rmf_local_attention(
                fine, context, context, spatial_shapes)
            context_from_fine, entropy_cf = cls._rmf_local_attention(
                context, fine, fine, spatial_shapes)
            candidate = 0.5 * (fine_from_context + context_from_fine)
            entropy = 0.5 * (entropy_fc + entropy_cf)
        elif name == 'v6_qk':
            # Context supplies only correspondence (Q/K); values stay Fine.
            candidate, entropy = cls._rmf_local_attention(
                context, context, fine, spatial_shapes)
        elif name == 'v7_value':
            # Fine-native correspondence transports Context content only.
            candidate, entropy = cls._rmf_local_attention(
                fine, fine, context, spatial_shapes)
        elif name == 'v8_full':
            # Context supplies the complete Q/K/V candidate readout.
            candidate, entropy = cls._rmf_local_attention(
                context, context, context, spatial_shapes)
        else:
            raise ValueError(f'Unknown multimodal operator: {name}.')
        output = cls._rmf_blend(fine.float(), candidate.float()).to(fine.dtype)
        shift = float((output.float() - fine.float()).norm().item())
        scale = float(fine.float().norm().item())
        return output, dict(
            normalized_attention_entropy=float(entropy),
            relative_memory_shift=shift / max(scale, 1e-8),
        )

    @classmethod
    def _rmf_agreement_operator(
            cls, fine, context, fine_anchor, context_anchor):
        """Route only role residuals supported in both aligned views."""
        fine_delta = fine.float() - fine_anchor.float()
        context_delta = context.float() - context_anchor.float()
        agreement = F.cosine_similarity(
            fine_delta, context_delta, dim=-1, eps=1e-6
        ).clamp_min(0.0).unsqueeze(-1)
        # Route only the supplementary Context role residual.  Absolute
        # Context appearance never replaces the Fine anchor in this family.
        transported = fine.float() + (
            float(RMF_BLEND) * agreement * context_delta)
        shift = float((transported - fine.float()).norm().item())
        scale = float(fine.float().norm().item())
        return transported.to(fine.dtype), dict(
            positive_agreement_fraction=float((agreement > 0).float().mean()),
            mean_positive_agreement=float(agreement.mean()),
            relative_memory_shift=shift / max(scale, 1e-8),
        )

    def _rmf_compact_package(self, package):
        """Safely discard queries that can never pass SAM3 admission."""
        result = dict(package)
        scores = torch.as_tensor(result['raw_scores']).float().flatten()
        count = min(int(scores.numel()), int(result['raw_masks'].shape[0]))
        scores = scores[:count]
        keep = scores > float(self.processor.confidence_threshold)
        result['raw_scores'] = scores[keep].cpu()
        result['raw_masks'] = result['raw_masks'][:count][keep].cpu()
        return result

    def _rmf_replay_package(self, state, memory, output_shape):
        with torch.no_grad(), self._rpt_autocast_context():
            self.processor.forward_grounding_from_encoder(state, memory)
            package = self._rpt_head_role_raw_package(state, output_shape)
        return self._rmf_compact_package(package)

    def _rmf_requirements(self):
        requirements = OrderedDict((prompt, None) for prompt in self.query_words)
        canonical = {}
        slots = self._rpt_role_selection['_best_overall_slots']
        fields = (
            ('presence', 'presence_candidates'),
            ('semantic', 'semantic_candidates'),
            ('instance', 'instance_candidates'),
        )
        for item in self._rpt_prompt_bank['classes']:
            class_anchor = item['official_prompts'][0]
            for (_, field), slot in zip(fields, slots):
                if slot:
                    prompt = item[field][slot - 1]
                    requirements.setdefault(prompt, None)
                    canonical[prompt] = class_anchor
        return tuple(requirements), canonical

    def _rmf_view_from_packages(self, packages, shape):
        query_indices = self.query_idx.detach().long().cpu()
        query_final = torch.stack([
            packages[prompt]['native_final'] for prompt in self.query_words])
        slots = self._rpt_role_selection['_best_overall_slots']
        fields = (
            ('presence', 'presence_candidates'),
            ('semantic', 'semantic_candidates'),
            ('instance', 'instance_candidates'),
        )
        classes = []
        for class_index, item in enumerate(self._rpt_prompt_bank['classes']):
            indices = torch.nonzero(
                query_indices == class_index, as_tuple=False).flatten().tolist()
            anchor = self._rft_merge_anchor_packages([
                packages[self.query_words[index]] for index in indices])
            selected = {}
            selected_prompts = {}
            for (role, field), slot in zip(fields, slots):
                if slot:
                    prompt = item[field][slot - 1]
                    selected[role] = packages[prompt]
                    selected_prompts[role] = prompt
                else:
                    selected[role] = None
                    selected_prompts[role] = list(item['official_prompts'])
            classes.append(dict(
                anchor=anchor, selected=selected,
                selected_prompts=selected_prompts))
        return dict(shape=shape, query_final=query_final, classes=classes)

    def _rmf_compact_view(self, view):
        for item in view['classes']:
            item['anchor'] = self._rmf_compact_package(item['anchor'])
            for role, package in item['selected'].items():
                if package is not None:
                    item['selected'][role] = self._rmf_compact_package(package)
        return view

    def _rmf_compose_class(
            self, view, class_index, active_roles, roi, instance_cache):
        data = view['classes'][class_index]
        alpha, clip = 0.5, 0.25
        presence = torch.as_tensor(
            data['anchor']['presence']).float().reshape(())
        if ('p' in active_roles
                and data['selected']['presence'] is not None):
            presence, _ = self._rft_bounded(
                presence, data['selected']['presence']['presence'],
                alpha, clip)
        semantic = self._rvf_crop(data['anchor']['semantic'], roi)
        if ('s' in active_roles
                and data['selected']['semantic'] is not None):
            candidate = self._rvf_crop(
                data['selected']['semantic']['semantic'], roi)
            semantic, _ = self._rft_bounded(
                semantic, candidate, alpha, clip)
        admission = presence
        if self._rpt_role_selection['best_overall'][
                'admission'] == 'anchor_admission':
            admission = torch.as_tensor(
                data['anchor']['presence']).float().reshape(())

        def instance(package):
            key = (
                id(package), float(presence), float(admission), tuple(roi))
            if key not in instance_cache:
                instance_cache[key] = self._rvf_instance(
                    package, presence, admission, view['shape'], roi)[0]
            return instance_cache[key]

        anchor_instance = instance(data['anchor'])
        instance_map = anchor_instance
        if ('i' in active_roles
                and data['selected']['instance'] is not None):
            candidate = instance(data['selected']['instance'])
            instance_map, _ = self._rft_bounded(
                anchor_instance, candidate, alpha, clip)
        return self._rft_role_final(
            semantic, instance_map, presence, semantic.shape)

    @staticmethod
    def _rmf_resize(value, output_shape):
        value = torch.as_tensor(value).float()
        if tuple(value.shape[-2:]) == tuple(output_shape):
            return value
        return F.interpolate(
            value.unsqueeze(0), size=output_shape, mode='bilinear',
            align_corners=False).squeeze(0)

    def _rmf_view_outputs(self, view, roi, output_shape):
        base = self._rvf_crop(
            self._aggregate_query_logits_to_classes(view['query_final']), roi)
        base = self._rmf_resize(base, output_shape)
        instance_cache = {}
        anchor_roles = torch.stack([
            self._rmf_compose_class(
                view, class_index, frozenset(), roi, instance_cache)
            for class_index in range(self.num_cls)
        ])
        anchor_roles = self._rmf_resize(anchor_roles, output_shape)
        outputs = dict(anchor=base)
        for scope in RMF_SCOPES:
            active = frozenset(scope) if scope != 'psi' else frozenset('psi')
            role_output = torch.stack([
                self._rmf_compose_class(
                    view, class_index, active, roi, instance_cache)
                for class_index in range(self.num_cls)
            ])
            role_output = self._rmf_resize(role_output, output_shape)
            outputs[scope] = (
                base + role_output - anchor_roles).clamp(0.0, 1.0)
        return outputs

    @staticmethod
    def _rmf_canvas(fine, context=None, resolution=1008):
        panel = int(resolution) // 2
        resampling = getattr(Image, 'Resampling', Image).BILINEAR
        canvas = Image.new('RGB', (resolution, resolution), (128, 128, 128))
        canvas.paste(fine.resize((panel, panel), resampling), (0, 0))
        if context is not None:
            canvas.paste(context.resize((panel, panel), resampling), (panel, 0))
        return canvas, (0, 0, panel, panel)

    def _rmf_process_unit(self, unit):
        requirements, canonical = self._rmf_requirements()
        output_shape = (unit['fine'].height, unit['fine'].width)
        fine_shape = output_shape
        cache = self._rpt_prepare_text_cache()
        with torch.no_grad(), self._rpt_autocast_context():
            context_state = self.processor.set_image(unit['context'])

        # Encode Context first and keep only target-aligned CPU memories.  Fine
        # and Context backbones are never resident together, which bounds the
        # per-GPU memory of this deliberately forward-heavy screen.
        context_memories = {}
        context_shapes = None
        for prompt in requirements:
            with torch.no_grad(), self._rpt_autocast_context():
                self._rpt_set_cached_prompt(context_state, prompt)
                memory = context_state[
                    'encoder_hidden_states'].detach().clone()
                shapes = self._rmf_shapes(
                    context_state['encoder_spatial_shapes'])
            if context_shapes is None:
                context_shapes = shapes
            elif shapes != context_shapes:
                raise RuntimeError('Context encoder spatial shapes drifted.')
            context_memories[prompt] = self._rmf_align_memory(
                memory, shapes, unit['context_roi'], unit['context'].size,
                shapes).detach().cpu()
        del context_state
        with torch.no_grad(), self._rpt_autocast_context():
            fine_state = self.processor.set_image(unit['fine'])

        package_sets = {
            name: {} for name in (
                'native', 'v4_transport', 'v5_agreement',
                'v6_qk', 'v7_value', 'v8_full')}
        canonical_memories = {}
        operator_stats = defaultdict(list)
        native_query = {}
        for prompt in requirements:
            with torch.no_grad(), self._rpt_autocast_context():
                self._rpt_set_cached_prompt(fine_state, prompt)
                fine_memory = fine_state[
                    'encoder_hidden_states'].detach().clone()
                fine_shapes = self._rmf_shapes(
                    fine_state['encoder_spatial_shapes'])
                native = self._rmf_compact_package(
                    self._rpt_head_role_raw_package(
                        fine_state, output_shape))
            if fine_shapes != context_shapes:
                raise RuntimeError(
                    'Fine and Context encoder spatial shapes do not match.')
            aligned_context = context_memories[prompt].to(fine_memory.device)
            package_sets['native'][prompt] = native
            if prompt in self.query_words:
                native_query[prompt] = native['native_final']
            if not getattr(self, '_rmf_replay_parity_checked', False):
                replay = self._rmf_replay_package(
                    fine_state, fine_memory, output_shape)
                parity = max(float((
                    torch.as_tensor(replay[field]).float()
                    - torch.as_tensor(native[field]).float()
                ).abs().max()) for field in (
                    'semantic', 'native_instance', 'native_final', 'presence'))
                self._rmf_replay_parity_max_abs = parity
                self._rmf_replay_parity_checked = True
                if (self.role_prompt_tta_strict_integrity
                        and parity
                        > self.role_prompt_tta_integrity_tolerance):
                    raise RuntimeError(
                        'Native encoder replay differs from SAM3 grounding: '
                        f'max_abs={parity}.')
            for method in ('v4_transport', 'v6_qk', 'v7_value', 'v8_full'):
                memory, stats = self._rmf_operator(
                    method, fine_memory, aligned_context, fine_shapes)
                package_sets[method][prompt] = self._rmf_replay_package(
                    fine_state, memory, output_shape)
                operator_stats[method].append(stats)

            if prompt in canonical:
                anchor_prompt = canonical[prompt]
                anchor = canonical_memories[anchor_prompt]
                memory, stats = self._rmf_agreement_operator(
                    fine_memory, aligned_context,
                    anchor['fine'].to(fine_memory.device),
                    anchor['context'].to(fine_memory.device))
                package_sets['v5_agreement'][prompt] = (
                    self._rmf_replay_package(
                        fine_state, memory, output_shape))
                operator_stats['v5_agreement'].append(stats)
            else:
                package_sets['v5_agreement'][prompt] = native

            if prompt in self.query_words:
                # The first official alias is the class anchor used by V5.
                for item in self._rpt_prompt_bank['classes']:
                    if prompt == item['official_prompts'][0]:
                        canonical_memories[prompt] = dict(
                            fine=fine_memory.detach().cpu(),
                            context=aligned_context.detach().cpu())
                        break

        views = {
            name: self._rmf_view_from_packages(packages, fine_shape)
            for name, packages in package_sets.items()
        }

        fine_canvas, canvas_roi = self._rmf_canvas(unit['fine'])
        shared_canvas, _ = self._rmf_canvas(unit['fine'], unit['context'])
        canvas_views = {
            'v2_canvas_fine': self._rmf_compact_view(
                self._rvf_ground_view(fine_canvas)),
            'v2_canvas_shared': self._rmf_compact_view(
                self._rvf_ground_view(shared_canvas)),
        }

        method_outputs = {
            'v1_role_text': self._rmf_view_outputs(
                views['native'], unit['fine_roi'], output_shape)}
        for method in ('v4_transport', 'v5_agreement',
                       'v6_qk', 'v7_value', 'v8_full'):
            method_outputs[method] = self._rmf_view_outputs(
                views[method], unit['fine_roi'], output_shape)
        for method, view in canvas_views.items():
            method_outputs[method] = self._rmf_view_outputs(
                view, canvas_roi, output_shape)

        variants = OrderedDict()
        variants['rmf_v0_official'] = method_outputs[
            'v1_role_text']['anchor']
        for method in RMF_ANCHOR_METHODS:
            variants[f'rmf_{method}_anchor'] = method_outputs[method]['anchor']
        for method in RMF_TEXT_METHODS:
            for scope in RMF_SCOPES:
                variants[f'rmf_{method}_{scope}'] = method_outputs[method][scope]
        if tuple(variants) != RMF_VARIANTS:
            raise RuntimeError('Multimodal fusion variant order drifted.')

        stats = {}
        for method, rows in operator_stats.items():
            fields = sorted(set(key for row in rows for key in row))
            stats[method] = {
                field: sum(float(row.get(field, 0.0)) for row in rows)
                / max(len(rows), 1)
                for field in fields
            }
            stats[method]['prompt_count'] = len(rows)
        return torch.stack([
            native_query[prompt] for prompt in self.query_words
        ]), variants, stats, len(cache['prompts'])

    def _rmf_predict_image(self, image, image_path):
        height, width = image.height, image.width
        counts = torch.zeros((1, height, width), dtype=torch.float32)
        query_sum = torch.zeros(
            (self.num_queries, height, width), dtype=torch.float32)
        raw = OrderedDict((
            name,
            torch.zeros((self.num_cls, height, width), dtype=torch.float32),
        ) for name in RMF_VARIANTS)
        unit_rows, mechanism_rows = [], defaultdict(list)
        for unit_index, unit in enumerate(
                self._rvf_visual_units(image, image_path)):
            query, variants, stats, cache_count = self._rmf_process_unit(unit)
            x1, y1, x2, y2 = unit['target_box']
            query_sum[:, y1:y2, x1:x2] += query
            counts[:, y1:y2, x1:x2] += 1.0
            for name, value in variants.items():
                raw[name][:, y1:y2, x1:x2] += value
            for method, row in stats.items():
                mechanism_rows[method].append(row)
            unit_rows.append(dict(
                unit_index=int(unit_index),
                target_box=list(unit['target_box']),
                fine_size=[unit['fine'].width, unit['fine'].height],
                context_size=[unit['context'].width, unit['context'].height],
                context_roi=list(unit['context_roi']),
                source_prefix=unit['metadata']['source_prefix'],
                source_canvas=unit['metadata']['source_canvas'],
                context_box=unit['metadata']['context_box'],
                contributor_tiles=unit['metadata']['contributor_tiles'],
                cached_prompt_count=int(cache_count),
            ))
        if torch.any(counts == 0):
            raise RuntimeError('Multimodal fusion units left uncovered pixels.')
        query_mean = query_sum / counts
        for name in raw:
            raw[name] /= counts

        if self._rvf_fine_matches_official(image):
            official_query = query_mean
            official = self._aggregate_query_logits_to_classes(query_mean)
            best_text = raw['rmf_v1_role_text_psi']
            reused_fine = True
        else:
            official_query, official, best_text = (
                self._rvf_accumulate_official_units(image))
            reused_fine = False

        variants = OrderedDict()
        variants['rmf_v0_official'] = official
        raw_v0 = raw['rmf_v0_official']
        for method in RMF_ANCHOR_METHODS:
            name = f'rmf_{method}_anchor'
            variants[name] = (
                official + raw[name] - raw_v0).clamp(0.0, 1.0)
        for method in RMF_TEXT_METHODS:
            for scope in RMF_SCOPES:
                name = f'rmf_{method}_{scope}'
                if method == 'v1_role_text':
                    variants[name] = (
                        official + raw[name] - raw_v0).clamp(0.0, 1.0)
                else:
                    reference_name = f'rmf_v1_role_text_{scope}'
                    variants[name] = (
                        variants[reference_name]
                        + raw[name] - raw[reference_name]
                    ).clamp(0.0, 1.0)
        # Preserve the already validated overall-best endpoint exactly.
        variants['rmf_v1_role_text_psi'] = best_text
        if tuple(variants) != RMF_VARIANTS:
            raise RuntimeError('Multimodal fusion output order drifted.')

        v0_error = float((variants['rmf_v0_official'] - official).abs().max())
        v1_error = float((
            variants['rmf_v1_role_text_psi'] - best_text).abs().max())
        if (self.role_prompt_tta_strict_integrity
                and max(v0_error, v1_error)
                > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                'V0/V1 protected endpoint identity failed: '
                f'{max(v0_error, v1_error)}.')
        mechanisms = {}
        for method, rows in mechanism_rows.items():
            fields = sorted(set(key for row in rows for key in row))
            mechanisms[method] = {
                field: sum(float(row.get(field, 0.0)) for row in rows)
                / max(len(rows), 1)
                for field in fields
            }
        metadata = dict(
            schema_version=RMF_SCHEMA_VERSION,
            protocol=RMF_PROTOCOL,
            variants=list(RMF_VARIANTS),
            excluded_methods=[
                'text_native_exemplar', 'support_image',
                'retrieval_reference_bank'],
            blend=float(RMF_BLEND),
            local_window=int(RMF_WINDOW),
            operator_definitions=dict(
                v2_canvas_fine='Fine panel plus blank-panel control',
                v2_canvas_shared='Fine and aligned Context on one canvas',
                v4_transport='bidirectional local cross-view transport',
                v5_agreement='positive Fine/Context role-residual agreement',
                v6_qk='Context Q/K with Fine V',
                v7_value='Context V with Fine Q/K',
                v8_full='Context Q/K/V candidate'),
            source_mode=self._rvf_config['source_mode'],
            fine_size=int(self._rvf_config['fine_size']),
            context_size=int(self._rvf_config['context_size']),
            selected_slots=list(
                self._rpt_role_selection['_best_overall_slots']),
            selected_admission=self._rpt_role_selection[
                'best_overall']['admission'],
            selected_prompts=[
                dict(
                    class_name=item['name'],
                    presence=(
                        item['presence_candidates'][slots[0] - 1]
                        if slots[0] else list(item['official_prompts'])),
                    semantic=(
                        item['semantic_candidates'][slots[1] - 1]
                        if slots[1] else list(item['official_prompts'])),
                    instance=(
                        item['instance_candidates'][slots[2] - 1]
                        if slots[2] else list(item['official_prompts'])),
                )
                for item in self._rpt_prompt_bank['classes']
                for slots in [
                    self._rpt_role_selection['_best_overall_slots']]
            ],
            official_reused_fine=bool(reused_fine),
            v0_identity_max_abs=v0_error,
            v1_identity_max_abs=v1_error,
            native_replay_parity_max_abs=float(getattr(
                self, '_rmf_replay_parity_max_abs', 0.0)),
            mechanism_summaries=mechanisms,
            units=unit_rows,
        )
        return official_query.to(self.device), variants, metadata

    def _rmf_record_image(self, variants, metadata, data_sample, image_path):
        if not self.dump_role_prompt_tta_stats:
            return
        gt = data_sample.gt_sem_seg.data.squeeze().detach().long().cpu()
        valid = gt != 255
        predictions = {
            name: self._rpt_threshold(value).to(torch.int16)
            for name, value in variants.items()
        }
        rows = OrderedDict()
        for name in RMF_VARIANTS:
            if name == 'rmf_v0_official':
                reference_name = name
            elif name.startswith('rmf_v1_role_text_'):
                reference_name = 'rmf_v0_official'
            elif name.endswith('_anchor'):
                reference_name = 'rmf_v0_official'
            else:
                scope = name.rsplit('_', 1)[-1]
                reference_name = f'rmf_v1_role_text_{scope}'
            prediction = predictions[name]
            reference = predictions[reference_name]
            changed = valid & (prediction != reference)
            improved = changed & (prediction == gt) & (reference != gt)
            harmed = changed & (prediction != gt) & (reference == gt)
            reference_fg = reference != self.bg_idx
            prediction_fg = prediction != self.bg_idx
            oracle = reference.clone()
            use_variant = valid & (prediction == gt) & (reference != gt)
            oracle[use_variant] = prediction[use_variant]
            rows[name] = dict(
                reference_variant=reference_name,
                confusion=self._rpt_confusion(prediction, gt, valid),
                oracle_confusion=self._rpt_confusion(oracle, gt, valid),
                changed_pixels=int(changed.sum()),
                improved_pixels=int(improved.sum()),
                harmed_pixels=int(harmed.sum()),
                help_minus_harm=int(improved.sum()) - int(harmed.sum()),
                background_to_foreground_pixels=int((
                    changed & ~reference_fg & prediction_fg).sum()),
                foreground_to_background_pixels=int((
                    changed & reference_fg & ~prediction_fg).sum()),
                foreground_to_foreground_pixels=int((
                    changed & reference_fg & prediction_fg).sum()),
            )
        payload = dict(metadata, variants=rows)
        record = dict(
            schema_version=RMF_SCHEMA_VERSION,
            rank=int(os.environ.get('RANK', 0)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            dataset_name=self.role_prompt_tta_dataset_name,
            img_path=image_path,
            class_names=list(self.class_names),
            valid_pixels=int(valid.sum()),
            prob_thd=float(self.prob_thd),
            role_multimodal_fusion=payload,
        )
        self._rpt_write_stats(record)
