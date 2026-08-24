"""Frozen SAM3 Joint Role--View profiling and final inference."""

import json
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

from role_functional_text_definitions import (
    JOINT_ROLE_VIEW_OPERATOR_SPECS,
    JOINT_ROLE_VIEW_FINAL_PROTOCOL,
    JOINT_ROLE_VIEW_FINAL_SCHEMA_VERSION,
    JOINT_ROLE_VIEW_PROTOCOL,
    JOINT_ROLE_VIEW_SCHEMA_VERSION,
    ROLE_VISUAL_FIELD_PROTOCOL,
    ROLE_VISUAL_FIELD_SCHEMA_VERSION,
    load_joint_role_view_registry,
    load_joint_role_view_final_registry,
)
from sam3.model.data_misc import interpolate as sam3_interpolate
from tiled_context import CoordinateTileIndex


RVF_PROTOCOL = ROLE_VISUAL_FIELD_PROTOCOL
RVF_SCHEMA_VERSION = ROLE_VISUAL_FIELD_SCHEMA_VERSION
JRV_PROTOCOL = JOINT_ROLE_VIEW_PROTOCOL
JRV_SCHEMA_VERSION = JOINT_ROLE_VIEW_SCHEMA_VERSION
JRV_OPERATOR_SPECS = JOINT_ROLE_VIEW_OPERATOR_SPECS
JRV_FINAL_PROTOCOL = JOINT_ROLE_VIEW_FINAL_PROTOCOL
JRV_FINAL_SCHEMA_VERSION = JOINT_ROLE_VIEW_FINAL_SCHEMA_VERSION


def load_visual_field_registry(path, dataset):
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if (int(payload.get('schema_version', -1)) != RVF_SCHEMA_VERSION
            or payload.get('protocol') != RVF_PROTOCOL):
        raise ValueError(f'{path} is not a {RVF_PROTOCOL} registry.')
    key = str(dataset).lower()
    record = dict(payload.get('datasets', {}).get(key, {}))
    if not record:
        raise ValueError(f'{path} has no visual-field entry for {key!r}.')
    record['fine_size'] = int(record['fine_size'])
    record['context_size'] = int(record['context_size'])
    if (record['fine_size'] <= 0
            or record['context_size'] <= record['fine_size']):
        raise ValueError(f'{key}: context_size must exceed fine_size.')
    if record.get('source_mode') not in ('image', 'coordinate_tiles'):
        raise ValueError(f'{key}: invalid source_mode.')
    return record


class RoleVisualFieldMixin:
    """Compare all P/S/I allocations over two aligned real visual fields."""

    def _rvf_initialize(
            self, role_prompt_tta_visual_field_diagnosis=False,
            role_prompt_tta_visual_field_registry=None,
            role_prompt_tta_visual_field_mode='joint_role_view_final',
            role_prompt_tta_joint_profile_registry=None,
            role_prompt_tta_joint_final_registry=None,
            role_prompt_tta_final_profile='audit'):
        self.role_prompt_tta_visual_field_diagnosis = bool(
            role_prompt_tta_visual_field_diagnosis)
        self.role_prompt_tta_visual_field_registry = (
            role_prompt_tta_visual_field_registry)
        self.role_prompt_tta_visual_field_mode = str(
            role_prompt_tta_visual_field_mode)
        self._rvf_config = None
        self._jrv_config = None
        self._jrv_final_config = None
        self.role_prompt_tta_final_profile = str(
            role_prompt_tta_final_profile)
        self._rvf_tile_indices = {}
        if not self.role_prompt_tta_visual_field_diagnosis:
            return
        if not role_prompt_tta_visual_field_registry:
            raise ValueError(
                'Visual-field diagnosis requires its registry JSON.')
        if self.role_prompt_tta_visual_field_mode not in (
                'joint_role_view', 'joint_role_view_final'):
            raise ValueError(
                'role_prompt_tta_visual_field_mode must be '
                "'joint_role_view' or 'joint_role_view_final'.")
        if self._rpt_role_selection is None:
            raise ValueError(
                'Visual-field diagnosis requires the role-text selection.')
        self._rvf_config = load_visual_field_registry(
            role_prompt_tta_visual_field_registry,
            self.role_prompt_tta_dataset_name)
        if self.role_prompt_tta_visual_field_mode == 'joint_role_view':
            if not role_prompt_tta_joint_profile_registry:
                raise ValueError(
                    'Joint Role--View mode requires its profile registry.')
            self._jrv_config = load_joint_role_view_registry(
                role_prompt_tta_joint_profile_registry,
                self.role_prompt_tta_dataset_name)
        elif self.role_prompt_tta_visual_field_mode == 'joint_role_view_final':
            if not role_prompt_tta_joint_final_registry:
                raise ValueError(
                    'Final Joint Role--View mode requires its registry.')
            self._jrv_final_config = load_joint_role_view_final_registry(
                role_prompt_tta_joint_final_registry,
                self.role_prompt_tta_dataset_name)
            if (self.role_prompt_tta_final_profile != 'audit'
                    and self.role_prompt_tta_final_profile
                    not in self._jrv_final_config['profiles']):
                raise ValueError(
                    'Unknown final Joint Role--View profile: '
                    f'{self.role_prompt_tta_final_profile!r}.')

    def _uses_role_visual_field(self):
        return bool(getattr(
            self, 'role_prompt_tta_visual_field_diagnosis', False))

    @staticmethod
    def _rvf_grid_boxes(width, height, crop, stride=None):
        stride = int(crop if stride is None else stride)
        crop = int(crop)
        height_grids = max(height - crop + stride - 1, 0) // stride + 1
        width_grids = max(width - crop + stride - 1, 0) // stride + 1
        boxes = []
        for row in range(height_grids):
            for column in range(width_grids):
                y1, x1 = row * stride, column * stride
                y2, x2 = min(y1 + crop, height), min(x1 + crop, width)
                y1, x1 = max(y2 - crop, 0), max(x2 - crop, 0)
                box = (x1, y1, x2, y2)
                if box not in boxes:
                    boxes.append(box)
        return boxes

    @staticmethod
    def _rvf_context_box(target_box, size, image_size):
        x1, y1, x2, y2 = target_box
        width, height = image_size
        context_width = min(int(size), int(width))
        context_height = min(int(size), int(height))
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        left = min(max(int(round(center_x - context_width / 2.0)), 0),
                   width - context_width)
        top = min(max(int(round(center_y - context_height / 2.0)), 0),
                  height - context_height)
        return left, top, left + context_width, top + context_height

    def _rvf_ground_view(self, image, role_candidates=None):
        """Ground official aliases and only the requested Role candidates."""
        output_shape = (image.height, image.width)
        cache = self._rpt_prepare_text_cache()
        with torch.no_grad(), self._rpt_autocast_context():
            state = self.processor.set_image(image)
        requirements = {
            prompt: {'final', 'semantic', 'raw'}
            for prompt in self.query_words
        }
        slots = self._rpt_role_selection['_best_overall_slots']
        candidate_slots = [slots]
        if role_candidates is not None:
            candidate_slots = [
                tuple(value['slots']) for value in role_candidates]
        fields = (
            ('presence', 'presence_candidates'),
            ('semantic', 'semantic_candidates'),
            ('instance', 'instance_candidates'),
        )
        for item in self._rpt_prompt_bank['classes']:
            for requested in candidate_slots:
                for (role, field), slot in zip(fields, requested):
                    if slot:
                        requirements.setdefault(
                            item[field][slot - 1], set()).add(
                                'raw' if role == 'instance' else role)

        packages = {}
        query_finals = {}
        for prompt, needs in requirements.items():
            native_reference = None
            with torch.no_grad(), self._rpt_autocast_context():
                if ('final' in needs
                        and not self._rpt_native_parity_checked):
                    self.processor.reset_all_prompts(state)
                    self.processor.set_text_prompt(prompt, state)
                    native_reference = self._rpt_prompt_components(
                        state, output_shape)
                self._rpt_set_cached_prompt(state, prompt)
                presence = state['presence_score'].detach().float().mean()
                semantic = None
                if 'semantic' in needs or 'final' in needs:
                    semantic = state['semantic_mask_logits'].squeeze().float()
                    if semantic.shape != output_shape:
                        semantic = sam3_interpolate(
                            semantic.reshape(1, 1, *semantic.shape),
                            size=output_shape, mode='bilinear',
                            align_corners=False).squeeze().float()
                raw_masks = state.get('raw_masks_logits_lowres')
                raw_scores = state.get('raw_object_score')
                if 'raw' not in needs:
                    raw_masks = None
                    raw_scores = None
                if not isinstance(raw_masks, torch.Tensor):
                    raw_masks = torch.empty((0, 1, 1), dtype=torch.float32)
                if not isinstance(raw_scores, torch.Tensor):
                    raw_scores = torch.empty((0,), dtype=torch.float32)
                package = dict(
                    semantic=(
                        semantic.detach().float().cpu()
                        if semantic is not None else None),
                    presence=presence.detach().cpu(),
                    raw_masks=raw_masks.detach().cpu(),
                    raw_scores=raw_scores.detach().cpu(),
                )
                if 'final' in needs:
                    native = self._rpt_prompt_components(state, output_shape)
                    query_finals[prompt] = native['final'].detach().float().cpu()
                    if native_reference is not None:
                        error = max(
                            float((native['final'].float()
                                   - native_reference['final'].float()
                                   ).abs().max()),
                            float((native['semantic'].float()
                                   - native_reference['semantic'].float()
                                   ).abs().max()),
                            float((native['presence'].float()
                                   - native_reference['presence'].float()
                                   ).abs().max()),
                        )
                        self._rpt_native_parity_max_abs = error
                        self._rpt_native_parity_checked = True
                        if (self.role_prompt_tta_strict_integrity
                                and error
                                > self.role_prompt_tta_integrity_tolerance):
                            raise RuntimeError(
                                'Cached visual-field prompt differs from '
                                f'native SAM3: max_abs={error}.')
                packages[prompt] = package

        query_packages = [packages[prompt] for prompt in self.query_words]
        query_final = torch.stack([
            query_finals[prompt] for prompt in self.query_words])
        query_indices = self.query_idx.detach().long().cpu()
        classes = []
        for class_index, item in enumerate(self._rpt_prompt_bank['classes']):
            official_indices = torch.nonzero(
                query_indices == class_index, as_tuple=False
            ).flatten().tolist()
            official_packages = [
                query_packages[index] for index in official_indices]
            masks = [value['raw_masks'] for value in official_packages
                     if value['raw_masks'].numel()]
            scores = [value['raw_scores'] for value in official_packages
                      if value['raw_scores'].numel()]
            anchor = dict(
                semantic=torch.stack([
                    value['semantic'] for value in official_packages
                ]).max(dim=0)[0],
                presence=torch.stack([
                    value['presence'].reshape(())
                    for value in official_packages
                ]).max(),
                raw_masks=(torch.cat(masks) if masks else torch.empty(
                    (0, 1, 1), dtype=torch.float32)),
                raw_scores=(torch.cat(scores) if scores else torch.empty(
                    (0,), dtype=torch.float32)),
            )
            selected = {}
            selected_prompts = {}
            for (role, field), slot in zip(fields, slots):
                if slot:
                    prompt = item[field][slot - 1]
                    selected[role] = packages.get(prompt)
                    selected_prompts[role] = (
                        prompt if prompt in packages
                        else list(item['official_prompts']))
                else:
                    selected[role] = None
                    selected_prompts[role] = list(item['official_prompts'])
            role_packages = {}
            role_indices = dict(presence=0, semantic=1, instance=2)
            for role, field in fields:
                index = role_indices[role]
                requested = sorted({
                    int(value[index]) for value in candidate_slots
                    if int(value[index]) > 0
                })
                role_packages[role] = {
                    slot: packages[item[field][slot - 1]]
                    for slot in requested
                }
            classes.append(dict(
                anchor=anchor,
                selected=selected,
                selected_prompts=selected_prompts,
                role_packages=role_packages,
            ))
        del state
        return dict(
            shape=output_shape,
            query_final=query_final,
            classes=classes,
            cache_prompt_count=len(cache['prompts']),
            grounding_prompt_count=len(requirements),
        )

    @staticmethod
    def _rvf_crop(value, roi):
        x1, y1, x2, y2 = roi
        return torch.as_tensor(value).float()[..., y1:y2, x1:x2]

    def _rvf_instance(
            self, package, amplitude_presence, admission_presence,
            view_shape, roi, chunk_size=8):
        """Rebuild one instance map in bounded chunks, then crop target ROI."""
        target_height = int(roi[3] - roi[1])
        target_width = int(roi[2] - roi[0])
        empty = torch.zeros((target_height, target_width), dtype=torch.float32)
        if not self.use_transformer_decoder:
            return empty, 0
        count = min(
            int(package['raw_masks'].shape[0]),
            int(package['raw_scores'].numel()))
        if count == 0:
            return empty, 0
        scores = package['raw_scores'][:count].to(self.device).float()
        admission = torch.as_tensor(
            admission_presence, device=self.device).float().reshape(())
        amplitude = torch.as_tensor(
            amplitude_presence, device=self.device).float().reshape(())
        keep = scores * admission > float(self.processor.confidence_threshold)
        selected = torch.nonzero(keep, as_tuple=False).flatten()
        if not selected.numel():
            return empty, 0
        maximum = None
        with torch.no_grad(), self._rpt_autocast_context():
            for start in range(0, int(selected.numel()), int(chunk_size)):
                selected_chunk = selected[start:start + int(chunk_size)]
                masks = package['raw_masks'][
                    selected_chunk.cpu()].to(self.device)
                masks = sam3_interpolate(
                    masks.unsqueeze(1), size=view_shape, mode='bilinear',
                    align_corners=False).sigmoid().squeeze(1).float()
                if self.instance_score_type == 'raw':
                    weights = scores[selected_chunk]
                else:
                    weights = scores[selected_chunk] * amplitude
                chunk = (masks * weights[:, None, None]).max(dim=0)[0]
                maximum = chunk if maximum is None else torch.maximum(
                    maximum, chunk)
        return self._rvf_crop(maximum.detach().cpu(), roi), int(
            selected.numel())

    def _rvf_compose_class(self, fine, context, class_index, composition,
                           family, fine_roi, context_roi, instance_cache,
                           slots=None, admission_mode=None,
                           role_update='residual'):
        views = {'F': (fine, fine_roi), 'C': (context, context_roi)}
        p_view, s_view, i_view = composition
        p_data, _ = views[p_view]
        s_data, s_roi = views[s_view]
        i_data, _ = views[i_view]
        p_class = p_data['classes'][class_index]
        s_class = s_data['classes'][class_index]
        i_class = i_data['classes'][class_index]
        slots = tuple(
            self._rpt_role_selection['_best_overall_slots']
            if slots is None else slots)
        admission_mode = (
            self._rpt_role_selection['best_overall']['admission']
            if admission_mode is None else str(admission_mode))
        alpha, clip = 0.5, 0.25
        if role_update not in ('residual', 'direct'):
            raise ValueError(f'Unknown role update: {role_update!r}.')

        presence = torch.as_tensor(
            p_class['anchor']['presence']).float().reshape(())
        presence_package = p_class['role_packages']['presence'].get(slots[0])
        if family == 'text' and presence_package is not None:
            candidate_presence = torch.as_tensor(
                presence_package['presence']).float().reshape(())
            if role_update == 'direct':
                presence = candidate_presence
            else:
                presence, _ = self._rft_bounded(
                    presence, candidate_presence, alpha, clip)

        semantic = self._rvf_crop(s_class['anchor']['semantic'], s_roi)
        semantic_package = s_class['role_packages']['semantic'].get(slots[1])
        if family == 'text' and semantic_package is not None:
            candidate = self._rvf_crop(
                semantic_package['semantic'], s_roi)
            if role_update == 'direct':
                semantic = candidate
            else:
                semantic, _ = self._rft_bounded(
                    semantic, candidate, alpha, clip)

        admission = presence
        if (family == 'text'
                and admission_mode == 'anchor_admission'):
            admission = torch.as_tensor(
                i_class['anchor']['presence']).float().reshape(())

        def instance(package, source):
            key = (
                source, id(package), float(presence.item()),
                float(torch.as_tensor(admission).item()))
            if key not in instance_cache:
                data, roi = views[source]
                instance_cache[key] = self._rvf_instance(
                    package, presence, admission, data['shape'], roi)[0]
            return instance_cache[key]

        anchor_instance = instance(i_class['anchor'], i_view)
        instance_map = anchor_instance
        instance_package = i_class['role_packages']['instance'].get(slots[2])
        if family == 'text' and instance_package is not None:
            candidate = instance(instance_package, i_view)
            if role_update == 'direct':
                instance_map = candidate
            else:
                instance_map, _ = self._rft_bounded(
                    anchor_instance, candidate, alpha, clip)
        return self._rft_role_final(
            semantic, instance_map, presence, semantic.shape)

    def _rvf_visual_units(self, image, image_path):
        fine_size = self._rvf_config['fine_size']
        context_size = self._rvf_config['context_size']
        if self._rvf_config['source_mode'] == 'coordinate_tiles':
            directory = os.path.dirname(os.path.abspath(image_path))
            if directory not in self._rvf_tile_indices:
                self._rvf_tile_indices[directory] = CoordinateTileIndex(
                    directory)
            context, context_roi, metadata = self._rvf_tile_indices[
                directory].read_context(image_path, context_size)
            return [dict(
                target_box=(0, 0, image.width, image.height),
                fine=image,
                fine_roi=(0, 0, image.width, image.height),
                context=context,
                context_roi=context_roi,
                metadata=metadata,
            )]

        units = []
        for target_box in self._rvf_grid_boxes(
                image.width, image.height, fine_size):
            context_box = self._rvf_context_box(
                target_box, context_size, image.size)
            cx1, cy1, _, _ = context_box
            x1, y1, x2, y2 = target_box
            context_roi = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
            units.append(dict(
                target_box=target_box,
                fine=image.crop(target_box),
                fine_roi=(0, 0, x2 - x1, y2 - y1),
                context=image.crop(context_box),
                context_roi=context_roi,
                metadata=dict(
                    source_prefix=os.path.basename(image_path),
                    source_canvas=[image.width, image.height],
                    target_box=list(target_box),
                    context_box=list(context_box),
                    context_size=[
                        context_box[2] - context_box[0],
                        context_box[3] - context_box[1]],
                    contributor_tiles=1,
                ),
            ))
        return units

    def _rvf_fine_matches_official(self, image):
        if self._rvf_config['source_mode'] == 'coordinate_tiles':
            return (image.width <= self._rvf_config['fine_size']
                    and image.height <= self._rvf_config['fine_size'])
        use_sliding = (
            self.slide_crop > 0
            and (self.slide_crop < image.width
                 or self.slide_crop < image.height))
        if use_sliding:
            return (int(self.slide_crop) == self._rvf_config['fine_size']
                    and int(self.slide_stride) == self._rvf_config['fine_size'])
        return (image.width <= self._rvf_config['fine_size']
                and image.height <= self._rvf_config['fine_size'])

    def _jrv_global_source(self, image):
        """Choose the complementary observation, not a dataset label rule."""
        if self._rvf_config['source_mode'] == 'coordinate_tiles':
            return 'aligned_context'
        use_sliding = (
            self.slide_crop > 0
            and (self.slide_crop < image.width
                 or self.slide_crop < image.height))
        if use_sliding:
            return 'aligned_context'
        if (image.width > self._rvf_config['fine_size']
                or image.height > self._rvf_config['fine_size']):
            return 'full_image'
        return 'aligned_context'

    def _jrv_ground_target(
            self, view, roi, candidates, include_direct=False,
            include_residual=True):
        """Compose every retained Role candidate on one coherent view."""
        official = self._rvf_crop(
            self._aggregate_query_logits_to_classes(view['query_final']), roi)
        cache = {}
        anchor = None
        if include_residual:
            anchor = torch.stack([
                self._rvf_compose_class(
                    view, view, class_index, 'FFF', 'anchor', roi, roi, cache,
                    slots=(0, 0, 0), admission_mode='native')
                for class_index in range(self.num_cls)
            ])
        role_maps = OrderedDict()
        direct_maps = OrderedDict()
        for candidate in candidates:
            if include_residual:
                role_value = torch.stack([
                    self._rvf_compose_class(
                        view, view, class_index, 'FFF', 'text', roi, roi,
                        cache, slots=candidate['slots'],
                        admission_mode=candidate['admission'])
                    for class_index in range(self.num_cls)
                ])
                role_maps[candidate['id']] = (
                    official + role_value - anchor).clamp(0.0, 1.0)
            if include_direct:
                direct_maps[candidate['id']] = torch.stack([
                    self._rvf_compose_class(
                        view, view, class_index, 'FFF', 'text', roi, roi,
                        cache, slots=candidate['slots'],
                        admission_mode=candidate['admission'],
                        role_update='direct')
                    for class_index in range(self.num_cls)
                ])
        if include_direct:
            return official, role_maps, direct_maps
        return official, role_maps

    def _jrv_accumulate_visual_units(
            self, image, image_path, need_context, candidates,
            include_direct=False, include_residual=True):
        """Accumulate all retained Role candidates for Local/Context views."""
        height, width = image.height, image.width
        counts = torch.zeros((1, height, width), dtype=torch.float32)
        local_query_sum = torch.zeros(
            (self.num_queries, height, width), dtype=torch.float32)
        local_official_sum = torch.zeros(
            (self.num_cls, height, width), dtype=torch.float32)
        local_role_sums = (
            OrderedDict(
                (value['id'], torch.zeros_like(local_official_sum))
                for value in candidates)
            if include_residual else None)
        local_direct_sums = (
            OrderedDict(
                (value['id'], torch.zeros_like(local_official_sum))
                for value in candidates)
            if include_direct else None)
        context_query_sum = (
            torch.zeros_like(local_query_sum) if need_context else None)
        context_official_sum = (
            torch.zeros_like(local_official_sum) if need_context else None)
        context_role_sums = (
            OrderedDict(
                (value['id'], torch.zeros_like(local_official_sum))
                for value in candidates)
            if need_context and include_residual else None)
        context_direct_sums = (
            OrderedDict(
                (value['id'], torch.zeros_like(local_official_sum))
                for value in candidates)
            if need_context and include_direct else None)
        image_encoder_calls = 0
        grounding_calls = 0
        unit_stats, context_cache = [], {}
        for unit_index, unit in enumerate(
                self._rvf_visual_units(image, image_path)):
            x1, y1, x2, y2 = unit['target_box']
            fine = self._rvf_ground_view(unit['fine'], candidates)
            grounded = self._jrv_ground_target(
                fine, unit['fine_roi'], candidates,
                include_direct=include_direct,
                include_residual=include_residual)
            local_official, local_roles = grounded[:2]
            local_direct = grounded[2] if include_direct else None
            image_encoder_calls += 1
            grounding_calls += int(fine['grounding_prompt_count'])
            local_query_sum[:, y1:y2, x1:x2] += self._rvf_crop(
                fine['query_final'], unit['fine_roi'])
            local_official_sum[:, y1:y2, x1:x2] += local_official
            if include_residual:
                for identifier, value in local_roles.items():
                    local_role_sums[identifier][:, y1:y2, x1:x2] += value
            if include_direct:
                for identifier, value in local_direct.items():
                    local_direct_sums[identifier][:, y1:y2, x1:x2] += value
            counts[:, y1:y2, x1:x2] += 1.0

            if need_context:
                context_key = (
                    unit['metadata']['source_prefix'],
                    tuple(unit['metadata']['source_canvas']),
                    tuple(unit['metadata']['context_box']),
                )
                if context_key not in context_cache:
                    context_cache[context_key] = self._rvf_ground_view(
                        unit['context'], candidates)
                context = context_cache[context_key]
                grounded = self._jrv_ground_target(
                    context, unit['context_roi'], candidates,
                    include_direct=include_direct,
                    include_residual=include_residual)
                context_official, context_roles = grounded[:2]
                context_direct = grounded[2] if include_direct else None
                context_query_sum[:, y1:y2, x1:x2] += self._rvf_crop(
                    context['query_final'], unit['context_roi'])
                context_official_sum[:, y1:y2, x1:x2] += context_official
                if include_residual:
                    for identifier, value in context_roles.items():
                        context_role_sums[identifier][
                            :, y1:y2, x1:x2] += value
                if include_direct:
                    for identifier, value in context_direct.items():
                        context_direct_sums[identifier][:, y1:y2, x1:x2] += value

            unit_stats.append(dict(
                unit_index=int(unit_index),
                target_box=list(unit['target_box']),
                local_size=[unit['fine'].width, unit['fine'].height],
                context_size=[unit['context'].width, unit['context'].height],
                context_roi=list(unit['context_roi']),
                source_prefix=unit['metadata']['source_prefix'],
                source_canvas=unit['metadata']['source_canvas'],
                context_box=unit['metadata']['context_box'],
                contributor_tiles=unit['metadata']['contributor_tiles'],
            ))
            del fine, local_roles

        if torch.any(counts == 0):
            raise RuntimeError('Joint Role--View units left uncovered pixels.')

        def finalize(query_sum, official_sum, role_sums, direct_sums=None):
            query = query_sum / counts
            exact = self._aggregate_query_logits_to_classes(query)
            unit_official = official_sum / counts
            roles = (
                OrderedDict(
                    (identifier, (
                        exact + value / counts - unit_official
                    ).clamp(0.0, 1.0))
                    for identifier, value in role_sums.items())
                if role_sums is not None else None)
            directs = (
                OrderedDict(
                    (identifier, value / counts)
                    for identifier, value in direct_sums.items())
                if direct_sums is not None else None)
            return query, exact, roles, directs

        local_query, local_anchor, local_roles, local_direct = finalize(
            local_query_sum, local_official_sum, local_role_sums,
            local_direct_sums)
        result = dict(
            local_query=local_query,
            local_anchor=local_anchor,
            local_roles=local_roles,
            local_direct=local_direct,
            context_anchor=None,
            context_roles=None,
            context_direct=None,
            units=unit_stats,
            unique_context_views=len(context_cache),
            image_encoder_calls=image_encoder_calls + len(context_cache),
            grounding_calls=grounding_calls + sum(
                int(value['grounding_prompt_count'])
                for value in context_cache.values()),
        )
        if need_context:
            _, context_anchor, context_roles, context_direct = finalize(
                context_query_sum, context_official_sum, context_role_sums,
                context_direct_sums)
            result.update(
                context_anchor=context_anchor,
                context_roles=context_roles,
                context_direct=context_direct,
            )
        return result

    def _jrv_accumulate_official_units(
            self, image, candidates, include_direct=False,
            include_residual=True):
        """Reproduce the official observation while retaining Role candidates."""
        use_sliding = (
            self.slide_crop > 0
            and (self.slide_crop < image.width
                 or self.slide_crop < image.height))
        boxes = (self._rvf_grid_boxes(
            image.width, image.height, self.slide_crop, self.slide_stride)
            if use_sliding else [(0, 0, image.width, image.height)])
        height, width = image.height, image.width
        counts = torch.zeros((1, height, width), dtype=torch.float32)
        query_sum = torch.zeros(
            (self.num_queries, height, width), dtype=torch.float32)
        official_sum = torch.zeros(
            (self.num_cls, height, width), dtype=torch.float32)
        role_sums = (
            OrderedDict(
                (value['id'], torch.zeros_like(official_sum))
                for value in candidates)
            if include_residual else None)
        direct_sums = (
            OrderedDict(
                (value['id'], torch.zeros_like(official_sum))
                for value in candidates)
            if include_direct else None)
        image_encoder_calls = 0
        grounding_calls = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            view = self._rvf_ground_view(image.crop(box), candidates)
            roi = (0, 0, x2 - x1, y2 - y1)
            grounded = self._jrv_ground_target(
                view, roi, candidates, include_direct=include_direct,
                include_residual=include_residual)
            official, roles = grounded[:2]
            directs = grounded[2] if include_direct else None
            image_encoder_calls += 1
            grounding_calls += int(view['grounding_prompt_count'])
            query_sum[:, y1:y2, x1:x2] += view['query_final']
            official_sum[:, y1:y2, x1:x2] += official
            if include_residual:
                for identifier, value in roles.items():
                    role_sums[identifier][:, y1:y2, x1:x2] += value
            if include_direct:
                for identifier, value in directs.items():
                    direct_sums[identifier][:, y1:y2, x1:x2] += value
            counts[:, y1:y2, x1:x2] += 1.0
            del view, roles
        query = query_sum / counts
        exact = self._aggregate_query_logits_to_classes(query)
        unit_official = official_sum / counts
        roles = (
            OrderedDict(
                (identifier, (
                    exact + value / counts - unit_official
                ).clamp(0.0, 1.0))
                for identifier, value in role_sums.items())
            if role_sums is not None else None)
        directs = (
            OrderedDict(
                (identifier, value / counts)
                for identifier, value in direct_sums.items())
            if include_direct else None)
        return query, exact, roles, directs, dict(
            image_encoder_calls=image_encoder_calls,
            grounding_calls=grounding_calls)

    @staticmethod
    def _jrv_fuse_views(local, global_value, family, global_weight, rho):
        """One common operator family spanning endpoints, means and max."""
        local = torch.as_tensor(local).float().clamp(0.0, 1.0)
        global_value = torch.as_tensor(global_value).float().clamp(0.0, 1.0)
        weight = float(global_weight)
        if family == 'endpoint':
            return global_value if weight >= 0.5 else local
        if family == 'max':
            return torch.maximum(local, global_value)
        if family == 'power':
            exponent = float(rho)
            return (
                (1.0 - weight) * local.pow(exponent)
                + weight * global_value.pow(exponent)
            ).clamp_min(0.0).pow(1.0 / exponent).clamp(0.0, 1.0)
        if family == 'logit':
            epsilon = 1e-6
            local_safe = local.clamp(epsilon, 1.0 - epsilon)
            global_safe = global_value.clamp(epsilon, 1.0 - epsilon)
            local_logit = torch.log(local_safe / (1.0 - local_safe))
            global_logit = torch.log(global_safe / (1.0 - global_safe))
            return torch.sigmoid(
                (1.0 - weight) * local_logit + weight * global_logit)
        raise ValueError(f'Unknown Joint Role--View family: {family!r}.')

    @staticmethod
    def _jrv_operator_spec(name):
        if name == 'reference':
            return None
        for operator, family, weight, rho in JRV_OPERATOR_SPECS:
            if operator == name:
                return family, float(weight), rho
        raise ValueError(f'Unknown Joint Role--View operator: {name!r}.')

    def _jrv_final_fuse(self, endpoints, operator, reference_endpoint):
        if operator == 'reference':
            key = 'local' if reference_endpoint == 'local' else 'global_value'
            return endpoints[key]
        family, weight, rho = self._jrv_operator_spec(operator)
        return self._jrv_fuse_views(
            endpoints['local'], endpoints['global_value'],
            family, weight, rho)

    def _jrv_final_candidates(self):
        candidates = self._jrv_final_config['role_candidates']
        if self.role_prompt_tta_final_profile == 'audit':
            return candidates
        profile = self._jrv_final_config['profiles'][
            self.role_prompt_tta_final_profile]
        required = {'anchor', profile['candidate']}
        return tuple(
            value for value in candidates if value['id'] in required)

    def _jrv_final_predict_image(self, image, image_path):
        """Run the shared audit bank or one compiled deployment profile."""
        audit = self.role_prompt_tta_final_profile == 'audit'
        candidates = self._jrv_final_candidates()
        profiles = self._jrv_final_config['profiles']
        active_profiles = (
            profiles if audit else {
                self.role_prompt_tta_final_profile:
                profiles[self.role_prompt_tta_final_profile]})
        include_residual = audit or any(
            value['update'] == 'residual' and value['candidate'] != 'anchor'
            for value in active_profiles.values())
        include_direct = audit or any(
            value['update'] == 'direct' and value['candidate'] != 'anchor'
            for value in active_profiles.values())
        role_only_deploy = (
            not audit
            and next(iter(active_profiles.values()))['operator'] == 'reference')

        if role_only_deploy:
            result = self._jrv_accumulate_official_units(
                image, candidates, include_direct=include_direct,
                include_residual=include_residual)
            (official_query, official, residual_roles, direct_roles,
             cost) = result
            reference_endpoint = 'global'
            global_source = 'official_observation'
            units = []
            unique_context_views = 0
            anchor_endpoints = dict(local=official, global_value=official)
            residual_endpoints = OrderedDict(
                (identifier, dict(local=value, global_value=value))
                for identifier, value in (residual_roles or {}).items())
            direct_endpoints = OrderedDict(
                (identifier, dict(local=value, global_value=value))
                for identifier, value in (direct_roles or {}).items())
            identity_error = 0.0
        else:
            global_source = self._jrv_global_source(image)
            unit_values = self._jrv_accumulate_visual_units(
                image, image_path,
                need_context=(global_source == 'aligned_context'),
                candidates=candidates, include_direct=include_direct,
                include_residual=include_residual)
            fine_matches = self._rvf_fine_matches_official(image)
            cost = dict(
                image_encoder_calls=int(
                    unit_values['image_encoder_calls']),
                grounding_calls=int(unit_values['grounding_calls']))
            if fine_matches:
                official_query = unit_values['local_query']
                official = unit_values['local_anchor']
                official_residual = unit_values['local_roles']
                official_direct = unit_values['local_direct']
            else:
                result = self._jrv_accumulate_official_units(
                    image, candidates, include_direct=include_direct,
                    include_residual=include_residual)
                (official_query, official, official_residual,
                 official_direct, official_cost) = result
                for key in cost:
                    cost[key] += int(official_cost[key])

            if global_source == 'full_image':
                global_anchor = official
                global_residual = official_residual
                global_direct = official_direct
            else:
                global_anchor = unit_values['context_anchor']
                global_residual = unit_values['context_roles']
                global_direct = unit_values['context_direct']
            anchor_endpoints = dict(
                local=unit_values['local_anchor'],
                global_value=global_anchor)
            residual_endpoints = OrderedDict(
                (value['id'], dict(
                    local=unit_values['local_roles'][value['id']],
                    global_value=global_residual[value['id']]))
                for value in candidates) if include_residual else OrderedDict()
            direct_endpoints = OrderedDict(
                (value['id'], dict(
                    local=unit_values['local_direct'][value['id']],
                    global_value=global_direct[value['id']]))
                for value in candidates) if include_direct else OrderedDict()
            reference_endpoint = 'local' if fine_matches else 'global'
            reference = anchor_endpoints[
                'local' if reference_endpoint == 'local' else 'global_value']
            identity_error = float((reference - official).abs().max())
            units = unit_values['units']
            unique_context_views = int(
                unit_values['unique_context_views'])

        if (self.role_prompt_tta_strict_integrity
                and identity_error > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                'Final Joint Role--View reference identity failed: '
                f'{identity_error}.')

        profile_scores = OrderedDict()
        for name, profile in active_profiles.items():
            identifier = profile['candidate']
            if identifier == 'anchor':
                endpoints = anchor_endpoints
            elif profile['update'] == 'direct':
                endpoints = direct_endpoints[identifier]
            else:
                endpoints = residual_endpoints[identifier]
            profile_scores[name] = self._jrv_final_fuse(
                endpoints, profile['operator'], reference_endpoint)

        selected_profile = (
            self._jrv_final_config['primary_profile'] if audit
            else self.role_prompt_tta_final_profile)
        self._jrv_final_official_query = official_query.detach().float().cpu()
        bundle = dict(
            joint_role_view_final=True,
            profiles=profile_scores,
            selected_profile=selected_profile,
        )
        cost.update(
            candidate_count=len(candidates),
            profile_count=len(active_profiles),
            compiled=not audit,
            unique_cached_text_prompts=len(
                self._rpt_prepare_text_cache()['prompts']),
        )
        self._inference_profile_cost = dict(cost)
        metadata = dict(
            schema_version=JRV_FINAL_SCHEMA_VERSION,
            protocol=JRV_FINAL_PROTOCOL,
            execution='audit' if audit else 'compiled',
            selected_profile=selected_profile,
            profiles={
                name: dict(value) for name, value in active_profiles.items()},
            role_candidates=[dict(
                id=value['id'], slots=list(value['slots']),
                admission=value['admission']) for value in candidates],
            global_source=global_source,
            reference_endpoint=reference_endpoint,
            local_size=int(self._rvf_config['fine_size']),
            context_size=int(self._rvf_config['context_size']),
            source_mode=self._rvf_config['source_mode'],
            reference_identity_max_abs=identity_error,
            native_prompt_parity_max_abs=float(
                self._rpt_native_parity_max_abs or 0.0),
            cost=dict(cost),
            units=units,
            unique_context_views=unique_context_views,
        )
        return official_query.to(self.device), bundle, metadata

    def _jrv_predict_image(self, image, image_path):
        """Build the shared Local/Global evidence bank for joint profiling."""
        candidates = self._jrv_config['role_candidates']
        global_source = self._jrv_global_source(image)
        unit_values = self._jrv_accumulate_visual_units(
            image, image_path, need_context=(global_source == 'aligned_context'),
            candidates=candidates)
        fine_matches = self._rvf_fine_matches_official(image)
        if fine_matches:
            official_query = unit_values['local_query']
            official = unit_values['local_anchor']
            official_roles = unit_values['local_roles']
        else:
            official_result = self._jrv_accumulate_official_units(
                image, candidates)
            official_query, official, official_roles = official_result[:3]

        if global_source == 'full_image':
            global_anchor = official
            global_roles = official_roles
        else:
            global_anchor = unit_values['context_anchor']
            global_roles = unit_values['context_roles']
        bundle = dict(
            joint_role_view=True,
            anchor=dict(
                local=unit_values['local_anchor'],
                global_value=global_anchor),
            roles=OrderedDict(
                (candidate['id'], dict(
                    local=unit_values['local_roles'][candidate['id']],
                    global_value=global_roles[candidate['id']]))
                for candidate in candidates),
        )
        reference_endpoint = 'local' if fine_matches else 'global'
        reference_anchor = bundle['anchor'][
            'local' if reference_endpoint == 'local' else 'global_value']
        identity_error = float((reference_anchor - official).abs().max())
        if (self.role_prompt_tta_strict_integrity
                and identity_error > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                f'Joint Role--View reference identity failed: {identity_error}.')
        metadata = dict(
            schema_version=JRV_SCHEMA_VERSION,
            protocol=JRV_PROTOCOL,
            global_source=global_source,
            reference_endpoint=reference_endpoint,
            operator_specs=[dict(
                name=name, family=family, global_weight=float(weight),
                rho=(None if rho is None else float(rho)))
                for name, family, weight, rho in JRV_OPERATOR_SPECS],
            role_candidates=[dict(
                id=value['id'], slots=list(value['slots']),
                admission=value['admission'],
                source_miou=float(value['source_miou']))
                for value in candidates],
            anchor_candidate=self._jrv_config['anchor_candidate'],
            current_role_candidate=self._jrv_config[
                'current_role_candidate'],
            prior_view_operator=self._jrv_config['prior_view_operator'],
            prior_view_miou=float(self._jrv_config['prior_view_miou']),
            local_size=int(self._rvf_config['fine_size']),
            context_size=int(self._rvf_config['context_size']),
            source_mode=self._rvf_config['source_mode'],
            reference_identity_max_abs=identity_error,
            native_prompt_parity_max_abs=float(
                self._rpt_native_parity_max_abs or 0.0),
            units=unit_values['units'],
            unique_context_views=int(unit_values['unique_context_views']),
        )
        return official_query.to(self.device), bundle, metadata

    def _rvf_predict_image(self, image, image_path):
        if self.role_prompt_tta_visual_field_mode == 'joint_role_view_final':
            return self._jrv_final_predict_image(image, image_path)
        if self.role_prompt_tta_visual_field_mode == 'joint_role_view':
            return self._jrv_predict_image(image, image_path)
        raise RuntimeError('Unsupported Joint Role--View mode.')

    def _rvf_record_image(
            self, variants, metadata, data_sample, image_path):
        if self.role_prompt_tta_visual_field_mode == 'joint_role_view_final':
            return self._jrv_final_record_image(
                variants, metadata, data_sample, image_path)
        if self.role_prompt_tta_visual_field_mode == 'joint_role_view':
            return self._jrv_record_image(
                variants, metadata, data_sample, image_path)
        raise RuntimeError('Unsupported Joint Role--View mode.')

    def _rvf_resize_variants(variants, output_shape):
        """Resize flat experiment maps or the compact Joint endpoint bank."""
        if variants.get('joint_role_view_final', False):
            return dict(
                joint_role_view_final=True,
                selected_profile=variants['selected_profile'],
                profiles={
                    name: (
                        value if tuple(value.shape[-2:]) == tuple(output_shape)
                        else F.interpolate(
                            value.unsqueeze(0), size=output_shape,
                            mode='bilinear',
                            align_corners=False).squeeze(0))
                    for name, value in variants['profiles'].items()
                },
            )
        if not variants.get('joint_role_view', False):
            return {
                name: F.interpolate(
                    value.unsqueeze(0), size=output_shape,
                    mode='bilinear', align_corners=False).squeeze(0)
                for name, value in variants.items()
            }

        def resize(value):
            if tuple(value.shape[-2:]) == tuple(output_shape):
                return value
            return F.interpolate(
                value.unsqueeze(0), size=output_shape,
                mode='bilinear', align_corners=False).squeeze(0)

        return dict(
            joint_role_view=True,
            anchor={
                name: resize(value)
                for name, value in variants['anchor'].items()
            },
            roles=OrderedDict(
                (identifier, {
                    name: resize(value) for name, value in endpoints.items()
                })
                for identifier, endpoints in variants['roles'].items()),
        )

    def _jrv_final_record_image(
            self, bundle, metadata, data_sample, image_path):
        """Record final profiles and their exact changes from official."""
        if not self.dump_role_prompt_tta_stats:
            return
        gt = data_sample.gt_sem_seg.data.squeeze().detach().long().cpu()
        valid = gt != 255
        official_query = self._aggregate_query_logits_to_classes(
            self._jrv_final_official_query).detach().float().cpu()
        if official_query.shape[-2:] != gt.shape[-2:]:
            official_query = F.interpolate(
                official_query.unsqueeze(0), size=gt.shape[-2:],
                mode='bilinear', align_corners=False).squeeze(0)
        official_prediction = self._rpt_threshold(
            official_query).to(torch.int16)
        rows = OrderedDict()
        candidates = {
            value['id']: value for value in metadata['role_candidates']}
        for name, score in bundle['profiles'].items():
            prediction = self._rpt_threshold(score).to(torch.int16)
            changed = valid & (prediction != official_prediction)
            improved = changed & (prediction == gt) & (
                official_prediction != gt)
            harmed = changed & (prediction != gt) & (
                official_prediction == gt)
            official_fg = official_prediction != self.bg_idx
            prediction_fg = prediction != self.bg_idx
            rows[name] = dict(
                **metadata['profiles'][name],
                slots=list(candidates[
                    metadata['profiles'][name]['candidate']]['slots']),
                admission=candidates[
                    metadata['profiles'][name]['candidate']]['admission'],
                confusion=self._rpt_confusion(prediction, gt, valid),
                changed_pixels=int(changed.sum()),
                improved_pixels=int(improved.sum()),
                harmed_pixels=int(harmed.sum()),
                help_minus_harm=int(improved.sum()) - int(harmed.sum()),
                foreground_added_pixels=int((
                    valid & ~official_fg & prediction_fg).sum()),
                foreground_removed_pixels=int((
                    valid & official_fg & ~prediction_fg).sum()),
                foreground_class_switch_pixels=int((
                    valid & official_fg & prediction_fg
                    & (official_prediction != prediction)).sum()),
            )
        official_row = dict(
            candidate='anchor', update='residual', operator='reference',
            confusion=self._rpt_confusion(
                official_prediction, gt, valid),
            changed_pixels=0, improved_pixels=0, harmed_pixels=0,
            help_minus_harm=0, foreground_added_pixels=0,
            foreground_removed_pixels=0, foreground_class_switch_pixels=0,
        )
        pairwise = {}
        for stem in ('joint', 'fast_joint', 'role_only'):
            residual = f'{stem}_residual'
            direct = f'{stem}_direct'
            if residual not in bundle['profiles'] or direct not in bundle['profiles']:
                continue
            residual_prediction = self._rpt_threshold(
                bundle['profiles'][residual]).to(torch.int16)
            direct_prediction = self._rpt_threshold(
                bundle['profiles'][direct]).to(torch.int16)
            disagreement = valid & (
                residual_prediction != direct_prediction)
            pairwise[f'{residual}_vs_{direct}'] = dict(
                disagreement_pixels=int(disagreement.sum()),
                residual_only_correct=int((
                    disagreement & (residual_prediction == gt)
                    & (direct_prediction != gt)).sum()),
                direct_only_correct=int((
                    disagreement & (direct_prediction == gt)
                    & (residual_prediction != gt)).sum()),
            )
        record = dict(
            schema_version=JRV_FINAL_SCHEMA_VERSION,
            rank=int(os.environ.get('RANK', 0)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            dataset_name=self.role_prompt_tta_dataset_name,
            img_path=image_path,
            class_names=list(self.class_names),
            valid_pixels=int(valid.sum()),
            prob_thd=float(self.prob_thd),
            joint_role_view_final=dict(
                metadata, official=official_row,
                profiles=rows, residual_direct_pairwise=pairwise),
        )
        self._rpt_write_stats(record)

    def _jrv_record_image(
            self, bundle, metadata, data_sample, image_path):
        """Evaluate all Joint profiles without retaining every fused score map."""
        if not self.dump_role_prompt_tta_stats:
            return
        gt = data_sample.gt_sem_seg.data.squeeze().detach().long().cpu()
        valid = gt != 255
        reference_key = (
            'local' if metadata['reference_endpoint'] == 'local'
            else 'global_value')
        official_score = bundle['anchor'][reference_key]
        official_prediction = self._rpt_threshold(
            official_score).to(torch.int16)

        def transition(prediction):
            changed = valid & (prediction != official_prediction)
            improved = changed & (prediction == gt) & (
                official_prediction != gt)
            harmed = changed & (prediction != gt) & (
                official_prediction == gt)
            return dict(
                changed_pixels=int(changed.sum()),
                improved_pixels=int(improved.sum()),
                harmed_pixels=int(harmed.sum()),
                help_minus_harm=int(improved.sum()) - int(harmed.sum()),
            )

        rows = OrderedDict()
        rows['jrv_official'] = dict(
            candidate_id=metadata['anchor_candidate'],
            slots=[0, 0, 0],
            admission='native',
            operator=metadata['reference_endpoint'],
            operator_family='reference',
            global_weight=(
                0.0 if metadata['reference_endpoint'] == 'local' else 1.0),
            rho=None,
            confusion=self._rpt_confusion(
                official_prediction, gt, valid),
            **transition(official_prediction),
        )
        candidate_lookup = {
            value['id']: value for value in metadata['role_candidates']}

        def add_family(prefix, candidate, endpoints):
            for operator in metadata['operator_specs']:
                score = self._jrv_fuse_views(
                    endpoints['local'], endpoints['global_value'],
                    operator['family'], operator['global_weight'],
                    operator['rho'])
                prediction = self._rpt_threshold(score).to(torch.int16)
                name = f'jrv_{prefix}__{operator["name"]}'
                rows[name] = dict(
                    candidate_id=candidate['id'],
                    slots=list(candidate['slots']),
                    admission=candidate['admission'],
                    operator=operator['name'],
                    operator_family=operator['family'],
                    global_weight=float(operator['global_weight']),
                    rho=operator['rho'],
                    confusion=self._rpt_confusion(prediction, gt, valid),
                    **transition(prediction),
                )

        anchor_id = metadata['anchor_candidate']
        add_family('anchor', candidate_lookup[anchor_id], bundle['anchor'])
        for identifier, endpoints in bundle['roles'].items():
            if identifier == anchor_id:
                continue
            add_family(identifier, candidate_lookup[identifier], endpoints)

        complementarity = OrderedDict()
        endpoint_families = OrderedDict(anchor=bundle['anchor'])
        endpoint_families.update(
            (identifier, endpoints)
            for identifier, endpoints in bundle['roles'].items()
            if identifier != anchor_id)
        for identifier, endpoints in endpoint_families.items():
            local_prediction = self._rpt_threshold(
                endpoints['local']).to(torch.int16)
            global_prediction = self._rpt_threshold(
                endpoints['global_value']).to(torch.int16)
            local_correct = valid & (local_prediction == gt)
            global_correct = valid & (global_prediction == gt)
            local_only = local_correct & ~global_correct
            global_only = global_correct & ~local_correct
            oracle = local_prediction.clone()
            oracle[global_only] = global_prediction[global_only]
            complementarity[identifier] = dict(
                local_only_correct_pixels=int(local_only.sum()),
                global_only_correct_pixels=int(global_only.sum()),
                disagreement_pixels=int((
                    valid & (local_prediction != global_prediction)).sum()),
                oracle_confusion=self._rpt_confusion(oracle, gt, valid),
            )

        record = dict(
            schema_version=JRV_SCHEMA_VERSION,
            rank=int(os.environ.get('RANK', 0)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            dataset_name=self.role_prompt_tta_dataset_name,
            img_path=image_path,
            class_names=list(self.class_names),
            valid_pixels=int(valid.sum()),
            prob_thd=float(self.prob_thd),
            joint_role_view=dict(
                metadata, variants=rows,
                complementarity=complementarity),
        )
        self._rpt_write_stats(record)
