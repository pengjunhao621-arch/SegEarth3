"""Role-specific Fine/Context visual-field diagnosis for frozen SAM3."""

import json
import os
from collections import OrderedDict

import torch

from role_functional_text_definitions import (
    ROLE_VISUAL_FIELD_COMPOSITIONS,
    ROLE_VISUAL_FIELD_PROTOCOL,
    ROLE_VISUAL_FIELD_SCHEMA_VERSION,
    ROLE_VISUAL_FIELD_VARIANT_NAMES,
)
from sam3.model.data_misc import interpolate as sam3_interpolate
from tiled_context import CoordinateTileIndex


RVF_PROTOCOL = ROLE_VISUAL_FIELD_PROTOCOL
RVF_SCHEMA_VERSION = ROLE_VISUAL_FIELD_SCHEMA_VERSION
RVF_COMPOSITIONS = ROLE_VISUAL_FIELD_COMPOSITIONS
RVF_VARIANT_NAMES = ROLE_VISUAL_FIELD_VARIANT_NAMES


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
            role_prompt_tta_visual_field_registry=None):
        self.role_prompt_tta_visual_field_diagnosis = bool(
            role_prompt_tta_visual_field_diagnosis)
        self.role_prompt_tta_visual_field_registry = (
            role_prompt_tta_visual_field_registry)
        self._rvf_config = None
        self._rvf_tile_indices = {}
        if not self.role_prompt_tta_visual_field_diagnosis:
            return
        if not role_prompt_tta_visual_field_registry:
            raise ValueError(
                'Visual-field diagnosis requires its registry JSON.')
        if self._rpt_role_selection is None:
            raise ValueError(
                'Visual-field diagnosis requires the role-text selection.')
        self._rvf_config = load_visual_field_registry(
            role_prompt_tta_visual_field_registry,
            self.role_prompt_tta_dataset_name)

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

    def _rvf_ground_view(self, image):
        """Ground only official aliases and the registered role candidates."""
        output_shape = (image.height, image.width)
        cache = self._rpt_prepare_text_cache()
        with torch.no_grad(), self._rpt_autocast_context():
            state = self.processor.set_image(image)
        requirements = {
            prompt: {'final', 'semantic', 'raw'}
            for prompt in self.query_words
        }
        slots = self._rpt_role_selection['_best_overall_slots']
        fields = (
            ('presence', 'presence_candidates'),
            ('semantic', 'semantic_candidates'),
            ('instance', 'instance_candidates'),
        )
        for item in self._rpt_prompt_bank['classes']:
            for (role, field), slot in zip(fields, slots):
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
                    selected[role] = packages[prompt]
                    selected_prompts[role] = prompt
                else:
                    selected[role] = None
                    selected_prompts[role] = list(item['official_prompts'])
            classes.append(dict(
                anchor=anchor,
                selected=selected,
                selected_prompts=selected_prompts,
            ))
        del state
        return dict(
            shape=output_shape,
            query_final=query_final,
            classes=classes,
            cache_prompt_count=len(cache['prompts']),
        )

    @staticmethod
    def _rvf_crop(value, roi):
        x1, y1, x2, y2 = roi
        return torch.as_tensor(value).float()[y1:y2, x1:x2]

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
                           family, fine_roi, context_roi, instance_cache):
        views = {'F': (fine, fine_roi), 'C': (context, context_roi)}
        p_view, s_view, i_view = composition
        p_data, _ = views[p_view]
        s_data, s_roi = views[s_view]
        i_data, _ = views[i_view]
        p_class = p_data['classes'][class_index]
        s_class = s_data['classes'][class_index]
        i_class = i_data['classes'][class_index]
        alpha, clip = 0.5, 0.25

        presence = torch.as_tensor(
            p_class['anchor']['presence']).float().reshape(())
        if family == 'text' and p_class['selected']['presence'] is not None:
            presence, _ = self._rft_bounded(
                presence, p_class['selected']['presence']['presence'],
                alpha, clip)

        semantic = self._rvf_crop(s_class['anchor']['semantic'], s_roi)
        if family == 'text' and s_class['selected']['semantic'] is not None:
            candidate = self._rvf_crop(
                s_class['selected']['semantic']['semantic'], s_roi)
            semantic, _ = self._rft_bounded(
                semantic, candidate, alpha, clip)

        admission = presence
        if (family == 'text'
                and self._rpt_role_selection['best_overall']['admission']
                == 'anchor_admission'):
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
        if family == 'text' and i_class['selected']['instance'] is not None:
            candidate = instance(i_class['selected']['instance'], i_view)
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

    def _rvf_accumulate_visual_units(self, image, image_path):
        height, width = image.height, image.width
        counts = torch.zeros((1, height, width), dtype=torch.float32)
        query_sum = torch.zeros(
            (self.num_queries, height, width), dtype=torch.float32)
        raw = OrderedDict(
            (name, torch.zeros(
                (self.num_cls, height, width), dtype=torch.float32))
            for name in RVF_VARIANT_NAMES[2:])
        unit_stats = []
        for unit_index, unit in enumerate(
                self._rvf_visual_units(image, image_path)):
            fine = self._rvf_ground_view(unit['fine'])
            context = self._rvf_ground_view(unit['context'])
            x1, y1, x2, y2 = unit['target_box']
            query_sum[:, y1:y2, x1:x2] += fine['query_final']
            counts[:, y1:y2, x1:x2] += 1.0
            official = self._aggregate_query_logits_to_classes(
                fine['query_final'])
            anchor_roles, text_roles = {}, {}
            instance_cache = {}
            for composition in RVF_COMPOSITIONS:
                anchor_roles[composition] = torch.stack([
                    self._rvf_compose_class(
                        fine, context, class_index, composition, 'anchor',
                        unit['fine_roi'], unit['context_roi'], instance_cache)
                    for class_index in range(self.num_cls)
                ])
                text_roles[composition] = torch.stack([
                    self._rvf_compose_class(
                        fine, context, class_index, composition, 'text',
                        unit['fine_roi'], unit['context_roi'], instance_cache)
                    for class_index in range(self.num_cls)
                ])
            anchor_fff = anchor_roles['FFF']
            best_fine = (
                official + text_roles['FFF'] - anchor_fff).clamp(0.0, 1.0)
            for composition in RVF_COMPOSITIONS:
                raw[f'rvf_anchor_{composition}'][:, y1:y2, x1:x2] += (
                    official + anchor_roles[composition] - anchor_fff
                ).clamp(0.0, 1.0)
                raw[f'rvf_text_{composition}'][:, y1:y2, x1:x2] += (
                    best_fine + text_roles[composition]
                    - text_roles['FFF']).clamp(0.0, 1.0)
            unit_stats.append(dict(
                unit_index=int(unit_index),
                target_box=list(unit['target_box']),
                fine_size=[unit['fine'].width, unit['fine'].height],
                context_size=[unit['context'].width, unit['context'].height],
                context_roi=list(unit['context_roi']),
                source_prefix=unit['metadata']['source_prefix'],
                source_canvas=unit['metadata']['source_canvas'],
                context_box=unit['metadata']['context_box'],
                contributor_tiles=unit['metadata']['contributor_tiles'],
            ))
            del fine, context, anchor_roles, text_roles
        if torch.any(counts == 0):
            raise RuntimeError('Visual-field Fine units left uncovered pixels.')
        query_mean = query_sum / counts
        exact_fine = self._aggregate_query_logits_to_classes(query_mean)
        for name in raw:
            raw[name] /= counts
        anchor_fff = raw['rvf_anchor_FFF'].clone()
        for composition in RVF_COMPOSITIONS:
            name = f'rvf_anchor_{composition}'
            raw[name] = (
                exact_fine + raw[name] - anchor_fff).clamp(0.0, 1.0)
        fine_best = (
            exact_fine + raw['rvf_text_FFF'] - anchor_fff).clamp(0.0, 1.0)
        text_fff = raw['rvf_text_FFF'].clone()
        for composition in RVF_COMPOSITIONS:
            name = f'rvf_text_{composition}'
            raw[name] = (
                fine_best + raw[name] - text_fff).clamp(0.0, 1.0)
        return query_mean, exact_fine, fine_best, raw, unit_stats

    def _rvf_accumulate_official_units(self, image):
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
        unit_official = torch.zeros(
            (self.num_cls, height, width), dtype=torch.float32)
        unit_best = torch.zeros_like(unit_official)
        for box in boxes:
            x1, y1, x2, y2 = box
            view = self._rvf_ground_view(image.crop(box))
            official = self._aggregate_query_logits_to_classes(
                view['query_final'])
            cache = {}
            roi = (0, 0, x2 - x1, y2 - y1)
            roles_anchor = torch.stack([
                self._rvf_compose_class(
                    view, view, class_index, 'FFF', 'anchor',
                    roi, roi, cache)
                for class_index in range(self.num_cls)
            ])
            roles_text = torch.stack([
                self._rvf_compose_class(
                    view, view, class_index, 'FFF', 'text',
                    roi, roi, cache)
                for class_index in range(self.num_cls)
            ])
            best = (official + roles_text - roles_anchor).clamp(0.0, 1.0)
            query_sum[:, y1:y2, x1:x2] += view['query_final']
            unit_official[:, y1:y2, x1:x2] += official
            unit_best[:, y1:y2, x1:x2] += best
            counts[:, y1:y2, x1:x2] += 1.0
            del view
        query_mean = query_sum / counts
        exact = self._aggregate_query_logits_to_classes(query_mean)
        unit_official /= counts
        unit_best /= counts
        best_exact = (exact + unit_best - unit_official).clamp(0.0, 1.0)
        return query_mean, exact, best_exact

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

    def _rvf_predict_image(self, image, image_path):
        (fine_query, fine_exact, fine_best, raw,
         unit_stats) = self._rvf_accumulate_visual_units(image, image_path)
        if self._rvf_fine_matches_official(image):
            official_query, official, best_text = (
                fine_query, fine_exact, fine_best)
            reused_fine = True
        else:
            official_query, official, best_text = (
                self._rvf_accumulate_official_units(image))
            reused_fine = False
        variants = OrderedDict(
            (name, torch.empty_like(official)) for name in RVF_VARIANT_NAMES)
        variants['rvf_official'].copy_(official)
        variants['rvf_best_text'].copy_(best_text)
        anchor_reference = raw['rvf_anchor_FFF']
        text_reference = raw['rvf_text_FFF']
        for composition in RVF_COMPOSITIONS:
            variants[f'rvf_anchor_{composition}'].copy_((
                official + raw[f'rvf_anchor_{composition}']
                - anchor_reference).clamp(0.0, 1.0))
            variants[f'rvf_text_{composition}'].copy_((
                best_text + raw[f'rvf_text_{composition}']
                - text_reference).clamp(0.0, 1.0))
        identity_error = max(
            float((variants['rvf_anchor_FFF'] - official).abs().max()),
            float((variants['rvf_text_FFF'] - best_text).abs().max()))
        if (self.role_prompt_tta_strict_integrity
                and identity_error > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                f'Visual-field FFF identity failed: {identity_error}.')
        metadata = dict(
            schema_version=RVF_SCHEMA_VERSION,
            protocol=RVF_PROTOCOL,
            variants=list(RVF_VARIANT_NAMES),
            compositions=list(RVF_COMPOSITIONS),
            fine_size=int(self._rvf_config['fine_size']),
            context_size=int(self._rvf_config['context_size']),
            source_mode=self._rvf_config['source_mode'],
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
            native_prompt_parity_max_abs=float(
                self._rpt_native_parity_max_abs or 0.0),
            fff_identity_max_abs=float(identity_error),
            units=unit_stats,
        )
        return official_query.to(self.device), variants, metadata

    def _rvf_record_image(
            self, variants, metadata, data_sample, image_path):
        if not self.dump_role_prompt_tta_stats:
            return
        gt = data_sample.gt_sem_seg.data.squeeze().detach().long().cpu()
        valid = gt != 255
        predictions = {
            name: self._rpt_threshold(value).to(torch.int16)
            for name, value in variants.items()
        }
        rows = OrderedDict()
        for name in RVF_VARIANT_NAMES:
            reference_name = (
                'rvf_best_text' if name.startswith('rvf_text_')
                else 'rvf_official')
            prediction = predictions[name]
            reference = predictions[reference_name]
            changed = valid & (prediction != reference)
            improved = changed & (prediction == gt) & (reference != gt)
            harmed = changed & (prediction != gt) & (reference == gt)
            rows[name] = dict(
                reference_variant=reference_name,
                confusion=self._rpt_confusion(prediction, gt, valid),
                changed_pixels=int(changed.sum()),
                improved_pixels=int(improved.sum()),
                harmed_pixels=int(harmed.sum()),
                help_minus_harm=int(improved.sum()) - int(harmed.sum()),
            )
        record = dict(
            schema_version=RVF_SCHEMA_VERSION,
            rank=int(os.environ.get('RANK', 0)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            dataset_name=self.role_prompt_tta_dataset_name,
            img_path=image_path,
            class_names=list(self.class_names),
            valid_pixels=int(valid.sum()),
            prob_thd=float(self.prob_thd),
            role_visual_field=dict(metadata, variants=rows),
        )
        self._rpt_write_stats(record)
