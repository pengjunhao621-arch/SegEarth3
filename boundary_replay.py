"""Boundary-local mask refinement and anchor-clamped role replay.

This module is active only for ``boundary_replay_v1``.  It keeps the official
SAM3 prediction path intact and reuses frozen native layers for causal
counterfactuals around the selected Role-Functional Text composition.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from role_functional_text_definitions import (
    BOUNDARY_GUIDES,
    BOUNDARY_PRIMARY_STRENGTH,
    BOUNDARY_REPLAY_VARIANT_NAMES,
    BOUNDARY_STRENGTHS,
)
from sam3.model.data_misc import interpolate as sam3_interpolate
from sam3.model.model_misc import gen_sineembed_for_position, inverse_sigmoid


class BoundaryReplayMixin:
    """Frozen diagnostic helpers mixed into the SegEarth-OV3 segmentor."""

    def _br_initialize(self):
        self._br_anchor_replay_checked = False
        self._br_anchor_replay_max_abs = 0.0

    @staticmethod
    def _br_variant_names():
        return BOUNDARY_REPLAY_VARIANT_NAMES

    def _br_prompt_only(self, state, prompt):
        self._rpt_load_cached_prompt(state, prompt)
        model = self.processor.model
        prompt_tensor, prompt_mask, _ = model._encode_prompt(
            state['backbone_out'], self.processor.find_stage,
            state['geometric_prompt'])
        return prompt_tensor, prompt_mask

    def _br_decoder_step(
            self, decoder, layer_index, query, presence, reference_boxes,
            encoder_out, prompt, prompt_mask):
        reference_points_input = (
            reference_boxes[:, :, None]
            * torch.cat([
                encoder_out['valid_ratios'], encoder_out['valid_ratios']
            ], -1)[None, :])
        query_sine_embed = gen_sineembed_for_position(
            reference_points_input[:, :, 0, :], decoder.d_model)
        query_pos = decoder.ref_point_head(query_sine_embed)
        cross_attention_mask = None
        if decoder.boxRPB != 'none':
            spatial_shapes = encoder_out['spatial_shapes']
            if int(spatial_shapes.shape[0]) != 1:
                raise RuntimeError(
                    'boundary_replay_v1 supports the native single-scale '
                    'decoder memory only.')
            cross_attention_mask = decoder._get_rpb_matrix(
                reference_boxes,
                (spatial_shapes[0, 0], spatial_shapes[0, 1]),
            ).flatten(0, 1)
        query_out, presence_out = decoder.layers[layer_index](
            tgt=query,
            tgt_query_pos=query_pos,
            tgt_query_sine_embed=query_sine_embed,
            tgt_key_padding_mask=None,
            tgt_reference_points=reference_points_input,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            memory=encoder_out['encoder_hidden_states'],
            memory_key_padding_mask=encoder_out['padding_mask'],
            memory_level_start_index=encoder_out['level_start_index'],
            memory_spatial_shapes=encoder_out['spatial_shapes'],
            memory_pos=encoder_out['pos_embed'],
            self_attn_mask=None,
            cross_attn_mask=cross_attention_mask,
            dac=False,
            dac_use_selfatt_ln=decoder.dac_use_selfatt_ln,
            presence_token=presence,
            act_ckpt_enable=False,
        )
        box_input = (
            decoder.norm(query_out)
            if decoder.use_normed_output_consistently else query_out)
        delta = decoder.bbox_embed(box_input)
        reference_out = (
            inverse_sigmoid(reference_boxes) + delta).sigmoid().detach()
        return query_out, presence_out, reference_out

    def _br_trace_decoder(self, encoder_out, prompt, prompt_mask):
        decoder = self.processor.model.transformer.decoder
        if self.processor.model.training:
            raise RuntimeError('boundary_replay_v1 requires frozen eval mode.')
        memory = encoder_out['encoder_hidden_states']
        batch = int(memory.shape[1])
        query = decoder.query_embed.weight.unsqueeze(1).repeat(1, batch, 1)
        reference = decoder.reference_points.weight.unsqueeze(1).repeat(
            1, batch, 1).sigmoid()
        presence = None
        if decoder.presence_token is not None:
            presence = decoder.presence_token.weight[None].expand(
                1, batch, -1)

        trace = dict(
            query_inputs=[], presence_inputs=[], reference_inputs=[],
            query_outputs=[], presence_outputs=[], reference_outputs=[],
            normalized_queries=[], presence_logits=[])
        for layer_index in range(decoder.num_layers):
            trace['query_inputs'].append(query)
            trace['presence_inputs'].append(presence)
            trace['reference_inputs'].append(reference)
            query, presence, reference = self._br_decoder_step(
                decoder, layer_index, query, presence, reference,
                encoder_out, prompt, prompt_mask)
            trace['query_outputs'].append(query)
            trace['presence_outputs'].append(presence)
            trace['reference_outputs'].append(reference)
            trace['normalized_queries'].append(decoder.norm(query))
            logit = decoder.presence_token_head(
                decoder.presence_token_out_norm(presence)).squeeze(-1)
            if decoder.clamp_presence_logits:
                logit = logit.clamp(
                    min=-decoder.clamp_presence_logit_max_val,
                    max=decoder.clamp_presence_logit_max_val)
            trace['presence_logits'].append(logit)
        return trace

    def _br_build_outputs(
            self, context, normalized_queries, reference_inputs,
            presence_logits, score_prompt, score_prompt_mask,
            segmentation_prompt, segmentation_prompt_mask):
        model = self.processor.model
        hs = torch.stack(normalized_queries).transpose(1, 2)
        references = torch.stack(reference_inputs).transpose(1, 2)
        decoder_presence = torch.stack(presence_logits).transpose(1, 2)
        out = {
            'encoder_hidden_states': context['encoder_out'][
                'encoder_hidden_states'],
            'prev_encoder_out': {
                'encoder_out': context['encoder_out'],
                'backbone_out': context['backbone_out'],
            },
        }
        model._update_scores_and_boxes(
            out, hs, references, score_prompt, score_prompt_mask,
            dec_presence_out=decoder_presence)
        model._run_segmentation_heads(
            out=out,
            backbone_out=context['backbone_out'],
            img_ids=self.processor.find_stage.img_ids,
            vis_feat_sizes=context['encoder_out']['vis_feat_sizes'],
            encoder_hidden_states=context['encoder_out'][
                'encoder_hidden_states'],
            prompt=segmentation_prompt,
            prompt_mask=segmentation_prompt_mask,
            hs=hs,
        )
        return out

    def _br_package_from_outputs(self, outputs, output_shape):
        raw_masks = outputs['pred_masks'].reshape(
            -1, *outputs['pred_masks'].shape[-2:]).detach().float().cpu()
        raw_scores = outputs['pred_logits'].sigmoid().reshape(
            -1).detach().float().cpu()
        semantic = outputs['semantic_seg'].float()
        if semantic.shape[-2:] != output_shape:
            semantic = sam3_interpolate(
                semantic, size=output_shape, mode='bilinear',
                align_corners=False)
        semantic = semantic.sigmoid().squeeze().detach().float().cpu()
        presence = outputs['presence_logit_dec'].sigmoid().reshape(
            ()).detach().float().cpu()
        return dict(
            semantic=semantic,
            presence=presence,
            raw_masks=raw_masks,
            raw_scores=raw_scores,
            raw_candidate_count=int(raw_scores.numel()),
        )

    def _br_build_anchor_context(self, state, prompt, output_shape):
        self._rpt_load_cached_prompt(state, prompt)
        model = self.processor.model
        prompt_tensor, prompt_mask, backbone_out = model._encode_prompt(
            state['backbone_out'], self.processor.find_stage,
            state['geometric_prompt'])
        backbone_out, encoder_out, _ = model._run_encoder(
            backbone_out, self.processor.find_stage, prompt_tensor,
            prompt_mask)
        trace = self._br_trace_decoder(
            encoder_out, prompt_tensor, prompt_mask)
        context = dict(
            prompt=prompt_tensor,
            prompt_mask=prompt_mask,
            backbone_out=backbone_out,
            encoder_out=encoder_out,
            trace=trace,
        )
        outputs = self._br_build_outputs(
            context,
            trace['normalized_queries'],
            trace['reference_inputs'],
            trace['presence_logits'],
            prompt_tensor, prompt_mask,
            prompt_tensor, prompt_mask)
        context['package'] = self._br_package_from_outputs(
            outputs, output_shape)
        return context

    @staticmethod
    def _br_package_error(left, right, fields):
        errors = []
        for field in fields:
            a = torch.as_tensor(left[field]).float()
            b = torch.as_tensor(right[field]).float()
            if a.shape != b.shape:
                return float('inf')
            errors.append(float((a - b).abs().max().item()))
        return max(errors or [0.0])

    def _br_replay_presence(self, context, prompt, prompt_mask):
        decoder = self.processor.model.transformer.decoder
        anchor = context['trace']
        presence = anchor['presence_inputs'][0]
        for layer_index in range(decoder.num_layers):
            _, presence, _ = self._br_decoder_step(
                decoder, layer_index,
                anchor['query_inputs'][layer_index],
                presence,
                anchor['reference_inputs'][layer_index],
                context['encoder_out'], prompt, prompt_mask)
        logit = decoder.presence_token_head(
            decoder.presence_token_out_norm(presence)).squeeze(-1)
        if decoder.clamp_presence_logits:
            logit = logit.clamp(
                min=-decoder.clamp_presence_logit_max_val,
                max=decoder.clamp_presence_logit_max_val)
        return logit.sigmoid().reshape(()).detach().float().cpu()

    def _br_replay_instance(
            self, context, prompt, prompt_mask, output_shape):
        decoder = self.processor.model.transformer.decoder
        anchor = context['trace']
        query = anchor['query_inputs'][0]
        reference = anchor['reference_inputs'][0]
        normalized, reference_inputs = [], []
        for layer_index in range(decoder.num_layers):
            reference_inputs.append(reference)
            query, _, reference = self._br_decoder_step(
                decoder, layer_index, query,
                anchor['presence_inputs'][layer_index], reference,
                context['encoder_out'], prompt, prompt_mask)
            normalized.append(decoder.norm(query))
        outputs = self._br_build_outputs(
            context, normalized, reference_inputs,
            anchor['presence_logits'],
            prompt, prompt_mask,
            context['prompt'], context['prompt_mask'])
        return self._br_package_from_outputs(outputs, output_shape)

    def _br_replay_semantic(
            self, context, prompt, prompt_mask, output_shape):
        anchor = context['trace']
        outputs = self._br_build_outputs(
            context,
            anchor['normalized_queries'],
            anchor['reference_inputs'],
            anchor['presence_logits'],
            context['prompt'], context['prompt_mask'],
            prompt, prompt_mask)
        return self._br_package_from_outputs(outputs, output_shape)

    def _br_check_anchor_replay(self, context, output_shape):
        if self._br_anchor_replay_checked:
            return self._br_anchor_replay_max_abs
        prompt, prompt_mask = context['prompt'], context['prompt_mask']
        anchor = context['package']
        presence = self._br_replay_presence(context, prompt, prompt_mask)
        instance = self._br_replay_instance(
            context, prompt, prompt_mask, output_shape)
        semantic = self._br_replay_semantic(
            context, prompt, prompt_mask, output_shape)
        errors = [
            float((presence - anchor['presence']).abs().item()),
            self._br_package_error(
                instance, anchor, ('raw_masks', 'raw_scores')),
            self._br_package_error(semantic, anchor, ('semantic',)),
        ]
        self._br_anchor_replay_max_abs = max(errors)
        self._br_anchor_replay_checked = True
        return self._br_anchor_replay_max_abs

    @staticmethod
    def _br_merge_replay_packages(packages, field):
        if field == 'presence':
            return dict(presence=torch.stack([
                torch.as_tensor(item['presence']).float().reshape(())
                for item in packages]).max())
        if field == 'semantic':
            return dict(semantic=torch.stack([
                item['semantic'].float() for item in packages
            ]).max(dim=0)[0])
        masks = [item['raw_masks'] for item in packages
                 if item['raw_masks'].numel()]
        scores = [item['raw_scores'] for item in packages
                  if item['raw_scores'].numel()]
        return dict(
            raw_masks=(torch.cat(masks) if masks else
                       torch.empty((0, 1, 1), dtype=torch.float32)),
            raw_scores=(torch.cat(scores) if scores else
                        torch.empty((0,), dtype=torch.float32)),
        )

    def _br_keep(self, package, admission_presence):
        count = min(
            int(package['raw_masks'].shape[0]),
            int(package['raw_scores'].numel()))
        if count == 0:
            return torch.zeros(0, dtype=torch.bool)
        return (
            package['raw_scores'][:count].float()
            * float(torch.as_tensor(admission_presence).float().item())
            > float(self.processor.confidence_threshold))

    def _br_refine_package(
            self, package, guide_feature, strength, selection, guide_name,
            query_meta=None):
        count = min(
            int(package['raw_masks'].shape[0]),
            int(package['raw_scores'].numel()))
        selection = torch.as_tensor(selection).bool().flatten()[:count]
        result = dict(package)
        if count == 0 or not selection.any():
            return result, []
        indices = torch.nonzero(selection, as_tuple=False).flatten()
        raw = package['raw_masks'][:count][selection].to(
            self.device).float()
        target_size = tuple(guide_feature.shape[-2:])
        low = F.interpolate(
            raw.unsqueeze(1), size=target_size, mode='bilinear',
            align_corners=False).squeeze(1)
        weighted = low.clone()
        denominator = torch.ones_like(low)
        feature = None if guide_name == 'uniform' else guide_feature.float()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            if feature is None:
                weight = torch.ones_like(low[:1])
            else:
                neighbour = self._rft_shift(feature, dy, dx)
                similarity = (feature * neighbour).sum(
                    dim=1, keepdim=False).clamp(-1.0, 1.0)
                weight = torch.exp((similarity - 1.0) / 0.10)
            weighted = weighted + weight * self._rft_shift(low, dy, dx)
            denominator = denominator + weight
        local_mean = weighted / denominator.clamp_min(1e-6)
        boundary_band = torch.exp(-low.abs())
        low_update = boundary_band * (local_mean - low)
        raw_update = F.interpolate(
            low_update.unsqueeze(1), size=raw.shape[-2:], mode='bilinear',
            align_corners=False).squeeze(1)
        refined = raw + float(strength) * raw_update
        all_masks = package['raw_masks'][:count].clone().float()
        all_masks[selection] = refined.detach().float().cpu()
        result['raw_masks'] = all_masks

        rows = []
        if query_meta is not None:
            before = low >= 0.0
            after = (
                low + float(strength) * low_update) >= 0.0
            for local_index, query_index in enumerate(indices.tolist()):
                row = dict(query_meta)
                row.update(
                    query_index=int(query_index),
                    object_score=float(
                        package['raw_scores'][query_index].item()),
                    diagnostic_height=int(target_size[0]),
                    diagnostic_width=int(target_size[1]),
                    _before_support=before[local_index].to(
                        torch.uint8).detach().cpu(),
                    _after_support=after[local_index].to(
                        torch.uint8).detach().cpu(),
                )
                rows.append(row)
        return result, rows

    def _br_class_variants(
            self, state, output_shape, class_index, class_name,
            official_prompts, official_packages, anchor, role_packages,
            official_final, role_anchor_final, pe_features,
            default_alpha, default_clip):
        selected = self._rpt_role_selection
        p_slot, s_slot, i_slot = selected['_best_overall_slots']
        admission_mode = selected['best_overall']['admission']
        anchor_presence = torch.as_tensor(anchor['presence']).float().reshape(())
        anchor_semantic = anchor['semantic'].float()

        full_packages = dict(
            presence=(role_packages['presence'][p_slot] if p_slot else None),
            semantic=(role_packages['semantic'][s_slot] if s_slot else None),
            instance=(role_packages['instance'][i_slot] if i_slot else None),
        )

        def compose(packages, anchor_package=anchor,
                    candidate_instance_package=None):
            presence = anchor_presence
            if packages.get('presence') is not None:
                presence, _ = self._rft_bounded(
                    anchor_presence, packages['presence']['presence'],
                    default_alpha, default_clip)
            semantic = anchor_semantic
            if packages.get('semantic') is not None:
                semantic, _ = self._rft_bounded(
                    anchor_semantic, packages['semantic']['semantic'],
                    default_alpha, default_clip)
            admission = (
                anchor_presence if admission_mode == 'anchor_admission'
                else presence)
            base, _ = self._rpt_head_role_instance_from_raw(
                anchor_package, presence, output_shape,
                admission_presence=admission,
                amplitude_presence=presence)
            instance_package = (
                candidate_instance_package
                if candidate_instance_package is not None
                else packages.get('instance'))
            instance = base
            if instance_package is not None:
                candidate, _ = self._rpt_head_role_instance_from_raw(
                    instance_package, presence, output_shape,
                    admission_presence=admission,
                    amplitude_presence=presence)
                instance, _ = self._rft_bounded(
                    base, candidate, default_alpha, default_clip)
            role_final = self._rft_role_final(
                semantic, instance, presence, output_shape)
            final = (
                official_final.float() + role_final.float()
                - role_anchor_final.float()).clamp(0.0, 1.0)
            return final, dict(
                presence=presence, semantic=semantic,
                admission=admission, instance=instance)

        best_full, full_state = compose(full_packages)
        values = OrderedDict(
            br_official=official_final.float(), br_best_full=best_full)

        prompt_tokens = {}
        anchor_contexts = []
        trace_errors = []
        for prompt, package in zip(official_prompts, official_packages):
            context = self._br_build_anchor_context(
                state, prompt, output_shape)
            trace_errors.append(self._br_package_error(
                context['package'], package,
                ('semantic', 'presence', 'raw_masks', 'raw_scores')))
            self._br_check_anchor_replay(context, output_shape)
            anchor_contexts.append(context)

        def tokens(prompt):
            if prompt not in prompt_tokens:
                prompt_tokens[prompt] = self._br_prompt_only(state, prompt)
            return prompt_tokens[prompt]

        replay = dict(presence=None, semantic=None, instance=None)
        for role, slot in (('presence', p_slot), ('semantic', s_slot),
                           ('instance', i_slot)):
            if slot == 0:
                continue
            prompt = self._rpt_prompt_bank['classes'][class_index][
                f'{role}_candidates'][slot - 1]
            prompt_tensor, prompt_mask = tokens(prompt)
            packages = []
            for context in anchor_contexts:
                if role == 'presence':
                    packages.append(dict(presence=self._br_replay_presence(
                        context, prompt_tensor, prompt_mask)))
                elif role == 'semantic':
                    packages.append(self._br_replay_semantic(
                        context, prompt_tensor, prompt_mask, output_shape))
                else:
                    packages.append(self._br_replay_instance(
                        context, prompt_tensor, prompt_mask, output_shape))
            replay[role] = self._br_merge_replay_packages(packages, role)

        for role, name in (('presence', 'br_replay_p'),
                           ('semantic', 'br_replay_s'),
                           ('instance', 'br_replay_i')):
            mixed = dict(full_packages)
            if replay[role] is not None:
                mixed[role] = replay[role]
            values[name] = compose(mixed)[0]
        values['br_replay_all'] = compose(replay)[0]

        official_keep = self._br_keep(anchor, anchor_presence)
        best_anchor_keep = self._br_keep(anchor, full_state['admission'])
        anchor_union = official_keep | best_anchor_keep
        candidate_keep = (
            self._br_keep(full_packages['instance'], full_state['admission'])
            if full_packages['instance'] is not None
            else torch.zeros(0, dtype=torch.bool))
        query_rows = []
        for guide_name in BOUNDARY_GUIDES:
            guide = pe_features[
                'block23' if guide_name == 'uniform' else guide_name]
            for strength_name, strength in BOUNDARY_STRENGTHS:
                primary = (
                    guide_name == 'block23'
                    and strength_name == BOUNDARY_PRIMARY_STRENGTH)
                refined_anchor, rows = self._br_refine_package(
                    anchor, guide, strength, anchor_union, guide_name,
                    query_meta=(dict(
                        class_index=int(class_index), class_name=class_name,
                        source='anchor', instance_slot=0,
                        admitted_official_count=int(official_keep.sum().item()),
                        admitted_best_count=int(best_anchor_keep.sum().item()),
                        guide=guide_name, strength=float(strength))
                                if primary else None))
                if primary:
                    for row in rows:
                        query_index = int(row['query_index'])
                        row['admitted_official'] = bool(
                            official_keep[query_index].item())
                        row['admitted_best'] = bool(
                            best_anchor_keep[query_index].item())
                    query_rows.extend(rows)

                official_instance, _ = (
                    self._rpt_head_role_instance_from_raw(
                        refined_anchor, anchor_presence, output_shape,
                        admission_presence=anchor_presence,
                        amplitude_presence=anchor_presence))
                official_role = self._rft_role_final(
                    anchor_semantic, official_instance, anchor_presence,
                    output_shape)
                values[
                    f'br_official_{guide_name}_{strength_name}'] = (
                        official_final.float() + official_role.float()
                        - role_anchor_final.float()).clamp(0.0, 1.0)

                refined_candidate = None
                if full_packages['instance'] is not None:
                    refined_candidate, rows = self._br_refine_package(
                        full_packages['instance'], guide, strength,
                        candidate_keep, guide_name,
                        query_meta=(dict(
                            class_index=int(class_index),
                            class_name=class_name,
                            source='candidate', instance_slot=int(i_slot),
                            admitted_official_count=0,
                            admitted_best_count=int(
                                candidate_keep.sum().item()),
                            guide=guide_name, strength=float(strength))
                                    if primary else None))
                    if primary:
                        for row in rows:
                            row['admitted_official'] = False
                            row['admitted_best'] = True
                        query_rows.extend(rows)
                values[f'br_best_{guide_name}_{strength_name}'] = compose(
                    full_packages,
                    anchor_package=refined_anchor,
                    candidate_instance_package=refined_candidate)[0]

        if tuple(values) != BOUNDARY_REPLAY_VARIANT_NAMES:
            raise RuntimeError('Boundary/replay variant order drifted.')
        trace_error = max(trace_errors or [0.0])
        if (self.role_prompt_tta_strict_integrity
                and max(trace_error, self._br_anchor_replay_max_abs)
                > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                'Anchor-clamped replay integrity failed: '
                f'trace={trace_error}, '
                f'anchor_replay={self._br_anchor_replay_max_abs}.')
        class_row = dict(
            class_index=int(class_index), class_name=class_name,
            selected_slots=[int(p_slot), int(s_slot), int(i_slot)],
            selected_admission=admission_mode,
            anchor_trace_max_abs=float(trace_error),
            anchor_replay_max_abs=float(self._br_anchor_replay_max_abs),
            replay_p_abs_mean=float((
                values['br_replay_p'] - best_full).abs().mean().item()),
            replay_s_abs_mean=float((
                values['br_replay_s'] - best_full).abs().mean().item()),
            replay_i_abs_mean=float((
                values['br_replay_i'] - best_full).abs().mean().item()),
            replay_all_abs_mean=float((
                values['br_replay_all'] - best_full).abs().mean().item()),
        )
        return values, query_rows, class_row
