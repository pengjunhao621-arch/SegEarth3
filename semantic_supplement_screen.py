"""Native-SAM3 two-prompt semantic supplement screening protocol."""

from collections import OrderedDict

import torch

from semantic_supplement_definitions import (
    FAMILIES,
    MAX_CANDIDATES,
    VARIANT_NAMES,
    aggregate_variant_name,
    candidate_variant_name,
)
from role_functional_text_definitions import (
    DEFAULT_SETTING as ROLE_TEXT_DEFAULT_SETTING,
    RESIDUAL_SETTINGS as ROLE_TEXT_RESIDUAL_SETTINGS,
    ROLE_FIELDS as ROLE_TEXT_ROLE_FIELDS,
    VARIANT_NAMES as ROLE_TEXT_VARIANT_NAMES,
    combo_variant_name as role_text_combo_variant_name,
    sensitivity_variant_name as role_text_sensitivity_variant_name,
)


def _safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


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


class SemanticSupplementScreenMixin:
    """Methods mixed into ``RolePromptTTAMixin``.

    Every prompt is grounded by unmodified SAM3.  The structural anchor's
    Presence and raw instance queries remain coupled.  Candidate text may only
    replace or residually amend the native semantic map after grounding.
    """

    def _ss_bounded_semantic(self, anchor_semantic, candidate_semantic):
        delta = candidate_semantic.float() - anchor_semantic.float()
        clipped = delta.clamp(
            -float(self.role_prompt_tta_semantic_residual_clip),
            float(self.role_prompt_tta_semantic_residual_clip),
        )
        semantic = (
            anchor_semantic.float()
            + float(self.role_prompt_tta_semantic_residual_alpha) * clipped
        ).clamp(0.0, 1.0)
        return semantic, delta, clipped

    def _ss_compose_semantic(
            self, semantic, anchor, anchor_instance, output_shape):
        semantic_package = dict(semantic=semantic)
        final, _, _ = self._rpt_head_role_compose(
            semantic_package,
            anchor,
            anchor,
            output_shape,
            instance_override=anchor_instance,
        )
        return final

    def _rpt_semantic_supplement_infer_single_view(
            self, image, return_stats=False, return_components=False,
            view_id=None, crop_box=None):
        """Evaluate two frozen prompt families with one image encoding."""
        width, height = image.size
        output_shape = (height, width)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        cache = self._rpt_prepare_text_cache()

        with torch.no_grad(), self._rpt_autocast_context():
            state = self.processor.set_image(image)
            baseline_rows = []
            parity_errors = []
            for prompt in self.query_words:
                native_row = None
                if not self._rpt_native_parity_checked:
                    self.processor.reset_all_prompts(state)
                    self.processor.set_text_prompt(prompt, state)
                    native_row = self._rpt_prompt_components(
                        state, output_shape)
                self._rpt_set_cached_prompt(state, prompt)
                cached_row = self._rpt_prompt_components(state, output_shape)
                if native_row is not None:
                    for key in (
                            'final', 'semantic', 'semantic_raw',
                            'instance', 'presence'):
                        parity_errors.append(float((
                            cached_row[key].float()
                            - native_row[key].float()).abs().max().item()))
                baseline_rows.append({
                    key: cached_row[key].detach().float().cpu()
                    for key in ('final', 'semantic', 'instance')
                })
            if not self._rpt_native_parity_checked:
                self._rpt_native_parity_max_abs = max(parity_errors or [0.0])
                self._rpt_native_parity_checked = True
                if (
                        self.role_prompt_tta_strict_integrity
                        and self._rpt_native_parity_max_abs
                        > self.role_prompt_tta_integrity_tolerance):
                    raise RuntimeError(
                        'Cached prompt path differs from protected native '
                        f'prompt path: max_abs='
                        f'{self._rpt_native_parity_max_abs}.')

        baseline_query = {
            key: torch.stack([row[key] for row in baseline_rows], dim=0)
            for key in ('final', 'semantic', 'instance')
        }
        baseline_class = self._rpt_query_to_class(
            baseline_query['final']).detach().float().cpu()
        class_variants = OrderedDict(
            (name, torch.empty(
                (int(self.num_cls), height, width),
                device='cpu', dtype=torch.float32))
            for name in VARIANT_NAMES)
        class_variants['baseline'].copy_(baseline_class)
        candidate_rows = []
        family_rows = []
        raw_recomposition_errors = []

        for class_idx, item in enumerate(self._rpt_prompt_bank['classes']):
            anchor_prompt = item['anchor']
            with torch.no_grad(), self._rpt_autocast_context():
                self._rpt_set_cached_prompt(state, anchor_prompt)
                anchor = self._rpt_head_role_raw_package(
                    state, output_shape)
            anchor_instance = self._rpt_head_role_instance_from_raw(
                anchor, anchor['presence'], output_shape)
            anchor_rebuilt, _, _ = self._rpt_head_role_compose(
                anchor, anchor, anchor, output_shape,
                instance_override=anchor_instance)
            anchor_final_error = float((
                anchor_rebuilt - anchor['native_final']).abs().max().item())
            anchor_instance_error = float((
                anchor_instance[0]
                - anchor['native_instance']).abs().max().item())
            raw_recomposition_errors.extend(
                [anchor_final_error, anchor_instance_error])
            class_variants['anchor_native'][class_idx].copy_(
                anchor['native_final'])

            anchor_cache_index = int(cache['index'][anchor_prompt])
            for family, field in FAMILIES:
                prompts = list(item.get(field, []))
                active_semantics = []
                for slot in range(1, MAX_CANDIDATES + 1):
                    available = slot <= len(prompts)
                    prompt = prompts[slot - 1] if available else anchor_prompt
                    if available:
                        with torch.no_grad(), self._rpt_autocast_context():
                            self._rpt_set_cached_prompt(state, prompt)
                            candidate = self._rpt_head_role_raw_package(
                                state, output_shape)
                        candidate_instance = (
                            self._rpt_head_role_instance_from_raw(
                                candidate, candidate['presence'], output_shape))
                        candidate_rebuilt, _, _ = (
                            self._rpt_head_role_compose(
                                candidate, candidate, candidate, output_shape,
                                instance_override=candidate_instance))
                        native_error = float((
                            candidate_rebuilt
                            - candidate['native_final']).abs().max().item())
                        native_instance_error = float((
                            candidate_instance[0]
                            - candidate['native_instance']).abs().max().item())
                        raw_recomposition_errors.extend(
                            [native_error, native_instance_error])
                        active_semantics.append(candidate['semantic'])
                    else:
                        candidate = anchor
                        native_error = anchor_final_error
                        native_instance_error = anchor_instance_error

                    replacement = self._ss_compose_semantic(
                        candidate['semantic'], anchor, anchor_instance,
                        output_shape)
                    residual_semantic, delta, clipped_delta = (
                        self._ss_bounded_semantic(
                            anchor['semantic'], candidate['semantic']))
                    residual = self._ss_compose_semantic(
                        residual_semantic, anchor, anchor_instance,
                        output_shape)

                    path_maps = {
                        'shared_native': candidate['native_final'],
                        'semantic_replace': replacement,
                        'semantic_residual': residual,
                    }
                    for path, value in path_maps.items():
                        name = candidate_variant_name(family, slot, path)
                        class_variants[name][class_idx].copy_(
                            value.detach().float().cpu())

                    prompt_cache_index = int(cache['index'][prompt])
                    candidate_rows.append(dict(
                        class_index=int(class_idx),
                        class_name=item['name'],
                        family=family,
                        slot=int(slot),
                        available=bool(available),
                        anchor_prompt=anchor_prompt,
                        candidate_prompt=prompt,
                        anchor_token_count=int((
                            ~cache['language_mask'][anchor_cache_index].bool()
                        ).sum().item()),
                        candidate_token_count=int((
                            ~cache['language_mask'][prompt_cache_index].bool()
                        ).sum().item()),
                        candidate_word_count=len(prompt.split()),
                        anchor_presence=float(anchor['presence']),
                        candidate_presence=float(candidate['presence']),
                        presence_delta=float(
                            candidate['presence'] - anchor['presence']),
                        anchor_semantic_mean=float(
                            anchor['semantic'].mean().item()),
                        candidate_semantic_mean=float(
                            candidate['semantic'].mean().item()),
                        candidate_semantic_area_050=float((
                            candidate['semantic'] >= 0.5
                        ).float().mean().item()),
                        semantic_signed_delta_mean=float(delta.mean().item()),
                        semantic_abs_delta_mean=float(
                            delta.abs().mean().item()),
                        semantic_positive_delta_ratio=float((
                            delta > 0).float().mean().item()),
                        semantic_negative_delta_ratio=float((
                            delta < 0).float().mean().item()),
                        semantic_clip_ratio=float((
                            delta.abs()
                            > float(
                                self.role_prompt_tta_semantic_residual_clip)
                        ).float().mean().item()),
                        clipped_abs_delta_mean=float(
                            clipped_delta.abs().mean().item()),
                        anchor_instance_mean=float(
                            anchor['native_instance'].mean().item()),
                        candidate_instance_mean=float(
                            candidate['native_instance'].mean().item()),
                        anchor_native_kept_count=int(
                            anchor['native_kept_count']),
                        candidate_native_kept_count=int(
                            candidate['native_kept_count']),
                        candidate_raw_object_score_mean=(
                            float(candidate['raw_scores'].mean().item())
                            if candidate['raw_scores'].numel() else 0.0),
                        candidate_raw_object_score_max=(
                            float(candidate['raw_scores'].max().item())
                            if candidate['raw_scores'].numel() else 0.0),
                        native_recomposition_max_abs=native_error,
                        native_instance_recomposition_max_abs=(
                            native_instance_error),
                        path_final_means={
                            path: float(value.mean().item())
                            for path, value in path_maps.items()
                        },
                    ))
                    if available:
                        del candidate
                    del replacement, residual_semantic, residual, delta
                    del clipped_delta
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                if active_semantics:
                    mean_semantic = torch.stack(
                        active_semantics, dim=0).mean(dim=0)
                else:
                    mean_semantic = anchor['semantic']
                mean_replace = self._ss_compose_semantic(
                    mean_semantic, anchor, anchor_instance, output_shape)
                mean_residual_semantic, mean_delta, mean_clipped = (
                    self._ss_bounded_semantic(
                        anchor['semantic'], mean_semantic))
                mean_residual = self._ss_compose_semantic(
                    mean_residual_semantic, anchor, anchor_instance,
                    output_shape)
                class_variants[aggregate_variant_name(
                    family, 'mean_semantic_replace')][class_idx].copy_(
                        mean_replace.detach().float().cpu())
                class_variants[aggregate_variant_name(
                    family, 'mean_semantic_residual')][class_idx].copy_(
                        mean_residual.detach().float().cpu())
                family_rows.append(dict(
                    class_index=int(class_idx),
                    class_name=item['name'],
                    family=family,
                    active_candidate_count=len(active_semantics),
                    aggregate='mean',
                    semantic_signed_delta_mean=float(
                        mean_delta.mean().item()),
                    semantic_abs_delta_mean=float(
                        mean_delta.abs().mean().item()),
                    semantic_clip_ratio=float((
                        mean_delta.abs()
                        > float(self.role_prompt_tta_semantic_residual_clip)
                    ).float().mean().item()),
                    clipped_abs_delta_mean=float(
                        mean_clipped.abs().mean().item()),
                    replace_final_mean=float(mean_replace.mean().item()),
                    residual_final_mean=float(mean_residual.mean().item()),
                ))
                del (
                    active_semantics, mean_semantic, mean_replace,
                    mean_residual_semantic, mean_delta, mean_clipped,
                    mean_residual,
                )
            del anchor

        if tuple(class_variants) != VARIANT_NAMES:
            raise RuntimeError(
                'Semantic-supplement variant order drifted: '
                f'{tuple(class_variants)}')
        raw_recomposition_max_abs = max(raw_recomposition_errors or [0.0])
        if (
                self.role_prompt_tta_strict_integrity
                and raw_recomposition_max_abs
                > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                'Raw-query native recomposition failed: max_abs='
                f'{raw_recomposition_max_abs}.')

        anchor_vs_official = float((
            class_variants['anchor_native']
            - class_variants['baseline']).abs().max().item())
        diagnostic_cpu_bytes = int(sum(
            value.numel() * value.element_size()
            for value in class_variants.values()))
        view_stats = dict(
            schema_version=1,
            protocol=self.role_prompt_tta_protocol,
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            adaptation_unit=(
                'sam3_crop' if crop_box is not None else 'full_image'),
            baseline_reconstruction_max_abs=float(
                self._rpt_native_parity_max_abs or 0.0),
            raw_recomposition_max_abs=float(raw_recomposition_max_abs),
            anchor_vs_official_max_abs=anchor_vs_official,
            official_query_count=int(self.num_queries),
            canonical_class_count=int(self.num_cls),
            synonym_query_count=int(self.num_queries - self.num_cls),
            diagnostic_variant_count=len(class_variants),
            diagnostic_cpu_bytes=diagnostic_cpu_bytes,
            semantic_residual_alpha=float(
                self.role_prompt_tta_semantic_residual_alpha),
            semantic_residual_clip=float(
                self.role_prompt_tta_semantic_residual_clip),
            candidate_rows=candidate_rows,
            family_rows=family_rows,
            cuda_memory=dict(main=_cuda_memory_snapshot(self.device)),
        )

        primary_query = baseline_query['final'].to(self.device)
        components = dict(
            semantic_logits=baseline_query['semantic'].to(self.device),
            instance_logits=baseline_query['instance'].to(self.device),
            role_prompt_variant_class_logits=class_variants,
            role_prompt_view_stats=[view_stats],
        )
        stats = dict(
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            role_prompt_tta=view_stats,
        )
        if not return_stats and not return_components:
            return primary_query
        if return_stats and return_components:
            return primary_query, stats, components
        if return_stats:
            return primary_query, stats
        return primary_query, components

    @staticmethod
    def _rft_bounded(anchor, candidate, alpha, clip):
        anchor = torch.as_tensor(anchor).float()
        candidate = torch.as_tensor(candidate).float()
        delta = candidate - anchor
        return (
            anchor + float(alpha) * delta.clamp(-float(clip), float(clip))
        ).clamp(0.0, 1.0), delta

    @staticmethod
    def _rft_merge_anchor_packages(packages):
        """Merge official aliases role-wise without changing their grounding."""
        semantic = torch.stack(
            [item['semantic'].float() for item in packages], dim=0
        ).max(dim=0)[0]
        presence = torch.stack([
            torch.as_tensor(item['presence']).float().reshape(())
            for item in packages
        ]).max()
        masks = [item['raw_masks'] for item in packages
                 if item['raw_masks'].numel()]
        scores = [item['raw_scores'] for item in packages
                  if item['raw_scores'].numel()]
        raw_masks = (
            torch.cat(masks, dim=0)
            if masks else torch.empty((0, 1, 1), dtype=torch.float32))
        raw_scores = (
            torch.cat(scores, dim=0)
            if scores else torch.empty((0,), dtype=torch.float32))
        return dict(
            semantic=semantic,
            presence=presence,
            raw_masks=raw_masks,
            raw_scores=raw_scores,
            native_instance=torch.stack([
                item['native_instance'].float() for item in packages
            ], dim=0).max(dim=0)[0],
            native_final=torch.stack([
                item['native_final'].float() for item in packages
            ], dim=0).max(dim=0)[0],
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

    def _rpt_role_functional_text_infer_single_view(
            self, image, return_stats=False, return_components=False,
            view_id=None, crop_box=None):
        """Screen frozen role texts and residual stability in one image pass.

        Image features are encoded once.  Every text still follows SAM3's
        native grounding path.  The official class-file queries form a shared
        anchor; a candidate can alter only one selected output role.  The
        resulting role-space change is added back to the exact official class
        output, so the no-update composition is numerically the baseline.
        """
        width, height = image.size
        output_shape = (height, width)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        cache = self._rpt_prepare_text_cache()

        with torch.no_grad(), self._rpt_autocast_context():
            state = self.processor.set_image(image)

        prompt_packages = {}
        parity_errors = []

        def ground(prompt, check_native=False):
            if prompt in prompt_packages:
                return prompt_packages[prompt]
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
            prompt_packages[prompt] = package
            return package

        baseline_packages = [
            ground(prompt, check_native=True) for prompt in self.query_words
        ]
        if not self._rpt_native_parity_checked:
            self._rpt_native_parity_max_abs = max(parity_errors or [0.0])
            self._rpt_native_parity_checked = True
            if (
                    self.role_prompt_tta_strict_integrity
                    and self._rpt_native_parity_max_abs
                    > self.role_prompt_tta_integrity_tolerance):
                raise RuntimeError(
                    'Cached role-text prompt path differs from native SAM3: '
                    f'max_abs={self._rpt_native_parity_max_abs}.')

        baseline_query = {
            'final': torch.stack([
                item['native_final'] for item in baseline_packages], dim=0),
            'semantic': torch.stack([
                item['semantic'] for item in baseline_packages], dim=0),
            'instance': torch.stack([
                item['native_instance'] for item in baseline_packages], dim=0),
        }
        baseline_class = self._rpt_query_to_class(
            baseline_query['final']).detach().float().cpu()
        class_variants = OrderedDict(
            (name, torch.empty(
                (int(self.num_cls), height, width),
                device='cpu', dtype=torch.float32))
            for name in ROLE_TEXT_VARIANT_NAMES)
        class_variants['baseline'].copy_(baseline_class)

        settings = {
            name: (float(alpha), float(clip))
            for name, alpha, clip in ROLE_TEXT_RESIDUAL_SETTINGS
        }
        default_alpha, default_clip = settings[ROLE_TEXT_DEFAULT_SETTING]
        candidate_rows = []
        class_rows = []
        native_recomposition_errors = []

        query_indices = self.query_idx.detach().long().cpu()
        for class_idx, item in enumerate(self._rpt_prompt_bank['classes']):
            official_indices = torch.nonzero(
                query_indices == int(class_idx), as_tuple=False
            ).flatten().tolist()
            official_packages = [
                baseline_packages[index] for index in official_indices
            ]
            anchor = self._rft_merge_anchor_packages(official_packages)
            anchor_presence = anchor['presence'].float()
            anchor_semantic = anchor['semantic'].float()
            instance_cache = {}

            def instance_for(package, presence):
                key = (
                    id(package),
                    float(torch.as_tensor(presence).float().item()),
                )
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
            official_final = baseline_class[class_idx]

            class_variants['semantic_head_anchor'][class_idx].copy_(
                anchor_semantic)
            class_variants['instance_head_anchor'][class_idx].copy_(
                anchor_instance)

            role_packages = {}
            for role, field, count in ROLE_TEXT_ROLE_FIELDS:
                role_packages[role] = [None]
                for slot, prompt in enumerate(item[field], 1):
                    package = ground(prompt)
                    role_packages[role].append(package)
                    candidate_instance, candidate_kept = instance_for(
                        package, anchor_presence)
                    candidate_rebuilt, _, _ = self._rpt_head_role_compose(
                        package, package, package, output_shape)
                    native_error = float((
                        candidate_rebuilt - package['native_final']
                    ).abs().max().item())
                    native_recomposition_errors.append(native_error)
                    prompt_index = int(cache['index'][prompt])
                    candidate_rows.append(dict(
                        class_index=int(class_idx),
                        class_name=item['name'],
                        role=role,
                        slot=int(slot),
                        candidate_prompt=prompt,
                        token_count=int((
                            ~cache['language_mask'][prompt_index].bool()
                        ).sum().item()),
                        anchor_presence=float(anchor_presence.item()),
                        candidate_presence=float(
                            torch.as_tensor(package['presence']).float().item()),
                        presence_delta=float((
                            torch.as_tensor(package['presence']).float()
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
                        native_recomposition_max_abs=native_error,
                    ))

                    if role == 'semantic':
                        value, _ = self._rft_bounded(
                            anchor_semantic, package['semantic'],
                            default_alpha, default_clip)
                        class_variants[
                            f'semantic_head_s{slot}'][class_idx].copy_(value)
                    elif role == 'instance':
                        value, _ = self._rft_bounded(
                            anchor_instance, candidate_instance,
                            default_alpha, default_clip)
                        class_variants[
                            f'instance_head_i{slot}'][class_idx].copy_(value)

            def compose(presence_slot=0, semantic_slot=0, instance_slot=0,
                        setting=ROLE_TEXT_DEFAULT_SETTING,
                        shared_package=None):
                alpha, clip = settings[setting]
                presence_package = (
                    shared_package or (
                        role_packages['presence'][presence_slot]
                        if presence_slot else None))
                semantic_package = (
                    shared_package or (
                        role_packages['semantic'][semantic_slot]
                        if semantic_slot else None))
                instance_package = (
                    shared_package or (
                        role_packages['instance'][instance_slot]
                        if instance_slot else None))

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
                # Exact baseline identity at the no-update point, while role
                # changes are measured relative to the common role anchor.
                return (
                    official_final.float()
                    + role_final.float() - role_anchor_final.float()
                ).clamp(0.0, 1.0)

            for presence_slot in range(3):
                for semantic_slot in range(4):
                    for instance_slot in range(3):
                        name = role_text_combo_variant_name(
                            presence_slot, semantic_slot, instance_slot)
                        value = compose(
                            presence_slot, semantic_slot, instance_slot)
                        class_variants[name][class_idx].copy_(value)

            for role, _, count in ROLE_TEXT_ROLE_FIELDS:
                for slot in range(1, count + 1):
                    for setting, _, _ in ROLE_TEXT_RESIDUAL_SETTINGS:
                        if setting == ROLE_TEXT_DEFAULT_SETTING:
                            continue
                        kwargs = {f'{role}_slot': slot, 'setting': setting}
                        value = compose(**kwargs)
                        name = role_text_sensitivity_variant_name(
                            role, slot, setting)
                        class_variants[name][class_idx].copy_(value)

            # The semantic candidate is also sent through all roles as the
            # shared-text control.  The role-specific path differs only in
            # where that fully grounded package is allowed to contribute.
            for slot in range(1, 4):
                value = compose(
                    shared_package=role_packages['semantic'][slot])
                class_variants[
                    f'shared_semantic_s{slot}'][class_idx].copy_(value)

            class_rows.append(dict(
                class_index=int(class_idx),
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

        if tuple(class_variants) != ROLE_TEXT_VARIANT_NAMES:
            raise RuntimeError(
                'Role-functional text variant order drifted: '
                f'{tuple(class_variants)}')
        identity_error = float((
            class_variants['combo_p0_s0_i0']
            - class_variants['baseline']).abs().max().item())
        recomposition_error = max(native_recomposition_errors or [0.0])
        if (
                self.role_prompt_tta_strict_integrity
                and max(identity_error, recomposition_error)
                > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                'Role-functional text integrity failed: '
                f'identity={identity_error}, '
                f'native_recomposition={recomposition_error}.')

        view_stats = dict(
            schema_version=1,
            protocol=self.role_prompt_tta_protocol,
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            adaptation_unit=(
                'sam3_crop' if crop_box is not None else 'full_image'),
            baseline_reconstruction_max_abs=float(
                self._rpt_native_parity_max_abs or 0.0),
            no_update_identity_max_abs=identity_error,
            native_recomposition_max_abs=recomposition_error,
            official_query_count=int(self.num_queries),
            canonical_class_count=int(self.num_cls),
            diagnostic_variant_count=len(class_variants),
            diagnostic_cpu_bytes=int(sum(
                value.numel() * value.element_size()
                for value in class_variants.values())),
            default_residual_setting=ROLE_TEXT_DEFAULT_SETTING,
            residual_settings=[dict(name=name, alpha=alpha, clip=clip)
                               for name, alpha, clip
                               in ROLE_TEXT_RESIDUAL_SETTINGS],
            candidate_rows=candidate_rows,
            class_rows=class_rows,
            cuda_memory=dict(main=_cuda_memory_snapshot(self.device)),
        )

        primary_query = baseline_query['final'].to(self.device)
        components = dict(
            semantic_logits=baseline_query['semantic'].to(self.device),
            instance_logits=baseline_query['instance'].to(self.device),
            role_prompt_variant_class_logits=class_variants,
            role_prompt_view_stats=[view_stats],
        )
        stats = dict(
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            role_prompt_tta=view_stats,
        )
        if not return_stats and not return_components:
            return primary_query
        if return_stats and return_components:
            return primary_query, stats, components
        if return_stats:
            return primary_query, stats
        return primary_query, components
