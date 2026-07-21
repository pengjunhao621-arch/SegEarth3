import hashlib
import json
import os

import torch
from torch import nn
import torch.nn.functional as F


REVIEWER_SOURCE_NAMES = (
    'final',
    'semantic',
    'instance',
    'presence',
    'fusion_no_presence',
    'semantic_presence',
    'instance_presence',
    'raw_object',
    'raw_presence',
    'encoder_level_0',
    'encoder_level_1',
    'encoder_level_2',
)

REVIEWER_VARIANT_SOURCES = {
    'output_only': ('final',),
    'three_head': ('final', 'semantic', 'instance', 'presence'),
    'internal_trajectory': REVIEWER_SOURCE_NAMES,
}


def reviewer_variant_mask(variant, device=None):
    variant = str(variant).strip().lower()
    if variant not in REVIEWER_VARIANT_SOURCES:
        raise ValueError(
            f'Unknown reviewer variant {variant!r}; expected one of '
            f'{sorted(REVIEWER_VARIANT_SOURCES)}.')
    enabled = set(REVIEWER_VARIANT_SOURCES[variant])
    return torch.tensor(
        [name in enabled for name in REVIEWER_SOURCE_NAMES],
        dtype=torch.bool,
        device=device,
    )


class InternalTrajectoryReviewer(nn.Module):
    """Small matched-capacity reviewer for all evidence variants."""

    def __init__(
            self,
            topk=3,
            source_feature_dim=3,
            candidate_aux_dim=8,
            hidden_dim=128,
            num_heads=4,
            source_layers=2,
            candidate_layers=2,
            dropout=0.1):
        super().__init__()
        self.topk = int(topk)
        self.source_feature_dim = int(source_feature_dim)
        self.candidate_aux_dim = int(candidate_aux_dim)
        self.hidden_dim = int(hidden_dim)

        self.source_projection = nn.Sequential(
            nn.Linear(self.source_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.source_embedding = nn.Parameter(
            torch.empty(len(REVIEWER_SOURCE_NAMES), hidden_dim))
        nn.init.normal_(self.source_embedding, std=0.02)

        source_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(
            source_layer, num_layers=source_layers)

        self.candidate_aux_projection = nn.Sequential(
            nn.Linear(self.candidate_aux_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.candidate_rank_embedding = nn.Parameter(
            torch.empty(self.topk, hidden_dim))
        nn.init.normal_(self.candidate_rank_embedding, std=0.02)

        candidate_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(
            candidate_layer, num_layers=candidate_layers)
        self.action_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.recoverable_head = nn.Linear(hidden_dim, 1)

    def forward(
            self, source_features, source_mask, candidate_aux,
            variant_source_mask):
        batch_size, topk, source_count, _ = source_features.shape
        if topk != self.topk:
            raise ValueError(
                f'Expected topk={self.topk}, got {topk}.')
        if source_count != len(REVIEWER_SOURCE_NAMES):
            raise ValueError(
                f'Expected {len(REVIEWER_SOURCE_NAMES)} sources, '
                f'got {source_count}.')

        active = (
            source_mask.bool()
            & variant_source_mask.view(1, 1, source_count).bool())
        flat_features = source_features.reshape(
            batch_size * topk, source_count, -1)
        flat_active = active.reshape(batch_size * topk, source_count)
        source_tokens = self.source_projection(flat_features)
        source_tokens = source_tokens + self.source_embedding.unsqueeze(0)
        source_tokens = self.source_encoder(
            source_tokens,
            src_key_padding_mask=~flat_active,
        )
        active_float = flat_active.float().unsqueeze(-1)
        candidate_tokens = (
            source_tokens * active_float).sum(dim=1)
        candidate_tokens = candidate_tokens / active_float.sum(
            dim=1).clamp_min(1.0)
        candidate_tokens = candidate_tokens.view(
            batch_size, topk, self.hidden_dim)
        candidate_tokens = (
            candidate_tokens
            + self.candidate_aux_projection(candidate_aux)
            + self.candidate_rank_embedding.unsqueeze(0)
        )
        candidate_tokens = self.candidate_encoder(candidate_tokens)

        action_logits = self.action_head(candidate_tokens).squeeze(-1)
        pooled = candidate_tokens.mean(dim=1)
        risk_logits = self.risk_head(pooled).squeeze(-1)
        recoverable_logits = self.recoverable_head(pooled).squeeze(-1)
        return {
            'action_logits': action_logits,
            'risk_logits': risk_logits,
            'recoverable_logits': recoverable_logits,
        }


def build_reviewer_from_checkpoint(checkpoint, device):
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model_kwargs = dict(checkpoint.get('model_kwargs') or {})
    model = InternalTrajectoryReviewer(**model_kwargs)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model, checkpoint


class RethinkingReviewerMixin:
    """Feature caching and prediction-preserving reviewer integration."""

    def _uses_rethinking_reviewer(self):
        return bool(
            self.dump_reviewer_cache
            or self.use_learned_reviewer
            or self.dump_learned_reviewer_stats)

    def _reviewer_internal_sources(self):
        if self.dump_reviewer_cache:
            enabled = set(REVIEWER_SOURCE_NAMES)
        else:
            enabled = set(REVIEWER_VARIANT_SOURCES.get(
                str(self.reviewer_variant).strip().lower(),
                REVIEWER_SOURCE_NAMES,
            ))
        return tuple(
            name for name in REVIEWER_SOURCE_NAMES
            if name in enabled
            and name not in ('final', 'semantic', 'instance', 'presence')
        )

    def _reviewer_diag_shape(self, height, width):
        max_side = max(1, int(self.reviewer_max_side))
        scale = min(1.0, float(max_side) / float(max(height, width)))
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _aggregate_query_scalar_to_classes(self, query_values):
        class_values = torch.full(
            (self.num_cls,),
            float('-inf'),
            device=self.device,
            dtype=torch.float32,
        )
        query_values = query_values.detach().float().flatten()
        count = min(int(query_values.numel()), int(self.query_idx.numel()))
        for query_id in range(count):
            class_id = int(self.query_idx[query_id].item())
            class_values[class_id] = torch.maximum(
                class_values[class_id], query_values[query_id])
        class_values[~torch.isfinite(class_values)] = 0.0
        return class_values

    def _reviewer_map_bank(
            self, base_logits, components, output_shape):
        maps = {}
        maps['final'] = self._interpolate_float32(
            base_logits.detach().unsqueeze(0), output_shape).squeeze(0)
        for name in ('semantic', 'instance'):
            value = None if components is None else components.get(
                f'{name}_logits')
            if isinstance(value, torch.Tensor):
                maps[name] = self._interpolate_float32(
                    value.detach().unsqueeze(0), output_shape).squeeze(0)

        presence_query = (
            None if components is None
            else components.get('presence_query_scores'))
        if isinstance(presence_query, torch.Tensor):
            presence = self._aggregate_query_scalar_to_classes(
                presence_query)
            maps['presence'] = presence[:, None, None].expand(
                -1, output_shape[0], output_shape[1])

        internal_maps = (
            {} if components is None
            else components.get('internal_source_maps', {}))
        for name in self._reviewer_internal_sources():
            value = internal_maps.get(name)
            if isinstance(value, torch.Tensor):
                maps[name] = self._interpolate_float32(
                    value.detach().unsqueeze(0), output_shape).squeeze(0)
        return maps

    def _reviewer_candidates(self, final_logits, base_pred):
        class_count, height, width = final_logits.shape
        topk = max(2, int(self.reviewer_topk))
        top_count = min(class_count, topk + 1)
        top_idx = torch.topk(
            final_logits, k=top_count, dim=0).indices
        candidates = torch.full(
            (topk, height, width),
            -1,
            dtype=torch.long,
            device=final_logits.device,
        )
        candidates[0] = base_pred.long()
        for slot in range(1, topk):
            for source_rank in range(top_count):
                proposed = top_idx[source_rank]
                duplicate = (
                    candidates[:slot] == proposed.unsqueeze(0)).any(dim=0)
                fill = (candidates[slot] < 0) & ~duplicate
                candidates[slot][fill] = proposed[fill]
        candidates[candidates < 0] = candidates[0].expand_as(
            candidates)[candidates < 0]
        return candidates

    def _build_reviewer_features(
            self, base_logits, base_pred, components):
        output_shape = self._reviewer_diag_shape(
            int(base_logits.shape[-2]), int(base_logits.shape[-1]))
        final_logits = self._interpolate_float32(
            base_logits.detach().unsqueeze(0), output_shape).squeeze(0)
        base_pred_grid = F.interpolate(
            base_pred.detach().float().view(1, 1, *base_pred.shape),
            size=output_shape,
            mode='nearest',
        ).view(*output_shape).long()
        candidates = self._reviewer_candidates(
            final_logits, base_pred_grid)
        map_bank = self._reviewer_map_bank(
            base_logits, components, output_shape)

        pixel_count = int(output_shape[0] * output_shape[1])
        candidate_flat = candidates.view(
            int(self.reviewer_topk), pixel_count).transpose(0, 1)
        source_features = torch.zeros(
            (
                pixel_count,
                int(self.reviewer_topk),
                len(REVIEWER_SOURCE_NAMES),
                3,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        source_mask = torch.zeros(
            (
                pixel_count,
                int(self.reviewer_topk),
                len(REVIEWER_SOURCE_NAMES),
            ),
            device=self.device,
            dtype=torch.bool,
        )

        for source_id, source_name in enumerate(REVIEWER_SOURCE_NAMES):
            source_map = map_bank.get(source_name)
            if not isinstance(source_map, torch.Tensor):
                continue
            source_map = source_map.detach().float()
            if source_map.shape[-2:] != output_shape:
                source_map = self._interpolate_float32(
                    source_map.unsqueeze(0), output_shape).squeeze(0)
            flat = source_map.view(self.num_cls, pixel_count).transpose(0, 1)
            finite = torch.isfinite(flat)
            clean = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            valid_count = finite.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (clean * finite.float()).sum(
                dim=1, keepdim=True) / valid_count
            variance = (
                (clean - mean).square() * finite.float()).sum(
                    dim=1, keepdim=True) / valid_count
            zscore = (clean - mean) / variance.sqrt().clamp_min(1e-5)
            gathered = torch.gather(clean, 1, candidate_flat)
            gathered_z = torch.gather(zscore, 1, candidate_flat)
            gathered_valid = torch.gather(finite, 1, candidate_flat)
            baseline_z = gathered_z[:, :1]
            candidate_rank = (
                clean.unsqueeze(1) > gathered.unsqueeze(-1)).float().sum(
                    dim=-1)
            rank_quality = 1.0 - candidate_rank / float(
                max(1, self.num_cls - 1))
            source_features[:, :, source_id, 0] = gathered_z
            source_features[:, :, source_id, 1] = (
                gathered_z - baseline_z)
            source_features[:, :, source_id, 2] = rank_quality
            source_mask[:, :, source_id] = gathered_valid

        final_flat = final_logits.view(
            self.num_cls, pixel_count).transpose(0, 1)
        final_gathered = torch.gather(final_flat, 1, candidate_flat)
        final_mean = final_flat.mean(dim=1, keepdim=True)
        final_std = final_flat.std(
            dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        final_z = (final_gathered - final_mean) / final_std
        baseline_final = final_gathered[:, :1]
        raw_top1 = final_flat.argmax(dim=1, keepdim=True)

        one_hot = F.one_hot(
            base_pred_grid.clamp(0, self.num_cls - 1),
            num_classes=self.num_cls,
        ).permute(2, 0, 1).float().unsqueeze(0)
        local = F.avg_pool2d(
            one_hot,
            kernel_size=int(self.reviewer_local_kernel),
            stride=1,
            padding=int(self.reviewer_local_kernel) // 2,
        ).squeeze(0).view(self.num_cls, pixel_count).transpose(0, 1)
        local_support = torch.gather(local, 1, candidate_flat)

        slot_rank = torch.linspace(
            1.0,
            0.0,
            steps=int(self.reviewer_topk),
            device=self.device,
        ).view(1, -1).expand(pixel_count, -1)
        threshold_reject = (
            final_flat.max(dim=1, keepdim=True)[0]
            < float(self.prob_thd)).float()
        candidate_aux = torch.stack(
            (
                torch.zeros_like(final_gathered).scatter_(
                    1,
                    torch.zeros(
                        (pixel_count, 1),
                        dtype=torch.long,
                        device=self.device,
                    ),
                    1.0,
                ),
                final_z,
                final_gathered - baseline_final,
                slot_rank,
                local_support,
                threshold_reject.expand(-1, int(self.reviewer_topk)),
                (candidate_flat == int(self.bg_idx)).float(),
                (candidate_flat == raw_top1).float(),
            ),
            dim=-1,
        )
        return {
            'source_features': source_features,
            'source_mask': source_mask,
            'candidate_aux': candidate_aux,
            'candidates': candidate_flat,
            'base_pred': base_pred_grid.reshape(-1),
            'output_shape': output_shape,
        }

    @staticmethod
    def _sample_reviewer_indices(category_masks, max_samples, seed):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed) & 0x7fffffff)
        categories = []
        for mask in category_masks:
            categories.append(torch.nonzero(
                mask.detach().cpu(), as_tuple=False).flatten())
        target = max(1, int(max_samples) // max(1, len(categories)))
        selected = []
        leftovers = []
        for indices in categories:
            if indices.numel() > target:
                order = torch.randperm(
                    indices.numel(), generator=generator)
                selected.append(indices[order[:target]])
                leftovers.append(indices[order[target:]])
            else:
                selected.append(indices)
        selected_count = sum(int(value.numel()) for value in selected)
        remaining = max(0, int(max_samples) - selected_count)
        if remaining > 0 and leftovers:
            pool = torch.cat(leftovers)
            if pool.numel() > remaining:
                order = torch.randperm(
                    pool.numel(), generator=generator)
                pool = pool[order[:remaining]]
            selected.append(pool)
        if not selected:
            return torch.empty(0, dtype=torch.long)
        result = torch.cat(selected)
        if result.numel() > 1:
            order = torch.randperm(result.numel(), generator=generator)
            result = result[order]
        return result

    def _dump_reviewer_cache_record(
            self, image_path, data_sample, base_logits, base_pred, components):
        if not self.dump_reviewer_cache:
            return
        gt = data_sample.gt_sem_seg.data
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        gt = gt.to(self.device)
        features = self._build_reviewer_features(
            base_logits, base_pred, components)
        output_shape = features['output_shape']
        gt_grid = F.interpolate(
            gt.detach().float().view(1, 1, *gt.shape),
            size=output_shape,
            mode='nearest',
        ).view(-1).long()
        candidates = features['candidates']
        base_pred_grid = features['base_pred']
        valid = (gt_grid >= 0) & (gt_grid < int(self.num_cls))
        baseline_correct = valid & (base_pred_grid == gt_grid)
        matches = candidates == gt_grid.unsqueeze(1)
        recoverable = valid & ~baseline_correct & matches.any(dim=1)
        unrecoverable = valid & ~baseline_correct & ~matches.any(dim=1)
        if candidates.shape[1] > 1:
            alt_margin = features['candidate_aux'][:, 1:, 2].amax(dim=1)
        else:
            alt_margin = torch.full_like(gt_grid.float(), float('-inf'))
        hard_keep = (
            baseline_correct
            & (alt_margin >= -float(self.reviewer_cache_hard_keep_margin))
        )
        easy_keep = baseline_correct & ~hard_keep
        action_target = torch.full_like(gt_grid, -100)
        action_target[baseline_correct] = 0
        if recoverable.any():
            action_target[recoverable] = matches[recoverable].float().argmax(
                dim=1).long()

        digest = hashlib.sha1(
            str(image_path).encode('utf-8')).hexdigest()
        seed = int(digest[:8], 16)
        selected = self._sample_reviewer_indices(
            (easy_keep, hard_keep, recoverable, unrecoverable),
            int(self.reviewer_cache_samples_per_image),
            seed,
        )
        if selected.numel() == 0:
            return
        selected_device = selected.to(self.device)
        sample_weight = torch.ones(
            selected_device.numel(),
            device=self.device,
            dtype=torch.float32,
        )
        for category in (
                easy_keep, hard_keep, recoverable, unrecoverable):
            category_total = int(category.sum().item())
            selected_category = category[selected_device]
            selected_count = int(selected_category.sum().item())
            if category_total > 0 and selected_count > 0:
                sample_weight[selected_category] = (
                    float(category_total) / float(selected_count))
        record = {
            'format_version': 1,
            'source_names': REVIEWER_SOURCE_NAMES,
            'dataset_name': self.reviewer_dataset_name,
            'image_path': str(image_path),
            'num_classes': int(self.num_cls),
            'bg_idx': int(self.bg_idx),
            'prob_thd': float(self.prob_thd),
            'confidence_threshold': float(self.confidence_threshold),
            'output_shape': tuple(int(v) for v in output_shape),
            'source_features': features['source_features'][
                selected_device].half().cpu(),
            'source_mask': features['source_mask'][
                selected_device].cpu(),
            'candidate_aux': features['candidate_aux'][
                selected_device].half().cpu(),
            'candidates': candidates[selected_device].short().cpu(),
            'gt': gt_grid[selected_device].short().cpu(),
            'base_pred': base_pred_grid[selected_device].short().cpu(),
            'risk_target': (
                valid & ~baseline_correct)[selected_device].float().cpu(),
            'recoverable_target': recoverable[
                selected_device].float().cpu(),
            'action_target': action_target[selected_device].cpu(),
            'baseline_correct': baseline_correct[
                selected_device].cpu(),
            'hard_keep_target': hard_keep[selected_device].cpu(),
            'sample_weight': sample_weight.cpu(),
        }
        rank = int(os.environ.get('RANK', 0))
        output_dir = os.path.join(
            str(self.reviewer_cache_dir), f'rank{rank}')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{digest}.pt')
        temporary_path = output_path + '.tmp'
        torch.save(record, temporary_path)
        os.replace(temporary_path, output_path)

    def _load_learned_reviewer(self):
        if self._learned_reviewer_model is not None:
            return self._learned_reviewer_model
        if not self.reviewer_checkpoint:
            raise ValueError(
                'use_learned_reviewer=True requires reviewer_checkpoint.')
        model, checkpoint = build_reviewer_from_checkpoint(
            self.reviewer_checkpoint, self.device)
        checkpoint_topk = int(
            (checkpoint.get('model_kwargs') or {}).get('topk', 3))
        if checkpoint_topk != int(self.reviewer_topk):
            raise ValueError(
                f'Reviewer checkpoint topk={checkpoint_topk} does not '
                f'match config topk={self.reviewer_topk}.')
        checkpoint_variant = str(
            checkpoint.get('variant') or self.reviewer_variant)
        if checkpoint_variant != str(self.reviewer_variant):
            raise ValueError(
                f'Reviewer checkpoint variant={checkpoint_variant!r} does '
                f'not match config variant={self.reviewer_variant!r}.')
        self._reviewer_checkpoint_meta = checkpoint
        self._learned_reviewer_model = model
        return model

    def _apply_learned_reviewer(
            self, base_logits, base_pred, components):
        if not self.use_learned_reviewer:
            return base_logits, None
        features = self._build_reviewer_features(
            base_logits, base_pred, components)
        model = self._load_learned_reviewer()
        variant_mask = reviewer_variant_mask(
            self.reviewer_variant, device=self.device)
        row_count = int(features['source_features'].shape[0])
        action_parts = []
        risk_parts = []
        recoverable_parts = []
        with torch.no_grad():
            for start in range(
                    0, row_count, int(self.reviewer_inference_batch_size)):
                end = min(
                    row_count,
                    start + int(self.reviewer_inference_batch_size))
                output = model(
                    features['source_features'][start:end],
                    features['source_mask'][start:end],
                    features['candidate_aux'][start:end],
                    variant_mask,
                )
                action_parts.append(output['action_logits'])
                risk_parts.append(output['risk_logits'])
                recoverable_parts.append(output['recoverable_logits'])
        action_logits = torch.cat(action_parts)
        risk = torch.sigmoid(torch.cat(risk_parts))
        recoverable = torch.sigmoid(torch.cat(recoverable_parts))
        action_probability = torch.softmax(action_logits, dim=1)
        selected_slot = action_probability.argmax(dim=1)
        selected_probability = torch.gather(
            action_probability, 1, selected_slot.unsqueeze(1)).squeeze(1)
        keep_probability = action_probability[:, 0]

        checkpoint = self._reviewer_checkpoint_meta or {}
        gate_threshold = (
            float(self.reviewer_gate_threshold)
            if self.reviewer_gate_threshold is not None
            else float(checkpoint.get('gate_threshold', 0.5)))
        action_margin = (
            float(self.reviewer_action_margin)
            if self.reviewer_action_margin is not None
            else float(checkpoint.get('action_margin', 0.0)))
        gate_score = risk * recoverable
        selected_class = torch.gather(
            features['candidates'],
            1,
            selected_slot.unsqueeze(1),
        ).squeeze(1)
        change = (
            (selected_slot > 0)
            & (gate_score >= gate_threshold)
            & ((selected_probability - keep_probability) >= action_margin)
            & (selected_class != features['base_pred'])
        )

        grid_shape = features['output_shape']
        selected_grid = selected_class.view(*grid_shape).float()
        change_grid = change.view(*grid_shape).float()
        full_shape = base_logits.shape[-2:]
        selected_full = F.interpolate(
            selected_grid.view(1, 1, *grid_shape),
            size=full_shape,
            mode='nearest',
        ).view(*full_shape).long()
        change_full = F.interpolate(
            change_grid.view(1, 1, *grid_shape),
            size=full_shape,
            mode='nearest',
        ).view(*full_shape) > 0.5

        reviewed = base_logits.clone()
        flat_reviewed = reviewed.view(self.num_cls, -1)
        flat_selected = selected_full.reshape(-1)
        flat_change = change_full.reshape(-1)
        if flat_change.any():
            positions = torch.nonzero(
                flat_change, as_tuple=False).flatten()
            chosen = flat_selected[positions]
            current_top = flat_reviewed[:, positions].max(dim=0)[0]
            target = current_top + float(self.reviewer_min_boost)
            foreground = chosen != int(self.bg_idx)
            target[foreground] = torch.maximum(
                target[foreground],
                torch.full_like(
                    target[foreground],
                    float(self.prob_thd)
                    + float(self.reviewer_min_boost),
                ),
            )
            flat_reviewed[chosen, positions] = target

        context = {
            'gate_threshold': gate_threshold,
            'action_margin': action_margin,
            'grid_pixels': row_count,
            'changed_grid_pixels': int(change.sum().item()),
            'changed_grid_ratio': float(change.float().mean().item()),
            'mean_risk': float(risk.mean().item()),
            'mean_recoverable': float(recoverable.mean().item()),
        }
        return reviewed, context

    def _write_learned_reviewer_stats(self, record):
        if not self.dump_learned_reviewer_stats:
            return
        if self._learned_reviewer_stats_file is None:
            path = (
                self.learned_reviewer_stats_path
                or './work_dirs/rethinking_reviewer/stats.jsonl')
            rank = int(os.environ.get('RANK', 0))
            root, extension = os.path.splitext(path)
            path = f'{root}.rank{rank}{extension or ".jsonl"}'
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self._learned_reviewer_stats_file = open(
                path, 'a', buffering=1)
        self._learned_reviewer_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')
