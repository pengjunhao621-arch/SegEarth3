"""Role-aware, label-free test-time prompt adaptation for SegEarth-OV3.

The module deliberately keeps SAM3's native joint grounding path intact.  It
separates semantic, instance, Presence, and RemoteCLIP evidence only when
constructing a bounded per-image prompt update.  The adapted language condition
is then sent back through the same SAM3 encoder, decoder, heads, candidate
filter, and SegEarth-OV3 fusion used by the protected baseline.

The inexpensive output-space optimization is retained as a diagnostic control.
The main ``full_regrounded_e2e`` variant additionally performs a bounded
gradient update through frozen SAM3 grounding, one eligible class at a time, so
the method remains feasible on the older server stack.
"""

import contextlib
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from sam3.model.data_misc import interpolate as sam3_interpolate

from role_prompt_tta_definitions import (
    SCHEMA_VERSION,
    VARIANT_NAMES as V1_VARIANT_NAMES,
)
from prompt_functional_atlas_definitions import (
    PROMPT_COUNT as ATLAS_PROMPT_COUNT,
    PROMPT_SLOTS as ATLAS_PROMPT_SLOTS,
    SCHEMA_VERSION as ATLAS_SCHEMA_VERSION,
    VARIANT_NAMES as ATLAS_VARIANT_NAMES,
)
from head_role_prompt_conflict_definitions import (
    DESCRIPTION_SLOTS as HEAD_ROLE_DESCRIPTION_SLOTS,
    HEAD_PATHS as HEAD_ROLE_PATHS,
    PROMPT_COUNT as HEAD_ROLE_PROMPT_COUNT,
    SCHEMA_VERSION as HEAD_ROLE_SCHEMA_VERSION,
    VARIANT_NAMES as HEAD_ROLE_VARIANT_NAMES,
    variant_name as head_role_variant_name,
)
from semantic_supplement_definitions import (
    PROTOCOL as SEMANTIC_SUPPLEMENT_PROTOCOL,
    SCHEMA_VERSION as SEMANTIC_SUPPLEMENT_SCHEMA_VERSION,
    VARIANT_NAMES as SEMANTIC_SUPPLEMENT_VARIANT_NAMES,
    load_semantic_supplement_bank,
)
from semantic_supplement_screen import SemanticSupplementScreenMixin

# Backward-compatible export used by existing configs/tests/summarization.
VARIANT_NAMES = V1_VARIANT_NAMES

def _safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def _safe_log(value, eps):
    return torch.log(value.clamp_min(float(eps)))


def resolve_remoteclip_device(requested, main_device):
    """Resolve an optional RemoteCLIP device without changing DDP ownership.

    ``aux`` reserves one additional visible CUDA device per local DDP rank.
    For example, with four visible GPUs and two ranks, SAM3 uses cuda:0/1 and
    RemoteCLIP uses cuda:2/3.  This is model placement, not model parallelism.
    """
    main_device = torch.device(main_device)
    requested = 'same' if requested is None else str(requested).strip().lower()
    if requested in ('', 'same', 'main'):
        return main_device
    if requested == 'aux':
        if main_device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError(
                'role_prompt_tta_remoteclip_device=aux requires CUDA.')
        local_rank = int(os.environ.get('LOCAL_RANK', main_device.index or 0))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        candidate = local_rank + local_world_size
        device_count = int(torch.cuda.device_count())
        if candidate >= device_count:
            raise RuntimeError(
                'RemoteCLIP aux placement needs at least two visible CUDA '
                'devices per local evaluation rank. For two ranks use, for '
                'example, GPU_LIST=0,1,2,3 NPROC=2 '
                'REMOTECLIP_DEVICE=aux; do not use NPROC=4.')
        return torch.device(f'cuda:{candidate}')
    device = torch.device(requested)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                f'RemoteCLIP device {requested!r} requires CUDA.')
        index = 0 if device.index is None else int(device.index)
        if index >= int(torch.cuda.device_count()):
            raise RuntimeError(
                f'RemoteCLIP device {requested!r} is not visible; '
                f'visible CUDA device count={torch.cuda.device_count()}.')
        device = torch.device(f'cuda:{index}')
    return device


def cuda_memory_snapshot(device):
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


def _ranked_jsonl_path(path):
    stem, suffix = os.path.splitext(path)
    return f'{stem}.rank{int(os.environ.get("RANK", 0))}{suffix or ".jsonl"}'


def _tensor_tree_grad_safe_clone(value, device=None):
    """Clone inference-mode image tensors into ordinary tensors for grad."""
    if isinstance(value, torch.Tensor):
        # Sam3Processor.set_image is decorated with inference_mode.  Such
        # tensors cannot be saved for backward even when only prompt weights
        # require gradients, so a real clone is required here.
        result = value.detach().clone()
        return result.to(device) if device is not None else result
    if isinstance(value, dict):
        return {
            key: _tensor_tree_grad_safe_clone(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_tensor_tree_grad_safe_clone(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _tensor_tree_grad_safe_clone(item, device) for item in value)
    return value


def load_prompt_bank(path, expected_class_names=None):
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if int(payload.get('schema_version', -1)) != SCHEMA_VERSION:
        raise ValueError(
            f'Prompt bank {path} has schema_version='
            f'{payload.get("schema_version")!r}; expected {SCHEMA_VERSION}.')
    classes = payload.get('classes')
    if not isinstance(classes, list) or not classes:
        raise ValueError(f'Prompt bank {path} must contain a non-empty classes list.')
    ids = [int(item.get('id', -1)) for item in classes]
    if ids != list(range(len(classes))):
        raise ValueError(
            f'Prompt bank {path} class IDs must be contiguous and ordered; got {ids}.')
    prompt_count = None
    for item in classes:
        descriptions = item.get('descriptions')
        if not isinstance(descriptions, list) or len(descriptions) < 2:
            raise ValueError(
                f'Class {item.get("name")!r} needs at least two descriptions.')
        if prompt_count is None:
            prompt_count = len(descriptions)
        elif len(descriptions) != prompt_count:
            raise ValueError('All classes must use the same description count.')
        name = str(item.get('name', '')).strip()
        if descriptions[0].strip().lower() != name.lower():
            raise ValueError(
                f'Class {name!r} must use its literal class name as description 0.')
        for description in descriptions:
            if name.lower() not in str(description).lower():
                raise ValueError(
                    f'Every description must retain literal class identity {name!r}: '
                    f'{description!r}')
    if expected_class_names is not None:
        expected = [str(name).strip().lower() for name in expected_class_names]
        observed = [str(item['name']).strip().lower() for item in classes]
        if observed != expected:
            raise ValueError(
                f'Prompt bank class order mismatch. expected={expected}, observed={observed}')
    payload['_path'] = os.path.abspath(path)
    payload['_prompt_count'] = int(prompt_count)
    return payload


def initial_prompt_weights(class_count, prompt_count, anchor_mass, device):
    if prompt_count < 2:
        return torch.ones(class_count, prompt_count, device=device)
    anchor_mass = float(anchor_mass)
    if not (0.0 < anchor_mass < 1.0):
        raise ValueError('role_prompt_tta_anchor_mass must be in (0, 1).')
    weights = torch.full(
        (class_count, prompt_count),
        (1.0 - anchor_mass) / float(prompt_count - 1),
        device=device,
        dtype=torch.float32,
    )
    weights[:, 0] = anchor_mass
    return weights


def mask_aware_language_fusion(features, masks, weights, eps=1e-6):
    """Fuse complete padded SAM3 language sequences without using pad tokens.

    Args:
        features: ``[K, L, D]`` contextual SAM3 language features.
        masks: ``[K, L]`` where True denotes a padded/invalid token.
        weights: ``[K]`` normalized description weights.
    """
    if features.ndim != 3 or masks.ndim != 2 or weights.ndim != 1:
        raise ValueError('Unexpected language fusion tensor ranks.')
    if features.shape[:2] != masks.shape or features.shape[0] != weights.shape[0]:
        raise ValueError('Language fusion shape mismatch.')
    valid = (~masks.bool()).to(dtype=features.dtype)
    weighted_valid = weights.to(features.dtype)[:, None] * valid
    denominator = weighted_valid.sum(dim=0)
    numerator = (
        features * weighted_valid[:, :, None]
    ).sum(dim=0)
    fused = numerator / denominator.clamp_min(float(eps))[:, None]
    fused_mask = denominator <= float(eps)
    fused[fused_mask] = 0
    return fused, fused_mask


def normalized_entropy(class_logits, temperature=1.0, eps=1e-6):
    if class_logits.ndim != 3:
        raise ValueError('class_logits must be [C,H,W].')
    temperature = max(float(temperature), float(eps))
    probabilities = torch.softmax(class_logits.float() / temperature, dim=0)
    entropy = -(probabilities * _safe_log(probabilities, eps)).sum(dim=0)
    normalizer = math.log(max(int(class_logits.shape[0]), 2))
    return probabilities, entropy / normalizer


def effective_prompt_count(weights, eps=1e-6):
    entropy = -(weights * _safe_log(weights, eps)).sum(dim=-1)
    return torch.exp(entropy)


def prompt_weights_from_state(
        anchor_weights, visual_affinity, presence_gate, theta,
        visual_strength, delta_max, eps=1e-6):
    logits = _safe_log(anchor_weights, eps)
    if visual_affinity is not None:
        logits = logits + (
            float(visual_strength)
            * presence_gate[:, None]
            * visual_affinity.float()
        )
    if theta is not None:
        logits = logits + (
            float(delta_max)
            * presence_gate[:, None]
            * torch.tanh(theta))
    return torch.softmax(logits, dim=-1)


def prompt_weight_objective(
        semantic_raw, weights, anchor_weights, anchor_top1,
        temperature, anchor_lambda, class_balanced, eps=1e-6):
    class_logits = (
        weights[:, :, None, None] * semantic_raw.float()
    ).sum(dim=1)
    _, entropy = normalized_entropy(class_logits, temperature, eps)
    if class_balanced:
        class_terms = []
        for class_idx in range(int(class_logits.shape[0])):
            mask = anchor_top1 == class_idx
            if mask.any():
                class_terms.append(entropy[mask].mean())
        entropy_loss = (
            torch.stack(class_terms).mean()
            if class_terms else entropy.mean())
    else:
        entropy_loss = entropy.mean()
    kl = (
        weights * (
            _safe_log(weights, eps) - _safe_log(anchor_weights, eps)
        )
    ).sum(dim=-1).mean()
    loss = entropy_loss + float(anchor_lambda) * kl
    return loss, entropy_loss, kl, class_logits


def optimize_surrogate_weights(
        semantic_raw, anchor_weights, visual_affinity, presence_gate,
        anchor_top1, steps, lr, temperature, anchor_lambda,
        visual_strength, delta_max, class_balanced=True, eps=1e-6):
    # MMEngine's test loop executes model prediction under torch.no_grad().
    # This helper is an explicit test-time optimizer, so its tiny theta-only
    # graph must be constructed inside a local grad-enabled scope. The input
    # maps remain detached and no model parameter is registered with Adam.
    with torch.enable_grad():
        theta = torch.zeros_like(anchor_weights, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=float(lr))
        trajectory = []
        for step in range(int(steps)):
            optimizer.zero_grad()
            weights = prompt_weights_from_state(
                anchor_weights, visual_affinity, presence_gate, theta,
                visual_strength, delta_max, eps)
            loss, entropy_loss, kl, _ = prompt_weight_objective(
                semantic_raw, weights, anchor_weights, anchor_top1,
                temperature, anchor_lambda, class_balanced, eps)
            loss.backward()
            if theta.grad is None:
                raise RuntimeError(
                    'Surrogate prompt weights are disconnected from the '
                    'test-time objective.')
            grad_norm = theta.grad.detach().float().norm()
            optimizer.step()
            trajectory.append(dict(
                step=int(step + 1),
                loss=float(loss.detach().item()),
                entropy=float(entropy_loss.detach().item()),
                anchor_kl=float(kl.detach().item()),
                grad_norm=float(grad_norm.item()),
            ))
    with torch.no_grad():
        weights = prompt_weights_from_state(
            anchor_weights, visual_affinity, presence_gate, theta,
            visual_strength, delta_max, eps)
    return weights.detach(), theta.detach(), trajectory


class RemoteCLIPRuntime:
    """Lazy, unregistered RemoteCLIP runtime using SCORE's OpenCLIP copy."""

    def __init__(self, checkpoint, source_root, model_name, device):
        self.checkpoint = checkpoint
        self.source_root = source_root
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.text_cache = None

    def _load(self):
        if self.model is not None:
            return
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(
                'RemoteCLIP checkpoint is required by this experiment: '
                f'{self.checkpoint}')
        source_path = os.path.join(
            os.path.abspath(self.source_root), 'open_clip_training', 'src')
        if not os.path.isdir(source_path):
            raise FileNotFoundError(
                f'Could not find SCORE OpenCLIP source directory: {source_path}')
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        import open_clip  # pylint: disable=import-error,import-outside-toplevel

        model = open_clip.create_model(self.model_name, pretrained=None)
        checkpoint = torch.load(self.checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        cleaned = OrderedDict()
        for key, value in checkpoint.items():
            cleaned[key[7:] if key.startswith('module.') else key] = value
        incompatible = model.load_state_dict(cleaned, strict=False)
        missing = [key for key in incompatible.missing_keys if 'attn_mask' not in key]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                'RemoteCLIP checkpoint does not match ViT-L-14. '
                f'missing={missing[:8]}, unexpected={incompatible.unexpected_keys[:8]}')
        model = model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode_text(self, descriptions):
        self._load()
        if self.text_cache is None:
            tokens = self.tokenizer(list(descriptions)).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens, normalize=True)
            self.text_cache = features.detach().float()
        return self.text_cache

    def encode_image_dense(self, image):
        self._load()
        array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = F.interpolate(
            tensor, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).to(self.device)
        visual = self.model.visual
        with torch.no_grad():
            outputs = visual(tensor, get_embedding=True)
            layer_idx = int(visual.transformer.layers)
            patch = outputs[layer_idx].permute(1, 0, 2)
            patch = visual.ln_post(patch)
            if visual.proj is not None:
                patch = patch @ visual.proj
            patch = F.normalize(patch.float(), dim=-1)
            side = int(round(math.sqrt(int(patch.shape[1]))))
            if side * side != int(patch.shape[1]):
                raise RuntimeError(
                    f'RemoteCLIP patch token count is not square: {patch.shape[1]}')
            dense = patch.reshape(1, side, side, patch.shape[-1]).permute(0, 3, 1, 2)
            global_feature = F.normalize(
                outputs['final_cls_token'].float(), dim=-1)
        return dense.detach(), global_feature.detach()


class RolePromptTTAMixin(SemanticSupplementScreenMixin):
    """Mixin integrated by :class:`SegEarthOV3Segmentation`."""

    def _rpt_initialize(
            self, use_role_prompt_tta=False,
            dump_role_prompt_tta_stats=False,
            role_prompt_tta_protocol='v1',
            role_prompt_tta_dataset_name=None,
            role_prompt_tta_prompt_bank=None,
            role_prompt_tta_stats_path=None,
            role_prompt_tta_artifact_dir=None,
            role_prompt_tta_remoteclip_checkpoint=(
                'weights/remoteclip/RemoteCLIP-ViT-L-14.pt'),
            role_prompt_tta_remoteclip_source_root='SCORE-main',
            role_prompt_tta_remoteclip_model='ViT-L-14',
            role_prompt_tta_remoteclip_device='same',
            role_prompt_tta_primary_variant='full_regrounded_e2e',
            role_prompt_tta_anchor_mass=0.50,
            role_prompt_tta_visual_strength=1.0,
            role_prompt_tta_presence_threshold=0.05,
            role_prompt_tta_seed_fraction=0.20,
            role_prompt_tta_min_seed_pixels=2,
            role_prompt_tta_temperature=1.0,
            role_prompt_tta_steps=3,
            role_prompt_tta_lr=0.05,
            role_prompt_tta_anchor_lambda=0.05,
            role_prompt_tta_delta_max=1.0,
            role_prompt_tta_class_balanced_entropy=True,
            role_prompt_tta_e2e_steps=1,
            role_prompt_tta_e2e_lr=0.02,
            role_prompt_tta_e2e_max_classes=4,
            role_prompt_tta_e2e_measure_post=False,
            role_prompt_tta_adapt_background=False,
            role_prompt_tta_eps=1e-6,
            role_prompt_tta_strict_integrity=True,
            role_prompt_tta_integrity_tolerance=1e-5,
            role_prompt_tta_save_npz=True,
            role_prompt_tta_artifact_max_side=128,
            role_prompt_tta_max_saved_images=8,
            role_prompt_tta_semantic_residual_alpha=0.50,
            role_prompt_tta_semantic_residual_clip=0.25):
        self.use_role_prompt_tta = bool(use_role_prompt_tta)
        self.dump_role_prompt_tta_stats = bool(dump_role_prompt_tta_stats)
        self.role_prompt_tta_protocol = str(role_prompt_tta_protocol)
        self.role_prompt_tta_dataset_name = role_prompt_tta_dataset_name
        self.role_prompt_tta_prompt_bank = role_prompt_tta_prompt_bank
        self.role_prompt_tta_stats_path = role_prompt_tta_stats_path
        self.role_prompt_tta_artifact_dir = role_prompt_tta_artifact_dir
        self.role_prompt_tta_remoteclip_checkpoint = (
            role_prompt_tta_remoteclip_checkpoint)
        self.role_prompt_tta_remoteclip_source_root = (
            role_prompt_tta_remoteclip_source_root)
        self.role_prompt_tta_remoteclip_model = role_prompt_tta_remoteclip_model
        self.role_prompt_tta_remoteclip_device = str(
            role_prompt_tta_remoteclip_device)
        self.role_prompt_tta_primary_variant = str(role_prompt_tta_primary_variant)
        self.role_prompt_tta_anchor_mass = float(role_prompt_tta_anchor_mass)
        self.role_prompt_tta_visual_strength = float(
            role_prompt_tta_visual_strength)
        self.role_prompt_tta_presence_threshold = float(
            role_prompt_tta_presence_threshold)
        self.role_prompt_tta_seed_fraction = float(role_prompt_tta_seed_fraction)
        self.role_prompt_tta_min_seed_pixels = int(
            role_prompt_tta_min_seed_pixels)
        self.role_prompt_tta_temperature = float(role_prompt_tta_temperature)
        self.role_prompt_tta_steps = int(role_prompt_tta_steps)
        self.role_prompt_tta_lr = float(role_prompt_tta_lr)
        self.role_prompt_tta_anchor_lambda = float(
            role_prompt_tta_anchor_lambda)
        self.role_prompt_tta_delta_max = float(role_prompt_tta_delta_max)
        self.role_prompt_tta_class_balanced_entropy = bool(
            role_prompt_tta_class_balanced_entropy)
        self.role_prompt_tta_e2e_steps = int(role_prompt_tta_e2e_steps)
        self.role_prompt_tta_e2e_lr = float(role_prompt_tta_e2e_lr)
        self.role_prompt_tta_e2e_max_classes = int(
            role_prompt_tta_e2e_max_classes)
        self.role_prompt_tta_e2e_measure_post = bool(
            role_prompt_tta_e2e_measure_post)
        self.role_prompt_tta_adapt_background = bool(
            role_prompt_tta_adapt_background)
        self.role_prompt_tta_eps = float(role_prompt_tta_eps)
        self.role_prompt_tta_strict_integrity = bool(
            role_prompt_tta_strict_integrity)
        self.role_prompt_tta_integrity_tolerance = float(
            role_prompt_tta_integrity_tolerance)
        self.role_prompt_tta_save_npz = bool(role_prompt_tta_save_npz)
        self.role_prompt_tta_artifact_max_side = int(
            role_prompt_tta_artifact_max_side)
        self.role_prompt_tta_max_saved_images = int(
            role_prompt_tta_max_saved_images)
        self.role_prompt_tta_semantic_residual_alpha = float(
            role_prompt_tta_semantic_residual_alpha)
        self.role_prompt_tta_semantic_residual_clip = float(
            role_prompt_tta_semantic_residual_clip)
        self._role_prompt_tta_stats_file = None
        self._role_prompt_tta_saved_images = 0
        self._rpt_text_cache = None
        self._rpt_remoteclip = None
        self._rpt_native_parity_checked = False
        self._rpt_native_parity_max_abs = None

        if self.role_prompt_tta_protocol not in (
                'v1', 'prompt_functional_atlas_v2',
                'head_role_prompt_conflict_v1',
                SEMANTIC_SUPPLEMENT_PROTOCOL):
            raise ValueError(
                f'Unknown role_prompt_tta_protocol='
                f'{self.role_prompt_tta_protocol!r}.')
        if self.role_prompt_tta_primary_variant not in self._rpt_variant_names():
            raise ValueError(
                f'Unknown role_prompt_tta_primary_variant='
                f'{self.role_prompt_tta_primary_variant!r}.')
        if (
                self.role_prompt_tta_protocol in (
                    'head_role_prompt_conflict_v1',
                    SEMANTIC_SUPPLEMENT_PROTOCOL)
                and self.role_prompt_tta_primary_variant != 'baseline'):
            raise ValueError(
                'Prompt screening protocols are diagnostic-only and must '
                'return the protected baseline as its primary prediction.')
        if not (0.0 <= self.role_prompt_tta_semantic_residual_alpha <= 1.0):
            raise ValueError(
                'role_prompt_tta_semantic_residual_alpha must be in [0, 1].')
        if not (0.0 < self.role_prompt_tta_semantic_residual_clip <= 1.0):
            raise ValueError(
                'role_prompt_tta_semantic_residual_clip must be in (0, 1].')
        if self._uses_role_prompt_tta():
            if not role_prompt_tta_prompt_bank:
                raise ValueError('role_prompt_tta_prompt_bank is required.')
            if self.role_prompt_tta_protocol == SEMANTIC_SUPPLEMENT_PROTOCOL:
                self._rpt_prompt_bank = load_semantic_supplement_bank(
                    role_prompt_tta_prompt_bank, self.class_names)
            else:
                self._rpt_prompt_bank = load_prompt_bank(
                    role_prompt_tta_prompt_bank, self.class_names)
            if (
                    self.role_prompt_tta_protocol
                    == 'prompt_functional_atlas_v2'
                    and int(self._rpt_prompt_bank['_prompt_count'])
                    != ATLAS_PROMPT_COUNT):
                raise ValueError(
                    'Prompt functional atlas v2 requires exactly '
                    f'{ATLAS_PROMPT_COUNT} role-controlled prompts per class.')
            if (
                    self.role_prompt_tta_protocol
                    == 'head_role_prompt_conflict_v1'
                    and int(self._rpt_prompt_bank['_prompt_count'])
                    != HEAD_ROLE_PROMPT_COUNT):
                raise ValueError(
                    'Head-role prompt conflict v1 requires exactly '
                    f'{HEAD_ROLE_PROMPT_COUNT} prompts per class.')
            descriptions = [
                description
                for item in self._rpt_prompt_bank['classes']
                for description in item['descriptions']
            ]
            if self.role_prompt_tta_protocol in (
                    'head_role_prompt_conflict_v1',
                    SEMANTIC_SUPPLEMENT_PROTOCOL):
                # This audit is native-SAM3 only.  Avoid loading RemoteCLIP or
                # constructing any test-time optimization graph.
                self._rpt_remoteclip = None
                self._rpt_remoteclip_descriptions = None
            else:
                self._rpt_remoteclip = RemoteCLIPRuntime(
                    checkpoint=os.path.abspath(
                        self.role_prompt_tta_remoteclip_checkpoint),
                    source_root=os.path.abspath(
                        self.role_prompt_tta_remoteclip_source_root),
                    model_name=self.role_prompt_tta_remoteclip_model,
                    device=resolve_remoteclip_device(
                        self.role_prompt_tta_remoteclip_device, self.device),
                )
                self._rpt_remoteclip_descriptions = descriptions
            # SAM3 is frozen only for the enabled TTA experiment.  The official
            # baseline's default construction remains untouched.
            for parameter in self.processor.model.parameters():
                parameter.requires_grad_(False)
        else:
            self._rpt_prompt_bank = None
            self._rpt_remoteclip_descriptions = None

    def _uses_role_prompt_tta(self):
        return bool(
            getattr(self, 'use_role_prompt_tta', False)
            or getattr(self, 'dump_role_prompt_tta_stats', False))

    def _rpt_uses_class_space_variants(self):
        return getattr(self, 'role_prompt_tta_protocol', 'v1') in (
            'head_role_prompt_conflict_v1',
            SEMANTIC_SUPPLEMENT_PROTOCOL,
        )

    def _rpt_variant_names(self):
        if getattr(self, 'role_prompt_tta_protocol', 'v1') \
                == SEMANTIC_SUPPLEMENT_PROTOCOL:
            return SEMANTIC_SUPPLEMENT_VARIANT_NAMES
        if getattr(self, 'role_prompt_tta_protocol', 'v1') \
                == 'head_role_prompt_conflict_v1':
            return HEAD_ROLE_VARIANT_NAMES
        if getattr(self, 'role_prompt_tta_protocol', 'v1') \
                == 'prompt_functional_atlas_v2':
            return ATLAS_VARIANT_NAMES
        return V1_VARIANT_NAMES

    def _rpt_record_schema_version(self):
        if getattr(self, 'role_prompt_tta_protocol', 'v1') \
                == SEMANTIC_SUPPLEMENT_PROTOCOL:
            return SEMANTIC_SUPPLEMENT_SCHEMA_VERSION
        if getattr(self, 'role_prompt_tta_protocol', 'v1') \
                == 'head_role_prompt_conflict_v1':
            return HEAD_ROLE_SCHEMA_VERSION
        if getattr(self, 'role_prompt_tta_protocol', 'v1') \
                == 'prompt_functional_atlas_v2':
            return ATLAS_SCHEMA_VERSION
        return SCHEMA_VERSION

    def _rpt_autocast_context(self):
        return (
            torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            if self.device.type == 'cuda'
            else contextlib.nullcontext())

    def _rpt_prepare_text_cache(self):
        if self._rpt_text_cache is not None:
            return self._rpt_text_cache
        bank_descriptions = [
            description
            for item in self._rpt_prompt_bank['classes']
            for description in item['descriptions']
        ]
        all_prompts = []
        for prompt in list(self.query_words) + bank_descriptions:
            if prompt not in all_prompts:
                all_prompts.append(prompt)
        with torch.no_grad(), self._rpt_autocast_context():
            text_outputs = self.processor.model.backbone.forward_text(
                all_prompts, device=self.device)
        index = {prompt: idx for idx, prompt in enumerate(all_prompts)}
        language_features = text_outputs['language_features'].detach().clone()
        language_mask = text_outputs['language_mask'].detach().clone()
        # Preserve the protected baseline exactly: original query strings are
        # encoded one at a time, matching set_text_prompt.  The head-role
        # audit extends this identity rule to every description because its
        # claim depends on independently native grounding, not batched-text
        # numerical behavior. Other protocols retain their original cache.
        prompts_to_encode_individually = (
            all_prompts
            if self.role_prompt_tta_protocol in (
                'head_role_prompt_conflict_v1',
                SEMANTIC_SUPPLEMENT_PROTOCOL)
            else self.query_words)
        with torch.no_grad(), self._rpt_autocast_context():
            for prompt in prompts_to_encode_individually:
                single = self.processor.model.backbone.forward_text(
                    [prompt], device=self.device)
                prompt_index = index[prompt]
                language_features[:, prompt_index:prompt_index + 1] = (
                    single['language_features'])
                language_mask[prompt_index:prompt_index + 1] = (
                    single['language_mask'])
        self._rpt_text_cache = dict(
            prompts=all_prompts,
            index=index,
            language_features=language_features,
            language_mask=language_mask,
        )
        return self._rpt_text_cache

    def _rpt_set_cached_prompt(self, state, prompt):
        cache = self._rpt_prepare_text_cache()
        index = int(cache['index'][prompt])
        self.processor.reset_all_prompts(state)
        state['backbone_out']['language_features'] = (
            cache['language_features'][:, index:index + 1])
        state['backbone_out']['language_mask'] = (
            cache['language_mask'][index:index + 1])
        state['geometric_prompt'] = self.processor.model._get_dummy_prompt()
        return self.processor._forward_grounding(state)

    def _rpt_prompt_components(self, state, output_shape):
        height, width = output_shape
        semantic = state['semantic_mask_logits'].squeeze().float()
        semantic_raw = state['semantic_mask_raw_logits'].squeeze().float()
        if semantic.shape != output_shape:
            semantic = F.interpolate(
                semantic.view(1, 1, *semantic.shape),
                size=output_shape, mode='bilinear', align_corners=False,
            ).squeeze()
        if semantic_raw.shape != output_shape:
            semantic_raw = F.interpolate(
                semantic_raw.view(1, 1, *semantic_raw.shape),
                size=output_shape, mode='bilinear', align_corners=False,
            ).squeeze()
        instance = torch.zeros(
            (height, width), device=self.device, dtype=torch.float32)
        masks = state.get('masks_logits')
        if self.use_transformer_decoder and isinstance(masks, torch.Tensor):
            for instance_idx in range(int(masks.shape[0])):
                mask = masks[instance_idx].squeeze().float()
                if mask.shape != output_shape:
                    mask = F.interpolate(
                        mask.view(1, 1, *mask.shape), size=output_shape,
                        mode='bilinear', align_corners=False).squeeze()
                score = self._get_instance_score(state, instance_idx).float()
                instance = torch.maximum(instance, mask * score)
        final = torch.maximum(semantic, instance)
        presence = state['presence_score'].detach().float().mean()
        if self.use_presence_score:
            final = final * presence
        return dict(
            final=final,
            semantic=semantic,
            semantic_raw=semantic_raw,
            instance=instance,
            presence=presence,
            raw_candidate_count=int(
                state.get('raw_keep_mask', torch.zeros(0)).numel()),
            kept_candidate_count=int(
                state.get('raw_keep_mask', torch.zeros(0)).bool().sum().item()),
        )

    def _rpt_collect_prompt_outputs(self, image):
        width, height = image.size
        output_shape = (height, width)
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
                baseline_rows.append(cached_row)
                if native_row is not None:
                    for key in (
                            'final', 'semantic', 'semantic_raw',
                            'instance', 'presence'):
                        parity_errors.append(float((
                            cached_row[key].float()
                            - native_row[key].float()).abs().max().item()))
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
            bank_rows = []
            for item in self._rpt_prompt_bank['classes']:
                class_rows = []
                for prompt in item['descriptions']:
                    self._rpt_set_cached_prompt(state, prompt)
                    row = self._rpt_prompt_components(state, output_shape)
                    row['prompt'] = prompt
                    class_rows.append(row)
                bank_rows.append(class_rows)
        baseline = {
            key: torch.stack([row[key] for row in baseline_rows], dim=0)
            for key in ('final', 'semantic', 'semantic_raw', 'instance', 'presence')
        }
        bank = {
            key: torch.stack([
                torch.stack([row[key] for row in class_rows], dim=0)
                for class_rows in bank_rows
            ], dim=0)
            for key in ('final', 'semantic_raw', 'instance', 'presence')
        }
        candidate_stats = []
        for class_rows in bank_rows:
            rows = []
            for row in class_rows:
                prompt = row['prompt']
                prompt_index = int(cache['index'][prompt])
                token_count = int((
                    ~cache['language_mask'][prompt_index].bool()
                ).sum().item())
                rows.append(dict(
                    prompt=prompt,
                    token_count=token_count,
                    word_count=len(prompt.split()),
                    character_count=len(prompt),
                    presence=float(row['presence'].item()),
                    raw_candidate_count=int(row['raw_candidate_count']),
                    kept_candidate_count=int(row['kept_candidate_count']),
                    semantic_mean=float(row['semantic'].mean().item()),
                    instance_mean=float(row['instance'].mean().item()),
                    final_mean=float(row['final'].mean().item()),
                ))
            candidate_stats.append(rows)
        return state, cache, baseline, bank, candidate_stats

    def _rpt_query_to_class(self, query_maps):
        return self._rpt_aggregate_query_logits_to_classes(query_maps)

    def _rpt_class_to_query(self, class_maps):
        return class_maps[self.query_idx]

    def _rpt_seed_and_visual_evidence(
            self, image, anchor_raw_class, anchor_presence):
        dense, global_feature = self._rpt_remoteclip.encode_image_dense(image)
        text = self._rpt_remoteclip.encode_text(
            self._rpt_remoteclip_descriptions)
        # Only compact dense/text evidence crosses devices.  Keeping the
        # frozen RemoteCLIP parameters on an auxiliary GPU removes them from
        # the SAM3 E2E backward peak without changing any scores.
        if dense.device != self.device:
            dense = dense.to(self.device)
        if text.device != self.device:
            text = text.to(self.device)
        if global_feature.device != self.device:
            global_feature = global_feature.to(self.device)
        class_count = int(anchor_raw_class.shape[0])
        prompt_count = int(self._rpt_prompt_bank['_prompt_count'])
        text = text.view(class_count, prompt_count, -1)
        grid_shape = dense.shape[-2:]
        raw_grid = F.interpolate(
            anchor_raw_class.unsqueeze(0), size=grid_shape,
            mode='bilinear', align_corners=False).squeeze(0)
        probabilities, entropy = normalized_entropy(
            raw_grid, self.role_prompt_tta_temperature,
            self.role_prompt_tta_eps)
        top1 = probabilities.argmax(dim=0)
        image_mean = entropy.mean()
        affinity = torch.zeros(
            class_count, prompt_count, device=self.device,
            dtype=torch.float32)
        seed_counts = []
        seed_entropy = []
        prototype_norms = []
        seed_masks = []
        for class_idx in range(class_count):
            eligible = (top1 == class_idx) & (entropy <= image_mean)
            indices = torch.nonzero(eligible.flatten(), as_tuple=False).flatten()
            keep_count = int(math.ceil(
                float(indices.numel()) * self.role_prompt_tta_seed_fraction))
            keep_count = min(int(indices.numel()), max(keep_count, 0))
            selected_mask = torch.zeros_like(eligible)
            if keep_count >= self.role_prompt_tta_min_seed_pixels:
                selected_entropy = entropy.flatten()[indices]
                selected = indices[torch.topk(
                    -selected_entropy, k=keep_count).indices]
                selected_mask.flatten()[selected] = True
                features = dense[0, :, selected_mask]
                prototype = features.mean(dim=1)
                prototype_norm = prototype.norm()
                if float(prototype_norm.item()) > self.role_prompt_tta_eps:
                    prototype = F.normalize(prototype, dim=0)
                    class_affinity = torch.mv(text[class_idx], prototype)
                    affinity[class_idx] = class_affinity - class_affinity.mean()
                prototype_norms.append(float(prototype_norm.item()))
                seed_entropy.append(float(entropy[selected_mask].mean().item()))
            else:
                prototype_norms.append(0.0)
                seed_entropy.append(None)
            seed_counts.append(int(selected_mask.sum().item()))
            seed_masks.append(selected_mask)
        threshold = self.role_prompt_tta_presence_threshold
        presence_gate = (
            (anchor_presence.float() - threshold)
            / max(1.0 - threshold, self.role_prompt_tta_eps)
        ).clamp(0.0, 1.0)
        has_seed = torch.tensor(
            [count >= self.role_prompt_tta_min_seed_pixels for count in seed_counts],
            device=self.device, dtype=torch.float32)
        seed_gate = has_seed.clone()
        presence_gate = presence_gate * seed_gate
        if not self.role_prompt_tta_adapt_background:
            presence_gate[int(self.bg_idx)] = 0.0
            seed_gate[int(self.bg_idx)] = 0.0
        return dict(
            affinity=affinity,
            presence_gate=presence_gate,
            seed_gate=seed_gate,
            seed_masks=torch.stack(seed_masks, dim=0),
            seed_counts=seed_counts,
            seed_entropy=seed_entropy,
            prototype_norms=prototype_norms,
            remoteclip_global_norm=float(global_feature.norm().item()),
            remoteclip_device=str(self._rpt_remoteclip.device),
            anchor_entropy_mean=float(entropy.mean().item()),
            anchor_entropy_std=float(entropy.std(unbiased=False).item()),
            grid_shape=list(grid_shape),
        )

    def _rpt_bank_language_tensors(self, cache):
        features = []
        masks = []
        for item in self._rpt_prompt_bank['classes']:
            class_features = []
            class_masks = []
            for prompt in item['descriptions']:
                index = cache['index'][prompt]
                class_features.append(cache['language_features'][:, index])
                class_masks.append(cache['language_mask'][index])
            features.append(torch.stack(class_features, dim=0))
            masks.append(torch.stack(class_masks, dim=0))
        return torch.stack(features, dim=0), torch.stack(masks, dim=0)

    def _rpt_reground_weights(self, state, cache, weights, output_shape):
        features, masks = self._rpt_bank_language_tensors(cache)
        rows = []
        fusion_stats = []
        with torch.no_grad(), self._rpt_autocast_context():
            for class_idx in range(int(weights.shape[0])):
                fused, fused_mask = mask_aware_language_fusion(
                    features[class_idx], masks[class_idx], weights[class_idx],
                    self.role_prompt_tta_eps)
                self.processor.reset_all_prompts(state)
                state['backbone_out']['language_features'] = fused[:, None]
                state['backbone_out']['language_mask'] = fused_mask[None]
                state['geometric_prompt'] = self.processor.model._get_dummy_prompt()
                self.processor._forward_grounding(state)
                rows.append(self._rpt_prompt_components(state, output_shape))
                anchor = features[class_idx, 0].float()
                valid = ~masks[class_idx, 0].bool()
                delta = fused.float()[valid] - anchor[valid]
                anchor_norm = anchor[valid].norm().clamp_min(
                    self.role_prompt_tta_eps)
                fusion_stats.append(dict(
                    class_index=int(class_idx),
                    valid_tokens=int((~fused_mask).sum().item()),
                    raw_candidate_count=int(
                        rows[-1]['raw_candidate_count']),
                    kept_candidate_count=int(
                        rows[-1]['kept_candidate_count']),
                    residual_to_anchor_norm=float(
                        (delta.norm() / anchor_norm).item()),
                ))
        outputs = {
            key: torch.stack([row[key] for row in rows], dim=0)
            for key in ('final', 'semantic', 'semantic_raw', 'instance', 'presence')
        }
        return outputs, fusion_stats

    def _rpt_reground_anchor_residual(
            self, state, cache, target_weights, output_shape, alpha):
        """Re-ground a bounded residual while preserving the anchor mask.

        This is an intentionally conservative diagnostic.  Only token
        positions valid in the literal class-name sequence are modified; the
        literal sequence length/mask remains unchanged.  It tests whether the
        failure is caused by the magnitude of language movement or by any
        position-wise contextual-sequence interpolation at all.
        """
        alpha = float(alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError('anchor residual alpha must be in [0, 1].')
        features, masks = self._rpt_bank_language_tensors(cache)
        rows = []
        fusion_stats = []
        with torch.no_grad(), self._rpt_autocast_context():
            for class_idx in range(int(target_weights.shape[0])):
                target, target_mask = mask_aware_language_fusion(
                    features[class_idx], masks[class_idx],
                    target_weights[class_idx], self.role_prompt_tta_eps)
                anchor = features[class_idx, 0]
                anchor_mask = masks[class_idx, 0].bool()
                overlap = (~anchor_mask) & (~target_mask.bool())
                fused = anchor.clone()
                # Under CUDA autocast the cached SAM3 anchor is BF16, while
                # mask-aware fusion can promote the weighted target to FP32.
                # Boolean index assignment does not cast implicitly. Compute
                # the bounded residual in FP32 for numerical stability, then
                # restore SAM3's native language-feature dtype explicitly.
                updated = (
                    anchor[overlap].float()
                    + alpha * (
                        target[overlap].float()
                        - anchor[overlap].float()))
                fused[overlap] = updated.to(dtype=fused.dtype)
                fused_mask = anchor_mask.clone()
                self.processor.reset_all_prompts(state)
                state['backbone_out']['language_features'] = fused[:, None]
                state['backbone_out']['language_mask'] = fused_mask[None]
                state['geometric_prompt'] = self.processor.model._get_dummy_prompt()
                self.processor._forward_grounding(state)
                rows.append(self._rpt_prompt_components(state, output_shape))
                valid = ~anchor_mask
                anchor_norm = anchor[valid].float().norm().clamp_min(
                    self.role_prompt_tta_eps)
                residual = fused[valid].float() - anchor[valid].float()
                fusion_stats.append(dict(
                    class_index=int(class_idx),
                    alpha=alpha,
                    valid_tokens=int(valid.sum().item()),
                    overlap_tokens=int(overlap.sum().item()),
                    anchor_dtype=str(anchor.dtype),
                    target_dtype=str(target.dtype),
                    fused_dtype=str(fused.dtype),
                    raw_candidate_count=int(rows[-1]['raw_candidate_count']),
                    kept_candidate_count=int(rows[-1]['kept_candidate_count']),
                    residual_to_anchor_norm=float((
                        residual.norm() / anchor_norm).item()),
                ))
        outputs = {
            key: torch.stack([row[key] for row in rows], dim=0)
            for key in ('final', 'semantic', 'semantic_raw', 'instance', 'presence')
        }
        return outputs, fusion_stats

    def _rpt_head_change_block(self, name, outputs, baseline_class):
        rows = []
        for class_idx in range(self.num_cls):
            semantic_mask = outputs['semantic'][class_idx] >= 0.5
            instance_mask = outputs['instance'][class_idx] >= 0.5
            intersection = (semantic_mask & instance_mask).sum().item()
            union = (semantic_mask | instance_mask).sum().item()
            rows.append(dict(
                class_index=int(class_idx),
                class_name=self.class_names[class_idx],
                semantic_mean_delta=float((
                    outputs['semantic'][class_idx]
                    - baseline_class['semantic'][class_idx]).mean().item()),
                semantic_abs_delta=float((
                    outputs['semantic'][class_idx]
                    - baseline_class['semantic'][class_idx]).abs().mean().item()),
                instance_mean_delta=float((
                    outputs['instance'][class_idx]
                    - baseline_class['instance'][class_idx]).mean().item()),
                instance_abs_delta=float((
                    outputs['instance'][class_idx]
                    - baseline_class['instance'][class_idx]).abs().mean().item()),
                presence_delta=float((
                    outputs['presence'][class_idx]
                    - baseline_class['presence'][class_idx]).item()),
                final_mean_delta=float((
                    outputs['final'][class_idx]
                    - baseline_class['final'][class_idx]).mean().item()),
                semantic_instance_mask_iou=_safe_div(intersection, union),
            ))
        return dict(variant=name, classes=rows)

    def _rpt_e2e_refine_weights(
            self, state, cache, start_weights, anchor_raw_class,
            visual_evidence):
        if self.role_prompt_tta_e2e_steps <= 0:
            return start_weights, [], []
        features, masks = self._rpt_bank_language_tensors(cache)
        class_count = int(start_weights.shape[0])
        seed_counts = visual_evidence['seed_counts']
        gate = visual_evidence['presence_gate']
        eligible = []
        for class_idx in range(class_count):
            if (
                    class_idx == int(self.bg_idx)
                    and not self.role_prompt_tta_adapt_background):
                continue
            if (
                    seed_counts[class_idx] >= self.role_prompt_tta_min_seed_pixels
                    and float(gate[class_idx].item()) > 0.0):
                eligible.append(class_idx)
        eligible.sort(
            key=lambda idx: float(gate[idx].item()) * seed_counts[idx],
            reverse=True)
        if self.role_prompt_tta_e2e_max_classes > 0:
            eligible = eligible[:self.role_prompt_tta_e2e_max_classes]

        # set_image() creates inference-mode tensors which cannot be saved for
        # backward. Replace those cached tensors in-place with ordinary clones
        # so state and the E2E path share one copy instead of retaining both a
        # full inference cache and a full grad-safe cache on the same GPU.
        image_backbone = {}
        for key in tuple(state['backbone_out']):
            if key in ('language_features', 'language_mask', 'language_embeds'):
                continue
            cloned = _tensor_tree_grad_safe_clone(
                state['backbone_out'][key], self.device)
            state['backbone_out'][key] = cloned
            image_backbone[key] = cloned
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        refined = start_weights.detach().clone()
        trajectories = []
        updated_classes = []
        model = self.processor.model
        for class_idx in eligible:
            gate_value = gate[class_idx].detach().float()
            theta = torch.zeros_like(
                refined[class_idx], dtype=torch.float32,
                device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam(
                [theta], lr=self.role_prompt_tta_e2e_lr)
            other_indices = [idx for idx in range(class_count) if idx != class_idx]
            other_raw = anchor_raw_class[other_indices].max(dim=0)[0].detach()
            seed_mask = visual_evidence['seed_masks'][class_idx].float()[None, None]
            class_trace = []
            for step in range(self.role_prompt_tta_e2e_steps):
                optimizer.zero_grad()
                memory_before = cuda_memory_snapshot(self.device)
                autocast = (
                    torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                    if self.device.type == 'cuda'
                    else contextlib.nullcontext())
                # MMEngine's validation loop wraps model prediction in
                # torch.no_grad().  Re-enable gradients before *constructing*
                # the fused language sequence, otherwise theta is silently
                # disconnected even though forward_grounding runs with grad.
                with torch.enable_grad(), autocast:
                    weights = torch.softmax(
                        _safe_log(
                            refined[class_idx], self.role_prompt_tta_eps)
                        + self.role_prompt_tta_delta_max
                        * gate_value * torch.tanh(theta),
                        dim=0)
                    fused, fused_mask = mask_aware_language_fusion(
                        features[class_idx], masks[class_idx], weights,
                        self.role_prompt_tta_eps)
                    backbone = dict(image_backbone)
                    backbone['language_features'] = fused[:, None]
                    backbone['language_mask'] = fused_mask[None]
                    geometry = model._get_dummy_prompt()
                    outputs = model.forward_grounding(
                        backbone_out=backbone,
                        find_input=self.processor.find_stage,
                        geometric_prompt=geometry,
                        find_target=None,
                    )
                    current = outputs['semantic_seg'].float()
                    competitor = F.interpolate(
                        other_raw[None, None], size=current.shape[-2:],
                        mode='bilinear', align_corners=False)
                    mask = F.interpolate(
                        seed_mask, size=current.shape[-2:], mode='nearest').bool()
                    binary = torch.sigmoid(
                        (current - competitor)
                        / max(self.role_prompt_tta_temperature,
                              self.role_prompt_tta_eps))
                    entropy = -(
                        binary * _safe_log(binary, self.role_prompt_tta_eps)
                        + (1.0 - binary) * _safe_log(
                            1.0 - binary, self.role_prompt_tta_eps))
                    entropy_loss = (
                        entropy[mask].mean() if mask.any() else entropy.mean())
                    kl = (
                        weights * (
                            _safe_log(weights, self.role_prompt_tta_eps)
                            - _safe_log(
                                start_weights[class_idx],
                                self.role_prompt_tta_eps)
                        )
                    ).sum()
                    loss = entropy_loss + self.role_prompt_tta_anchor_lambda * kl
                memory_after_forward = cuda_memory_snapshot(self.device)
                loss.backward()
                memory_after_backward = cuda_memory_snapshot(self.device)
                grad_norm = theta.grad.detach().float().norm()
                optimizer.step()
                class_trace.append(dict(
                    step=int(step + 1),
                    loss=float(loss.detach().item()),
                    binary_entropy=float(entropy_loss.detach().item()),
                    anchor_kl=float(kl.detach().item()),
                    grad_norm=float(grad_norm.item()),
                    cuda_memory=dict(
                        before_forward=memory_before,
                        after_forward=memory_after_forward,
                        after_backward=memory_after_backward,
                    ),
                ))
                del (
                    outputs, current, competitor, binary, entropy,
                    entropy_loss, kl, loss, backbone, fused, fused_mask,
                    weights, geometry, memory_before, memory_after_forward,
                    memory_after_backward,
                )
            with torch.no_grad():
                refined[class_idx] = torch.softmax(
                    _safe_log(refined[class_idx], self.role_prompt_tta_eps)
                    + self.role_prompt_tta_delta_max
                    * gate_value * torch.tanh(theta), dim=0)
            if self.role_prompt_tta_e2e_measure_post and class_trace:
                with torch.no_grad(), self._rpt_autocast_context():
                    post_weights = refined[class_idx]
                    post_fused, post_fused_mask = mask_aware_language_fusion(
                        features[class_idx], masks[class_idx], post_weights,
                        self.role_prompt_tta_eps)
                    post_backbone = dict(image_backbone)
                    post_backbone['language_features'] = post_fused[:, None]
                    post_backbone['language_mask'] = post_fused_mask[None]
                    post_outputs = model.forward_grounding(
                        backbone_out=post_backbone,
                        find_input=self.processor.find_stage,
                        geometric_prompt=model._get_dummy_prompt(),
                        find_target=None,
                    )
                    post_current = post_outputs['semantic_seg'].float()
                    post_competitor = F.interpolate(
                        other_raw[None, None],
                        size=post_current.shape[-2:], mode='bilinear',
                        align_corners=False)
                    post_mask = F.interpolate(
                        seed_mask, size=post_current.shape[-2:],
                        mode='nearest').bool()
                    post_binary = torch.sigmoid(
                        (post_current - post_competitor)
                        / max(self.role_prompt_tta_temperature,
                              self.role_prompt_tta_eps))
                    post_entropy = -(
                        post_binary * _safe_log(
                            post_binary, self.role_prompt_tta_eps)
                        + (1.0 - post_binary) * _safe_log(
                            1.0 - post_binary, self.role_prompt_tta_eps))
                    post_entropy_loss = (
                        post_entropy[post_mask].mean()
                        if post_mask.any() else post_entropy.mean())
                    post_kl = (
                        post_weights * (
                            _safe_log(
                                post_weights, self.role_prompt_tta_eps)
                            - _safe_log(
                                start_weights[class_idx],
                                self.role_prompt_tta_eps)
                        )
                    ).sum()
                    post_loss = (
                        post_entropy_loss
                        + self.role_prompt_tta_anchor_lambda * post_kl)
                class_trace[-1].update(dict(
                    post_update_loss=float(post_loss.float().item()),
                    post_update_binary_entropy=float(
                        post_entropy_loss.float().item()),
                    post_update_anchor_kl=float(post_kl.float().item()),
                    post_update_weight_l1=float((
                        refined[class_idx]
                        - start_weights[class_idx]).abs().sum().item()),
                    post_update_cuda_memory=cuda_memory_snapshot(self.device),
                ))
                del (
                    post_weights, post_fused, post_fused_mask,
                    post_backbone, post_outputs, post_current,
                    post_competitor, post_mask, post_binary, post_entropy,
                    post_entropy_loss, post_kl, post_loss,
                )
            trajectories.append(dict(
                class_index=int(class_idx),
                class_name=self.class_names[class_idx],
                steps=class_trace,
            ))
            updated_classes.append(int(class_idx))
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        return refined.detach(), trajectories, updated_classes

    def _rpt_aggregate_query_logits_to_classes(self, query_logits):
        """CPU-safe equivalent of the protected baseline class reduction."""
        query_idx = self.query_idx.detach().long().to(query_logits.device)
        class_logits = []
        for class_idx in range(int(self.num_cls)):
            selected = query_logits[query_idx == class_idx]
            if selected.numel() == 0:
                raise RuntimeError(
                    f'No baseline query maps class {class_idx}.')
            class_logits.append(selected.max(dim=0)[0])
        return torch.stack(class_logits, dim=0)

    def _rpt_weights_stats(self, name, weights):
        rows = []
        effective = effective_prompt_count(weights, self.role_prompt_tta_eps)
        for class_idx, item in enumerate(self._rpt_prompt_bank['classes']):
            values = weights[class_idx].detach().float()
            rows.append(dict(
                class_index=int(class_idx),
                class_name=item['name'],
                weights=[float(value) for value in values.cpu().tolist()],
                effective_prompt_count=float(effective[class_idx].item()),
                winner_index=int(values.argmax().item()),
                winner_prompt=item['descriptions'][int(values.argmax().item())],
                max_weight=float(values.max().item()),
            ))
        return dict(name=name, classes=rows)

    def _rpt_head_role_raw_package(self, state, output_shape):
        """Detach one native SAM3 prompt into auditable head-role sources."""
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
            # Keep every raw query, not only candidates retained by the
            # original prompt's Presence value. Counterfactual Presence must
            # be allowed to change the native hard candidate set.
            raw_masks=raw_masks.detach().cpu(),
            raw_scores=raw_scores.detach().cpu(),
            native_kept_count=int(native['kept_candidate_count']),
            raw_candidate_count=int(native['raw_candidate_count']),
        )

    def _rpt_head_role_instance_from_raw(
            self, raw_package, presence, output_shape):
        """Reapply SAM3's native query gate with a chosen Presence source."""
        height, width = output_shape
        if not self.use_transformer_decoder:
            return torch.zeros((height, width), dtype=torch.float32), 0
        raw_masks = raw_package['raw_masks']
        raw_scores = raw_package['raw_scores']
        count = min(int(raw_masks.shape[0]), int(raw_scores.numel()))
        if count <= 0:
            return torch.zeros((height, width), dtype=torch.float32), 0
        raw_scores_device = raw_scores[:count].to(self.device)
        presence_device = torch.as_tensor(
            presence,
            device=self.device,
            dtype=raw_scores_device.dtype,
        )
        with torch.no_grad(), self._rpt_autocast_context():
            candidate_scores = raw_scores_device * presence_device
            keep = candidate_scores > float(
                self.processor.confidence_threshold)
        kept_count = int(keep.sum().item())
        if kept_count == 0:
            return torch.zeros((height, width), dtype=torch.float32), 0
        selected_masks = raw_masks[:count].to(self.device)[keep]
        with torch.no_grad(), self._rpt_autocast_context():
            # Match Sam3Processor._forward_grounding exactly.  SAM3's
            # compatibility interpolator promotes BF16 masks to FP32 for the
            # resize and restores the original dtype afterwards; direct
            # F.interpolate fails on the project's PyTorch 1.13 server because
            # upsample_bilinear2d has no BF16 implementation there.
            selected_masks = sam3_interpolate(
                selected_masks.unsqueeze(1),
                size=output_shape,
                mode='bilinear',
                align_corners=False,
            ).sigmoid().squeeze(1)
            if self.instance_score_type == 'raw':
                amplitudes = raw_scores_device[keep].to(
                    dtype=selected_masks.dtype)
            else:
                amplitudes = candidate_scores[keep].to(
                    dtype=selected_masks.dtype)
            instance = (
                selected_masks.float() * amplitudes.float()[:, None, None]
            ).max(dim=0)[0]
        return instance.detach().float().cpu(), kept_count

    def _rpt_head_role_compose(
            self, semantic_package, instance_package, presence_package,
            output_shape, instance_override=None):
        semantic = (
            semantic_package['semantic']
            if self.use_sem_seg
            else torch.zeros(output_shape, dtype=torch.float32))
        presence = float(
            torch.as_tensor(presence_package['presence']).float().item())
        if instance_override is None:
            instance, kept_count = self._rpt_head_role_instance_from_raw(
                instance_package, presence, output_shape)
        else:
            instance, kept_count = instance_override
        final = torch.maximum(semantic.float(), instance.float())
        if self.use_presence_score:
            final = final * presence
        return final, instance, kept_count

    @staticmethod
    def _rpt_head_role_mask_iou(left, right, threshold=0.5):
        left = left >= float(threshold)
        right = right >= float(threshold)
        intersection = int((left & right).sum().item())
        union = int((left | right).sum().item())
        return _safe_div(intersection, union)

    def _rpt_head_role_infer_single_view(
            self, image, return_stats=False, return_components=False,
            view_id=None, crop_box=None):
        """Assign independently grounded prompts to SAM3 output roles.

        The primary prediction is always the exact protected baseline.  Every
        alternative description is grounded normally by SAM3 before its
        semantic, raw instance-query, or Presence result is used.  No token or
        contextual-language feature is synthesized across prompts.
        """
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
            for name in HEAD_ROLE_VARIANT_NAMES)
        class_variants['baseline'].copy_(baseline_class)
        head_role_rows = []
        raw_recomposition_errors = []

        for class_idx, item in enumerate(self._rpt_prompt_bank['classes']):
            literal_prompt = item['descriptions'][0]
            with torch.no_grad(), self._rpt_autocast_context():
                self._rpt_set_cached_prompt(state, literal_prompt)
                literal = self._rpt_head_role_raw_package(
                    state, output_shape)
            literal_rebuilt_instance = self._rpt_head_role_instance_from_raw(
                literal, literal['presence'], output_shape)
            literal_rebuilt, _, _ = self._rpt_head_role_compose(
                literal, literal, literal, output_shape,
                instance_override=literal_rebuilt_instance)
            literal_error = float((
                literal_rebuilt - literal['native_final']).abs().max().item())
            literal_instance_error = float((
                literal_rebuilt_instance[0]
                - literal['native_instance']).abs().max().item())
            raw_recomposition_errors.extend(
                [literal_error, literal_instance_error])
            class_variants['literal_native'][class_idx].copy_(
                literal['native_final'])

            for prompt_slot, prompt_role in HEAD_ROLE_DESCRIPTION_SLOTS:
                description_prompt = item['descriptions'][prompt_slot]
                with torch.no_grad(), self._rpt_autocast_context():
                    self._rpt_set_cached_prompt(state, description_prompt)
                    description = self._rpt_head_role_raw_package(
                        state, output_shape)

                description_native_instance = (
                    self._rpt_head_role_instance_from_raw(
                        description, description['presence'], output_shape))
                description_rebuilt, _, _ = self._rpt_head_role_compose(
                    description, description, description, output_shape,
                    instance_override=description_native_instance)
                native_error = float((
                    description_rebuilt
                    - description['native_final']).abs().max().item())
                native_instance_error = float((
                    description_native_instance[0]
                    - description['native_instance']).abs().max().item())
                raw_recomposition_errors.extend(
                    [native_error, native_instance_error])

                description_instance_literal_presence = (
                    self._rpt_head_role_instance_from_raw(
                        description, literal['presence'], output_shape))
                literal_instance_description_presence = (
                    self._rpt_head_role_instance_from_raw(
                        literal, description['presence'], output_shape))

                path_maps = {}
                for path_name, semantic_source, instance_source, presence_source \
                        in HEAD_ROLE_PATHS:
                    if path_name == 'native':
                        final = description['native_final']
                        instance = description['native_instance']
                        kept_count = description['native_kept_count']
                    else:
                        semantic_package = (
                            description if semantic_source == 'description'
                            else literal)
                        instance_package = (
                            description if instance_source == 'description'
                            else literal)
                        presence_package = (
                            description if presence_source == 'description'
                            else literal)
                        if (
                                instance_source == 'description'
                                and presence_source == 'literal'):
                            instance_override = (
                                description_instance_literal_presence)
                        elif (
                                instance_source == 'literal'
                                and presence_source == 'description'):
                            instance_override = (
                                literal_instance_description_presence)
                        elif (
                                instance_source == 'literal'
                                and presence_source == 'literal'):
                            instance_override = literal_rebuilt_instance
                        else:
                            instance_override = description_native_instance
                        final, instance, kept_count = (
                            self._rpt_head_role_compose(
                                semantic_package,
                                instance_package,
                                presence_package,
                                output_shape,
                                instance_override=instance_override,
                            ))
                    name = head_role_variant_name(prompt_slot, path_name)
                    class_variants[name][class_idx].copy_(
                        final.detach().float().cpu())
                    path_maps[path_name] = dict(
                        final=final,
                        instance=instance,
                        kept_count=int(kept_count),
                    )

                raw_scores = description['raw_scores']
                literal_cache_index = int(cache['index'][literal_prompt])
                description_cache_index = int(
                    cache['index'][description_prompt])
                head_role_rows.append(dict(
                    class_index=int(class_idx),
                    class_name=item['name'],
                    prompt_slot=int(prompt_slot),
                    prompt_role=prompt_role,
                    literal_prompt=literal_prompt,
                    description_prompt=description_prompt,
                    literal_token_count=int((
                        ~cache['language_mask'][literal_cache_index].bool()
                    ).sum().item()),
                    description_token_count=int((
                        ~cache['language_mask'][description_cache_index].bool()
                    ).sum().item()),
                    description_word_count=len(description_prompt.split()),
                    description_character_count=len(description_prompt),
                    literal_presence=float(literal['presence']),
                    description_presence=float(description['presence']),
                    presence_delta=float(
                        description['presence'] - literal['presence']),
                    literal_semantic_mean=float(
                        literal['semantic'].mean().item()),
                    description_semantic_mean=float(
                        description['semantic'].mean().item()),
                    description_semantic_area_050=float((
                        description['semantic'] >= 0.5).float().mean().item()),
                    literal_instance_mean=float(
                        literal['native_instance'].mean().item()),
                    description_instance_mean=float(
                        description['native_instance'].mean().item()),
                    semantic_abs_change=float((
                        description['semantic']
                        - literal['semantic']).abs().mean().item()),
                    instance_abs_change_native=float((
                        description['native_instance']
                        - literal['native_instance']).abs().mean().item()),
                    semantic_instance_iou=float(
                        self._rpt_head_role_mask_iou(
                            description['semantic'],
                            description['native_instance'])),
                    raw_object_score_mean=(
                        float(raw_scores.mean().item())
                        if raw_scores.numel() else 0.0),
                    raw_object_score_max=(
                        float(raw_scores.max().item())
                        if raw_scores.numel() else 0.0),
                    raw_candidate_count=int(
                        description['raw_candidate_count']),
                    literal_raw_candidate_count=int(
                        literal['raw_candidate_count']),
                    literal_native_kept_count=int(
                        literal['native_kept_count']),
                    native_kept_count=int(
                        description['native_kept_count']),
                    description_instance_literal_presence_kept_count=int(
                        description_instance_literal_presence[1]),
                    literal_instance_description_presence_kept_count=int(
                        literal_instance_description_presence[1]),
                    native_recomposition_max_abs=native_error,
                    native_instance_recomposition_max_abs=(
                        native_instance_error),
                    path_final_means={
                        name: float(values['final'].mean().item())
                        for name, values in path_maps.items()
                    },
                    path_kept_counts={
                        name: int(values['kept_count'])
                        for name, values in path_maps.items()
                    },
                ))
                del description
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            del literal

        if tuple(class_variants) != HEAD_ROLE_VARIANT_NAMES:
            raise RuntimeError(
                'Head-role prompt-conflict variant order drifted: '
                f'{tuple(class_variants)}')
        raw_recomposition_max_abs = max(raw_recomposition_errors or [0.0])
        if (
                self.role_prompt_tta_strict_integrity
                and raw_recomposition_max_abs
                > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                'Raw-query native recomposition failed: max_abs='
                f'{raw_recomposition_max_abs}.')

        literal_vs_official = float((
            class_variants['literal_native']
            - class_variants['baseline']).abs().max().item())
        diagnostic_cpu_bytes = int(sum(
            value.numel() * value.element_size()
            for value in class_variants.values()))
        view_stats = dict(
            schema_version=HEAD_ROLE_SCHEMA_VERSION,
            protocol=self.role_prompt_tta_protocol,
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            adaptation_unit='sam3_crop' if crop_box is not None else 'full_image',
            baseline_reconstruction_max_abs=float(
                self._rpt_native_parity_max_abs or 0.0),
            raw_recomposition_max_abs=float(raw_recomposition_max_abs),
            literal_vs_official_max_abs=literal_vs_official,
            official_query_count=int(self.num_queries),
            canonical_class_count=int(self.num_cls),
            synonym_query_count=int(self.num_queries - self.num_cls),
            diagnostic_variant_count=len(class_variants),
            diagnostic_cpu_bytes=diagnostic_cpu_bytes,
            head_role_rows=head_role_rows,
            cuda_memory=dict(main=cuda_memory_snapshot(self.device)),
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

    def _rpt_atlas_infer_single_view(
            self, image, return_stats=False, return_components=False,
            view_id=None, crop_box=None):
        """Run the v2 functional atlas without making a diagnostic the method.

        All prompt interventions share one SAM3 image encoding.  Variant maps
        are transferred to CPU as soon as they are produced so large images do
        not retain every diagnostic output on the SAM3 GPU simultaneously.
        """
        width, height = image.size
        output_shape = (height, width)
        remoteclip_device = self._rpt_remoteclip.device
        memory_devices = [self.device]
        if remoteclip_device != self.device:
            memory_devices.append(remoteclip_device)
        for memory_device in memory_devices:
            if memory_device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(memory_device)

        state, cache, baseline, bank, candidate_stats = (
            self._rpt_collect_prompt_outputs(image))
        baseline_class = {
            key: self._rpt_query_to_class(value)
            for key, value in baseline.items()
        }
        class_count = int(self.num_cls)
        prompt_count = int(self._rpt_prompt_bank['_prompt_count'])
        if prompt_count != ATLAS_PROMPT_COUNT:
            raise RuntimeError(
                f'Atlas expected {ATLAS_PROMPT_COUNT} prompts; '
                f'got {prompt_count}.')

        anchor_weights = initial_prompt_weights(
            class_count, prompt_count,
            self.role_prompt_tta_anchor_mass, self.device)
        visual = self._rpt_seed_and_visual_evidence(
            image, baseline_class['semantic_raw'],
            baseline_class['presence'])
        zeros = torch.zeros_like(visual['affinity'])
        ones = torch.ones_like(visual['presence_gate'])
        if not self.role_prompt_tta_adapt_background:
            ones[int(self.bg_idx)] = 0.0
        visual_weights = prompt_weights_from_state(
            anchor_weights, visual['affinity'], visual['presence_gate'], None,
            self.role_prompt_tta_visual_strength,
            self.role_prompt_tta_delta_max, self.role_prompt_tta_eps)
        entropy_weights, _, entropy_trajectory = optimize_surrogate_weights(
            bank['semantic_raw'], anchor_weights, zeros, ones,
            baseline_class['semantic_raw'].argmax(dim=0),
            self.role_prompt_tta_steps, self.role_prompt_tta_lr,
            self.role_prompt_tta_temperature,
            self.role_prompt_tta_anchor_lambda, 0.0,
            self.role_prompt_tta_delta_max,
            self.role_prompt_tta_class_balanced_entropy,
            self.role_prompt_tta_eps)
        full_weights, _, full_trajectory = optimize_surrogate_weights(
            bank['semantic_raw'], anchor_weights, visual['affinity'],
            visual['presence_gate'],
            baseline_class['semantic_raw'].argmax(dim=0),
            self.role_prompt_tta_steps, self.role_prompt_tta_lr,
            self.role_prompt_tta_temperature,
            self.role_prompt_tta_anchor_lambda,
            self.role_prompt_tta_visual_strength,
            self.role_prompt_tta_delta_max,
            self.role_prompt_tta_class_balanced_entropy,
            self.role_prompt_tta_eps)
        e2e_weights, e2e_trajectory, e2e_classes = (
            self._rpt_e2e_refine_weights(
                state, cache, full_weights,
                baseline_class['semantic_raw'], visual))

        literal_weights = torch.zeros_like(anchor_weights)
        literal_weights[:, 0] = 1.0
        full_bg_literal_weights = full_weights.detach().clone()
        full_bg_literal_weights[int(self.bg_idx)] = (
            literal_weights[int(self.bg_idx)])
        e2e_bg_literal_weights = e2e_weights.detach().clone()
        e2e_bg_literal_weights[int(self.bg_idx)] = (
            literal_weights[int(self.bg_idx)])

        prompt_presence = bank['presence'].float().clamp_min(0.0)
        presence_choice = prompt_presence.argmax(dim=1)
        class_indices = torch.arange(class_count, device=self.device)
        presence_selected = bank['final'][class_indices, presence_choice]
        presence_denominator = prompt_presence.sum(dim=1, keepdim=True)
        presence_weights = prompt_presence / presence_denominator.clamp_min(
            self.role_prompt_tta_eps)
        empty_presence = presence_denominator.squeeze(1) <= self.role_prompt_tta_eps
        if empty_presence.any():
            presence_weights[empty_presence] = literal_weights[empty_presence]

        query_variants = OrderedDict()

        def store_query(name, query_logits):
            query_variants[name] = query_logits.detach().float().cpu()

        def store_class(name, class_logits):
            store_query(name, self._rpt_class_to_query(class_logits))

        def weighted_output(weights):
            return (weights[:, :, None, None] * bank['final']).sum(dim=1)

        store_query('baseline', baseline['final'])
        for prompt_idx, (variant_name, _) in enumerate(ATLAS_PROMPT_SLOTS):
            store_class(variant_name, bank['final'][:, prompt_idx])
        temporary = bank['final'].mean(dim=1)
        store_class('pool_mean', temporary)
        del temporary
        temporary = bank['final'].max(dim=1)[0]
        store_class('pool_max', temporary)
        del temporary
        for name, weights in (
                ('anchor_output', anchor_weights),
                ('visual_output', visual_weights),
                ('entropy_output', entropy_weights),
                ('full_output', full_weights),
                ('full_bg_literal_output', full_bg_literal_weights)):
            temporary = weighted_output(weights)
            store_class(name, temporary)
            del temporary
        store_class('presence_selected_output', presence_selected)
        del presence_selected
        for name, weights in (
                ('presence_weighted_output', presence_weights),
                ('e2e_output', e2e_weights),
                ('e2e_bg_literal_output', e2e_bg_literal_weights)):
            temporary = weighted_output(weights)
            store_class(name, temporary)
            del temporary

        reground_stats = {}
        head_change = []
        literal_reground, literal_fusion = self._rpt_reground_weights(
            state, cache, literal_weights, output_shape)
        store_class('literal_regrounded', literal_reground['final'])
        literal_parity = {
            key: float((
                literal_reground[key].float()
                - baseline_class[key].float()).abs().max().item())
            for key in ('final', 'semantic', 'semantic_raw', 'instance', 'presence')
        }
        reground_stats['literal_regrounded'] = literal_fusion
        head_change.append(self._rpt_head_change_block(
            'literal_regrounded', literal_reground, baseline_class))
        del literal_reground
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        for alpha, name in (
                (0.10, 'anchor_residual_010_regrounded'),
                (0.25, 'anchor_residual_025_regrounded')):
            outputs, fusion = self._rpt_reground_anchor_residual(
                state, cache, full_weights, output_shape, alpha)
            store_class(name, outputs['final'])
            reground_stats[name] = fusion
            head_change.append(self._rpt_head_change_block(
                name, outputs, baseline_class))
            del outputs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        full_sequence, full_sequence_fusion = self._rpt_reground_weights(
            state, cache, full_weights, output_shape)
        store_class('full_sequence_regrounded', full_sequence['final'])
        reground_stats['full_sequence_regrounded'] = full_sequence_fusion
        head_change.append(self._rpt_head_change_block(
            'full_sequence_regrounded', full_sequence, baseline_class))
        del full_sequence
        if tuple(query_variants) != ATLAS_VARIANT_NAMES:
            raise RuntimeError(
                'Prompt functional atlas variant order drifted: '
                f'{tuple(query_variants)}')

        weight_stats = [
            self._rpt_weights_stats('literal', literal_weights),
            self._rpt_weights_stats('anchor', anchor_weights),
            self._rpt_weights_stats('visual', visual_weights),
            self._rpt_weights_stats('entropy', entropy_weights),
            self._rpt_weights_stats('full', full_weights),
            self._rpt_weights_stats('full_bg_literal', full_bg_literal_weights),
            self._rpt_weights_stats('presence_weighted', presence_weights),
            self._rpt_weights_stats('e2e', e2e_weights),
            self._rpt_weights_stats(
                'e2e_bg_literal', e2e_bg_literal_weights),
        ]
        view_stats = dict(
            schema_version=ATLAS_SCHEMA_VERSION,
            protocol=self.role_prompt_tta_protocol,
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            adaptation_unit='sam3_crop' if crop_box is not None else 'full_image',
            baseline_reconstruction_max_abs=float(
                self._rpt_native_parity_max_abs or 0.0),
            literal_reground_parity_max_abs=literal_parity,
            prompt_slots=[dict(
                index=int(idx), variant=name, role=role)
                for idx, (name, role) in enumerate(ATLAS_PROMPT_SLOTS)],
            candidate_stats=candidate_stats,
            presence_selected_indices=[
                int(value) for value in presence_choice.detach().cpu().tolist()],
            seed_counts=visual['seed_counts'],
            seed_entropy=visual['seed_entropy'],
            prototype_norms=visual['prototype_norms'],
            presence_gate=[
                float(value)
                for value in visual['presence_gate'].cpu().tolist()],
            visual_affinity=(
                visual['affinity'].detach().float().cpu().tolist()),
            remoteclip_grid_shape=visual['grid_shape'],
            remoteclip_device=visual['remoteclip_device'],
            seed_indices=[
                torch.nonzero(mask.flatten(), as_tuple=False)
                .flatten().detach().cpu().tolist()
                for mask in visual['seed_masks']
            ],
            remoteclip_global_norm=visual['remoteclip_global_norm'],
            anchor_entropy_mean=visual['anchor_entropy_mean'],
            anchor_entropy_std=visual['anchor_entropy_std'],
            weight_stats=weight_stats,
            entropy_trajectory=entropy_trajectory,
            full_trajectory=full_trajectory,
            e2e_trajectory=e2e_trajectory,
            e2e_updated_classes=e2e_classes,
            language_fusion_stats=reground_stats,
            head_change=head_change,
            cuda_memory=dict(
                main=cuda_memory_snapshot(self.device),
                remoteclip=cuda_memory_snapshot(remoteclip_device),
            ),
        )

        if self.role_prompt_tta_primary_variant == 'baseline':
            primary_query = baseline['final']
        else:
            primary_query = query_variants[
                self.role_prompt_tta_primary_variant].to(self.device)
        components = dict(
            semantic_logits=baseline['semantic'],
            instance_logits=baseline['instance'],
            role_prompt_variant_query_logits=query_variants,
            role_prompt_view_stats=[view_stats],
        )
        stats = dict(
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            role_prompt_tta=view_stats,
        )
        del bank, baseline_class
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        if not return_stats and not return_components:
            return primary_query
        if return_stats and return_components:
            return primary_query, stats, components
        if return_stats:
            return primary_query, stats
        return primary_query, components

    def _rpt_infer_single_view(
            self, image, return_stats=False, return_components=False,
            view_id=None, crop_box=None):
        if self.role_prompt_tta_protocol == SEMANTIC_SUPPLEMENT_PROTOCOL:
            return self._rpt_semantic_supplement_infer_single_view(
                image,
                return_stats=return_stats,
                return_components=return_components,
                view_id=view_id,
                crop_box=crop_box,
            )
        if self.role_prompt_tta_protocol == 'head_role_prompt_conflict_v1':
            return self._rpt_head_role_infer_single_view(
                image,
                return_stats=return_stats,
                return_components=return_components,
                view_id=view_id,
                crop_box=crop_box,
            )
        if self.role_prompt_tta_protocol == 'prompt_functional_atlas_v2':
            return self._rpt_atlas_infer_single_view(
                image,
                return_stats=return_stats,
                return_components=return_components,
                view_id=view_id,
                crop_box=crop_box,
            )
        width, height = image.size
        output_shape = (height, width)
        memory_devices = [self.device]
        remoteclip_device = self._rpt_remoteclip.device
        if remoteclip_device != self.device:
            memory_devices.append(remoteclip_device)
        for memory_device in memory_devices:
            if memory_device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(memory_device)
        state, cache, baseline, bank, candidate_stats = (
            self._rpt_collect_prompt_outputs(image))
        baseline_class = {
            key: self._rpt_query_to_class(value)
            for key, value in baseline.items()
        }
        anchor_weights = initial_prompt_weights(
            self.num_cls, self._rpt_prompt_bank['_prompt_count'],
            self.role_prompt_tta_anchor_mass, self.device)
        visual = self._rpt_seed_and_visual_evidence(
            image, baseline_class['semantic_raw'],
            baseline_class['presence'])
        zeros = torch.zeros_like(visual['affinity'])
        ones = torch.ones_like(visual['presence_gate'])
        if not self.role_prompt_tta_adapt_background:
            ones[int(self.bg_idx)] = 0.0
        uniform_weights = torch.full_like(
            anchor_weights, 1.0 / float(anchor_weights.shape[1]))
        visual_weights = prompt_weights_from_state(
            anchor_weights, visual['affinity'], visual['presence_gate'], None,
            self.role_prompt_tta_visual_strength,
            self.role_prompt_tta_delta_max, self.role_prompt_tta_eps)
        entropy_weights, _, entropy_trajectory = optimize_surrogate_weights(
            bank['semantic_raw'], anchor_weights, zeros, ones,
            baseline_class['semantic_raw'].argmax(dim=0),
            self.role_prompt_tta_steps, self.role_prompt_tta_lr,
            self.role_prompt_tta_temperature,
            self.role_prompt_tta_anchor_lambda, 0.0,
            self.role_prompt_tta_delta_max,
            self.role_prompt_tta_class_balanced_entropy,
            self.role_prompt_tta_eps)
        full_weights, _, full_trajectory = optimize_surrogate_weights(
            bank['semantic_raw'], anchor_weights, visual['affinity'],
            visual['presence_gate'],
            baseline_class['semantic_raw'].argmax(dim=0),
            self.role_prompt_tta_steps, self.role_prompt_tta_lr,
            self.role_prompt_tta_temperature,
            self.role_prompt_tta_anchor_lambda,
            self.role_prompt_tta_visual_strength,
            self.role_prompt_tta_delta_max,
            self.role_prompt_tta_class_balanced_entropy,
            self.role_prompt_tta_eps)
        no_anchor_weights, _, no_anchor_trajectory = optimize_surrogate_weights(
            bank['semantic_raw'], uniform_weights, visual['affinity'],
            visual['presence_gate'],
            baseline_class['semantic_raw'].argmax(dim=0),
            self.role_prompt_tta_steps, self.role_prompt_tta_lr,
            self.role_prompt_tta_temperature, 0.0,
            self.role_prompt_tta_visual_strength,
            self.role_prompt_tta_delta_max,
            self.role_prompt_tta_class_balanced_entropy,
            self.role_prompt_tta_eps)
        no_presence_weights, _, no_presence_trajectory = optimize_surrogate_weights(
            bank['semantic_raw'], anchor_weights, visual['affinity'],
            visual['seed_gate'],
            baseline_class['semantic_raw'].argmax(dim=0),
            self.role_prompt_tta_steps, self.role_prompt_tta_lr,
            self.role_prompt_tta_temperature,
            self.role_prompt_tta_anchor_lambda,
            self.role_prompt_tta_visual_strength,
            self.role_prompt_tta_delta_max,
            self.role_prompt_tta_class_balanced_entropy,
            self.role_prompt_tta_eps)

        # Run the only activation-heavy gradient path before materializing the
        # four no-grad re-grounded diagnostic maps. They are independent of
        # E2E adaptation, so this reordering is numerically neutral and lowers
        # the peak live tensor set during backward.
        e2e_weights, e2e_trajectory, e2e_classes = (
            self._rpt_e2e_refine_weights(
                state, cache, full_weights,
                baseline_class['semantic_raw'], visual))
        anchor_reground, anchor_fusion = self._rpt_reground_weights(
            state, cache, anchor_weights, output_shape)
        uniform_reground, uniform_fusion = self._rpt_reground_weights(
            state, cache, uniform_weights, output_shape)
        visual_reground, visual_fusion = self._rpt_reground_weights(
            state, cache, visual_weights, output_shape)
        full_reground, full_fusion = self._rpt_reground_weights(
            state, cache, full_weights, output_shape)
        e2e_reground, e2e_fusion = self._rpt_reground_weights(
            state, cache, e2e_weights, output_shape)

        def weighted_output(weights):
            return (weights[:, :, None, None] * bank['final']).sum(dim=1)

        def weighted_components(weights):
            return dict(
                semantic=(weights[:, :, None, None]
                          * bank['semantic_raw'].sigmoid()).sum(dim=1),
                instance=(weights[:, :, None, None]
                          * bank['instance']).sum(dim=1),
            )

        class_variants = OrderedDict([
            ('baseline', baseline_class['final']),
            ('pool_max', bank['final'].max(dim=1)[0]),
            ('pool_mean', bank['final'].mean(dim=1)),
            ('anchor_output', weighted_output(anchor_weights)),
            ('uniform_output', weighted_output(uniform_weights)),
            ('visual_output', weighted_output(visual_weights)),
            ('entropy_output', weighted_output(entropy_weights)),
            ('full_output', weighted_output(full_weights)),
            ('full_no_anchor_output', weighted_output(no_anchor_weights)),
            ('full_no_presence_gate_output', weighted_output(no_presence_weights)),
            ('anchor_regrounded', anchor_reground['final']),
            ('uniform_regrounded', uniform_reground['final']),
            ('visual_regrounded', visual_reground['final']),
            ('full_regrounded_surrogate', full_reground['final']),
            ('full_regrounded_e2e', e2e_reground['final']),
        ])
        if tuple(class_variants) != V1_VARIANT_NAMES:
            raise RuntimeError('Role Prompt TTA variant order drifted.')
        query_variants = {
            name: (
                baseline['final'].detach().float().cpu()
                if name == 'baseline'
                else self._rpt_class_to_query(value).detach().float().cpu())
            for name, value in class_variants.items()
        }
        baseline_error = float(self._rpt_native_parity_max_abs or 0.0)
        if (
                self.role_prompt_tta_strict_integrity
                and baseline_error > self.role_prompt_tta_integrity_tolerance):
            raise RuntimeError(
                f'Baseline reconstruction failed: max_abs={baseline_error}.')

        primary = class_variants[self.role_prompt_tta_primary_variant]
        primary_query = self._rpt_class_to_query(primary)
        primary_component_variants = {
            'baseline': baseline_class,
            'pool_max': dict(
                semantic=bank['semantic_raw'].sigmoid().max(dim=1)[0],
                instance=bank['instance'].max(dim=1)[0]),
            'pool_mean': weighted_components(uniform_weights),
            'anchor_output': weighted_components(anchor_weights),
            'uniform_output': weighted_components(uniform_weights),
            'visual_output': weighted_components(visual_weights),
            'entropy_output': weighted_components(entropy_weights),
            'full_output': weighted_components(full_weights),
            'full_no_anchor_output': weighted_components(no_anchor_weights),
            'full_no_presence_gate_output': weighted_components(
                no_presence_weights),
            'anchor_regrounded': anchor_reground,
            'uniform_regrounded': uniform_reground,
            'visual_regrounded': visual_reground,
            'full_regrounded_surrogate': full_reground,
            'full_regrounded_e2e': e2e_reground,
        }
        primary_components = primary_component_variants[
            self.role_prompt_tta_primary_variant]
        semantic_query = self._rpt_class_to_query(primary_components['semantic'])
        instance_query = self._rpt_class_to_query(primary_components['instance'])

        weight_stats = [
            self._rpt_weights_stats('anchor', anchor_weights),
            self._rpt_weights_stats('uniform', uniform_weights),
            self._rpt_weights_stats('visual', visual_weights),
            self._rpt_weights_stats('entropy', entropy_weights),
            self._rpt_weights_stats('full', full_weights),
            self._rpt_weights_stats('full_no_anchor', no_anchor_weights),
            self._rpt_weights_stats('full_no_presence_gate', no_presence_weights),
            self._rpt_weights_stats('e2e', e2e_weights),
        ]
        def head_change_block(name, outputs):
            rows = []
            for class_idx in range(self.num_cls):
                semantic_mask = outputs['semantic'][class_idx] >= 0.5
                instance_mask = outputs['instance'][class_idx] >= 0.5
                intersection = (semantic_mask & instance_mask).sum().item()
                union = (semantic_mask | instance_mask).sum().item()
                rows.append(dict(
                    class_index=int(class_idx),
                    class_name=self.class_names[class_idx],
                    semantic_mean_delta=float((
                        outputs['semantic'][class_idx]
                        - baseline_class['semantic'][class_idx]).mean().item()),
                    semantic_abs_delta=float((
                        outputs['semantic'][class_idx]
                        - baseline_class['semantic'][class_idx]).abs().mean().item()),
                    instance_mean_delta=float((
                        outputs['instance'][class_idx]
                        - baseline_class['instance'][class_idx]).mean().item()),
                    instance_abs_delta=float((
                        outputs['instance'][class_idx]
                        - baseline_class['instance'][class_idx]).abs().mean().item()),
                    presence_delta=float((
                        outputs['presence'][class_idx]
                        - baseline_class['presence'][class_idx]).item()),
                    final_mean_delta=float((
                        outputs['final'][class_idx]
                        - baseline_class['final'][class_idx]).mean().item()),
                    semantic_instance_mask_iou=_safe_div(intersection, union),
                ))
            return dict(variant=name, classes=rows)

        head_change = [
            head_change_block('anchor_regrounded', anchor_reground),
            head_change_block('uniform_regrounded', uniform_reground),
            head_change_block('visual_regrounded', visual_reground),
            head_change_block('full_regrounded_surrogate', full_reground),
            head_change_block('full_regrounded_e2e', e2e_reground),
        ]
        view_stats = dict(
            schema_version=SCHEMA_VERSION,
            view_id=view_id,
            crop_box=crop_box,
            image_size=[width, height],
            adaptation_unit='sam3_crop' if crop_box is not None else 'full_image',
            baseline_reconstruction_max_abs=baseline_error,
            seed_counts=visual['seed_counts'],
            seed_entropy=visual['seed_entropy'],
            prototype_norms=visual['prototype_norms'],
            presence_gate=[float(value) for value in visual['presence_gate'].cpu().tolist()],
            seed_gate=[float(value) for value in visual['seed_gate'].cpu().tolist()],
            visual_affinity=visual['affinity'].detach().float().cpu().tolist(),
            remoteclip_grid_shape=visual['grid_shape'],
            remoteclip_device=visual['remoteclip_device'],
            seed_indices=[
                torch.nonzero(mask.flatten(), as_tuple=False)
                .flatten().detach().cpu().tolist()
                for mask in visual['seed_masks']
            ],
            remoteclip_global_norm=visual['remoteclip_global_norm'],
            anchor_entropy_mean=visual['anchor_entropy_mean'],
            anchor_entropy_std=visual['anchor_entropy_std'],
            weight_stats=weight_stats,
            entropy_trajectory=entropy_trajectory,
            full_trajectory=full_trajectory,
            no_anchor_trajectory=no_anchor_trajectory,
            no_presence_trajectory=no_presence_trajectory,
            e2e_trajectory=e2e_trajectory,
            e2e_updated_classes=e2e_classes,
            candidate_stats=candidate_stats,
            language_fusion_stats=dict(
                anchor=anchor_fusion,
                uniform=uniform_fusion,
                visual=visual_fusion,
                full=full_fusion,
                e2e=e2e_fusion,
            ),
            head_change=head_change,
            cuda_memory=dict(
                main=cuda_memory_snapshot(self.device),
                remoteclip=cuda_memory_snapshot(remoteclip_device),
            ),
        )
        components = dict(
            semantic_logits=semantic_query,
            instance_logits=instance_query,
            role_prompt_variant_query_logits=query_variants,
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

    def _rpt_threshold(self, logits):
        prediction = logits.argmax(dim=0)
        prediction[logits.max(dim=0)[0] < float(self.prob_thd)] = int(self.bg_idx)
        return prediction

    def _rpt_confusion(self, prediction, gt, valid):
        count = int(self.num_cls)
        encoded = gt[valid].long() * count + prediction[valid].long()
        matrix = torch.bincount(encoded, minlength=count * count).view(count, count)
        intersection = matrix.diag().float()
        gt_area = matrix.sum(dim=1).float()
        pred_area = matrix.sum(dim=0).float()
        union = gt_area + pred_area - intersection
        iou = intersection / union.clamp_min(1.0)
        valid_class = union > 0
        miou = iou[valid_class].mean() if valid_class.any() else torch.tensor(0.0)
        return dict(
            matrix=matrix.cpu().tolist(),
            intersection=intersection.cpu().tolist(),
            union=union.cpu().tolist(),
            iou=iou.cpu().tolist(),
            miou=float(miou.item() * 100.0),
            aacc=float(intersection.sum().item() / gt_area.sum().clamp_min(1.0).item() * 100.0),
        )

    def _rpt_enrich_seed_stats_with_gt(self, view_stats, gt):
        """Attach GT-only diagnostics; these values never affect adaptation."""
        enriched = []
        for raw_view in view_stats:
            view = dict(raw_view)
            crop_box = view.get('crop_box')
            if crop_box is None:
                crop_gt = gt
            else:
                x1, y1, x2, y2 = [int(value) for value in crop_box]
                crop_gt = gt[y1:y2, x1:x2]
            grid_shape = tuple(int(value) for value in view['remoteclip_grid_shape'])
            gt_grid = F.interpolate(
                crop_gt.float()[None, None], size=grid_shape,
                mode='nearest').squeeze().long()
            class_rows = []
            for class_idx, indices in enumerate(view.get('seed_indices', [])):
                if indices:
                    index = torch.tensor(
                        indices, device=gt_grid.device, dtype=torch.long)
                    labels = gt_grid.flatten()[index]
                    valid_labels = labels[labels != 255]
                else:
                    valid_labels = gt_grid.new_empty((0,), dtype=torch.long)
                histogram = torch.bincount(
                    valid_labels.clamp(0, int(self.num_cls) - 1),
                    minlength=int(self.num_cls)) if valid_labels.numel() else (
                        torch.zeros(
                            int(self.num_cls), device=gt_grid.device,
                            dtype=torch.long))
                correct = int(histogram[class_idx].item())
                total = int(valid_labels.numel())
                class_rows.append(dict(
                    class_index=int(class_idx),
                    class_name=self.class_names[class_idx],
                    valid_seed_pixels=total,
                    correct_seed_pixels=correct,
                    seed_purity=_safe_div(correct, total),
                    gt_histogram=histogram.detach().cpu().tolist(),
                ))
            view['seed_gt_diagnostics'] = class_rows
            enriched.append(view)
        return enriched

    def _rpt_record_image(self, variant_logits, view_stats, data_sample, image_path):
        if not self.dump_role_prompt_tta_stats:
            return
        if data_sample is None or not hasattr(data_sample, 'gt_sem_seg'):
            return
        gt_data = data_sample.gt_sem_seg.data.squeeze().detach().long().cpu()
        valid = gt_data != 255
        if not valid.any():
            return
        compact_predictions = (
            self.role_prompt_tta_protocol in (
                'head_role_prompt_conflict_v1',
                SEMANTIC_SUPPLEMENT_PROTOCOL))
        predictions = {
            name: (
                self._rpt_threshold(logits.detach().float().cpu())
                .to(torch.int16)
                if compact_predictions
                else self._rpt_threshold(logits.detach().float().cpu()))
            for name, logits in variant_logits.items()
        }
        baseline = predictions['baseline']
        variant_stats = {}
        variant_names = self._rpt_variant_names()
        if tuple(variant_logits) != tuple(variant_names):
            raise RuntimeError(
                'Recorded prompt variant order drifted: '
                f'{tuple(variant_logits)}')
        for name in variant_names:
            pred = predictions[name]
            changed = valid & (pred != baseline)
            improved = changed & (pred == gt_data) & (baseline != gt_data)
            harmed = changed & (pred != gt_data) & (baseline == gt_data)
            wrong_to_wrong = changed & (pred != gt_data) & (baseline != gt_data)
            variant_stats[name] = dict(
                confusion=self._rpt_confusion(pred, gt_data, valid),
                changed_pixels=int(changed.sum().item()),
                changed_ratio=_safe_div(int(changed.sum().item()), int(valid.sum().item())),
                improved_pixels=int(improved.sum().item()),
                harmed_pixels=int(harmed.sum().item()),
                wrong_to_wrong_pixels=int(wrong_to_wrong.sum().item()),
                help_minus_harm=int(improved.sum().item() - harmed.sum().item()),
                change_precision=_safe_div(
                    int(improved.sum().item()), int(changed.sum().item())),
            )
        record = dict(
            schema_version=self._rpt_record_schema_version(),
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
                protocol=self.role_prompt_tta_protocol,
                remoteclip_device=(
                    None if self._rpt_remoteclip is None
                    else str(self._rpt_remoteclip.device)),
                anchor_mass=self.role_prompt_tta_anchor_mass,
                visual_strength=self.role_prompt_tta_visual_strength,
                presence_threshold=self.role_prompt_tta_presence_threshold,
                seed_fraction=self.role_prompt_tta_seed_fraction,
                min_seed_pixels=self.role_prompt_tta_min_seed_pixels,
                temperature=self.role_prompt_tta_temperature,
                surrogate_steps=self.role_prompt_tta_steps,
                surrogate_lr=self.role_prompt_tta_lr,
                anchor_lambda=self.role_prompt_tta_anchor_lambda,
                delta_max=self.role_prompt_tta_delta_max,
                class_balanced_entropy=self.role_prompt_tta_class_balanced_entropy,
                semantic_residual_alpha=(
                    self.role_prompt_tta_semantic_residual_alpha),
                semantic_residual_clip=(
                    self.role_prompt_tta_semantic_residual_clip),
                e2e_steps=self.role_prompt_tta_e2e_steps,
                e2e_lr=self.role_prompt_tta_e2e_lr,
                e2e_max_classes=self.role_prompt_tta_e2e_max_classes,
                e2e_measure_post=self.role_prompt_tta_e2e_measure_post,
                adapt_background=self.role_prompt_tta_adapt_background,
            ),
            variants=variant_stats,
            views=(
                [dict(view) for view in view_stats]
                if self.role_prompt_tta_protocol in (
                    'head_role_prompt_conflict_v1',
                    SEMANTIC_SUPPLEMENT_PROTOCOL)
                else self._rpt_enrich_seed_stats_with_gt(
                    view_stats, gt_data)),
        )
        self._rpt_write_stats(record)
        self._rpt_save_artifact(image_path, predictions, gt_data, valid)

    def _rpt_write_stats(self, record):
        if self._role_prompt_tta_stats_file is None:
            path = self.role_prompt_tta_stats_path or (
                './work_dirs/role_prompt_tta/role_prompt_tta.jsonl')
            path = _ranked_jsonl_path(path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self._role_prompt_tta_stats_file = open(
                path, 'a', buffering=1, encoding='utf-8')
        self._role_prompt_tta_stats_file.write(
            json.dumps(record, ensure_ascii=False) + '\n')

    def _rpt_save_artifact(self, image_path, predictions, gt, valid):
        if (
                not self.role_prompt_tta_save_npz
                or self._role_prompt_tta_saved_images
                >= self.role_prompt_tta_max_saved_images):
            return
        directory = self.role_prompt_tta_artifact_dir or (
            './work_dirs/role_prompt_tta/artifacts')
        rank_dir = os.path.join(
            directory, f'rank{int(os.environ.get("RANK", 0))}')
        os.makedirs(rank_dir, exist_ok=True)
        height, width = gt.shape[-2:]
        scale = min(
            1.0,
            float(self.role_prompt_tta_artifact_max_side)
            / float(max(height, width)))
        target = (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))))

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
            os.path.join(rank_dir, f'{self._role_prompt_tta_saved_images:04d}_{stem}.npz'),
            **arrays)
        self._role_prompt_tta_saved_images += 1
