"""Trainable, baseline-protected Prompt-SAM3 v1 segmentor.

This variant freezes SAM3 (and optional RemoteCLIP), learns only a compact
category-agnostic prompt synthesizer, and keeps SAM3's semantic, instance, and
presence grounding path.  The official ``SegEarthOV3Segmentation`` class and
its evaluation configs are not modified.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS
from PIL import Image

from prompt_remoteclip import FrozenRemoteCLIPGlobalEncoder
from prompt_synthesis import COMPONENT_MODES, PromptSynthesisModule
from sam3.model.data_misc import FindStage
from segearthov3_segmentor import SegEarthOV3Segmentation


def _soft_dice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    smooth: float = 1.0,
) -> torch.Tensor:
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).to(probability.dtype)
    valid_mask = valid.unsqueeze(1).to(probability.dtype)
    probability = probability * valid_mask
    one_hot = one_hot * valid_mask
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def _ranked_path(path: str) -> str:
    rank = int(os.environ.get("RANK", "0"))
    stem, suffix = os.path.splitext(path)
    return f"{stem}_rank{rank}{suffix}"


@MODELS.register_module()
class PromptSegEarthOV3Segmentation(SegEarthOV3Segmentation):
    """Prompt-SAM3 realization B with switchable image-global source."""

    def __init__(
        self,
        *args,
        global_feature_source: str = "sam3_global",
        global_feature_dim: Optional[int] = None,
        remoteclip_checkpoint: str = (
            "weights/remoteclip/RemoteCLIP-ViT-L-14.pt"
        ),
        remoteclip_source_root: str = "SCORE-main/open_clip_training/src",
        num_shared_tokens: int = 4,
        num_state_prototypes: int = 4,
        num_global_tokens: int = 1,
        prompt_residual_scale: float = 0.05,
        prompt_initial_gate: float = 0.10,
        prompt_attention_temperature: float = 0.07,
        prompt_component_mode: str = "full",
        prompt_batch_size: int = 1,
        use_grounding_checkpoint: bool = True,
        sam_image_size: int = 1008,
        loss_ce_weight: float = 1.0,
        loss_dice_weight: float = 1.0,
        loss_anchor_weight: float = 0.05,
        ignore_index: int = 255,
        score_epsilon: float = 1e-6,
        diagnostic_dir: Optional[str] = None,
        diagnostic_panel_file: Optional[str] = None,
        diagnostic_max_images: int = 32,
        diagnostic_max_side: int = 128,
        data_preprocessor: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if prompt_component_mode not in COMPONENT_MODES:
            raise ValueError(
                f"prompt_component_mode must be one of {COMPONENT_MODES}"
            )
        global_feature_source = str(global_feature_source).lower()
        if global_feature_source not in ("sam3_global", "remoteclip_global"):
            raise ValueError(
                "global_feature_source must be 'sam3_global' or "
                "'remoteclip_global'"
            )

        # The transformed tensor becomes the single source of truth.  BGR->RGB
        # is done once here; both frozen encoders then see this exact augmented
        # RGB view before their own resize/normalization.
        preprocessor_cfg = dict(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=ignore_index,
        )
        if data_preprocessor:
            preprocessor_cfg.update(data_preprocessor)
        self.data_preprocessor = SegDataPreProcessor(**preprocessor_cfg)

        self.global_feature_source = global_feature_source
        self.prompt_component_mode = prompt_component_mode
        self.prompt_batch_size = int(prompt_batch_size)
        self.use_grounding_checkpoint = bool(use_grounding_checkpoint)
        self.sam_image_size = int(sam_image_size)
        if self.sam_image_size != int(self.processor.resolution):
            raise ValueError(
                "sam_image_size must match the baseline processor resolution "
                f"({self.processor.resolution})"
            )
        self.loss_ce_weight = float(loss_ce_weight)
        self.loss_dice_weight = float(loss_dice_weight)
        self.loss_anchor_weight = float(loss_anchor_weight)
        self.ignore_index = int(ignore_index)
        self.score_epsilon = float(score_epsilon)
        self.diagnostic_dir = diagnostic_dir
        self.diagnostic_panel_file = diagnostic_panel_file
        self.diagnostic_max_images = int(diagnostic_max_images)
        self.diagnostic_max_side = int(diagnostic_max_side)
        self._diagnostic_written = set()
        self._diagnostic_panel = self._read_diagnostic_panel(
            diagnostic_panel_file
        )

        sam_model = self.processor.model
        sam_model.eval()
        for parameter in sam_model.parameters():
            parameter.requires_grad_(False)
        self._ddp_dummy_param.requires_grad_(False)

        language_backbone = sam_model.backbone.language_backbone
        context_length = int(language_backbone.context_length)
        tokenized = language_backbone.tokenizer(
            self.query_words, context_length=context_length
        ).to(self.device)
        with torch.no_grad():
            static_inputs = language_backbone.encoder.token_embedding(tokenized)
            static_mask, static_memory, _ = language_backbone(
                self.query_words, device=self.device
            )
        # Persistent=False is essential: target datasets can have a different
        # number of class synonyms, while the trainable parameters are Q-free.
        self.register_buffer("prompt_tokenized", tokenized, persistent=False)
        self.register_buffer(
            "prompt_static_inputs", static_inputs.detach(), persistent=False
        )
        self.register_buffer(
            "prompt_static_language",
            static_memory.transpose(0, 1).detach(),
            persistent=False,
        )
        self.register_buffer(
            "prompt_padding_mask", static_mask.detach(), persistent=False
        )

        inferred_global_dim = 256 if global_feature_source == "sam3_global" else 768
        if global_feature_dim is not None and int(global_feature_dim) != inferred_global_dim:
            raise ValueError(
                f"{global_feature_source} requires global_feature_dim="
                f"{inferred_global_dim}, got {global_feature_dim}"
            )
        self.prompt_synthesizer = PromptSynthesisModule(
            spatial_dim=int(self.prompt_static_language.shape[-1]),
            language_dim=int(self.prompt_static_language.shape[-1]),
            text_width=int(self.prompt_static_inputs.shape[-1]),
            global_dim=inferred_global_dim,
            context_length=context_length,
            num_shared_tokens=num_shared_tokens,
            num_state_prototypes=num_state_prototypes,
            num_global_tokens=num_global_tokens,
            residual_scale=prompt_residual_scale,
            initial_gate=prompt_initial_gate,
            attention_temperature=prompt_attention_temperature,
        )

        remote_encoder = None
        if global_feature_source == "remoteclip_global":
            remote_encoder = FrozenRemoteCLIPGlobalEncoder(
                checkpoint_path=remoteclip_checkpoint,
                source_root=remoteclip_source_root,
                device=self.device,
            )
        # Keep the frozen external model out of Module registration/state_dict.
        object.__setattr__(self, "_remoteclip_global_encoder", remote_encoder)

    @staticmethod
    def _read_diagnostic_panel(path: Optional[str]) -> Optional[set]:
        if not path:
            return None
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Diagnostic panel file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            return {
                os.path.basename(line.strip())
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            }

    def train(self, mode: bool = True):
        super().train(mode)
        # The parent SAM3 is a deliberately unregistered frozen object.
        self.processor.model.eval()
        if self._remoteclip_global_encoder is not None:
            self._remoteclip_global_encoder.model.eval()
        return self

    def _canonical_rgb(self, inputs: torch.Tensor) -> torch.Tensor:
        rgb = inputs.to(device=self.device, dtype=torch.float32)
        if rgb.detach().amax() > 1.5:
            rgb = rgb / 255.0
        return rgb.clamp(0.0, 1.0)

    def _encode_sam_image(
        self, rgb_01: torch.Tensor
    ) -> Tuple[dict, torch.Tensor]:
        with torch.no_grad():
            # Reuse the baseline processor's tensor resize/normalization exactly;
            # unlike ``set_image``, this does not detach/reopen the source image.
            sam_image = torch.stack(
                [self.processor.transform(image) for image in rgb_01], dim=0
            )
            backbone_out = self.processor.model.backbone.forward_image(sam_image)
        spatial = backbone_out["vision_features"].detach()
        return backbone_out, spatial

    def _global_feature(
        self, rgb_01: torch.Tensor, spatial: torch.Tensor
    ) -> torch.Tensor:
        if self.global_feature_source == "sam3_global":
            return F.normalize(
                F.adaptive_avg_pool2d(spatial.float(), 1).flatten(1),
                dim=-1,
                eps=1e-6,
            ).detach()
        return self._remoteclip_global_encoder(rgb_01)

    def _encode_dynamic_text(
        self,
        spatial: torch.Tensor,
        global_feature: torch.Tensor,
        component_mode: str,
    ) -> Tuple[dict, Dict[str, torch.Tensor], torch.Tensor]:
        dynamic_inputs, diagnostics = self.prompt_synthesizer(
            spatial_feature=spatial,
            static_inputs_embeds=self.prompt_static_inputs,
            static_language=self.prompt_static_language,
            padding_mask=self.prompt_padding_mask,
            global_feature=global_feature,
            component_mode=component_mode,
        )
        if dynamic_inputs.shape[0] != 1:
            raise RuntimeError(
                "Prompt-SAM3 v1 currently requires batch_size=1 per GPU; "
                "use gradient accumulation for the effective batch size."
            )
        language_backbone = self.processor.model.backbone.language_backbone
        text_mask, text_memory, text_inputs = language_backbone.forward_embeddings(
            self.prompt_tokenized, dynamic_inputs[0]
        )
        text_output = {
            "language_features": text_memory,
            "language_mask": text_mask,
            "language_embeds": text_inputs,
        }
        active = ~text_mask
        dynamic_qld = text_memory.transpose(0, 1)
        cosine = F.cosine_similarity(
            dynamic_qld.float(), self.prompt_static_language.float(), dim=-1
        )
        diagnostics["static_dynamic_cosine"] = cosine.masked_select(active).mean()
        diagnostics["dynamic_language"] = dynamic_qld
        diagnostics["dynamic_inputs"] = dynamic_inputs[0]
        return text_output, diagnostics, dynamic_qld

    def _find_stage(self, count: int) -> FindStage:
        return FindStage(
            img_ids=torch.zeros(count, device=self.device, dtype=torch.long),
            text_ids=torch.arange(count, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _reduce_grounding_output(
        self,
        output: dict,
        output_size: Tuple[int, int],
        hard_candidates: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = F.interpolate(
            output["pred_masks"].float(),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        object_raw = output["pred_logits"].float().sigmoid().squeeze(-1)
        presence = output["presence_logit_dec"].float().sigmoid().reshape(-1)
        object_presence = object_raw * presence.unsqueeze(1)
        object_score = (
            object_raw
            if self.instance_score_type == "raw"
            else object_presence
        )
        weighted_masks = masks * object_score.unsqueeze(-1).unsqueeze(-1)
        if hard_candidates:
            keep = object_presence > self.confidence_threshold
            weighted_masks = weighted_masks.masked_fill(
                ~keep.unsqueeze(-1).unsqueeze(-1), 0.0
            )
        instance = weighted_masks.amax(dim=1)
        semantic = F.interpolate(
            output["semantic_seg"].float(),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).sigmoid().squeeze(1)
        if not self.use_sem_seg:
            semantic = torch.zeros_like(semantic)
        final = torch.maximum(instance, semantic)
        if self.use_presence_score:
            final = final * presence.view(-1, 1, 1)
        return final, semantic, instance, presence

    def _ground_queries(
        self,
        image_backbone: dict,
        text_output: dict,
        output_size: Tuple[int, int],
        hard_candidates: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.processor.model
        final_parts: List[torch.Tensor] = []
        semantic_parts: List[torch.Tensor] = []
        instance_parts: List[torch.Tensor] = []
        presence_parts: List[torch.Tensor] = []
        for start in range(0, self.num_queries, self.prompt_batch_size):
            end = min(start + self.prompt_batch_size, self.num_queries)
            count = end - start
            language_mask = text_output["language_mask"][start:end]
            language_embeds = text_output["language_embeds"][:, start:end]

            def run_grounding(
                language_features,
                _count=count,
                _language_mask=language_mask,
                _language_embeds=language_embeds,
            ):
                backbone_out = dict(image_backbone)
                backbone_out.update(
                    language_features=language_features,
                    language_mask=_language_mask,
                    language_embeds=_language_embeds,
                )
                output = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=self._find_stage(_count),
                    find_target=None,
                    geometric_prompt=model._get_dummy_prompt(_count),
                )
                return self._reduce_grounding_output(
                    output, output_size, hard_candidates=hard_candidates
                )

            language_features = text_output["language_features"][:, start:end]
            if self.training and self.use_grounding_checkpoint:
                final, semantic, instance, presence = checkpoint(
                    run_grounding,
                    language_features,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                final, semantic, instance, presence = run_grounding(
                    language_features
                )
            final_parts.append(final)
            semantic_parts.append(semantic)
            instance_parts.append(instance)
            presence_parts.append(presence)
        return (
            torch.cat(final_parts, dim=0),
            torch.cat(semantic_parts, dim=0),
            torch.cat(instance_parts, dim=0),
            torch.cat(presence_parts, dim=0),
        )

    def _forward_prompt_scores(
        self,
        rgb_01: torch.Tensor,
        component_mode: Optional[str] = None,
        hard_candidates: bool = False,
    ) -> Dict[str, Union[torch.Tensor, dict]]:
        if rgb_01.shape[0] != 1:
            raise RuntimeError("Prompt-SAM3 v1 supports one image per GPU")
        mode = component_mode or self.prompt_component_mode
        image_backbone, spatial = self._encode_sam_image(rgb_01)
        global_feature = self._global_feature(rgb_01, spatial)
        text_output, diagnostics, _ = self._encode_dynamic_text(
            spatial, global_feature, mode
        )
        query_final, query_semantic, query_instance, query_presence = (
            self._ground_queries(
                image_backbone,
                text_output,
                output_size=tuple(rgb_01.shape[-2:]),
                hard_candidates=hard_candidates,
            )
        )
        class_final = self._aggregate_query_logits_to_classes(query_final)
        class_semantic = self._aggregate_query_logits_to_classes(query_semantic)
        class_instance = self._aggregate_query_logits_to_classes(query_instance)
        class_presence = self._aggregate_query_scores_differentiable(
            query_presence
        )
        return {
            "final": class_final,
            "semantic": class_semantic,
            "instance": class_instance,
            "presence": class_presence,
            "query_final": query_final,
            "diagnostics": diagnostics,
        }

    def _aggregate_query_scores_differentiable(
        self, query_scores: torch.Tensor
    ) -> torch.Tensor:
        if self.num_cls == self.num_queries:
            return query_scores
        result = []
        for class_index in range(self.num_cls):
            mask = self.query_idx == class_index
            result.append(query_scores[mask].amax(dim=0))
        return torch.stack(result)

    def _normalized_class_probability(
        self, class_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = class_scores.clamp_min(self.score_epsilon).log().unsqueeze(0)
        return logits, logits.softmax(dim=1)

    def loss(self, inputs, data_samples):
        rgb = self._canonical_rgb(inputs)
        output = self._forward_prompt_scores(
            rgb, component_mode="full", hard_candidates=False
        )
        logits, probability = self._normalized_class_probability(output["final"])
        target = torch.stack(
            [sample.gt_sem_seg.data.squeeze(0) for sample in data_samples]
        ).to(device=logits.device, dtype=torch.long)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
            probability = logits.softmax(dim=1)

        per_pixel_ce = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = target != self.ignore_index
        ce = per_pixel_ce[valid].mean()
        dice = _soft_dice_loss(
            probability,
            target,
            num_classes=self.num_cls,
            ignore_index=self.ignore_index,
        )
        anchor = 1.0 - output["diagnostics"]["static_dynamic_cosine"]

        diagnostics = output["diagnostics"]
        losses = {
            "loss_ce": self.loss_ce_weight * ce,
            "loss_dice": self.loss_dice_weight * dice,
            "loss_anchor": self.loss_anchor_weight * anchor,
            "metric_raw_ce": ce.detach(),
            "metric_raw_dice": dice.detach(),
            "metric_raw_anchor": anchor.detach(),
            "metric_ce_q50": per_pixel_ce[valid].detach().quantile(0.50),
            "metric_ce_q90": per_pixel_ce[valid].detach().quantile(0.90),
            "metric_ce_q99": per_pixel_ce[valid].detach().quantile(0.99),
        }
        for key in (
            "gate_shared",
            "gate_state",
            "gate_global",
            "shared_residual_norm",
            "state_residual_norm",
            "global_residual_norm",
            "attention_entropy",
            "attention_peak",
            "prototype_pair_cosine",
            "prototype_effective_rank",
            "static_dynamic_cosine",
        ):
            losses[f"metric_prompt_{key}"] = diagnostics[key].detach()
        return losses

    def _slide_prompt_scores(
        self, rgb_01: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, dict]]:
        height, width = rgb_01.shape[-2:]
        crop = int(self.slide_crop)
        stride = int(self.slide_stride)
        if crop <= 0 or (height <= crop and width <= crop):
            return self._forward_prompt_scores(
                rgb_01,
                component_mode=self.prompt_component_mode,
                hard_candidates=True,
            )
        if stride <= 0:
            stride = crop
        h_grids = max(height - crop + stride - 1, 0) // stride + 1
        w_grids = max(width - crop + stride - 1, 0) // stride + 1
        accumulators = {
            "final": rgb_01.new_zeros(self.num_cls, height, width),
            "semantic": rgb_01.new_zeros(self.num_cls, height, width),
            "instance": rgb_01.new_zeros(self.num_cls, height, width),
        }
        count = rgb_01.new_zeros(1, height, width)
        presence_sum = rgb_01.new_zeros(self.num_cls)
        last_diagnostics = None
        scalar_diagnostic_sum = {}
        views = 0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y2 = min(h_idx * stride + crop, height)
                x2 = min(w_idx * stride + crop, width)
                y1 = max(y2 - crop, 0)
                x1 = max(x2 - crop, 0)
                crop_output = self._forward_prompt_scores(
                    rgb_01[..., y1:y2, x1:x2],
                    component_mode=self.prompt_component_mode,
                    hard_candidates=True,
                )
                for key in accumulators:
                    accumulators[key][..., y1:y2, x1:x2] += crop_output[key]
                count[..., y1:y2, x1:x2] += 1
                presence_sum += crop_output["presence"]
                last_diagnostics = crop_output["diagnostics"]
                for key, value in crop_output["diagnostics"].items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        scalar_diagnostic_sum[key] = (
                            scalar_diagnostic_sum.get(key, 0.0) + value
                        )
                views += 1
        count = count.clamp_min(1.0)
        for key, value in scalar_diagnostic_sum.items():
            last_diagnostics[key] = value / max(views, 1)
        return {
            **{key: value / count for key, value in accumulators.items()},
            "presence": presence_sum / max(views, 1),
            "diagnostics": last_diagnostics,
        }

    def _should_dump_diagnostic(self, image_path: str) -> bool:
        if not self.diagnostic_dir or self.diagnostic_max_images <= 0:
            return False
        basename = os.path.basename(image_path)
        if basename in self._diagnostic_written:
            return False
        if self._diagnostic_panel is not None:
            return basename in self._diagnostic_panel
        return len(self._diagnostic_written) < self.diagnostic_max_images

    @staticmethod
    def _resize_for_dump(tensor: torch.Tensor, max_side: int) -> torch.Tensor:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        height, width = tensor.shape[-2:]
        scale = min(1.0, float(max_side) / max(height, width))
        if scale < 1.0:
            tensor = F.interpolate(
                tensor.unsqueeze(0).float(),
                size=(max(1, round(height * scale)), max(1, round(width * scale))),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor

    def _dump_diagnostic(
        self,
        image_path: str,
        output: Dict[str, Union[torch.Tensor, dict]],
        prediction: torch.Tensor,
        data_sample,
        rgb_01: torch.Tensor,
    ) -> None:
        if not self._should_dump_diagnostic(image_path):
            return
        basename = os.path.basename(image_path)
        stem = Path(basename).stem
        rank_dir = os.path.join(
            self.diagnostic_dir,
            self.prompt_component_mode,
            f"rank{int(os.environ.get('RANK', '0'))}",
        )
        os.makedirs(rank_dir, exist_ok=True)
        diagnostics = output["diagnostics"]
        arrays = {
            key: self._resize_for_dump(output[key].detach(), self.diagnostic_max_side)
            .cpu()
            .to(torch.float16)
            .numpy()
            for key in ("final", "semantic", "instance")
        }
        arrays["presence"] = output["presence"].detach().cpu().to(torch.float16).numpy()
        arrays["prediction"] = prediction.detach().cpu().to(torch.int16).numpy()
        if getattr(data_sample, "gt_sem_seg", None) is not None:
            arrays["ground_truth"] = (
                data_sample.gt_sem_seg.data.squeeze(0).detach().cpu().to(torch.int16).numpy()
            )
        attention = diagnostics.get("attention")
        if attention is not None:
            arrays["state_attention"] = attention.detach().cpu().to(torch.float16).numpy()
        for key in ("state_prototypes", "class_anchor", "dynamic_inputs", "dynamic_language"):
            value = diagnostics.get(key)
            if value is not None:
                arrays[key] = value.detach().cpu().to(torch.float16).numpy()
        np.savez_compressed(os.path.join(rank_dir, f"{stem}.npz"), **arrays)
        scalar_stats = {
            key: float(value.detach().float().mean().cpu())
            for key, value in diagnostics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        scalar_stats.update(
            image=basename,
            global_feature_source=self.global_feature_source,
            component_mode=self.prompt_component_mode,
        )
        with open(os.path.join(rank_dir, f"{stem}.json"), "w", encoding="utf-8") as handle:
            json.dump(scalar_stats, handle, indent=2, ensure_ascii=False)
        self._save_visual_panel(
            os.path.join(rank_dir, f"{stem}_panel.png"),
            rgb_01,
            prediction,
            arrays.get("ground_truth"),
        )
        self._diagnostic_written.add(basename)

    @staticmethod
    def _label_palette(num_classes: int) -> np.ndarray:
        palette = np.zeros((max(num_classes, 1), 3), dtype=np.uint8)
        for class_id in range(1, num_classes):
            # Deterministic high-contrast bit palette used only for diagnostics.
            value = class_id
            for bit in range(8):
                palette[class_id, 0] |= ((value >> 0) & 1) << (7 - bit)
                palette[class_id, 1] |= ((value >> 1) & 1) << (7 - bit)
                palette[class_id, 2] |= ((value >> 2) & 1) << (7 - bit)
                value >>= 3
        return palette

    def _save_visual_panel(
        self,
        path: str,
        rgb_01: torch.Tensor,
        prediction: torch.Tensor,
        ground_truth: Optional[np.ndarray],
    ) -> None:
        original_size = tuple(prediction.shape[-2:])
        scale = min(
            1.0,
            float(self.diagnostic_max_side) / max(original_size),
        )
        target_size = (
            max(1, round(original_size[0] * scale)),
            max(1, round(original_size[1] * scale)),
        )
        if target_size != original_size:
            prediction = F.interpolate(
                prediction.view(1, 1, *original_size).float(),
                size=target_size,
                mode="nearest",
            ).view(*target_size).long()
        rgb = rgb_01
        if rgb.shape[-2:] != target_size:
            rgb = F.interpolate(
                rgb,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        rgb_np = (
            rgb[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            * 255.0
        ).round().astype(np.uint8)
        pred_np = prediction.detach().cpu().numpy().astype(np.int64)
        palette = self._label_palette(self.num_cls)
        pred_color = palette[np.clip(pred_np, 0, self.num_cls - 1)]
        overlay = np.rint(0.55 * rgb_np + 0.45 * pred_color).astype(np.uint8)
        if ground_truth is None:
            gt_color = np.zeros_like(pred_color)
            error = overlay
        else:
            gt = ground_truth.astype(np.int64)
            if gt.shape != pred_np.shape:
                gt = np.asarray(
                    Image.fromarray(gt.astype(np.int32), mode="I").resize(
                        (pred_np.shape[1], pred_np.shape[0]),
                        resample=getattr(Image, "Resampling", Image).NEAREST,
                    )
                )
            safe_gt = np.clip(gt, 0, self.num_cls - 1)
            gt_color = palette[safe_gt]
            error = rgb_np.copy()
            mismatch = (gt != self.ignore_index) & (pred_np != gt)
            error[mismatch] = np.rint(
                0.25 * error[mismatch] + 0.75 * np.array([255, 0, 0])
            ).astype(np.uint8)
        height, width = pred_np.shape
        panel = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        panel[:height, :width] = rgb_np
        panel[:height, width:] = gt_color
        panel[height:, :width] = overlay
        panel[height:, width:] = error
        Image.fromarray(panel).save(path)

    @torch.no_grad()
    def predict(self, inputs, data_samples):
        rgb = self._canonical_rgb(inputs)
        for index, data_sample in enumerate(data_samples):
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                output = self._slide_prompt_scores(rgb[index : index + 1])
            scores = output["final"]
            ori_shape = tuple(data_sample.metainfo.get("ori_shape", scores.shape[-2:]))
            if scores.shape[-2:] != ori_shape:
                scores = F.interpolate(
                    scores.unsqueeze(0),
                    size=ori_shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            max_scores, prediction = scores.max(dim=0)
            prediction[max_scores < self.prob_thd] = self.bg_idx
            image_path = data_sample.metainfo.get("img_path", f"sample_{index}")
            self._dump_diagnostic(
                image_path,
                output,
                prediction,
                data_sample,
                rgb[index : index + 1],
            )
            data_sample.set_data(
                {
                    "seg_logits": PixelData(data=scores),
                    "pred_sem_seg": PixelData(data=prediction.unsqueeze(0)),
                    "prompt_anchor": (
                        1.0
                        - output["diagnostics"]["static_dynamic_cosine"].detach()
                    ),
                }
            )
        return data_samples

    def _forward(self, inputs, data_samples=None):
        rgb = self._canonical_rgb(inputs)
        return self._forward_prompt_scores(
            rgb,
            component_mode=self.prompt_component_mode,
            hard_candidates=not self.training,
        )["final"].unsqueeze(0)
