"""Validation diagnostics for Prompt-SAM3 v1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class PromptValidationMetric(BaseMetric):
    """Report calibration and soft-vs-hard behavior beside standard mIoU."""

    default_prefix = "prompt"

    def __init__(self, ignore_index=255, num_bins=15, collect_device="cpu", prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = int(ignore_index)
        self.num_bins = int(num_bins)

    @staticmethod
    def _pixel_tensor(sample: Mapping, key: str) -> torch.Tensor:
        """Read PixelData after MMEngine Evaluator converts it to a dict."""
        if not isinstance(sample, Mapping):
            raise TypeError(
                "PromptValidationMetric expects evaluator samples as mappings, "
                f"but received {type(sample).__name__}"
            )
        if key not in sample:
            raise KeyError(
                f"PromptValidationMetric sample is missing {key!r}; "
                f"available keys: {sorted(sample)}"
            )
        pixel_data = sample[key]
        if isinstance(pixel_data, Mapping):
            tensor = pixel_data.get("data")
        else:
            tensor = getattr(pixel_data, "data", None)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"PromptValidationMetric expected {key!r}['data'] to be a "
                f"Tensor, but received {type(tensor).__name__}"
            )
        return tensor

    def process(self, data_batch: dict, data_samples: list) -> None:
        for sample in data_samples:
            score = self._pixel_tensor(sample, "seg_logits").detach().float().cpu()
            target = (
                self._pixel_tensor(sample, "gt_sem_seg")
                .squeeze(0)
                .detach()
                .long()
                .cpu()
            )
            prediction = (
                self._pixel_tensor(sample, "pred_sem_seg")
                .squeeze(0)
                .detach()
                .long()
                .cpu()
            )
            if score.ndim != 3:
                raise ValueError(
                    "seg_logits must have shape [C,H,W], got "
                    f"{tuple(score.shape)}"
                )
            if target.shape != score.shape[-2:] or prediction.shape != target.shape:
                raise ValueError(
                    "Prompt validation spatial shapes disagree: "
                    f"score={tuple(score.shape)}, target={tuple(target.shape)}, "
                    f"prediction={tuple(prediction.shape)}"
                )
            valid = target != self.ignore_index
            if not valid.any():
                raise ValueError(
                    "Prompt validation sample contains no non-ignored pixels"
                )
            valid_target = target[valid]
            if valid_target.min() < 0 or valid_target.max() >= score.shape[0]:
                raise ValueError(
                    "Prompt validation target ids fall outside the score "
                    f"channels: min={int(valid_target.min())}, "
                    f"max={int(valid_target.max())}, classes={score.shape[0]}"
                )
            probability = score.clamp_min(1e-6)
            probability = probability / probability.sum(dim=0, keepdim=True).clamp_min(1e-6)
            confidence, soft_prediction = probability.max(dim=0)
            safe_target = target.masked_fill(~valid, 0).clamp_min(0)
            nll = -probability.gather(0, safe_target.unsqueeze(0)).squeeze(0).clamp_min(1e-6).log()

            bin_count = torch.zeros(self.num_bins, dtype=torch.float64)
            bin_confidence = torch.zeros(self.num_bins, dtype=torch.float64)
            bin_correct = torch.zeros(self.num_bins, dtype=torch.float64)
            valid_confidence = confidence[valid]
            valid_correct = (soft_prediction[valid] == target[valid]).to(torch.float64)
            bin_index = torch.clamp(
                (valid_confidence * self.num_bins).long(), max=self.num_bins - 1
            )
            for index in range(self.num_bins):
                selected = bin_index == index
                bin_count[index] = selected.sum()
                if selected.any():
                    bin_confidence[index] = valid_confidence[selected].double().sum()
                    bin_correct[index] = valid_correct[selected].sum()

            one_hot = F.one_hot(
                safe_target, num_classes=score.shape[0]
            ).permute(2, 0, 1).to(torch.float64)
            valid_float = valid.unsqueeze(0).to(torch.float64)
            probability64 = probability.to(torch.float64) * valid_float
            one_hot = one_hot * valid_float
            dice_intersection = (probability64 * one_hot).sum(dim=(1, 2))
            dice_denominator = probability64.sum(dim=(1, 2)) + one_hot.sum(dim=(1, 2))
            prompt_anchor = sample.get("prompt_anchor")
            prompt_anchor_value = (
                0.0
                if prompt_anchor is None
                else float(prompt_anchor.detach().float().mean().cpu())
            )

            self.results.append(
                dict(
                    nll_sum=float(nll[valid].double().sum()),
                    valid_count=int(valid.sum()),
                    soft_correct=int((soft_prediction[valid] == target[valid]).sum()),
                    hard_correct=int((prediction[valid] == target[valid]).sum()),
                    high_conf_errors=int(
                        ((confidence[valid] >= 0.9) & (soft_prediction[valid] != target[valid])).sum()
                    ),
                    high_conf_count=int((confidence[valid] >= 0.9).sum()),
                    bin_count=bin_count.numpy(),
                    bin_confidence=bin_confidence.numpy(),
                    bin_correct=bin_correct.numpy(),
                    dice_intersection=dice_intersection.numpy(),
                    dice_denominator=dice_denominator.numpy(),
                    prompt_anchor=prompt_anchor_value,
                )
            )

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        valid_count = max(sum(item["valid_count"] for item in results), 1)
        nll = sum(item["nll_sum"] for item in results) / valid_count
        soft_acc = sum(item["soft_correct"] for item in results) / valid_count
        hard_acc = sum(item["hard_correct"] for item in results) / valid_count
        high_count = sum(item["high_conf_count"] for item in results)
        high_errors = sum(item["high_conf_errors"] for item in results)
        bin_count = np.sum([item["bin_count"] for item in results], axis=0)
        bin_confidence = np.sum([item["bin_confidence"] for item in results], axis=0)
        bin_correct = np.sum([item["bin_correct"] for item in results], axis=0)
        nonempty = bin_count > 0
        average_confidence = np.zeros_like(bin_count, dtype=np.float64)
        average_accuracy = np.zeros_like(bin_count, dtype=np.float64)
        average_confidence[nonempty] = bin_confidence[nonempty] / bin_count[nonempty]
        average_accuracy[nonempty] = bin_correct[nonempty] / bin_count[nonempty]
        ece = float(
            np.sum(np.abs(average_confidence - average_accuracy) * bin_count)
            / max(float(bin_count.sum()), 1.0)
        )
        dice_intersection = np.sum(
            [item["dice_intersection"] for item in results], axis=0
        )
        dice_denominator = np.sum(
            [item["dice_denominator"] for item in results], axis=0
        )
        dice_loss = float(
            1.0
            - np.mean(
                (2.0 * dice_intersection + 1.0)
                / (dice_denominator + 1.0)
            )
        )
        anchor = float(np.mean([item["prompt_anchor"] for item in results]))
        return dict(
            NLL=float(nll),
            DiceLoss=dice_loss,
            AnchorLoss=anchor,
            validation_objective=float(nll + dice_loss + 0.05 * anchor),
            ECE=ece,
            soft_pixel_accuracy=float(soft_acc),
            hard_pixel_accuracy=float(hard_acc),
            soft_minus_hard_accuracy=float(soft_acc - hard_acc),
            high_confidence_error_rate=float(high_errors / max(high_count, 1)),
        )
