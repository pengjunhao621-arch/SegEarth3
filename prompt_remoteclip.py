"""Frozen RemoteCLIP global-feature adapter used by Prompt-SAM3 v1."""

from __future__ import annotations

import os
import sys
from typing import Union

import torch
import torch.nn.functional as F


class FrozenRemoteCLIPGlobalEncoder:
    """Load the SCORE-bundled OpenCLIP implementation without Detectron2.

    This is deliberately a plain Python object rather than an ``nn.Module``.
    The frozen RemoteCLIP weights are therefore not registered in the MMSeg
    segmentor state dict and are never duplicated in experiment checkpoints.
    """

    def __init__(
        self,
        checkpoint_path: str,
        source_root: str = "SCORE-main/open_clip_training/src",
        model_name: str = "ViT-L-14",
        image_size: int = 224,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "RemoteCLIP checkpoint was not found at "
                f"{checkpoint_path!r}. Expected server location: "
                "weights/remoteclip/RemoteCLIP-ViT-L-14.pt"
            )
        source_root = os.path.abspath(source_root)
        if not os.path.isdir(os.path.join(source_root, "open_clip")):
            raise FileNotFoundError(
                "Bundled OpenCLIP source was not found under "
                f"{source_root!r}; copy SCORE-main with the experiment code."
            )
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from open_clip import create_model

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.model = create_model(
            model_name,
            pretrained=checkpoint_path,
            precision="fp32",
            device=self.device,
        ).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(
            (0.48145466, 0.4578275, 0.40821073),
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            (0.26862954, 0.26130258, 0.27577711),
            device=self.device,
        ).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, rgb_01: torch.Tensor) -> torch.Tensor:
        if rgb_01.ndim != 4 or rgb_01.shape[1] != 3:
            raise ValueError("RemoteCLIP input must have shape [B, 3, H, W]")
        image = F.interpolate(
            rgb_01.to(device=self.device, dtype=torch.float32),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        image = (image.clamp(0.0, 1.0) - self.mean) / self.std
        feature = self.model.encode_image(image, normalize=True)
        return feature.float().detach()
