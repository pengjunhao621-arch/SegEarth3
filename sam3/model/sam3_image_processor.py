# # Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# from typing import Dict, List

# import numpy as np
# import PIL
# import torch

# from sam3.model import box_ops

# from sam3.model.data_misc import FindStage, interpolate
# from torchvision.transforms import v2


# class Sam3Processor:
#     """ """

#     def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
#         self.model = model
#         self.resolution = resolution
#         self.device = device
#         self.transform = v2.Compose(
#             [
#                 v2.ToDtype(torch.uint8, scale=True),
#                 v2.Resize(size=(resolution, resolution)),
#                 v2.ToDtype(torch.float32, scale=True),
#                 v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#             ]
#         )
#         self.confidence_threshold = confidence_threshold

#         self.find_stage = FindStage(
#             img_ids=torch.tensor([0], device=device, dtype=torch.long),
#             text_ids=torch.tensor([0], device=device, dtype=torch.long),
#             input_boxes=None,
#             input_boxes_mask=None,
#             input_boxes_label=None,
#             input_points=None,
#             input_points_mask=None,
#         )

#     @torch.inference_mode()
#     def set_image(self, image, state=None):
#         """Sets the image on which we want to do predictions."""
#         if state is None:
#             state = {}

#         if isinstance(image, PIL.Image.Image):
#             width, height = image.size
#         elif isinstance(image, (torch.Tensor, np.ndarray)):
#             height, width = image.shape[-2:]
#         else:
#             raise ValueError("Image must be a PIL image or a tensor")

#         image = v2.functional.to_image(image).to(self.device)
#         image = self.transform(image).unsqueeze(0)

#         state["original_height"] = height
#         state["original_width"] = width
#         state["backbone_out"] = self.model.backbone.forward_image(image)
#         inst_interactivity_en = self.model.inst_interactive_predictor is not None # False
#         if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
#             sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
#             sam2_backbone_out["backbone_fpn"][0] = (
#                 self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
#                     sam2_backbone_out["backbone_fpn"][0]
#                 )
#             )
#             sam2_backbone_out["backbone_fpn"][1] = (
#                 self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
#                     sam2_backbone_out["backbone_fpn"][1]
#                 )
#             )
#         return state

#     @torch.inference_mode()
#     def set_image_batch(self, images: List[np.ndarray], state=None):
#         """Sets the image batch on which we want to do predictions."""
#         if state is None:
#             state = {}

#         if not isinstance(images, list):
#             raise ValueError("Images must be a list of PIL images or tensors")
#         assert len(images) > 0, "Images list must not be empty"
#         assert isinstance(
#             images[0], PIL.Image.Image
#         ), "Images must be a list of PIL images"

#         state["original_heights"] = [image.height for image in images]
#         state["original_widths"] = [image.width for image in images]

#         images = [
#             self.transform(v2.functional.to_image(image).to(self.device))
#             for image in images
#         ]
#         images = torch.stack(images, dim=0)
#         state["backbone_out"] = self.model.backbone.forward_image(images)
#         inst_interactivity_en = self.model.inst_interactive_predictor is not None
#         if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
#             sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
#             sam2_backbone_out["backbone_fpn"][0] = (
#                 self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
#                     sam2_backbone_out["backbone_fpn"][0]
#                 )
#             )
#             sam2_backbone_out["backbone_fpn"][1] = (
#                 self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
#                     sam2_backbone_out["backbone_fpn"][1]
#                 )
#             )
#         return state

#     @torch.inference_mode()
#     def set_text_prompt(self, prompt: str, state: Dict):
#         """Sets the text prompt and run the inference"""

#         if "backbone_out" not in state:
#             raise ValueError("You must call set_image before set_text_prompt")

#         text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
#         # text_outputs['language_features']: [32, 1, 256]
#         # text_outputs['language_mask']: [1, 32]
#         # text_outputs['language_embeds']: [32, 1, 1024]

#         # will erase the previous text prompt if any
#         state["backbone_out"].update(text_outputs)
#         if "geometric_prompt" not in state:
#             state["geometric_prompt"] = self.model._get_dummy_prompt()

#         return self._forward_grounding(state)

#     @torch.inference_mode()
#     def add_geometric_prompt(self, box: List, label: bool, state: Dict):
#         """Adds a box prompt and run the inference.
#         The image needs to be set, but not necessarily the text prompt.
#         The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
#         The label is True for a positive box, False for a negative box.
#         """
#         if "backbone_out" not in state:
#             raise ValueError("You must call set_image before set_text_prompt")

#         if "language_features" not in state["backbone_out"]:
#             # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
#             dummy_text_outputs = self.model.backbone.forward_text(
#                 ["visual"], device=self.device
#             )
#             state["backbone_out"].update(dummy_text_outputs)

#         if "geometric_prompt" not in state:
#             state["geometric_prompt"] = self.model._get_dummy_prompt()

#         # adding a batch and sequence dimension
#         boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
#         labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
#         state["geometric_prompt"].append_boxes(boxes, labels)

#         return self._forward_grounding(state)

#     def reset_all_prompts(self, state: Dict):
#         """Removes all the prompts and results"""
#         if "backbone_out" in state:
#             backbone_keys_to_del = [
#                 "language_features",
#                 "language_mask",
#                 "language_embeds",
#             ]
#             for key in backbone_keys_to_del:
#                 if key in state["backbone_out"]:
#                     del state["backbone_out"][key]

#         keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
#         for key in keys_to_del:
#             if key in state:
#                 del state[key]

#     @torch.inference_mode()
#     def set_confidence_threshold(self, threshold: float, state=None):
#         """Sets the confidence threshold for the masks"""
#         self.confidence_threshold = threshold
#         if state is not None and "boxes" in state:
#             # we need to filter the boxes again
#             # In principle we could do this more efficiently since we would only need
#             # to rerun the heads. But this is simpler and not too inefficient
#             return self._forward_grounding(state)
#         return state

#     @torch.inference_mode()
#     def _forward_grounding(self, state: Dict):
#         outputs = self.model.forward_grounding(
#             backbone_out=state["backbone_out"],
#             find_input=self.find_stage,
#             geometric_prompt=state["geometric_prompt"],
#             find_target=None,
#         )

#         out_bbox = outputs["pred_boxes"]
#         out_logits = outputs["pred_logits"]
#         out_masks = outputs["pred_masks"]
#         out_probs = out_logits.sigmoid()
#         presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
#         out_probs = (out_probs * presence_score).squeeze(-1)

#         keep = out_probs > self.confidence_threshold
#         out_probs = out_probs[keep]
#         out_masks = out_masks[keep]
#         out_bbox = out_bbox[keep]

#         # convert to [x0, y0, x1, y1] format
#         boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

#         img_h = state["original_height"]
#         img_w = state["original_width"]
#         scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
#         boxes = boxes * scale_fct[None, :]

#         out_masks = interpolate(
#             out_masks.unsqueeze(1),
#             (img_h, img_w),
#             mode="bilinear",
#             align_corners=False,
#         ).sigmoid()

#         out_semantic_masks = interpolate(
#             outputs["semantic_seg"],
#             (img_h, img_w),
#             mode="bilinear",
#             align_corners=False,
#         ).sigmoid()

#         state["masks_logits"] = out_masks
#         state["masks"] = out_masks > 0.5
#         state["boxes"] = boxes
#         state["scores"] = out_probs
#         state["semantic_mask_logits"] = out_semantic_masks # for SS
#         state["presence_score"] = presence_score.squeeze().squeeze()
#         state["object_score"] = out_probs
#         return state

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from typing import Dict, List

import numpy as np
import PIL
import torch

from sam3.model import box_ops

from sam3.model.data_misc import FindStage, interpolate
# 替换为兼容旧版的 transforms
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        
        # 兼容 PyTorch 1.13.1 的等效处理管线：不再使用 v2
        # 张量的类型转换和归一化（0-1）将在下方的 _convert_to_tensor 中完成
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(resolution, resolution)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _convert_to_tensor(self, image):
        """兼容替代 v2.functional.to_image 和 ToDtype 的辅助方法"""
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            # to_tensor 会自动将其转换为 float32 CHW 张量并缩放到 [0.0, 1.0]
            return F.to_tensor(image)
        elif isinstance(image, torch.Tensor):
            img_t = image.clone().detach()
            if img_t.dtype == torch.uint8:
                img_t = img_t.float() / 255.0
            elif img_t.dtype != torch.float32:
                img_t = img_t.float()
            return img_t
        else:
            raise ValueError("Image must be a PIL image, numpy array, or a tensor")

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        # 使用替代的等效张量转换逻辑
        image_tensor = self._convert_to_tensor(image).to(self.device)
        image_tensor = self.transform(image_tensor).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image_tensor)
        # ---------- 新增：提取并缓存视觉编码器最后一层特征 ----------
        # 从 backbone_out 中自动查找空间分辨率最小的特征图
        feat_candidate = None
        min_resolution = float('inf')
        for key, value in state["backbone_out"].items():
            if isinstance(value, torch.Tensor) and value.ndim >= 4:  # [B, C, H, W]
                # 假设 B=1，取 H*W
                _, _, h, w = value.shape
                resolution = h * w
                if resolution < min_resolution:
                    min_resolution = resolution
                    feat_candidate = value
        if feat_candidate is not None:
            state["vision_features"] = feat_candidate.squeeze(0)  # [C, H, W]
        else:
            # 若没有 4D 张量，尝试保留整个 backbone_out 中最小的 3D 特征
            for key, value in state["backbone_out"].items():
                if isinstance(value, torch.Tensor) and value.ndim == 3:
                    state["vision_features"] = value  # 希望是 [C, H, W]
                    break
        # ------------------------------------------------------------
        inst_interactivity_en = self.model.inst_interactive_predictor is not None # False
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(
            images[0], PIL.Image.Image
        ), "Images must be a list of PIL images"

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        # 同样使用替代的等效张量转换逻辑处理 Batch
        processed_images = [
            self.transform(self._convert_to_tensor(image).to(self.device))
            for image in images
        ]
        processed_images = torch.stack(processed_images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(processed_images)
        
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # text_outputs['language_features']: [32, 1, 256]
        # text_outputs['language_mask']: [1, 32]
        # text_outputs['language_embeds']: [32, 1, 1024]

        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = [
            "geometric_prompt",
            "boxes",
            "masks",
            "masks_logits",
            "scores",
            "raw_masks_logits_lowres",
            "raw_object_score",
            "raw_object_score_presence",
            "raw_keep_mask",
            "encoder_hidden_states",
            "encoder_prompt_after",
            "encoder_prompt_mask",
            "encoder_vis_feat_sizes",
            "encoder_spatial_shapes",
            "encoder_level_start_index",
            "semantic_mask_raw_logits",
        ]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        object_score_raw = out_logits.sigmoid().squeeze(-1)
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        object_score_presence = object_score_raw * presence_score.squeeze(-1)

        raw_object_score = object_score_raw.reshape(-1)
        raw_object_score_presence = object_score_presence.reshape(-1)
        raw_out_masks = out_masks.reshape(-1, *out_masks.shape[-2:])
        raw_out_bbox = out_bbox.reshape(-1, out_bbox.shape[-1])

        keep = (object_score_presence > self.confidence_threshold).reshape(-1)
        object_score_raw = raw_object_score[keep]
        object_score_presence = raw_object_score_presence[keep]
        out_masks = raw_out_masks[keep]
        out_bbox = raw_out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        out_semantic_raw = interpolate(
            outputs["semantic_seg"],
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )
        out_semantic_masks = out_semantic_raw.sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = object_score_presence
        state["raw_masks_logits_lowres"] = raw_out_masks
        state["raw_object_score"] = raw_object_score
        state["raw_object_score_presence"] = raw_object_score_presence
        state["raw_keep_mask"] = keep
        state["semantic_mask_logits"] = out_semantic_masks # for SS
        # Preserve the pre-sigmoid dense semantic evidence for calibrated
        # uncertainty diagnostics and test-time prompt adaptation.  This does
        # not alter any official prediction or filtering behavior.
        state["semantic_mask_raw_logits"] = out_semantic_raw
        state["presence_score"] = presence_score.squeeze().squeeze()
        state["object_score_raw"] = object_score_raw
        state["object_score_presence"] = object_score_presence
        state["object_score"] = object_score_presence
        encoder_out = outputs.get("prev_encoder_out", {}).get("encoder_out", {})
        state["encoder_hidden_states"] = outputs.get("encoder_hidden_states")
        state["encoder_prompt_after"] = encoder_out.get("prompt_after_enc")
        state["encoder_prompt_mask"] = encoder_out.get("prompt_mask")
        state["encoder_vis_feat_sizes"] = encoder_out.get("vis_feat_sizes")
        state["encoder_spatial_shapes"] = encoder_out.get("spatial_shapes")
        state["encoder_level_start_index"] = encoder_out.get("level_start_index")
        return state
