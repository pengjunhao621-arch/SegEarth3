"""SegEarth-OV3 semantic segmentation baseline and active text-role screen.

The official SAM3 inference path is the default.  The only optional research
path retained here is ``role_functional_text_screen_v1``; it returns the exact
baseline prediction while recording alternative role-specific text variants.
"""

import os

import torch
from torch import nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from PIL import Image

from role_functional_text_screen import RoleFunctionalTextScreenMixin
from sam3 import build_sam3_image_model
from sam3.model.data_misc import interpolate as sam3_interpolate
from sam3.model.sam3_image_processor import Sam3Processor


@MODELS.register_module()
class SegEarthOV3Segmentation(RoleFunctionalTextScreenMixin, BaseSegmentor):
    """Frozen SAM3 segmentor with one config-gated diagnostic extension."""

    def __init__(
            self,
            classname_path,
            device=torch.device('cuda'),
            prob_thd=0.0,
            bg_idx=0,
            slide_stride=0,
            slide_crop=0,
            confidence_threshold=0.5,
            use_sem_seg=True,
            use_presence_score=True,
            use_transformer_decoder=True,
            instance_score_type='presence',
            use_role_prompt_tta=False,
            dump_role_prompt_tta_stats=False,
            role_prompt_tta_protocol='role_functional_text_screen_v1',
            role_prompt_tta_dataset_name=None,
            role_prompt_tta_prompt_bank=None,
            role_prompt_tta_stats_path=None,
            role_prompt_tta_artifact_dir=None,
            role_prompt_tta_primary_variant='baseline',
            role_prompt_tta_strict_integrity=True,
            role_prompt_tta_integrity_tolerance=1e-5,
            role_prompt_tta_save_npz=False,
            role_prompt_tta_artifact_max_side=128,
            role_prompt_tta_max_saved_images=8,
            role_prompt_tta_pi_diagnosis=False,
            role_prompt_tta_pi_presence_slot=0,
            role_prompt_tta_pi_instance_slot=0,
            **kwargs):
        super().__init__()
        self.device = _resolve_inference_device(device)

        # SAM3 is deliberately kept outside nn.Module registration.  Several
        # SAM3 buffers are complex tensors that old NCCL cannot broadcast.
        self._ddp_dummy_param = nn.Parameter(
            torch.zeros(1, device=self.device), requires_grad=True)
        sam3_model = build_sam3_image_model(
            bpe_path='./sam3/assets/bpe_simple_vocab_16e6.txt.gz',
            checkpoint_path='weights/sam3/sam3.pt',
            device=('cuda' if self.device.type == 'cuda'
                    else str(self.device)),
        ).to(self.device).eval()
        self.processor = Sam3Processor(
            sam3_model,
            confidence_threshold=confidence_threshold,
            device=self.device,
        )

        self.query_words, query_idx = get_cls_idx(classname_path)
        self.num_cls = max(query_idx) + 1
        self.num_queries = len(query_idx)
        self.query_idx = torch.tensor(
            query_idx, dtype=torch.int64, device=self.device)
        self.class_names = _build_class_names(
            self.query_words, query_idx, self.num_cls)

        self.prob_thd = float(prob_thd)
        self.bg_idx = int(bg_idx)
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = float(confidence_threshold)
        self.use_sem_seg = bool(use_sem_seg)
        self.use_presence_score = bool(use_presence_score)
        self.use_transformer_decoder = bool(use_transformer_decoder)
        self.instance_score_type = str(instance_score_type)
        if self.instance_score_type not in ('presence', 'raw'):
            raise ValueError(
                "instance_score_type must be 'presence' or 'raw', "
                f'got {self.instance_score_type!r}.')

        # Only the currently selected role-functional screen is exposed by
        # project configs.  Legacy arguments are intentionally not forwarded.
        self._rpt_initialize(
            use_role_prompt_tta=use_role_prompt_tta,
            dump_role_prompt_tta_stats=dump_role_prompt_tta_stats,
            role_prompt_tta_protocol=role_prompt_tta_protocol,
            role_prompt_tta_dataset_name=role_prompt_tta_dataset_name,
            role_prompt_tta_prompt_bank=role_prompt_tta_prompt_bank,
            role_prompt_tta_stats_path=role_prompt_tta_stats_path,
            role_prompt_tta_artifact_dir=role_prompt_tta_artifact_dir,
            role_prompt_tta_primary_variant=role_prompt_tta_primary_variant,
            role_prompt_tta_strict_integrity=role_prompt_tta_strict_integrity,
            role_prompt_tta_integrity_tolerance=(
                role_prompt_tta_integrity_tolerance),
            role_prompt_tta_save_npz=role_prompt_tta_save_npz,
            role_prompt_tta_artifact_max_side=(
                role_prompt_tta_artifact_max_side),
            role_prompt_tta_max_saved_images=(
                role_prompt_tta_max_saved_images),
            role_prompt_tta_pi_diagnosis=(
                role_prompt_tta_pi_diagnosis),
            role_prompt_tta_pi_presence_slot=(
                role_prompt_tta_pi_presence_slot),
            role_prompt_tta_pi_instance_slot=(
                role_prompt_tta_pi_instance_slot),
        )

    def _get_instance_score(self, state, instance_index):
        if self.instance_score_type == 'raw':
            return state['object_score_raw'][instance_index]
        return state['object_score_presence'][instance_index]

    def _aggregate_query_logits_to_classes(self, query_logits):
        if self.num_cls == self.num_queries:
            return query_logits
        class_logits = []
        query_idx = self.query_idx.to(query_logits.device)
        for class_index in range(self.num_cls):
            class_logits.append(
                query_logits[query_idx == class_index].max(dim=0)[0])
        return torch.stack(class_logits, dim=0)

    def _inference_single_view(
            self, image, return_components=False, view_id=None,
            crop_box=None):
        if self._uses_role_prompt_tta():
            return self._rpt_infer_single_view(
                image,
                return_components=return_components,
                view_id=view_id,
                crop_box=crop_box,
            )
        return self._inference_single_view_native(
            image, return_components=return_components)

    def _inference_single_view_native(self, image, return_components=False):
        """Run the protected SegEarth-OV3/SAM3 baseline on one view."""
        width, height = image.size
        output_shape = (height, width)
        final_all = torch.zeros(
            (self.num_queries, height, width), device=self.device)
        semantic_all = (
            torch.zeros_like(final_all) if return_components else None)
        instance_all = (
            torch.zeros_like(final_all) if return_components else None)

        with torch.no_grad(), self._rpt_autocast_context():
            state = self.processor.set_image(image)
            for query_index, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(state)
                self.processor.set_text_prompt(query_word, state)
                instance = torch.zeros(
                    output_shape, device=self.device, dtype=torch.float32)

                if self.use_transformer_decoder:
                    masks = state['masks_logits']
                    for instance_index in range(int(masks.shape[0])):
                        mask = masks[instance_index].squeeze()
                        if mask.shape != output_shape:
                            mask = sam3_interpolate(
                                mask.reshape(1, 1, *mask.shape),
                                size=output_shape,
                                mode='bilinear',
                                align_corners=False,
                            ).squeeze()
                        score = self._get_instance_score(
                            state, instance_index)
                        instance = torch.maximum(
                            instance, mask.float() * score.float())

                semantic = torch.zeros_like(instance)
                if self.use_sem_seg:
                    semantic = state['semantic_mask_logits'].squeeze()
                    if semantic.shape != output_shape:
                        semantic = sam3_interpolate(
                            semantic.reshape(1, 1, *semantic.shape),
                            size=output_shape,
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze()
                    semantic = semantic.float()

                final = torch.maximum(semantic, instance)
                if self.use_presence_score:
                    final = final * state['presence_score'].float()
                final_all[query_index] = final
                if return_components:
                    semantic_all[query_index] = semantic
                    instance_all[query_index] = instance

        if not return_components:
            return final_all
        return final_all, {
            'semantic_logits': semantic_all,
            'instance_logits': instance_all,
        }

    def slide_inference(
            self, image, stride, crop_size, return_components=False):
        """Average overlapping SAM3 crop predictions."""
        image_width, image_height = image.size
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        height_stride, width_stride = stride
        height_crop, width_crop = crop_size

        predictions = torch.zeros(
            (self.num_queries, image_height, image_width),
            device=self.device)
        counts = torch.zeros(
            (1, image_height, image_width), device=self.device)
        semantic_predictions = (
            torch.zeros_like(predictions) if return_components else None)
        instance_predictions = (
            torch.zeros_like(predictions) if return_components else None)
        role_predictions = (
            {
                name: torch.zeros(
                    (self.num_cls, image_height, image_width),
                    dtype=torch.float32,
                    device='cpu')
                for name in self._rpt_variant_names()
            }
            if return_components and self._uses_role_prompt_tta()
            else None)
        role_view_stats = [] if role_predictions is not None else None
        pi_predictions = (
            {
                name: torch.zeros(
                    (self.num_cls, image_height, image_width),
                    dtype=torch.float32, device='cpu')
                for name in self._rpt_pi_variant_names()
            }
            if role_predictions is not None
            and self._rpt_uses_pi_diagnosis()
            else None)
        pi_mechanism_maps = (
            {
                name: torch.zeros(
                    (self.num_cls, image_height, image_width),
                    dtype=torch.float32, device='cpu')
                for name in self._rpt_pi_mechanism_map_names()
            }
            if pi_predictions is not None else None)

        height_grids = (
            max(image_height - height_crop + height_stride - 1, 0)
            // height_stride + 1)
        width_grids = (
            max(image_width - width_crop + width_stride - 1, 0)
            // width_stride + 1)
        for height_index in range(height_grids):
            for width_index in range(width_grids):
                y1 = height_index * height_stride
                x1 = width_index * width_stride
                y2 = min(y1 + height_crop, image_height)
                x2 = min(x1 + width_crop, image_width)
                y1 = max(y2 - height_crop, 0)
                x1 = max(x2 - width_crop, 0)
                crop = image.crop((x1, y1, x2, y2))
                if return_components:
                    crop_logits, crop_components = self._inference_single_view(
                        crop,
                        return_components=True,
                        view_id=f'crop_{height_index}_{width_index}',
                        crop_box=[x1, y1, x2, y2],
                    )
                else:
                    crop_logits = self._inference_single_view(crop)
                    crop_components = None
                predictions[:, y1:y2, x1:x2] += crop_logits
                counts[:, y1:y2, x1:x2] += 1
                if return_components:
                    semantic_predictions[:, y1:y2, x1:x2] += (
                        crop_components['semantic_logits'])
                    instance_predictions[:, y1:y2, x1:x2] += (
                        crop_components['instance_logits'])
                    if role_predictions is not None:
                        for name, value in crop_components[
                                'role_prompt_variant_class_logits'].items():
                            role_predictions[name][:, y1:y2, x1:x2] += value
                        role_view_stats.extend(
                            crop_components['role_prompt_view_stats'])
                        if pi_predictions is not None:
                            for name, value in crop_components[
                                    'role_prompt_pi_variant_class_logits'
                            ].items():
                                pi_predictions[name][:, y1:y2, x1:x2] += value
                            for name, value in crop_components[
                                    'role_prompt_pi_mechanism_class_maps'
                            ].items():
                                pi_mechanism_maps[name][
                                    :, y1:y2, x1:x2] += value

        if torch.any(counts == 0):
            raise RuntimeError('Sparse sliding-window coverage.')
        predictions = predictions / counts
        if not return_components:
            return predictions

        semantic_predictions = semantic_predictions / counts
        instance_predictions = instance_predictions / counts
        components = {
            'semantic_logits': semantic_predictions,
            'instance_logits': instance_predictions,
        }
        if role_predictions is not None:
            cpu_counts = counts.float().cpu()
            for name in role_predictions:
                role_predictions[name] = role_predictions[name] / cpu_counts
            # Preserve the official alias order: crop-average each query, then
            # class-max.  Role variants remain differentials around no update.
            exact_baseline = self._rpt_aggregate_query_logits_to_classes(
                predictions.detach().float().cpu())
            crop_anchor = role_predictions['combo_p0_s0_i0'].clone()
            for name in self._rpt_variant_names():
                if (name == 'baseline'
                        or name.startswith('semantic_head_')
                        or name.startswith('instance_head_')):
                    continue
                role_predictions[name] = (
                    exact_baseline + role_predictions[name] - crop_anchor
                ).clamp(0.0, 1.0)
            role_predictions['baseline'] = exact_baseline
            role_predictions['combo_p0_s0_i0'] = exact_baseline.clone()
            components['role_prompt_variant_class_logits'] = role_predictions
            components['role_prompt_view_stats'] = role_view_stats
            if pi_predictions is not None:
                for name in pi_predictions:
                    pi_predictions[name] = (
                        pi_predictions[name] / cpu_counts)
                pi_anchor = pi_predictions['pi_native_p0_i0'].clone()
                for name in pi_predictions:
                    pi_predictions[name] = (
                        exact_baseline + pi_predictions[name] - pi_anchor
                    ).clamp(0.0, 1.0)
                pi_predictions['pi_native_p0_i0'] = exact_baseline.clone()
                for name in pi_mechanism_maps:
                    pi_mechanism_maps[name] = (
                        pi_mechanism_maps[name] / cpu_counts)
                components['role_prompt_pi_variant_class_logits'] = (
                    pi_predictions)
                components['role_prompt_pi_mechanism_class_maps'] = (
                    pi_mechanism_maps)
        return predictions, components

    def predict(self, inputs, data_samples):
        if data_samples is None:
            raise ValueError('SegEarthOV3Segmentation requires image metadata.')
        for data_sample in data_samples:
            meta = data_sample.metainfo
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            original_shape = tuple(meta['ori_shape'][:2])
            needs_components = self._uses_role_prompt_tta()

            use_sliding = (
                self.slide_crop > 0
                and (self.slide_crop < image.width
                     or self.slide_crop < image.height))
            if use_sliding:
                result = self.slide_inference(
                    image, self.slide_stride, self.slide_crop,
                    return_components=needs_components)
            else:
                result = self._inference_single_view(
                    image,
                    return_components=needs_components,
                    view_id='full_image')
            if needs_components:
                query_logits, components = result
            else:
                query_logits, components = result, None

            if query_logits.shape[-2:] != original_shape:
                query_logits = F.interpolate(
                    query_logits.unsqueeze(0),
                    size=original_shape,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
                if components is not None:
                    for key in ('semantic_logits', 'instance_logits'):
                        components[key] = F.interpolate(
                            components[key].unsqueeze(0),
                            size=original_shape,
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(0)
                    for name, value in components[
                            'role_prompt_variant_class_logits'].items():
                        components['role_prompt_variant_class_logits'][name] = (
                            F.interpolate(
                                value.unsqueeze(0),
                                size=original_shape,
                                mode='bilinear',
                                align_corners=False,
                            ).squeeze(0))
                    if 'role_prompt_pi_variant_class_logits' in components:
                        for field in (
                                'role_prompt_pi_variant_class_logits',
                                'role_prompt_pi_mechanism_class_maps'):
                            for name, value in components[field].items():
                                components[field][name] = F.interpolate(
                                    value.unsqueeze(0),
                                    size=original_shape,
                                    mode='bilinear',
                                    align_corners=False,
                                ).squeeze(0)

            class_logits = self._aggregate_query_logits_to_classes(
                query_logits)
            prediction = class_logits.argmax(dim=0)
            prediction[class_logits.max(dim=0)[0] < self.prob_thd] = (
                self.bg_idx)

            if components is not None:
                self._rpt_record_image(
                    components['role_prompt_variant_class_logits'],
                    components.get('role_prompt_view_stats', []),
                    data_sample,
                    image_path,
                    pi_variant_logits=components.get(
                        'role_prompt_pi_variant_class_logits'),
                    pi_mechanism_maps=components.get(
                        'role_prompt_pi_mechanism_class_maps'),
                )
            data_sample.set_data({
                'seg_logits': PixelData(data=class_logits),
                'pred_sem_seg': PixelData(data=prediction.unsqueeze(0)),
            })
        return data_samples

    def _forward(self, data_samples):
        raise NotImplementedError

    def inference(self, img, batch_img_metas):
        raise NotImplementedError

    def encode_decode(self, inputs, batch_img_metas):
        raise NotImplementedError

    def extract_feat(self, inputs):
        raise NotImplementedError

    def loss(self, inputs, data_samples):
        raise NotImplementedError


def get_cls_idx(path):
    class_names = []
    class_indices = []
    with open(path, encoding='utf-8') as handle:
        for class_index, row in enumerate(handle):
            names = [name.strip() for name in row.split(',') if name.strip()]
            class_names.extend(names)
            class_indices.extend([class_index] * len(names))
    if not class_names:
        raise ValueError(f'No class prompts found in {path}.')
    return class_names, class_indices


def _build_class_names(query_words, query_idx, num_classes):
    names = [None] * int(num_classes)
    for prompt, class_index in zip(query_words, query_idx):
        if names[int(class_index)] is None:
            names[int(class_index)] = prompt
    return [name or f'class_{index}' for index, name in enumerate(names)]


def _resolve_inference_device(device):
    device = torch.device(device)
    if device.type != 'cuda':
        return device
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        device = torch.device(f'cuda:{int(local_rank)}')
    elif device.index is None:
        device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    return device
