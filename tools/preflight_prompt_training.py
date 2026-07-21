#!/usr/bin/env python3
"""Fail-fast compatibility and data checks for Prompt-SAM3 training."""

import argparse
import copy
import inspect
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope, TRANSFORMS
from mmengine.runner import Runner
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import DATASETS
from mmseg.structures import SegDataSample
from torch.utils.checkpoint import checkpoint

import custom_datasets  # noqa: F401
import custom_transforms  # noqa: F401
from prompt_experiment_hooks import PromptBestCheckpointHook
from prompt_experiment_metrics import PromptValidationMetric


SERVER_REFERENCE = {
    "torch": "1.13.1+cu116",
    "torchvision": "0.14.1+cu116",
    "mmcv": "2.0.0",
    "mmengine": "0.10.4",
    "mmsegmentation": "1.2.2",
    "numpy": "1.26.4",
    "opencv-python": "4.6.0.66",
    "pillow": "11.3.0",
    "transformers": "4.44.2",
}


def _distribution_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _print_runtime():
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    for package, reference in SERVER_REFERENCE.items():
        current = (
            torch.__version__
            if package == "torch"
            else _distribution_version(package)
        )
        marker = "reference" if current == reference else f"reference={reference}"
        print(f"{package}: {current} ({marker})")


def _validate_runtime(cfg):
    errors = []
    if not hasattr(torch, "autocast"):
        errors.append("torch.autocast is unavailable")
    if "use_reentrant" not in inspect.signature(checkpoint).parameters:
        errors.append("torch.utils.checkpoint lacks use_reentrant support")
    checkpoint_parameters = inspect.signature(Runner.save_checkpoint).parameters
    if "filename" not in checkpoint_parameters:
        errors.append("MMEngine Runner.save_checkpoint lacks filename support")

    randomness = cfg.get("randomness", {})
    if bool(randomness.get("deterministic", False)):
        errors.append(
            "randomness.deterministic must be False: Prompt-SAM3 differentiates "
            "through CUDA bilinear interpolation, which PyTorch rejects in "
            "global deterministic-algorithm mode"
        )

    dtype = str(cfg.optim_wrapper.get("dtype", "")).lower()
    if dtype == "bfloat16":
        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable for the configured BF16 training")
        elif not torch.cuda.is_bf16_supported():
            errors.append(
                "The visible GPU does not support BF16; use the predeclared "
                "FP16 fallback instead of starting this config"
            )

    if errors:
        raise RuntimeError(
            "Prompt training runtime check failed:\n- " + "\n- ".join(errors)
        )


def _resolve_project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _validate_assets(cfg):
    required = {
        "SAM3 checkpoint": ROOT / "weights/sam3/sam3.pt",
        "SAM3 tokenizer": ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        "class vocabulary": _resolve_project_path(cfg.model.classname_path),
    }
    if cfg.model.get("global_feature_source") == "remoteclip_global":
        required["RemoteCLIP checkpoint"] = _resolve_project_path(
            cfg.model.remoteclip_checkpoint
        )
        required["RemoteCLIP source"] = _resolve_project_path(
            cfg.model.remoteclip_source_root
        ) / "open_clip"

    missing = [
        f"{label}: {path}"
        for label, path in required.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Prompt training assets are missing:\n- " + "\n- ".join(missing)
        )
    print("Assets:", ", ".join(required))


def _validate_transforms(cfg):
    pipeline = cfg.train_dataloader.dataset.pipeline
    built = []
    for index, transform_cfg in enumerate(pipeline):
        transform = TRANSFORMS.build(copy.deepcopy(transform_cfg))
        if not callable(transform):
            raise TypeError(
                f"train pipeline transform {index} is not callable: "
                f"{type(transform)}"
            )
        built.append(type(transform).__name__)
    print("Train transforms:", ", ".join(built))


def _validate_dataset(cfg):
    dataset = DATASETS.build(copy.deepcopy(cfg.train_dataloader.dataset))
    if len(dataset) == 0:
        raise RuntimeError("OpenEarthMap training dataset is empty")
    sample = dataset[0]
    if (
        not isinstance(sample, dict)
        or "inputs" not in sample
        or "data_samples" not in sample
    ):
        raise TypeError(
            "The first training sample was not packed as expected; got "
            f"{type(sample)} with keys "
            f"{sorted(sample) if isinstance(sample, dict) else 'N/A'}"
        )
    data_sample = sample["data_samples"]
    if not hasattr(data_sample, "gt_sem_seg"):
        raise AttributeError("The first training sample has no gt_sem_seg")
    print(f"Train samples: {len(dataset)}")
    print(f"First input shape: {tuple(sample['inputs'].shape)}")
    print(f"First label shape: {tuple(data_sample.gt_sem_seg.data.shape)}")
    return sample


def _validate_data_preprocessor(cfg, sample):
    preprocessor_cfg = copy.deepcopy(
        dict(cfg.model.get("data_preprocessor", {}))
    )
    size = preprocessor_cfg.get("size")
    size_divisor = preprocessor_cfg.get("size_divisor")
    if (size is not None) == (size_divisor is not None):
        raise RuntimeError(
            "Prompt training data_preprocessor must define exactly one of "
            "`size` and `size_divisor` for MMSeg 1.2.2"
        )

    preprocessor = SegDataPreProcessor(**preprocessor_cfg)
    processed = preprocessor(
        dict(
            inputs=[sample["inputs"]],
            data_samples=[sample["data_samples"]],
        ),
        training=True,
    )
    batch_inputs = processed["inputs"]
    if batch_inputs.ndim != 4 or batch_inputs.shape[0] != 1:
        raise RuntimeError(
            "Prompt training data_preprocessor returned an invalid input "
            f"batch shape: {tuple(batch_inputs.shape)}"
        )
    batch_labels = processed["data_samples"][0].gt_sem_seg.data
    if tuple(batch_inputs.shape[-2:]) != tuple(batch_labels.shape[-2:]):
        raise RuntimeError(
            "Prompt training image/label shapes diverged after preprocessing: "
            f"image={tuple(batch_inputs.shape[-2:])}, "
            f"label={tuple(batch_labels.shape[-2:])}"
        )
    policy = f"size={tuple(size)}" if size is not None else (
        f"size_divisor={size_divisor}"
    )
    print(f"Train preprocessor: {policy}")
    print(f"First batch shape: {tuple(batch_inputs.shape)}")


def _validate_metric_contract():
    metric = PromptValidationMetric(ignore_index=255, num_bins=5)
    model_sample = SegDataSample()
    model_sample.set_data(
        {
            "seg_logits": PixelData(
                data=torch.tensor(
                    [
                        [[0.8, 0.2], [0.7, 0.1]],
                        [[0.2, 0.8], [0.3, 0.9]],
                    ],
                    dtype=torch.float32,
                )
            ),
            "gt_sem_seg": PixelData(
                data=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
            ),
            "pred_sem_seg": PixelData(
                data=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)
            ),
            "prompt_anchor": torch.tensor(0.25),
        }
    )
    evaluator_sample = model_sample.to_dict()
    metric.process(data_batch={}, data_samples=[evaluator_sample])
    values = metric.compute_metrics(metric.results)
    required = {
        "NLL",
        "DiceLoss",
        "AnchorLoss",
        "validation_objective",
        "ECE",
    }
    missing = required - set(values)
    if missing:
        raise RuntimeError(
            "Prompt validation metric omitted required values: "
            f"{sorted(missing)}"
        )
    if not all(np.isfinite(float(values[key])) for key in required):
        raise RuntimeError(
            "Prompt validation metric produced non-finite values: "
            f"{values}"
        )
    published_metrics = {
        "IoU/mIoU": 42.0,
        "prompt/validation_objective": values["validation_objective"],
    }
    if PromptBestCheckpointHook._find_metric(
        published_metrics, "mIoU"
    ) != 42.0:
        raise RuntimeError("Prompt best-checkpoint hook could not resolve mIoU")
    objective = PromptBestCheckpointHook._find_metric(
        published_metrics, "prompt/validation_objective"
    )
    if objective != values["validation_objective"]:
        raise RuntimeError(
            "Prompt best-checkpoint hook could not resolve validation objective"
        )
    print("Validation metric contract: evaluator dict -> PASS")
    print("Best-checkpoint metric resolution: prefixed metrics -> PASS")


def main():
    parser = argparse.ArgumentParser(
        description="Preflight Prompt-SAM3 server runtime and OpenEarthMap data"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/experiments/cfg_openearthmap_prompt_sam3_global.py",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    DefaultScope.get_instance(
        "prompt_preflight", scope_name=cfg.get("default_scope", "mmseg")
    )
    _print_runtime()
    _validate_runtime(cfg)
    _validate_metric_contract()
    _validate_assets(cfg)
    _validate_transforms(cfg)
    sample = _validate_dataset(cfg)
    _validate_data_preprocessor(cfg, sample)
    print(
        "PASS: Prompt-SAM3 runtime, transforms, first data sample, and "
        "training/validation contracts"
    )


if __name__ == "__main__":
    main()
