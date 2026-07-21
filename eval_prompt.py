"""Evaluate one frozen Prompt-SAM3 checkpoint on source or target datasets."""

import argparse
import json
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import custom_datasets  # noqa: F401
import custom_transforms  # noqa: F401
import prompt_experiment_hooks  # noqa: F401
import prompt_experiment_metrics  # noqa: F401
import prompt_synthesis_segmentor  # noqa: F401


MODEL_DATASET_KEYS = (
    "classname_path",
    "prob_thd",
    "bg_idx",
    "slide_stride",
    "slide_crop",
    "confidence_threshold",
    "use_sem_seg",
    "use_presence_score",
    "use_transformer_decoder",
    "instance_score_type",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config", help="The matching training config")
    parser.add_argument("dataset_config", help="An official configs/cfg_*.py file")
    parser.add_argument("checkpoint")
    parser.add_argument("--component-mode", default="full")
    parser.add_argument("--work-dir")
    parser.add_argument(
        "--cfg-options", nargs="+", action=DictAction, help="Final merged overrides"
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    return args


def bind_cuda_device(args):
    if args.launcher == "none" or not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())


def merge_for_target(experiment: Config, dataset: Config) -> Config:
    cfg = experiment.copy()
    for key in MODEL_DATASET_KEYS:
        if key in dataset.model:
            cfg.model[key] = dataset.model[key]
    cfg.test_dataloader = dataset.test_dataloader
    cfg.test_evaluator = [
        dataset.test_evaluator,
        dict(type="PromptValidationMetric", ignore_index=255, num_bins=15),
    ]
    cfg.dataset_type = dataset.dataset_type
    if "test_cfg" in dataset:
        cfg.test_cfg = dataset.test_cfg
    # Training-only hooks must not save or select checkpoints during target test.
    cfg.custom_hooks = []
    return cfg


def main():
    args = parse_args()
    bind_cuda_device(args)
    experiment = Config.fromfile(args.experiment_config)
    dataset = Config.fromfile(args.dataset_config)
    cfg = merge_for_target(experiment, dataset)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    cfg.model.prompt_component_mode = args.component_mode
    # A source-panel filename would suppress all target artifacts because the
    # basenames differ.  Target test defaults to its deterministic first 32;
    # callers may still provide a target-specific panel through cfg-options.
    cfg.model.diagnostic_panel_file = None
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    dataset_name = str(cfg.dataset_type).replace("Dataset", "").lower()
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        variant = cfg.model.global_feature_source
        cfg.work_dir = osp.join(
            "work_dirs", "prompt_sam3_eval", variant, dataset_name, args.component_mode
        )
    cfg.model.diagnostic_dir = osp.join(cfg.work_dir, "diagnostics")
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    if runner.rank == 0:
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(osp.join(cfg.work_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
