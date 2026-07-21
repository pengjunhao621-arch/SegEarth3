"""MMEngine training entry point for the optional Prompt-SAM3 variant."""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train Prompt-SAM3 v1")
    parser.add_argument("config")
    parser.add_argument("--work-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config values as key=value pairs.",
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


def main():
    args = parse_args()
    bind_cuda_device(args)
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get("work_dir"):
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume:
        cfg.resume = True
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()

