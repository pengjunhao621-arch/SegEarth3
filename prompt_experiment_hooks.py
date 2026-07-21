"""MMEngine diagnostics for Prompt-SAM3 v1 training."""

from __future__ import annotations

import json
import math
import os
from typing import Dict

import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


def _ranked_jsonl(path: str) -> str:
    rank = int(os.environ.get("RANK", "0"))
    stem, suffix = os.path.splitext(path)
    return f"{stem}_rank{rank}{suffix or '.jsonl'}"


def _l2_norm(tensors) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is not None:
            total += float(tensor.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


@HOOKS.register_module()
class PromptGradientDiagnosticHook(Hook):
    """Persist component gradient/parameter norms and optimizer state."""

    priority = "VERY_LOW"

    def __init__(
        self,
        interval: int = 20,
        path: str = "work_dirs/prompt_sam3/gradient_diagnostics.jsonl",
    ) -> None:
        self.interval = int(interval)
        self.path = _ranked_jsonl(path)
        self._handle = None
        self._gradient_handles = []
        self._captured_gradient_squares = {}
        self._runner = None
        self._current_amp_scale = 1.0

    def _open(self):
        if self._handle is None:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8", buffering=1)

    @staticmethod
    def _group(name: str) -> str:
        if "shared_context" in name or "context_" in name:
            return "shared"
        if any(
            token in name
            for token in (
                "state_",
                "class_to_state",
                "spatial_key",
                "spatial_value",
            )
        ):
            return "state"
        if "global_" in name:
            return "global"
        if "gate_logits" in name:
            return "gate"
        return "other"

    def before_train(self, runner) -> None:
        """Capture backward-time gradients before OptimWrapper clears them."""
        self._runner = runner
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            def capture(gradient, parameter_name=name):
                self._captured_gradient_squares[parameter_name] = (
                    (gradient.detach().float() / self._current_amp_scale)
                    .pow(2)
                    .sum()
                )
                return gradient

            self._gradient_handles.append(parameter.register_hook(capture))

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        self._captured_gradient_squares.clear()
        scaler = getattr(runner.optim_wrapper, "loss_scaler", None)
        self._current_amp_scale = (
            max(float(scaler.get_scale()), 1.0)
            if scaler is not None and hasattr(scaler, "get_scale")
            else 1.0
        )

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        if not self.every_n_train_iters(runner, self.interval):
            return
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        grouped: Dict[str, dict] = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = self._group(name)
            grouped.setdefault(group, {"parameters": [], "gradients": []})
            grouped[group]["parameters"].append(parameter)
            grouped[group]["gradients"].append(parameter.grad)
        record = {
            "iter": int(runner.iter + 1),
            "epoch": int(getattr(runner, "epoch", 0)),
        }
        for group, values in grouped.items():
            record[f"parameter_norm/{group}"] = _l2_norm(values["parameters"])
            captured_squares = [
                value
                for name, value in self._captured_gradient_squares.items()
                if self._group(name) == group
            ]
            record[f"gradient_norm/{group}"] = math.sqrt(
                float(torch.stack(captured_squares).sum().cpu())
                if captured_squares
                else 0.0
            )
        record["gradient_norm/total"] = math.sqrt(
            float(
                torch.stack(list(self._captured_gradient_squares.values()))
                .sum()
                .cpu()
            )
            if self._captured_gradient_squares
            else 0.0
        )
        try:
            lr_values = runner.optim_wrapper.get_lr()
            record["learning_rate"] = {
                key: [float(value) for value in values]
                for key, values in lr_values.items()
            }
        except Exception:
            pass
        scaler = getattr(runner.optim_wrapper, "loss_scaler", None)
        if scaler is not None and hasattr(scaler, "get_scale"):
            record["amp_scale"] = float(scaler.get_scale())
        if torch.cuda.is_available():
            record["cuda_max_memory_mb"] = float(
                torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            )
        self._open()
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        checkpoint.setdefault("meta", {})["prompt_sam3"] = {
            "global_feature_source": model.global_feature_source,
            "prompt_component_mode": model.prompt_component_mode,
            "query_words": list(model.query_words),
            "trainable_parameters": [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
            "checkpoint_excludes_frozen_sam3": True,
            "checkpoint_excludes_frozen_remoteclip": True,
        }

    def after_run(self, runner) -> None:
        for handle in self._gradient_handles:
            handle.remove()
        self._gradient_handles = []
        self._runner = None
        self._captured_gradient_squares.clear()
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@HOOKS.register_module()
class PromptBestCheckpointHook(Hook):
    """Select by mIoU, then validation objective, then earliest iter."""

    priority = "LOWEST"

    def __init__(
        self,
        filename: str = "best_prompt_source_val.pth",
        record_file: str = "best_prompt_source_val.json",
        miou_key: str = "mIoU",
        nll_key: str = "prompt/validation_objective",
        tie_tolerance: float = 1e-12,
    ) -> None:
        self.filename = filename
        self.record_file = record_file
        self.miou_key = miou_key
        self.nll_key = nll_key
        self.tie_tolerance = float(tie_tolerance)
        self.best_miou = float("-inf")
        self.best_nll = float("inf")
        self.best_iter = None

    @staticmethod
    def _find_metric(metrics: dict, requested: str) -> float:
        if requested in metrics:
            return float(metrics[requested])
        suffix = requested.split("/")[-1]
        matches = [value for key, value in metrics.items() if key.split("/")[-1] == suffix]
        if len(matches) != 1:
            raise KeyError(
                f"Could not uniquely resolve metric {requested!r} from "
                f"{sorted(metrics)}"
            )
        return float(matches[0])

    def after_val_epoch(self, runner, metrics=None) -> None:
        if metrics is None:
            return
        miou = self._find_metric(metrics, self.miou_key)
        nll = self._find_metric(metrics, self.nll_key)
        better = miou > self.best_miou + self.tie_tolerance
        tied_better_nll = (
            abs(miou - self.best_miou) <= self.tie_tolerance
            and nll < self.best_nll - self.tie_tolerance
        )
        if not (better or tied_better_nll):
            return
        self.best_miou = miou
        self.best_nll = nll
        self.best_iter = int(runner.iter + 1)
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        meta = {
            "iter": self.best_iter,
            "source_val_mIoU": miou,
            "source_val_objective": nll,
            "selection_rule": (
                "max mIoU; tie=min CE+Dice+0.05*anchor; tie=earliest"
            ),
            "global_feature_source": model.global_feature_source,
            "prompt_component_mode": model.prompt_component_mode,
            "checkpoint_excludes_frozen_sam3": True,
            "checkpoint_excludes_frozen_remoteclip": True,
        }
        runner.save_checkpoint(
            runner.work_dir,
            filename=self.filename,
            save_optimizer=False,
            save_param_scheduler=False,
            meta=meta,
            by_epoch=False,
        )
        if runner.rank == 0:
            record_path = os.path.join(runner.work_dir, self.record_file)
            with open(record_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, ensure_ascii=False)


@HOOKS.register_module()
class PromptValidationArtifactHook(Hook):
    """Route each validation panel into its own iteration directory."""

    priority = "VERY_HIGH"

    def __init__(self, subdirectory: str = "diagnostics") -> None:
        self.subdirectory = subdirectory

    def before_val_epoch(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        model.diagnostic_dir = os.path.join(
            runner.work_dir,
            self.subdirectory,
            f"iter_{int(runner.iter + 1):06d}",
        )
        model._diagnostic_written.clear()
