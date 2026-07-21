import pytest
import torch
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

from prompt_experiment_metrics import PromptValidationMetric
from prompt_experiment_hooks import PromptBestCheckpointHook


def _evaluator_sample():
    sample = SegDataSample()
    sample.set_data(
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
    return sample.to_dict()


def test_prompt_metric_accepts_mmengine_evaluator_dict_contract():
    metric = PromptValidationMetric(ignore_index=255, num_bins=5)

    metric.process(data_batch={}, data_samples=[_evaluator_sample()])
    values = metric.compute_metrics(metric.results)

    assert values["hard_pixel_accuracy"] == pytest.approx(1.0)
    assert values["soft_pixel_accuracy"] == pytest.approx(1.0)
    assert values["AnchorLoss"] == pytest.approx(0.25)
    assert values["validation_objective"] > 0.0


def test_prompt_metric_rejects_object_samples_before_attribute_error():
    metric = PromptValidationMetric(ignore_index=255, num_bins=5)

    with pytest.raises(TypeError, match="expects evaluator samples as mappings"):
        metric.process(data_batch={}, data_samples=[object()])


def test_best_checkpoint_hook_resolves_evaluator_prefixes():
    metrics = {
        "IoU/mIoU": 42.0,
        "prompt/validation_objective": 0.75,
    }

    assert PromptBestCheckpointHook._find_metric(metrics, "mIoU") == 42.0
    assert (
        PromptBestCheckpointHook._find_metric(
            metrics, "prompt/validation_objective"
        )
        == 0.75
    )
