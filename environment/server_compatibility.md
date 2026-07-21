# Server Compatibility Reference

Captured from the user's `SegEarth` conda environment on 2026-07-21. This is
the compatibility target for local code changes; it is not a request to
upgrade the remote server.

## Critical runtime versions

| Component | Server version |
| --- | --- |
| Python | 3.9.23 |
| PyTorch | 1.13.1+cu116 |
| torchvision | 0.14.1+cu116 |
| torchaudio | 0.13.1+cu116 |
| CUDA toolkit/runtime | 11.6.x |
| cuBLAS | 11.9.2.110 |
| MMCV | 2.0.0 |
| MMEngine | 0.10.4 |
| MMSegmentation | 1.2.2 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| Pillow | 11.3.0 |
| opencv-python | 4.6.0.66 |
| opencv-python-headless | 4.8.0.76 |
| pytest | 8.4.2 |
| transformers | 4.44.2 |
| timm | 1.0.9 |
| rasterio | 1.4.3 |

## Audited Prompt-SAM3 constraints

- MMCV 2.0 custom data transforms must inherit `BaseTransform`; implementing
  `transform()` alone does not make an instance callable by MMEngine Compose.
- MMSegmentation 1.2.2 training calls `stack_batch`, which requires exactly
  one of `SegDataPreProcessor.size` and `size_divisor`. The Prompt-SAM3
  OpenEarthMap config uses fixed `size=(512, 512)` to match its train crop.
- MMEngine 0.10.4 converts each `BaseDataElement` model output to a plain
  nested dictionary before dispatching it to metrics. Custom metrics must read
  fields such as `sample['seg_logits']['data']`, matching MMSeg 1.2.2's
  `IoUMetric`; attribute access is valid in the model but not in a metric.
- PyTorch 1.13.1 supports `torch.autocast` and checkpoint
  `use_reentrant=False`, both used by Prompt-SAM3.
- Do not enable SAM3 `torch.compile` paths: `torch.compile` starts in PyTorch
  2.0. The current image-model builder leaves compilation disabled.
- Keep a fixed seed and `cudnn_benchmark=False`, but do not force global
  deterministic algorithms for Prompt-SAM3. The learned grounding path uses
  CUDA bilinear interpolation backward, which deterministic mode rejects.
- The primary config uses BF16. GPU BF16 capability must be checked on the
  server before training; use the declared FP16 fallback only if that check
  fails.
- Both GUI and headless OpenCV wheels are installed. Baseline execution has
  worked, so do not change them opportunistically; treat OpenCV import/ABI
  errors as a known environment risk if they appear.
- Triton 3.4.0 is present but is not used by the PyTorch 1.13 Prompt-SAM3 path.

Run the project preflight before a smoke/full training job. It constructs all
transforms and the dataset, loads one packed sample, and executes the actual
MMSeg training data-preprocessor path before SAM3 model construction:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/preflight_prompt_training.py \
  configs/experiments/cfg_openearthmap_prompt_sam3_global.py
```
