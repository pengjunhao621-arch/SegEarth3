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

## Audited project constraints

- MMCV 2.0 custom data transforms must inherit `BaseTransform`; implementing
  `transform()` alone does not make an instance callable by MMEngine Compose.
- MMSegmentation 1.2.2 training calls `stack_batch`, which requires exactly
  one of `SegDataPreProcessor.size` and `size_divisor`.
- MMEngine 0.10.4 converts each `BaseDataElement` model output to a plain
  nested dictionary before dispatching it to metrics. Custom metrics must read
  fields such as `sample['seg_logits']['data']`, matching MMSeg 1.2.2's
  `IoUMetric`; attribute access is valid in the model but not in a metric.
- Do not enable SAM3 `torch.compile` paths: `torch.compile` starts in PyTorch
  2.0. The current image-model builder leaves compilation disabled.
- Both GUI and headless OpenCV wheels are installed. Baseline execution has
  worked, so do not change them opportunistically; treat OpenCV import/ABI
  errors as a known environment risk if they appear.
- Triton 3.4.0 is present but is not required by the current inference path.

The Query Topology v1 diagnostic uses only APIs available in these recorded
versions: PyTorch tensor operations, NumPy, OpenCV connected components, and
JSON/CSV output. Run its bounded two-image smoke path before the complete
seven-dataset collection:

```bash
GPU_LIST=0 NPROC=1 \
ROOT=logs/query_topology_v1_smoke \
bash tools/run_query_topology_v1.sh smoke
```
