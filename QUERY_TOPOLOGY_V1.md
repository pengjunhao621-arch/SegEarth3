# Query Topology v1: prediction-preserving feasibility experiment

## 1. Focused research question

The same final pixel score can be supported by one spatially coherent SAM3
object query or assembled from many fragmented queries. Query Topology v1 asks
one narrow question:

> After controlling class, confidence, region area, and semantic/instance-head
> mixture, does the topology of the object-query sources contain reproducible
> information about region quality or about which fixed spatial candidate
> would be better?

This experiment does **not** assume that topology identifies the semantic
class. It does **not** train a selector and does **not** change a single
prediction. Its purpose is to determine whether "how the region was formed" is
a measurable, cross-dataset capability gap worth turning into a later method.

## 2. Faithful provenance hierarchy

The official SegEarth-OV3 inference path is retained. The diagnostic traces its
actual instance hierarchy:

1. for each text prompt and image/crop view, retained SAM3 object queries are
   mask-score multiplied and max-reduced;
2. overlapping sliding-window views are averaged for the same prompt;
3. synonym prompts are max-reduced into the dataset class;
4. semantic and instance evidence compete before the final class decision.

Every object query that survives the baseline confidence/presence filter is
captured. `raw_mask_oracle_topk=24` only bounds additional **unkept** queries;
it never truncates kept queries in this diagnostic.

The output records both:

- a hierarchical reconstruction of the prompt-level instance map; and
- atomic source identities `(view, text prompt, object query index)` for region
  topology.

`reconstruction_mae` compares the reconstructed prompt-level instance map with
the actual instance component returned by the baseline. A high error or an
incomplete kept-query count invalidates the result before any topology claim.
Small differences can remain because the baseline upsamples, max-reduces, and
sliding-window averages at full resolution before the diagnostic downsamples,
while the trace is reconstructed on a 256-pixel maximum-side analysis grid.

## 3. Region-level measurements

Predicted connected components are extracted from the unchanged final baseline
prediction. Ground truth is not consulted during extraction.

For each component, the JSONL/CSV records:

- class, area, mean/max final score, top1/top2 margin;
- mean semantic and instance evidence and the fraction of pixels whose winning
  prompt is instance-head dominated;
- supported fraction: pixels with atomic query contribution at least 0.05;
- dominant-source coverage: fraction of supported pixels won by the most
  frequent atomic query source;
- effective source count: `exp(entropy)` of integrated source contribution;
- switch-boundary density: adjacent supported pixels whose winning source
  identity changes;
- mean support count and consensus fraction (at least two sources);
- number of contributing views and text prompts;
- deterministic formation label:
  `semantic_dominant`, `weak_instance_support`,
  `coherent_single_source`, `overlap_consensus`, `fragmented_mosaic`, or
  `mixed_multi_source`;
- GT purity and local IoU, computed only after all prediction-derived features
  are fixed.

The labels are descriptive bins, not correctness rules.

## 4. Matched null and counterfactual spatial actions

### Spatial-roll null

Every atomic contribution map is deterministically rolled by a different
offset while preserving its total mass. The same topology metrics are measured
on this misaligned null. Offline analysis residualizes both real and null
metrics inside cells matched by:

- class;
- final-score bin;
- log2 area bin;
- semantic/instance-mixture bin.

A useful signal must have the predeclared direction and exceed the absolute
null correlation. Merely correlating with confidence or object size is not
accepted.

### Fixed GT-free candidate actions

For each baseline region, four spatial alternatives are built without GT:

- `dominant_query`: the overlapping individual query mask with greatest
  baseline-region overlap;
- `query_union`: the connected union of overlapping query masks;
- `query_consensus`: their at-least-two-query consensus component;
- `semantic_completion`: the overlapping semantic-head connected component.

All actions are evaluated against the class GT inside one shared padded window.
The best action is a **GT oracle for diagnosis only**. It says whether useful
spatial evidence already exists; it is not a deployable action selector.

## 5. Predeclared interpretation gates

The full verdict requires all seven declared datasets: UDD5, VDD, Vaihingen,
Potsdam, OpenEarthMap, LoveDA, and iSAID. Per dataset, the defaults require:

1. every prompt/view query record and all kept object queries were captured;
2. mean instance reconstruction MAE is at most 0.05;
3. at least 50 foreground regions;
4. at least 5% of regions have a fixed-action local-IoU gain of at least 0.02;
5. mean positive oracle gain is at least 0.01;
6. at least one predeclared conditional directional correlation exceeds its
   spatial-roll control by at least 0.02.

Any absent dataset gives `INCOMPLETE`. Any present dataset failing a gate gives
`NO_GO_FOR_QUERY_TOPOLOGY_METHOD`. Only an all-dataset pass gives
`GO_TO_PREDICTION_METHOD_DESIGN`.

These numerical gates are feasibility thresholds, not benchmark claims. A GO
result justifies designing a topology-aware correction and then testing actual
mIoU. A NO-GO result stops this branch rather than encouraging per-dataset
threshold tuning.

## 6. Server commands

Run from:

```bash
cd /home/PengJunhao/workspace/SegEarth-OV-3
```

### Unchanged official baseline

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  configs/cfg_udd5.py \
  --work-dir work_dirs/baseline_udd5 \
  --result-file work_dirs/baseline_udd5/results.xlsx
```

### Local two-image smoke test

This checks the complete SAM3 forward, query capture, JSONL, NPZ, and summary
path. It is expected to produce an `INCOMPLETE` scientific verdict.

```bash
GPU_LIST=0 NPROC=1 \
ROOT=logs/query_topology_v1_smoke \
bash tools/run_query_topology_v1.sh smoke
```

### Three-dataset diagnostic

This is useful for runtime and early failure inspection, but its verdict is
still `INCOMPLETE` because four datasets are absent.

```bash
GPU_LIST=0,1 NPROC=2 \
ROOT=logs/query_topology_v1_fast \
bash tools/run_query_topology_v1.sh fast
```

### Complete seven-dataset experiment

```bash
GPU_LIST=0,1 NPROC=2 \
ROOT=logs/query_topology_v1_all \
bash tools/run_query_topology_v1.sh all
```

The runner refuses to append to an existing per-dataset JSONL. Use a new
`ROOT` for a fresh run. This prevents duplicate images from silently changing
the statistics.

To summarize already collected files:

```bash
ROOT=logs/query_topology_v1_all \
python tools/summarize_query_topology.py \
  --inputs logs/query_topology_v1_all/*/topology.rank*.jsonl \
  --out-dir logs/query_topology_v1_all/summary
```

## 7. Outputs to return for analysis

Return:

```text
logs/query_topology_v1_all/
├── <dataset>/
│   ├── topology.rank*.jsonl
│   ├── results.xlsx
│   ├── run.log
│   └── artifacts/rank*/
│       ├── *.npz
│       └── *.sources.json
└── summary/
    ├── report.md
    ├── decision.json
    ├── dataset_gate.csv
    ├── image_summary.csv
    ├── region_topology.csv
    ├── conditional_signal.csv
    ├── formation_summary.csv
    └── action_oracle.csv
```

The first files to inspect are `report.md`, `dataset_gate.csv`,
`image_summary.csv`, `conditional_signal.csv`, and `action_oracle.csv`. Also
return any dataset `run.log` whose reconstruction or kept-query gate fails.
NPZ files are bounded to the first 16 images per rank and preserve pixel-level
prediction, GT, region id, winning prompt/head/source, support count, margin,
and reconstruction-error maps for exact follow-up inspection.

## 8. Known limits

- Atomic object-query identity is local to a SAM3 crop/view. Across sliding
  windows, distinct IDs are intentionally retained because the baseline
  averages those view-specific prompt maps.
- Query topology explains spatial formation, not semantic truth. A coherent
  query may confidently represent the wrong category.
- The spatial action oracle evaluates only four predeclared candidates and is
  not an upper bound on all possible segmentation corrections.
- Downsampled connected components can merge or remove very small objects.
  The fixed 256-side grid is shared across datasets to avoid dataset-specific
  tuning; small-object sensitivity can be tested later only as a declared
  sensitivity analysis.
- No prediction-changing method should be implemented from a partial or
  reconstruction-invalid result.
