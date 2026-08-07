# Semantic Supplement Screen V1

## Goal

Test whether one shared text prompt creates conflicting behavior across SAM3's
semantic, instance, and Presence roles, and whether a canonical structural
anchor plus a separate semantic supplement is a better organization.

This is an inference-only diagnostic. It does not train or update SAM3, does
not use RemoteCLIP, and never selects prompts online from evaluation labels.
The official SegEarth-OV3 prediction remains the primary model output.

## Frozen prompt families

Every class has a literal canonical anchor and two candidate families. The
JSON banks are frozen before evaluation.

- `lexical`: class-boundary-preserving aliases, spelling variants, singular
  or plural variants, and benchmark nomenclature. It excludes attributes,
  parts, subtypes, supertypes, scenes, and actions.
- `modifier`: one controlled overhead visual modifier that retains the
  literal class phrase. These prompts intentionally test additional semantic
  content and are not claimed to be category-equivalent aliases.

Candidate counts are allowed to differ by class. Missing slots are evaluated
as exact anchor fallbacks so every dataset has the same output schema; family
means use only candidates that actually exist.

## One-image-encoding comparison

For each prompt, the implementation preserves native SAM3 grounding. It does
not splice token embeddings or bypass the model's shared conditioning path.
The canonical anchor supplies the stable Presence and raw instance queries.
Each candidate produces three auditable paths:

1. `shared_native`: the candidate is used normally by all SAM3 roles.
2. `semantic_replace`: candidate semantic output is combined with the
   anchor's Presence and instance output.
3. `semantic_residual`: the anchor semantic output is retained and receives
   a bounded candidate correction:

   `S = clamp(S_anchor + alpha * clip(S_candidate - S_anchor, -tau, tau))`

The declared default is `alpha=0.50`, `tau=0.25`. No parameter is selected
from evaluation labels. The two fixed family rules average all available
candidate semantic maps before applying replacement or residual composition.

The protocol records 24 class-space variants in one evaluation pass: official
baseline, canonical anchor, 18 per-slot causal paths, and four family means.

## Interpretation

All causal comparisons use `anchor_native` as their reference because the
official baseline may already max over multiple query words. The official
baseline remains separately reported.

- `semantic_replace - shared_native` isolates the value of assigning
  different prompts to different roles.
- `semantic_residual - semantic_replace` measures whether retaining the
  canonical semantic base protects against prompt drift.
- `lexical` versus `modifier` measures whether category-equivalent wording
  and added overhead semantics behave differently.

`oracle_diagnostics.csv` is evaluation-label-selected analysis only. It is an
upper-bound/failure-localization report and must never be reported as a valid
inference method.

## Datasets and outputs

The fixed evaluation set is UDD5, VDD, Vaihingen, Potsdam, OpenEarthMap, and
LoveDA. iSAID is deliberately absent.

Each dataset directory contains the ordinary evaluation log/result plus
`screen.rank*.jsonl`. The summary directory contains dataset and class IoUs,
candidate causal effects, prompt-response statistics, integrity checks,
cross-dataset effects, oracle-only diagnostics, and `decision_report.md`.

## Commands

Run a server-runtime preflight:

```bash
python tools/preflight_semantic_supplement_screen.py --check-runtime-assets
```

Run one image from each dataset:

```bash
GPU_LIST=0 NPROC=1 SMOKE_SAMPLES=1 \
ROOT=logs/semantic_supplement_screen_v1_smoke \
bash tools/run_semantic_supplement_screen_v1.sh smoke
```

Run the fixed six-dataset screen and summarize it:

```bash
GPU_LIST=0,1 NPROC=2 \
ROOT=logs/semantic_supplement_screen_v1_full \
bash tools/run_semantic_supplement_screen_v1.sh all
```

Use a fresh `ROOT` for every run. The runner refuses to append to existing
rank records because duplicated images would invalidate aggregate metrics.
