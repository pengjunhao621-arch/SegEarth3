# SAM3 Prompt Functional Atlas v2

## 1. What re-grounding means

SAM3 first encodes an image once. For every text concept, its language
condition is then passed through the grounding encoder/decoder and the
semantic, instance, object-score and Presence heads.

The v1 `*_output` variants independently grounded each complete prompt and
combined the already-produced score maps. They did not change the language
condition consumed by SAM3.

The v1 `*_regrounded` variants instead averaged the contextual language-token
sequences from several prompts, installed that synthetic sequence back into
the cached image state, and called SAM3 grounding again. UDD5 showed that this
second operation, rather than prompt-weight optimization, caused the dominant
failure.

## 2. Directional questions

This atlas is a diagnostic, not a proposed final method. Its primary output is
the protected baseline while all interventions are evaluated from the same
image encoding.

It answers five questions:

1. Does SAM3 prefer a literal concept, short domain phrase, short visual noun
   phrase, context phrase, or long remote-sensing description?
2. Can independent prompt outputs be combined without creating a synthetic
   language sequence?
3. Does SAM3 Presence select the prompts that actually improve segmentation?
4. Does one-hot literal re-grounding reproduce the native cached baseline, and
   how quickly does a bounded contextual residual break it?
5. Does one E2E prompt-weight step lower its own post-update objective, and is
   that weight update useful before any re-grounding?

The five controlled prompt roles are fixed across all UDD5 classes. Ground
truth is used only after inference to calculate diagnostics.

## 3. Main controls

- `prompt_0_literal` through `prompt_4_long_description`: the same functional
  slot is used for every class.
- `full_bg_literal_output`: tests the previously missing literal-background
  preservation control.
- `presence_selected_output` and `presence_weighted_output`: test whether
  SAM3's own Presence is a useful prompt-quality signal. They are diagnostics,
  not assumed improvements.
- `e2e_output` and `e2e_bg_literal_output`: isolate adapted weights from the
  invalid sequence re-grounding path.
- `literal_regrounded`: must reproduce the literal baseline within tolerance.
- `anchor_residual_010/025_regrounded`: preserve the literal sequence mask and
  modify only its valid token positions by a 0.10/0.25 bounded residual.
- `full_sequence_regrounded`: retains v1 sequence fusion as a negative control.

## 4. Server commands

Preflight:

```bash
cd /home/PengJunhao/workspace/SegEarth-OV-3
python tools/preflight_prompt_functional_atlas.py --check-runtime-assets
```

One-image smoke using one SAM3 GPU and one RemoteCLIP GPU:

```bash
cd /home/PengJunhao/workspace/SegEarth-OV-3
ROOT=logs/prompt_functional_atlas_v2_smoke \
GPU_LIST=0,1 NPROC=1 REMOTECLIP_DEVICE=aux \
SMOKE_SAMPLES=1 E2E_STEPS=1 E2E_MAX_CLASSES=1 \
bash tools/run_prompt_functional_atlas.sh smoke
```

Full UDD5 atlas using two SAM3 ranks and two auxiliary RemoteCLIP GPUs:

```bash
cd /home/PengJunhao/workspace/SegEarth-OV-3
ROOT=logs/prompt_functional_atlas_v2_udd5 \
GPU_LIST=0,1,2,3 NPROC=2 REMOTECLIP_DEVICE=aux \
E2E_STEPS=1 E2E_MAX_CLASSES=4 \
bash tools/run_prompt_functional_atlas.sh all
```

The run refuses to append to existing JSONL files. Always use a fresh `ROOT`.

## 5. Outputs and decision order

`summary/` contains:

- `dataset_variants.csv`: exact multiclass performance and pixel transitions;
- `per_class.csv`: every variant's class IoU;
- `prompt_function.csv`: token length, Presence, candidates, head magnitude,
  selection frequency and slot-class IoU;
- `prompt_role_summary.csv`: each controlled prompt role's dataset mIoU and
  per-class wins/losses against the literal class name;
- `head_path.csv`: semantic/instance/Presence changes after each re-ground path;
- `e2e_update.csv`: pre/post objective, gradient and weight displacement;
- `comparisons.csv`: the direct structural contrasts;
- `correlations.csv` and `correlations_per_class.csv`: pooled and within-class
  descriptive associations, kept separate to expose class-confounding;
- `summary.json` and `decision_report.md`.

Interpret in this order:

1. Require exact baseline and literal re-ground identity.
2. Decide whether prompt form has a repeatable functional effect.
3. Decide whether independent output-space combination is worth retaining.
4. Decide whether Presence is an evaluator or only another prompt-sensitive
   output.
5. Only then inspect E2E adaptation and bounded re-ground residuals.

Focal loss, prototype diversity, dataset-wide training and new fusion modules
are outside this directional experiment.
