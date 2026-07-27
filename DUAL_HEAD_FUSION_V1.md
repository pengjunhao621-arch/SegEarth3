# Same-Forward Dual-Head Fusion Diagnostic v1

## 1. Purpose

This experiment tests whether SegEarth-OV3's semantic head, instance head,
Presence score, and object score are connected in the most useful structural
roles. It is a diagnostic formula bank, not a learned router and not a
top-1/top-2 selector.

All variants for one text prompt reuse the same SAM3 image encoding and the
same text-conditioned decoder outputs. No variant reruns SAM3. Ground truth is
read only after inference to compute diagnostics and never enters a fusion
formula.

For sliding-window inference, every variant remains in prompt/query space
while overlapping crops are averaged. Synonymous prompts are reduced to
classes only afterward, in the same order as the baseline. This avoids the
non-equivalent shortcut of taking a per-crop synonym maximum before averaging.

The official baseline remains unchanged. The bank is enabled only by
`dump_dual_head_fusion_stats=True`, and its maps never replace the returned
`seg_logits`.

## 2. Common notation

- `S`: semantic-head probability map for one text prompt.
- `M_q`: mask probability of instance query `q`.
- `r_q`: raw object score of query `q`.
- `p`: scalar Presence score of the text prompt.
- `K={q | p*r_q > tau}`: the native SegEarth-OV3 kept-query set.
- `I0=max_(q in K) p*r_q*M_q`: native instance branch.
- `I=max_(q in K) r_q*M_q`: role-consistent instance evidence.
- `M_w`: mask that wins `I` at a pixel.
- `q*`: the kept query with the largest `r_q`; `M*` and `r*` are its mask and
  object score.

The reconstructed baseline is:

```text
P0 = p * max(S, I0)
```

This reconstruction is checked at both prompt and final class-logit levels.
The run stops if it differs from the untouched baseline beyond `1e-5`.

## 3. Formula bank

| Variant | Formula / intervention | Question isolated |
|---|---|---|
| `p0_baseline` | `p*max(S,I0)` | Untouched SegEarth-OV3 reference |
| `semantic_only` | `p*S` | Semantic-head base |
| `instance_p0_path` | `p*I0` | Instance path after both Presence uses |
| `p1_role_once` | `p*max(S,I)` | Does each role need Presence only once? |
| `winner_score_once` | `p*max(S,r*M*)` | Is one representative query safer than pixelwise query max? |
| `proc_pgrf` | ProC PGRF below | Public-code-faithful ProC control |
| `proc_gate_debiased` | ProC native instance map, but remove repeated Presence from its residual gate | Isolate gate calibration |
| `proc_role_once` | ProC residual shape with role-consistent object-scored instance evidence | Complete role-consistent ProC adaptation |
| `uni_rcrf_floor` | protected semantic base plus agreement-gated positive instance residual | One-way role-consistent residual fusion |
| `uni_rcrf_region` | floored value agreement plus inside/ring contradiction attenuation | Does regional semantic contradiction identify harmful residuals? |
| `uni_rcrf_unfloored` | same, but agreement can suppress to zero | Is a nonzero residual floor needed? |
| `bi_s2i_only` | object scores corrected by semantic inside/ring contrast | Semantic-to-instance interaction only |
| `bi_i2s_only` | semantic map adjusted at instance boundaries | Instance-to-semantic interaction only |
| `bi_full` | both score correction and boundary feedback | Bidirectional interaction |
| `soft_or` | `p*[1-(1-S)(1-I)]` | Probabilistic union instead of max residual |
| `convex_25` | `p*(0.75S+0.25I)` | Naive blend control that can lower the semantic base |
| `boundary_residual` | instance positive residual only near mask uncertainty/boundary | Is instance evidence mainly useful at boundaries? |
| `interior_residual` | complementary interior residual | Is instance evidence mainly useful inside objects? |
| `raw_mask_residual` | `p*max(S,max M_q)` | Is object-score amplitude useful or harmful? |

### ProC-SAM3 control

The public ProC implementation first builds the native best-instance map with
Presence-weighted object scores. Let `r_max=max_q r_q` and:

```text
B = max(S, I0)
A = 1 - |S - B|
Delta = [B - S]+
k = p * (p*r_max) * A
P_proc = p * clamp(S + k*Delta, 1e-4, 1-1e-4)
```

This reproduces the public ProC PGRF operation order on SegEarth-OV3's native
candidate path: `I0` already uses `p*r_q`, the scalar maximum instance score is
also `p*r_max`, PGRF multiplies it by another `p`, and the prompt output is
finally multiplied by `p`. Candidate generation and semantic/instance tensors
still come from the current baseline's one decoder output.

`proc_gate_debiased` keeps `B=max(S,I0)` but replaces the residual gate's
`p*(p*r_max)` by `r_max`. `proc_role_once` additionally replaces `I0` by the
role-consistent `I=max r_q M_q`. These two controls distinguish a gate issue
from an instance-amplitude issue.

### Role-consistent residual fusion

The protected one-way version is:

```text
A_gamma = 1 - gamma*|S-M_w|, gamma=0.5
P_uni = p * [S + A_gamma*[I-S]+]
```

It never subtracts from `S`; the instance head can only add a bounded positive
residual.

The region ablation uses the winning query's normalized inside/ring contrast
`c` and `A_region=clip(1+0.5c,0.5,1)`. Positive or neutral regional support
does not amplify the residual; only a semantic contradiction attenuates it,
and attenuation is bounded to preserve small objects initially missed by the
semantic head.

For semantic-to-instance feedback, every kept query receives:

```text
c_q = (mean(S inside M_q) - mean(S in outer ring)) /
      (mean(S inside M_q) + mean(S in outer ring) + eps)
r'_q = clip(r_q*(1+eta*c_q), 0, 1), eta=0.25
```

The native candidate set is not re-filtered after this correction. This
isolates score interaction from candidate-selection changes.

For instance-to-semantic feedback:

```text
W = 4*M_w*(1-M_w)
S' = clip(S + beta*W*(M_w-S), 0, 1), beta=0.25
```

`bi_full` applies both interactions and then the same protected positive
residual structure.

## 4. What is recorded

Each rank writes one JSONL file with:

- exact confusion matrix for every variant;
- changed, improved, harmed, and wrong-to-wrong pixel counts against P0;
- direct structural contrasts such as P0→P1, ProC→ProC-role-once, and
  uni→bidirectional;
- per-class intersections, predictions, Presence, and maximum object score;
- candidate counts, inside/ring contrast, score correction, winner switches,
  residual areas, and boundary areas;
- high semantic/instance-agreement errors, including Presence, object score,
  and class margin;
- strict baseline reconstruction checks;
- a bounded number of compressed, downsampled NPZ artifacts.

The summary produces:

- `fusion_summary.csv`
- `fusion_class_iou.csv`
- `fusion_contrasts.csv`
- `fusion_mechanisms.csv`
- `high_agreement_errors.csv`
- `cross_dataset_ranking.csv`
- `integrity.csv`
- `decision.json`
- `report.md`

The cross-dataset `>=4/6` positive gate is only a prioritization signal. A
candidate must also show the expected mechanism, favorable improved/harmed
transitions, interpretable class behavior, and later seed stability.

## 5. Server commands

Run the static/formula/config preflight first:

```bash
cd /home/PengJunhao/workspace/SegEarth-OV-3
python tools/preflight_dual_head_fusion.py
```

Two-image smoke test on UDD5:

```bash
ROOT=logs/dual_head_fusion_v1_smoke \
GPU_LIST=0,1 NPROC=2 SMOKE_SAMPLES=2 \
bash tools/run_dual_head_fusion_v1.sh smoke
```

Full six-dataset collection and summary:

```bash
ROOT=logs/dual_head_fusion_v1 \
GPU_LIST=0,1 NPROC=2 \
bash tools/run_dual_head_fusion_v1.sh all
```

If collection completed but summary did not:

```bash
ROOT=logs/dual_head_fusion_v1 \
bash tools/run_dual_head_fusion_v1.sh summarize
```

One dataset can be run without changing code:

```bash
ROOT=logs/dual_head_fusion_v1_openearthmap \
DATASETS=openearthmap GPU_LIST=0,1 NPROC=2 \
bash tools/run_dual_head_fusion_v1.sh collect-all
```

The six-dataset protocol is UDD5, VDD, Vaihingen, Potsdam, OpenEarthMap, and
LoveDA. iSAID is deliberately excluded.
