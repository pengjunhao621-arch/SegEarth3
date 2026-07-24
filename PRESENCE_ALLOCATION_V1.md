# Presence Allocation Structural Audit

## Purpose

This experiment tests whether SegEarth-OV3 assigns SAM3 decoder presence to
the correct locations in its computation graph. It does not select top-1 or
top-2 classes, train a selector, or route predictions according to
ground-truth-derived rules.

The official baseline remains the only prediction returned to MMSegmentation.
Four fixed counterfactual graphs are reconstructed from the same SAM3 forward
and evaluated offline against the same ground truth.

## Baseline and predeclared variants

For one text prompt, let:

- `S` be the SAM3 semantic probability map;
- `m_q` be object-query `q`'s mask probability;
- `r_q` be its raw object score;
- `p` be the decoder-presence scalar;
- `tau` be the existing dataset confidence threshold.

The current SegEarth-OV3 path is:

```text
K0 = {q | r_q * p > tau}
I0 = max(q in K0) m_q * r_q * p
P0 = p * max(S, I0)
```

Thus presence controls an irreversible candidate gate, scales retained
instance evidence, and scales the fused map again.

The audit evaluates:

```text
P1 branch-once    = max(p * S, I0)
P2 delayed-gate   = p * max(S, max(q: r_q > tau) m_q * r_q)
P3 logit-prior    = sigmoid(logit(max(S, I_raw)) + logit(p))
P4 instance-scope = max(S, I0)
```

The planned contrasts are fixed before looking at results:

| Contrast | Isolated structural question |
|---|---|
| P1 − P0 | Does applying presence a second time suppress the instance branch? |
| P2 − P1 | Does placing presence inside the hard candidate gate discard useful masks? |
| P3 − P2 | Is probability multiplication the wrong algebra for using presence as a prior? |
| P4 − P1 | Should decoder presence remain local to the instance path instead of gating the semantic path? |

These are diagnostic counterfactuals, not four separately tuned methods.
No target-dataset-specific threshold or post-hoc per-dataset variant selection
is part of the protocol.

## Integrity and saved evidence

Each image record contains:

- exact confusion matrices for P0–P4;
- changed, improved, harmed, and wrong-to-wrong pixels;
- per-class IoU sufficient statistics;
- candidate counts before and after the presence-dependent gate;
- prompt/crop map area suppressed by semantic presence or duplicate instance
  presence;
- baseline class-logit and prediction reconstruction checks;
- prompt/crop-level mechanism statistics;
- a bounded set of compressed, downsampled `.npz` artifacts for changed
  images.

The run stops if the diagnostic P0 does not reconstruct the unchanged
baseline. At most 24 changed images per rank are saved by default, at a
maximum side length of 128 pixels. The summarizer removes only exact,
cross-rank duplicates introduced by distributed-sampler padding; same-rank or
inconsistent duplicates fail instead of silently changing totals.

## Run

From `/home/PengJunhao/workspace/SegEarth-OV-3`:

```bash
# Formula, baseline-default, config-inheritance, and summary-gate checks.
python tools/preflight_presence_allocation.py

# Two-image integrity smoke test on UDD5.
ROOT=logs/presence_allocation_v1_smoke \
GPU_LIST=0,1 NPROC=2 \
bash tools/run_presence_allocation_v1.sh smoke

# Full six-dataset protocol. iSAID is intentionally excluded.
ROOT=logs/presence_allocation_v1_full \
GPU_LIST=0,1 NPROC=2 \
bash tools/run_presence_allocation_v1.sh all
```

If collection is already complete and only the summary is needed:

```bash
ROOT=logs/presence_allocation_v1_full \
bash tools/run_presence_allocation_v1.sh summarize
```

Use a new `ROOT` for every rerun. The script refuses to append to an existing
rank JSONL because duplicate images would invalidate exact totals.

## Decision rule

A structural hypothesis is supported only when:

1. the integrity checks pass;
2. its mechanism is measurably activated;
3. its predeclared contrast improves more pixels than it harms;
4. the same fixed graph improves mIoU on all six complete datasets.

A gain on only some datasets is useful diagnostic evidence, but it does not
satisfy the project's required universal-improvement claim.
