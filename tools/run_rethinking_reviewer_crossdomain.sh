#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/rethinking_reviewer_crossdomain_miou}"
CACHE_ROOT="${CACHE_ROOT:-logs/rethinking_reviewer_stage1/cache/train}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_rethinking_reviewer.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_rethinking_reviewer.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_rethinking_reviewer.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_rethinking_reviewer.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_rethinking_reviewer.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_rethinking_reviewer.py" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

train_variant() {
    local variant="$1"
    local gpu_id="${GPU_ID:-0}"
    local out_dir="${ROOT}/checkpoints/${variant}"
    mkdir -p "${out_dir}" "${ROOT}/launch_logs"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" tools/train_rethinking_reviewer.py \
        --train-cache \
            "${CACHE_ROOT}/udd5" \
            "${CACHE_ROOT}/vdd" \
            "${CACHE_ROOT}/vaihingen" \
        --variant "${variant}" \
        --output-dir "${out_dir}" \
        --validation-fraction 0.10 \
        --epochs "${EPOCHS:-25}" \
        --batch-size "${BATCH_SIZE:-2048}" \
        --gate-thresholds "${GATE_THRESHOLDS:-0.40,0.50,0.60,0.70,0.80,0.90,1.01}" \
        --action-margins "${ACTION_MARGINS:-0.00,0.05,0.10,0.20}" \
        --seed "${SEED:-0}"
}

train_suite() {
    mkdir -p "${ROOT}/launch_logs"
    local variants=(
        "output_only"
        "three_head"
        "internal_trajectory"
    )
    local pids=()
    local index=0
    for variant in "${variants[@]}"; do
        ROOT="${ROOT}" CACHE_ROOT="${CACHE_ROOT}" GPU_ID="${index}" \
            PYTHON_BIN="${PYTHON_BIN}" \
            bash "$0" train "${variant}" \
            > "${ROOT}/launch_logs/train_${variant}.log" 2>&1 &
        pids+=("$!")
        index=$((index + 1))
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

eval_variant() {
    local dataset="$1"
    local variant="$2"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29701}"
    local config
    config="$(config_for "${dataset}")"
    local checkpoint="${ROOT}/checkpoints/${variant}/best.pth"
    local out_dir="${ROOT}/eval/${variant}/${dataset}"
    mkdir -p "${out_dir}"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    torchrun \
        --nproc_per_node=2 \
        --master_port="${port}" \
        eval.py "${config}" \
        --launcher pytorch \
        --work-dir "${out_dir}" \
        --result-file "${out_dir}/results.xlsx" \
        --cfg-options \
            model.dump_reviewer_cache=False \
            model.use_learned_reviewer=True \
            model.reviewer_checkpoint="${checkpoint}" \
            model.reviewer_variant="${variant}" \
            model.dump_learned_reviewer_stats=True \
            model.learned_reviewer_stats_path="${out_dir}/reviewer.jsonl"
}

eval_baseline() {
    local dataset="$1"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29801}"
    local config
    config="$(config_for "${dataset}")"
    local out_dir="${ROOT}/eval/baseline/${dataset}"
    mkdir -p "${out_dir}"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    torchrun \
        --nproc_per_node=2 \
        --master_port="${port}" \
        eval.py "${config}" \
        --launcher pytorch \
        --work-dir "${out_dir}" \
        --result-file "${out_dir}/results.xlsx"
}

run_parallel_jobs() {
    local action="$1"
    shift
    local jobs=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-29720}"
    while (( index < ${#jobs[@]} )); do
        read -r dataset_a variant_a <<< "${jobs[index]}"
        if [[ "${action}" == "baseline" ]]; then
            ROOT="${ROOT}" GPU_IDS=0,1 PORT=$((port_base + index)) \
                bash "$0" baseline "${dataset_a}" \
                > "${ROOT}/launch_logs/baseline_${dataset_a}.log" 2>&1 &
        else
            ROOT="${ROOT}" GPU_IDS=0,1 PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_a}" "${variant_a}" \
                > "${ROOT}/launch_logs/eval_${variant_a}_${dataset_a}.log" 2>&1 &
        fi
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#jobs[@]} )); then
            read -r dataset_b variant_b <<< "${jobs[index]}"
            if [[ "${action}" == "baseline" ]]; then
                ROOT="${ROOT}" GPU_IDS=2,3 PORT=$((port_base + index)) \
                    bash "$0" baseline "${dataset_b}" \
                    > "${ROOT}/launch_logs/baseline_${dataset_b}.log" 2>&1 &
            else
                ROOT="${ROOT}" GPU_IDS=2,3 PORT=$((port_base + index)) \
                    bash "$0" eval "${dataset_b}" "${variant_b}" \
                    > "${ROOT}/launch_logs/eval_${variant_b}_${dataset_b}.log" 2>&1 &
            fi
            pid_b=$!
            index=$((index + 1))
        fi
        wait "${pid_a}"
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}"
        fi
    done
}

eval_unseen_suite() {
    run_parallel_jobs eval \
        "openearthmap output_only" \
        "loveda output_only" \
        "potsdam output_only" \
        "openearthmap three_head" \
        "loveda three_head" \
        "potsdam three_head" \
        "openearthmap internal_trajectory" \
        "loveda internal_trajectory" \
        "potsdam internal_trajectory"
}

baseline_unseen_suite() {
    run_parallel_jobs baseline \
        "openearthmap baseline" \
        "loveda baseline" \
        "potsdam baseline"
}

case "${MODE}" in
    train)
        train_variant \
            "${2:?variant required: output_only|three_head|internal_trajectory}"
        ;;
    train-suite)
        train_suite
        ;;
    eval)
        eval_variant \
            "${2:?dataset required}" \
            "${3:?variant required}"
        ;;
    baseline)
        eval_baseline "${2:?dataset required}"
        ;;
    eval-unseen-suite)
        eval_unseen_suite
        ;;
    baseline-unseen-suite)
        baseline_unseen_suite
        ;;
    summarize)
        mkdir -p "${ROOT}/summary"
        "${PYTHON_BIN}" tools/summarize_rethinking_calibration.py \
            --checkpoints \
                "${ROOT}/checkpoints/output_only/best.pth" \
                "${ROOT}/checkpoints/three_head/best.pth" \
                "${ROOT}/checkpoints/internal_trajectory/best.pth" \
            --output "${ROOT}/summary/calibration_miou_summary.csv"
        "${PYTHON_BIN}" tools/summarize_rethinking_reviewer.py \
            --inputs "${ROOT}/eval"/*/*/reviewer.rank*.jsonl \
            --output "${ROOT}/summary/reviewer_change_summary.csv"
        "${PYTHON_BIN}" tools/collect_rethinking_results.py \
            --root "${ROOT}" \
            --datasets openearthmap loveda potsdam \
            --output "${ROOT}/summary/exact_unseen_miou_summary.csv"
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_rethinking_reviewer_crossdomain.sh train VARIANT
  bash tools/run_rethinking_reviewer_crossdomain.sh train-suite
  bash tools/run_rethinking_reviewer_crossdomain.sh eval DATASET VARIANT
  bash tools/run_rethinking_reviewer_crossdomain.sh baseline DATASET
  bash tools/run_rethinking_reviewer_crossdomain.sh eval-unseen-suite
  bash tools/run_rethinking_reviewer_crossdomain.sh baseline-unseen-suite
  bash tools/run_rethinking_reviewer_crossdomain.sh summarize

Environment overrides:
  ROOT, CACHE_ROOT, GPU_IDS, GPU_ID, PORT, PORT_BASE, EPOCHS,
  BATCH_SIZE, GATE_THRESHOLDS, ACTION_MARGINS, SEED, PYTHON_BIN
EOF
        exit 2
        ;;
esac
