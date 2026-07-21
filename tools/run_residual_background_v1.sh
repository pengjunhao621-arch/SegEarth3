#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/residual_background_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

DATASETS_SMALL=(udd5 vdd vaihingen)
DATASETS_MAIN=(udd5 vdd vaihingen potsdam openearthmap loveda)

config_for() {
    case "$1" in
        udd5) echo "configs/cfg_udd5.py" ;;
        vdd) echo "configs/cfg_vdd.py" ;;
        vaihingen) echo "configs/cfg_vaihingen.py" ;;
        potsdam) echo "configs/cfg_potsdam.py" ;;
        openearthmap) echo "configs/cfg_openearthmap.py" ;;
        loveda) echo "configs/cfg_loveda.py" ;;
        isaid) echo "configs/cfg_iSAID.py" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

variant_params() {
    local variant="$1"
    case "${variant}" in
        threshold_strict)
            RB_ROUTE="${RB_ROUTE:-threshold_only}"
            RB_SCORE_FACTOR="${RB_SCORE_FACTOR:-0.30}"
            RB_MIN_SCORE="${RB_MIN_SCORE:-0.02}"
            RB_MIN_SEMANTIC="${RB_MIN_SEMANTIC:-0.04}"
            RB_MIN_INSTANCE="${RB_MIN_INSTANCE:-0.01}"
            RB_MIN_PRESENCE="${RB_MIN_PRESENCE:-0.00}"
            RB_MIN_LOCAL="${RB_MIN_LOCAL:-0.60}"
            RB_MIN_FG_BG_MARGIN="${RB_MIN_FG_BG_MARGIN:-0.00}"
            RB_MIN_REL_MARGIN="${RB_MIN_REL_MARGIN:-0.04}"
            ;;
        threshold_balanced)
            RB_ROUTE="${RB_ROUTE:-threshold_only}"
            RB_SCORE_FACTOR="${RB_SCORE_FACTOR:-0.25}"
            RB_MIN_SCORE="${RB_MIN_SCORE:-0.02}"
            RB_MIN_SEMANTIC="${RB_MIN_SEMANTIC:-0.03}"
            RB_MIN_INSTANCE="${RB_MIN_INSTANCE:-0.01}"
            RB_MIN_PRESENCE="${RB_MIN_PRESENCE:-0.00}"
            RB_MIN_LOCAL="${RB_MIN_LOCAL:-0.55}"
            RB_MIN_FG_BG_MARGIN="${RB_MIN_FG_BG_MARGIN:-0.00}"
            RB_MIN_REL_MARGIN="${RB_MIN_REL_MARGIN:-0.02}"
            ;;
        weak_bg_strict)
            RB_ROUTE="${RB_ROUTE:-threshold_or_weak_bg}"
            RB_SCORE_FACTOR="${RB_SCORE_FACTOR:-0.35}"
            RB_MIN_SCORE="${RB_MIN_SCORE:-0.03}"
            RB_MIN_SEMANTIC="${RB_MIN_SEMANTIC:-0.05}"
            RB_MIN_INSTANCE="${RB_MIN_INSTANCE:-0.02}"
            RB_MIN_PRESENCE="${RB_MIN_PRESENCE:-0.00}"
            RB_MIN_LOCAL="${RB_MIN_LOCAL:-0.65}"
            RB_MIN_FG_BG_MARGIN="${RB_MIN_FG_BG_MARGIN:-0.00}"
            RB_MIN_REL_MARGIN="${RB_MIN_REL_MARGIN:-0.08}"
            ;;
        weak_bg_balanced)
            RB_ROUTE="${RB_ROUTE:-threshold_or_weak_bg}"
            RB_SCORE_FACTOR="${RB_SCORE_FACTOR:-0.25}"
            RB_MIN_SCORE="${RB_MIN_SCORE:-0.02}"
            RB_MIN_SEMANTIC="${RB_MIN_SEMANTIC:-0.04}"
            RB_MIN_INSTANCE="${RB_MIN_INSTANCE:-0.01}"
            RB_MIN_PRESENCE="${RB_MIN_PRESENCE:-0.00}"
            RB_MIN_LOCAL="${RB_MIN_LOCAL:-0.60}"
            RB_MIN_FG_BG_MARGIN="${RB_MIN_FG_BG_MARGIN:--0.03}"
            RB_MIN_REL_MARGIN="${RB_MIN_REL_MARGIN:-0.05}"
            ;;
        *)
            echo "Unknown variant: ${variant}" >&2
            exit 2
            ;;
    esac
}

run_eval() {
    local dataset="$1"
    local variant="${2:-threshold_balanced}"
    variant_params "${variant}"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-30631}"
    local config
    config="$(config_for "${dataset}")"
    local out_dir="${ROOT}/${variant}/${dataset}"
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
            model.use_residual_background_modeling=True \
            model.dump_residual_background_stats=True \
            model.residual_background_stats_path="${out_dir}/residual_background.jsonl" \
            model.residual_background_route="${RB_ROUTE}" \
            model.residual_background_score_factor="${RB_SCORE_FACTOR}" \
            model.residual_background_min_score="${RB_MIN_SCORE}" \
            model.residual_background_min_semantic="${RB_MIN_SEMANTIC}" \
            model.residual_background_min_instance="${RB_MIN_INSTANCE}" \
            model.residual_background_min_presence="${RB_MIN_PRESENCE}" \
            model.residual_background_min_local="${RB_MIN_LOCAL}" \
            model.residual_background_min_fg_bg_margin="${RB_MIN_FG_BG_MARGIN}" \
            model.residual_background_min_reliability_margin="${RB_MIN_REL_MARGIN}" \
            model.residual_background_semantic_weight="${RB_SEM_WEIGHT:-0.25}" \
            model.residual_background_instance_weight="${RB_INST_WEIGHT:-0.25}" \
            model.residual_background_presence_weight="${RB_PRES_WEIGHT:-0.10}" \
            model.residual_background_local_weight="${RB_LOCAL_WEIGHT:-0.10}" \
            model.residual_background_bg_weight="${RB_BG_WEIGHT:-0.50}" \
            model.residual_background_local_kernel="${RB_LOCAL_KERNEL:-7}" \
            model.residual_background_require_head_support="${RB_REQUIRE_HEAD_SUPPORT:-True}"
}

run_parallel() {
    local variant="$1"
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-30640}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" "${variant}" \
            > "${ROOT}/launch_logs/rb_${variant}_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" "${variant}" \
                > "${ROOT}/launch_logs/rb_${variant}_${dataset_b}.log" 2>&1 &
            pid_b=$!
            index=$((index + 1))
        fi
        wait "${pid_a}"
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}"
        fi
    done
}

summarize() {
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_residual_background_stats.py \
        --inputs "${ROOT}"/*/*/residual_background_rank*.jsonl \
        --output-dir "${ROOT}/summary"
}

case "${MODE}" in
    eval)
        run_eval "${2:?dataset required}" "${3:-threshold_balanced}"
        ;;
    small-threshold)
        run_parallel threshold_balanced "${DATASETS_SMALL[@]}"
        ;;
    small-weak)
        run_parallel weak_bg_strict "${DATASETS_SMALL[@]}"
        ;;
    suite-threshold)
        run_parallel threshold_balanced "${DATASETS_MAIN[@]}"
        ;;
    suite-weak)
        run_parallel weak_bg_strict "${DATASETS_MAIN[@]}"
        ;;
    summarize)
        summarize
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_residual_background_v1.sh eval DATASET [VARIANT]
  bash tools/run_residual_background_v1.sh small-threshold
  bash tools/run_residual_background_v1.sh small-weak
  bash tools/run_residual_background_v1.sh suite-threshold
  bash tools/run_residual_background_v1.sh suite-weak
  bash tools/run_residual_background_v1.sh summarize

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda, isaid

Variants:
  threshold_strict, threshold_balanced, weak_bg_strict, weak_bg_balanced

Environment overrides:
  ROOT, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  RB_ROUTE, RB_SCORE_FACTOR, RB_MIN_SCORE, RB_MIN_SEMANTIC,
  RB_MIN_INSTANCE, RB_MIN_PRESENCE, RB_MIN_LOCAL, RB_MIN_FG_BG_MARGIN,
  RB_MIN_REL_MARGIN, RB_SEM_WEIGHT, RB_INST_WEIGHT, RB_PRES_WEIGHT,
  RB_LOCAL_WEIGHT, RB_BG_WEIGHT, RB_LOCAL_KERNEL, RB_REQUIRE_HEAD_SUPPORT
EOF
        exit 2
        ;;
esac
