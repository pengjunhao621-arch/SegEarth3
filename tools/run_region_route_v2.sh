#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dev}"
ROOT="${ROOT:-logs/region_route_v2}"
BASE_PORT="${BASE_PORT:-29620}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PIDS=()
PORT="${BASE_PORT}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_region_route_v2.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_region_route_v2.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_region_route_v2.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_region_route_v2.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_region_route_v2.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_region_route_v2.py" ;;
        isaid) echo "configs/experiments/cfg_isaid_region_route_v2.py" ;;
        *) echo "Unknown dataset: $1" >&2; return 2 ;;
    esac
}

baseline_config_for() {
    case "$1" in
        udd5) echo "configs/cfg_udd5.py" ;;
        vdd) echo "configs/cfg_vdd.py" ;;
        vaihingen) echo "configs/cfg_vaihingen.py" ;;
        *) echo "Unknown baseline dataset: $1" >&2; return 2 ;;
    esac
}

wait_all() {
    local status=0
    local pid
    for pid in "${PIDS[@]}"; do
        wait "${pid}" || status=$?
    done
    PIDS=()
    return "${status}"
}

run_baseline() {
    local dataset="$1"
    local gpus="$2"
    local config
    local out_dir
    config="$(baseline_config_for "${dataset}")"
    out_dir="${ROOT}/baseline/${dataset}"
    mkdir -p "${out_dir}"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES="${gpus}" \
    "${TORCHRUN_BIN}" --nproc_per_node=2 --master_port="${PORT}" \
        eval.py "${config}" --launcher pytorch \
        --work-dir "${out_dir}/work" \
        --result-file "${out_dir}/results.xlsx" \
        > "${out_dir}/run.log" 2>&1 &
    PIDS+=("$!")
    PORT=$((PORT + 1))
}

run_route() {
    local phase="$1"
    local dataset="$2"
    local gpus="$3"
    local variant="$4"
    local agreement="$5"
    local strength="$6"
    local margin="$7"
    local config
    local tag
    local out_dir
    config="$(config_for "${dataset}")"
    tag="${variant}_a${agreement}_s${strength}_m${margin}"
    out_dir="${ROOT}/${phase}/${tag}/${dataset}"
    mkdir -p "${out_dir}"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES="${gpus}" \
    "${TORCHRUN_BIN}" --nproc_per_node=2 --master_port="${PORT}" \
        eval.py "${config}" --launcher pytorch \
        --work-dir "${out_dir}/work" \
        --result-file "${out_dir}/results.xlsx" \
        --cfg-options \
        model.use_region_contrastive_readout=True \
        model.dump_region_contrastive_readout_stats=True \
        model.region_readout_projection=region_winner \
        model.region_readout_route_variant="${variant}" \
        model.region_readout_formula_agreement="${agreement}" \
        model.region_readout_foreground_blend="${strength}" \
        model.region_readout_reject_blend="${strength}" \
        model.region_readout_background_blend="${strength}" \
        model.region_readout_min_margin="${margin}" \
        model.region_contrastive_readout_stats_path="${out_dir}/region.jsonl" \
        > "${out_dir}/run.log" 2>&1 &
    PIDS+=("$!")
    PORT=$((PORT + 1))
}

run_three_datasets() {
    local phase="$1"
    local variant="$2"
    local agreement="$3"
    local strength="$4"
    local margin="$5"
    run_route "${phase}" udd5 "0,1" \
        "${variant}" "${agreement}" "${strength}" "${margin}"
    run_route "${phase}" vdd "2,3" \
        "${variant}" "${agreement}" "${strength}" "${margin}"
    wait_all
    run_route "${phase}" vaihingen "0,1" \
        "${variant}" "${agreement}" "${strength}" "${margin}"
    wait_all
}

case "${MODE}" in
    baseline)
        run_baseline udd5 "0,1"
        run_baseline vdd "2,3"
        wait_all
        run_baseline vaihingen "0,1"
        wait_all
        ;;
    dev)
        while IFS=: read -r variant agreement; do
            run_three_datasets \
                dev "${variant}" "${agreement}" "0.25" "0.10"
        done <<'EOF'
threshold_disabled:1
foreground_contrast:1
reject_inside_mean:1
foreground_reject_contrast_mean:1
foreground_reject_zcontrast_mean:1
foreground_reject_contrast_mean:2
three_route_contrast_mean:1
EOF
        ;;
    sensitivity)
        BEST_VARIANT="${BEST_VARIANT:-foreground_reject_contrast_mean}"
        BEST_AGREEMENT="${BEST_AGREEMENT:-1}"
        while IFS=: read -r strength margin; do
            run_three_datasets \
                sensitivity "${BEST_VARIANT}" "${BEST_AGREEMENT}" \
                "${strength}" "${margin}"
        done <<'EOF'
0.15:0.10
0.40:0.10
0.25:0.05
0.25:0.15
EOF
        ;;
    expand)
        : "${BEST_VARIANT:?Set BEST_VARIANT before expand.}"
        BEST_AGREEMENT="${BEST_AGREEMENT:-1}"
        BEST_STRENGTH="${BEST_STRENGTH:-0.25}"
        BEST_MARGIN="${BEST_MARGIN:-0.10}"
        run_route expand potsdam "0,1" \
            "${BEST_VARIANT}" "${BEST_AGREEMENT}" \
            "${BEST_STRENGTH}" "${BEST_MARGIN}"
        run_route expand openearthmap "2,3" \
            "${BEST_VARIANT}" "${BEST_AGREEMENT}" \
            "${BEST_STRENGTH}" "${BEST_MARGIN}"
        wait_all
        run_route expand loveda "0,1" \
            "${BEST_VARIANT}" "${BEST_AGREEMENT}" \
            "${BEST_STRENGTH}" "${BEST_MARGIN}"
        run_route expand isaid "2,3" \
            "${BEST_VARIANT}" "${BEST_AGREEMENT}" \
            "${BEST_STRENGTH}" "${BEST_MARGIN}"
        wait_all
        ;;
    summarize)
        mkdir -p "${ROOT}/summary"
        "${PYTHON_BIN}" tools/summarize_region_contrastive_readout.py \
            --inputs \
            "${ROOT}/dev/*/*/region.rank*.jsonl" \
            "${ROOT}/sensitivity/*/*/region.rank*.jsonl" \
            "${ROOT}/expand/*/*/region.rank*.jsonl" \
            --out-dir "${ROOT}/summary" \
            --selection-datasets udd5,vdd,vaihingen
        ;;
    *)
        echo "Usage: bash $0 {baseline|dev|sensitivity|expand|summarize}" >&2
        exit 2
        ;;
esac
