#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fast}"
ROOT="${ROOT:-logs/region_hypothesis_v2}"
BASE_PORT="${BASE_PORT:-29920}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PORT="${BASE_PORT}"
PIDS=()

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_region_hypothesis_v2.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_region_hypothesis_v2.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_region_hypothesis_v2.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_region_hypothesis_v2.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_region_hypothesis_v2.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_region_hypothesis_v2.py" ;;
        isaid) echo "configs/experiments/cfg_isaid_region_hypothesis_v2.py" ;;
        *) echo "Unknown dataset: $1" >&2; return 2 ;;
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

run_diagnostic() {
    local phase="$1"
    local dataset="$2"
    local gpus="$3"
    local config
    local out_dir
    config="$(config_for "${dataset}")"
    out_dir="${ROOT}/${phase}/${dataset}"
    mkdir -p "${out_dir}"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES="${gpus}" \
    "${TORCHRUN_BIN}" --nproc_per_node=2 --master_port="${PORT}" \
        eval.py "${config}" --launcher pytorch \
        --work-dir "${out_dir}/work" \
        --result-file "${out_dir}/results.xlsx" \
        --cfg-options \
        model.use_region_contrastive_readout=False \
        model.dump_region_contrastive_readout_stats=False \
        model.dump_region_hypothesis_v2_stats=True \
        model.region_hypothesis_v2_stats_path="${out_dir}/rh2.jsonl" \
        > "${out_dir}/run.log" 2>&1 &
    PIDS+=("$!")
    PORT=$((PORT + 1))
}

summarize() {
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_region_hypothesis_v2.py \
        --inputs \
        "${ROOT}/fast/*/rh2.rank*.jsonl" \
        "${ROOT}/expand/*/rh2.rank*.jsonl" \
        --out-dir "${ROOT}/summary" \
        --selection-datasets udd5,vdd,vaihingen
}

case "${MODE}" in
    fast)
        run_diagnostic fast udd5 "0,1"
        run_diagnostic fast vdd "2,3"
        wait_all
        run_diagnostic fast vaihingen "0,1"
        wait_all
        ;;
    expand)
        run_diagnostic expand potsdam "0,1"
        run_diagnostic expand openearthmap "2,3"
        wait_all
        run_diagnostic expand loveda "0,1"
        run_diagnostic expand isaid "2,3"
        wait_all
        ;;
    summarize)
        summarize
        ;;
    all)
        run_diagnostic fast udd5 "0,1"
        run_diagnostic fast vdd "2,3"
        wait_all
        run_diagnostic fast vaihingen "0,1"
        wait_all
        run_diagnostic expand potsdam "0,1"
        run_diagnostic expand openearthmap "2,3"
        wait_all
        run_diagnostic expand loveda "0,1"
        run_diagnostic expand isaid "2,3"
        wait_all
        summarize
        ;;
    *)
        echo "Usage: bash $0 {fast|expand|summarize|all}" >&2
        exit 2
        ;;
esac
