#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fast}"
ROOT="${ROOT:-logs/candidate_region_quality}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_candidate_region_quality.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_candidate_region_quality.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_candidate_region_quality.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_candidate_region_quality.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_candidate_region_quality.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_candidate_region_quality.py" ;;
        isaid) echo "configs/experiments/cfg_isaid_candidate_region_quality.py" ;;
        *) echo "Unknown dataset: $1" >&2; return 2 ;;
    esac
}

datasets_for_mode() {
    if [[ -n "${DATASETS}" ]]; then
        echo "${DATASETS}"
    elif [[ "$1" == "fast" ]]; then
        echo "udd5 vdd vaihingen"
    else
        echo "udd5 vdd vaihingen potsdam openearthmap loveda isaid"
    fi
}

collect() {
    local dataset
    local config
    local out_dir
    for dataset in $(datasets_for_mode "$1"); do
        config="$(config_for "${dataset}")"
        out_dir="${ROOT}/${dataset}"
        mkdir -p "${out_dir}"
        echo "[candidate-region-quality] ${dataset} -> ${out_dir}"
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" eval.py "${config}" \
            --work-dir "${out_dir}/work" \
            --result-file "${out_dir}/results.xlsx" \
            --cfg-options \
            model.candidate_region_quality_stats_path="${out_dir}/regions.jsonl" \
            > "${out_dir}/run.log" 2>&1
    done
}

summarize() {
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_candidate_region_quality.py \
        --inputs "${ROOT}"/*/regions.rank*.jsonl \
        --out-dir "${ROOT}/summary" \
        --decision-source hybrid \
        --selection-datasets udd5,vdd,vaihingen
}

case "${MODE}" in
    fast)
        collect fast
        summarize
        ;;
    collect-all)
        collect all
        ;;
    summarize)
        summarize
        ;;
    all)
        collect all
        summarize
        ;;
    *)
        echo "Usage: bash $0 {fast|collect-all|summarize|all}" >&2
        exit 2
        ;;
esac
