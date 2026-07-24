#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${ROOT:-logs/presence_allocation_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
NPROC="${NPROC:-2}"
DATASETS="${DATASETS:-}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-2}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_presence_allocation.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_presence_allocation.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_presence_allocation.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_presence_allocation.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_presence_allocation.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_presence_allocation.py" ;;
        *) echo "Unknown or excluded dataset: $1" >&2; return 2 ;;
    esac
}

datasets_for_mode() {
    if [[ -n "${DATASETS}" ]]; then
        echo "${DATASETS}"
    elif [[ "$1" == "smoke" ]]; then
        echo "udd5"
    elif [[ "$1" == "fast" ]]; then
        echo "udd5 vdd vaihingen"
    else
        echo "udd5 vdd vaihingen potsdam openearthmap loveda"
    fi
}

run_eval() {
    local config="$1"
    local out_dir="$2"
    shift 2
    if [[ "${NPROC}" -gt 1 ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            torchrun --standalone --nproc_per_node="${NPROC}" \
            eval.py "${config}" \
            --launcher pytorch \
            --work-dir "${out_dir}/work" \
            --result-file "${out_dir}/results.xlsx" \
            "$@"
    else
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            "${PYTHON_BIN}" eval.py "${config}" \
            --work-dir "${out_dir}/work" \
            --result-file "${out_dir}/results.xlsx" \
            "$@"
    fi
}

collect() {
    local collection_mode="$1"
    local dataset
    local config
    local out_dir
    local existing
    for dataset in $(datasets_for_mode "${collection_mode}"); do
        config="$(config_for "${dataset}")"
        out_dir="${ROOT}/${dataset}"
        mkdir -p "${out_dir}"
        existing="$(find "${out_dir}" -maxdepth 1 \
            -name 'presence_allocation.rank*.jsonl' -print -quit)"
        if [[ -n "${existing}" ]]; then
            echo "Refusing to append duplicate records: ${existing}" >&2
            echo "Use a new ROOT or move the previous run first." >&2
            exit 2
        fi
        echo "[presence-allocation-v1] ${dataset} -> ${out_dir}"
        if [[ "${collection_mode}" == "smoke" ]]; then
            run_eval "${config}" "${out_dir}" \
                --cfg-options \
                test_dataloader.dataset.indices="${SMOKE_SAMPLES}" \
                model.presence_allocation_stats_path="${out_dir}/presence_allocation.jsonl" \
                model.presence_allocation_artifact_dir="${out_dir}/artifacts" \
                > "${out_dir}/run.log" 2>&1
        else
            run_eval "${config}" "${out_dir}" \
                --cfg-options \
                model.presence_allocation_stats_path="${out_dir}/presence_allocation.jsonl" \
                model.presence_allocation_artifact_dir="${out_dir}/artifacts" \
                > "${out_dir}/run.log" 2>&1
        fi
    done
}

summarize() {
    local inputs=()
    local path
    while IFS= read -r path; do
        inputs+=("${path}")
    done < <(find "${ROOT}" -mindepth 2 -maxdepth 2 \
        -name 'presence_allocation.rank*.jsonl' -type f | sort)
    if [[ "${#inputs[@]}" -eq 0 ]]; then
        echo "No presence_allocation.rank*.jsonl files found under ${ROOT}." >&2
        exit 2
    fi
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_presence_allocation.py \
        --inputs "${inputs[@]}" \
        --out-dir "${ROOT}/summary" \
        --expected-datasets \
        udd5 vdd vaihingen potsdam openearthmap loveda
}

case "${MODE}" in
    smoke)
        collect smoke
        summarize
        ;;
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
        echo "Usage: bash $0 {smoke|fast|collect-all|summarize|all}" >&2
        exit 2
        ;;
esac
