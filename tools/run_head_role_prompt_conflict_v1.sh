#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${ROOT:-logs/head_role_prompt_conflict_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
NPROC="${NPROC:-2}"
DATASETS="${DATASETS:-udd5 vdd vaihingen potsdam openearthmap loveda}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-1}"
SAVE_NPZ="${SAVE_NPZ:-True}"
INTEGRITY_TOLERANCE="${INTEGRITY_TOLERANCE:-1e-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_head_role_prompt_conflict.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_head_role_prompt_conflict.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_head_role_prompt_conflict.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_head_role_prompt_conflict.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_head_role_prompt_conflict.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_head_role_prompt_conflict.py" ;;
        *) echo "Unknown or excluded dataset: $1" >&2; return 2 ;;
    esac
}

validate_datasets() {
    local dataset
    for dataset in ${DATASETS}; do
        config_for "${dataset}" >/dev/null
    done
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
    local dataset config out_dir existing
    validate_datasets
    "${PYTHON_BIN}" tools/preflight_head_role_prompt_conflict.py \
        --check-runtime-assets
    for dataset in ${DATASETS}; do
        config="$(config_for "${dataset}")"
        out_dir="${ROOT}/${dataset}"
        mkdir -p "${out_dir}"
        existing="$(find "${out_dir}" -maxdepth 1 \
            -name 'head_role_conflict.rank*.jsonl' -print -quit)"
        if [[ -n "${existing}" ]]; then
            echo "Refusing to append duplicate records: ${existing}" >&2
            echo "Use a new ROOT or move the previous run first." >&2
            exit 2
        fi
        echo "[head-role-prompt-conflict-v1] ${dataset} -> ${out_dir}"
        common_options=(
            model.role_prompt_tta_stats_path="${out_dir}/head_role_conflict.jsonl"
            model.role_prompt_tta_artifact_dir="${out_dir}/artifacts"
            model.role_prompt_tta_primary_variant=baseline
            model.role_prompt_tta_integrity_tolerance="${INTEGRITY_TOLERANCE}"
            model.role_prompt_tta_save_npz="${SAVE_NPZ}"
        )
        if [[ "${collection_mode}" == "smoke" ]]; then
            run_eval "${config}" "${out_dir}" \
                --cfg-options \
                test_dataloader.dataset.indices="${SMOKE_SAMPLES}" \
                "${common_options[@]}" \
                > "${out_dir}/run.log" 2>&1
        else
            run_eval "${config}" "${out_dir}" \
                --cfg-options "${common_options[@]}" \
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
        -name 'head_role_conflict.rank*.jsonl' -type f | sort)
    if [[ "${#inputs[@]}" -eq 0 ]]; then
        echo "No head_role_conflict.rank*.jsonl files found under ${ROOT}." >&2
        exit 2
    fi
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_head_role_prompt_conflict.py \
        --inputs "${inputs[@]}" \
        --out-dir "${ROOT}/summary" \
        --expected-datasets \
        udd5 vdd vaihingen potsdam openearthmap loveda \
        --allow-incomplete \
        --integrity-tolerance "${INTEGRITY_TOLERANCE}"
}

case "${MODE}" in
    preflight)
        "${PYTHON_BIN}" tools/preflight_head_role_prompt_conflict.py \
            --check-runtime-assets
        ;;
    smoke)
        collect smoke
        summarize
        ;;
    collect-all)
        collect full
        ;;
    summarize)
        summarize
        ;;
    all)
        collect full
        summarize
        ;;
    *)
        echo "Usage: bash $0 {preflight|smoke|collect-all|summarize|all}" >&2
        exit 2
        ;;
esac
