#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${ROOT:-logs/prompt_functional_atlas_v2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
NPROC="${NPROC:-1}"
REMOTECLIP_DEVICE="${REMOTECLIP_DEVICE:-aux}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-1}"
E2E_STEPS="${E2E_STEPS:-1}"
E2E_MAX_CLASSES="${E2E_MAX_CLASSES:-4}"
SAVE_NPZ="${SAVE_NPZ:-True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

validate_device_layout() {
    if [[ "${REMOTECLIP_DEVICE}" != "aux" ]]; then
        return
    fi
    local visible_gpus required
    IFS=',' read -r -a visible_gpus <<< "${GPU_LIST}"
    required=$((2 * NPROC))
    if [[ "${#visible_gpus[@]}" -lt "${required}" ]]; then
        echo "REMOTECLIP_DEVICE=aux requires ${required} visible GPUs" >&2
        echo "for NPROC=${NPROC}; got GPU_LIST=${GPU_LIST}." >&2
        exit 2
    fi
}

run_eval() {
    local out_dir="$1"
    shift
    if [[ "${NPROC}" -gt 1 ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            torchrun --standalone --nproc_per_node="${NPROC}" \
            eval.py configs/experiments/cfg_udd5_prompt_functional_atlas.py \
            --launcher pytorch \
            --work-dir "${out_dir}/work" \
            --result-file "${out_dir}/results.xlsx" \
            "$@"
    else
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            "${PYTHON_BIN}" eval.py \
            configs/experiments/cfg_udd5_prompt_functional_atlas.py \
            --work-dir "${out_dir}/work" \
            --result-file "${out_dir}/results.xlsx" \
            "$@"
    fi
}

collect() {
    local collection_mode="$1"
    local out_dir="${ROOT}/udd5"
    local existing
    validate_device_layout
    "${PYTHON_BIN}" tools/preflight_prompt_functional_atlas.py \
        --check-runtime-assets
    mkdir -p "${out_dir}"
    existing="$(find "${out_dir}" -maxdepth 1 \
        -name 'prompt_atlas.rank*.jsonl' -print -quit)"
    if [[ -n "${existing}" ]]; then
        echo "Refusing to append duplicate records: ${existing}" >&2
        echo "Use a new ROOT or move the previous run first." >&2
        exit 2
    fi
    common_options=(
        model.role_prompt_tta_stats_path="${out_dir}/prompt_atlas.jsonl"
        model.role_prompt_tta_artifact_dir="${out_dir}/artifacts"
        model.role_prompt_tta_remoteclip_device="${REMOTECLIP_DEVICE}"
        model.role_prompt_tta_e2e_steps="${E2E_STEPS}"
        model.role_prompt_tta_e2e_max_classes="${E2E_MAX_CLASSES}"
        model.role_prompt_tta_save_npz="${SAVE_NPZ}"
    )
    if [[ "${collection_mode}" == "smoke" ]]; then
        run_eval "${out_dir}" \
            --cfg-options \
            test_dataloader.dataset.indices="${SMOKE_SAMPLES}" \
            "${common_options[@]}" \
            model.role_prompt_tta_e2e_max_classes=1 \
            > "${out_dir}/run.log" 2>&1
    else
        run_eval "${out_dir}" \
            --cfg-options "${common_options[@]}" \
            > "${out_dir}/run.log" 2>&1
    fi
}

summarize() {
    local inputs=()
    local path
    while IFS= read -r path; do
        inputs+=("${path}")
    done < <(find "${ROOT}/udd5" -maxdepth 1 \
        -name 'prompt_atlas.rank*.jsonl' -type f | sort)
    if [[ "${#inputs[@]}" -eq 0 ]]; then
        echo "No prompt_atlas.rank*.jsonl files found under ${ROOT}/udd5." >&2
        exit 2
    fi
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_prompt_functional_atlas.py \
        --inputs "${inputs[@]}" \
        --out-dir "${ROOT}/summary" \
        --expected-dataset udd5
}

case "${MODE}" in
    preflight)
        "${PYTHON_BIN}" tools/preflight_prompt_functional_atlas.py \
            --check-runtime-assets
        ;;
    smoke)
        collect smoke
        summarize
        ;;
    collect)
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
        echo "Usage: bash $0 {preflight|smoke|collect|summarize|all}" >&2
        exit 2
        ;;
esac
