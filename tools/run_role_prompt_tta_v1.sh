#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${ROOT:-logs/role_prompt_tta_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
NPROC="${NPROC:-2}"
DATASETS="${DATASETS:-}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-1}"
PRIMARY_VARIANT="${PRIMARY_VARIANT:-full_regrounded_e2e}"
E2E_STEPS="${E2E_STEPS:-1}"
E2E_MAX_CLASSES="${E2E_MAX_CLASSES:-4}"
REMOTECLIP_DEVICE="${REMOTECLIP_DEVICE:-same}"
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
        echo "REMOTECLIP_DEVICE=aux requires at least ${required} visible GPUs" >&2
        echo "for NPROC=${NPROC}; got GPU_LIST=${GPU_LIST}." >&2
        echo "Use two ranks with four GPUs, not four ranks: " >&2
        echo "GPU_LIST=0,1,2,3 NPROC=2 REMOTECLIP_DEVICE=aux" >&2
        exit 2
    fi
}

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_role_prompt_tta.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_role_prompt_tta.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_role_prompt_tta.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_role_prompt_tta.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_role_prompt_tta.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_role_prompt_tta.py" ;;
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
    local dataset config out_dir existing
    validate_device_layout
    "${PYTHON_BIN}" tools/preflight_role_prompt_tta.py \
        --check-runtime-assets
    for dataset in $(datasets_for_mode "${collection_mode}"); do
        config="$(config_for "${dataset}")"
        out_dir="${ROOT}/${dataset}"
        mkdir -p "${out_dir}"
        existing="$(find "${out_dir}" -maxdepth 1 \
            -name 'role_prompt_tta.rank*.jsonl' -print -quit)"
        if [[ -n "${existing}" ]]; then
            echo "Refusing to append duplicate records: ${existing}" >&2
            echo "Use a new ROOT or move the previous run first." >&2
            exit 2
        fi
        echo "[role-prompt-tta-v1] ${dataset} -> ${out_dir}"
        common_options=(
            model.role_prompt_tta_stats_path="${out_dir}/role_prompt_tta.jsonl"
            model.role_prompt_tta_artifact_dir="${out_dir}/artifacts"
            model.role_prompt_tta_primary_variant="${PRIMARY_VARIANT}"
            model.role_prompt_tta_e2e_steps="${E2E_STEPS}"
            model.role_prompt_tta_e2e_max_classes="${E2E_MAX_CLASSES}"
            model.role_prompt_tta_remoteclip_device="${REMOTECLIP_DEVICE}"
            model.role_prompt_tta_save_npz="${SAVE_NPZ}"
        )
        if [[ "${collection_mode}" == "smoke" ]]; then
            run_eval "${config}" "${out_dir}" \
                --cfg-options \
                test_dataloader.dataset.indices="${SMOKE_SAMPLES}" \
                "${common_options[@]}" \
                model.role_prompt_tta_e2e_max_classes=1 \
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
        -name 'role_prompt_tta.rank*.jsonl' -type f | sort)
    if [[ "${#inputs[@]}" -eq 0 ]]; then
        echo "No role_prompt_tta.rank*.jsonl files found under ${ROOT}." >&2
        exit 2
    fi
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_role_prompt_tta.py \
        --inputs "${inputs[@]}" \
        --out-dir "${ROOT}/summary" \
        --expected-datasets \
        udd5 vdd vaihingen potsdam openearthmap loveda \
        --positive-dataset-gate 4
}

case "${MODE}" in
    preflight)
        "${PYTHON_BIN}" tools/preflight_role_prompt_tta.py \
            --check-runtime-assets
        ;;
    smoke)
        collect smoke
        summarize
        ;;
    fast)
        collect fast
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
        echo "Usage: bash $0 {preflight|smoke|fast|collect-all|summarize|all}" >&2
        exit 2
        ;;
esac
