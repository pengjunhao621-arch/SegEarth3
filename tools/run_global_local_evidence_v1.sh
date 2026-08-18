#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${ROOT:-logs/global_local_evidence_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
NPROC="${NPROC:-2}"
DATASETS="${DATASETS:-udd5 vdd vaihingen potsdam openearthmap loveda}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-1}"
INTEGRITY_TOLERANCE="${INTEGRITY_TOLERANCE:-1e-5}"
VAIHINGEN_IMG_DIR="${VAIHINGEN_IMG_DIR:-/home/PengJunhao/workspace/data/vaihingen/img_dir/val}"
POTSDAM_IMG_DIR="${POTSDAM_IMG_DIR:-/home/PengJunhao/workspace/data/potsdam/img_dir/val}"
SKIP_TILED_PREFLIGHT="${SKIP_TILED_PREFLIGHT:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_role_functional_text_screen.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_role_functional_text_screen.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_role_functional_text_screen.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_role_functional_text_screen.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_role_functional_text_screen.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_role_functional_text_screen.py" ;;
        *) echo "Unknown or excluded dataset: $1" >&2; return 2 ;;
    esac
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

preflight() {
    "${PYTHON_BIN}" tools/preflight_role_functional_text_screen.py \
        --check-runtime-assets
    local dirs=()
    if [[ "${SKIP_TILED_PREFLIGHT}" == "1" ]]; then
        return
    fi
    [[ " ${DATASETS} " == *" vaihingen "* ]] && dirs+=("${VAIHINGEN_IMG_DIR}")
    [[ " ${DATASETS} " == *" potsdam "* ]] && dirs+=("${POTSDAM_IMG_DIR}")
    if [[ "${#dirs[@]}" -gt 0 ]]; then
        mkdir -p "${ROOT}/preflight"
        "${PYTHON_BIN}" tools/verify_tiled_context.py \
            "${dirs[@]}" \
            --output "${ROOT}/preflight/tiled_rgb_overlap.json"
    fi
}

collect() {
    local scope="$1"
    local dataset config out_dir existing
    preflight
    for dataset in ${DATASETS}; do
        config="$(config_for "${dataset}")"
        out_dir="${ROOT}/${dataset}"
        mkdir -p "${out_dir}"
        existing="$(find "${out_dir}" -maxdepth 1 \
            -name 'screen.rank*.jsonl' -print -quit)"
        if [[ -n "${existing}" ]]; then
            echo "Refusing to append duplicate records: ${existing}" >&2
            echo "Use a new ROOT or move the previous run first." >&2
            exit 2
        fi
        echo "[global-local-evidence-v1] ${dataset} -> ${out_dir}"
        options=(
            model.role_prompt_tta_prompt_bank="configs/prompt_banks/role_functional_text_v2/${dataset}.json"
            model.role_prompt_tta_selection_registry="configs/experiments/role_text_selections_v1.json"
            model.role_prompt_tta_visual_field_diagnosis=True
            model.role_prompt_tta_visual_field_registry="configs/experiments/role_visual_field_v1.json"
            model.role_prompt_tta_visual_field_mode=global_local
            model.role_prompt_tta_stats_path="${out_dir}/screen.jsonl"
            model.role_prompt_tta_primary_variant=baseline
            model.role_prompt_tta_integrity_tolerance="${INTEGRITY_TOLERANCE}"
            model.role_prompt_tta_save_npz=False
        )
        if [[ "${scope}" == "smoke" ]]; then
            run_eval "${config}" "${out_dir}" \
                --cfg-options \
                test_dataloader.dataset.indices="${SMOKE_SAMPLES}" \
                "${options[@]}" \
                > "${out_dir}/run.log" 2>&1
        else
            run_eval "${config}" "${out_dir}" \
                --cfg-options "${options[@]}" \
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
        -name 'screen.rank*.jsonl' -type f | sort)
    if [[ "${#inputs[@]}" -eq 0 ]]; then
        echo "No screen.rank*.jsonl files found under ${ROOT}." >&2
        exit 2
    fi
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_global_local_evidence.py \
        --inputs "${inputs[@]}" \
        --out-dir "${ROOT}/summary" \
        --expected-datasets \
        udd5 vdd vaihingen potsdam openearthmap loveda \
        --allow-incomplete
}

case "${MODE}" in
    preflight) preflight ;;
    smoke) collect smoke; summarize ;;
    collect-all) collect full ;;
    summarize) summarize ;;
    all) collect full; summarize ;;
    *)
        echo "Usage: $0 {preflight|smoke|collect-all|summarize|all}" >&2
        exit 2
        ;;
esac
