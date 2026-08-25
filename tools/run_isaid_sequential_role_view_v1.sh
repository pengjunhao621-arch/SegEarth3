#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-role-smoke}"
ROOT="${ROOT:-logs/isaid_sequential_role_view_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NPROC="${NPROC:-4}"
ISAID_ROOT="${ISAID_ROOT:-/home/PengJunhao/workspace/data/iSAID}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-1}"
INTEGRITY_TOLERANCE="${INTEGRITY_TOLERANCE:-1e-5}"
BASELINE_TOLERANCE="${BASELINE_TOLERANCE:-0.05}"
SOURCE_REGISTRY="${SOURCE_REGISTRY:-configs/experiments/joint_role_view_domain_extension_v1.json}"
SELECTED_REGISTRY="${SELECTED_REGISTRY:-}"
CONFIG="configs/cfg_iSAID.py"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

preflight() {
    # Discover the file without importing ``tests.*``. Some server
    # environments install an unrelated top-level ``tests`` package, which
    # shadows this repository's non-package tests directory.
    "${PYTHON_BIN}" -m unittest discover \
        -s tests -p 'test_isaid_sequential_role_view.py'
    if [[ ! -d "${ISAID_ROOT}/img_dir/val" \
            || ! -d "${ISAID_ROOT}/ann_dir/val" ]]; then
        echo "Invalid iSAID layout: ${ISAID_ROOT}" >&2
        exit 2
    fi
    if [[ ! -f weights/sam3/sam3.pt ]]; then
        echo "Missing weights/sam3/sam3.pt" >&2
        exit 2
    fi
}

run_eval() {
    local out_dir="$1"
    shift
    mkdir -p "${out_dir}"
    if [[ "${NPROC}" -gt 1 ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            torchrun --standalone --nproc_per_node="${NPROC}" \
            eval.py "${CONFIG}" \
            --launcher pytorch \
            --work-dir "${out_dir}/work" \
            --result-file "${out_dir}/results.json" \
            "$@"
    else
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            "${PYTHON_BIN}" eval.py "${CONFIG}" \
            --work-dir "${out_dir}/work" \
            --result-file "${out_dir}/results.json" \
            "$@"
    fi
}

collect() {
    local scope="$1"
    local role_only="$2"
    local registry="$3"
    local out_dir="${ROOT}/screen/isaid"
    local existing
    mkdir -p "${out_dir}"
    existing="$(find "${out_dir}" -maxdepth 1 \
        -name 'screen.rank*.jsonl' -print -quit)"
    if [[ -n "${existing}" ]]; then
        echo "Refusing to append screen records: ${existing}" >&2
        exit 2
    fi
    options=(
        test_dataloader.dataset.data_root="${ISAID_ROOT}"
        model.use_role_prompt_tta=True
        model.dump_role_prompt_tta_stats=True
        model.role_prompt_tta_protocol=role_functional_text_screen_v1
        model.role_prompt_tta_dataset_name=isaid
        model.role_prompt_tta_prompt_bank=configs/prompt_banks/role_functional_text_v3/isaid.json
        model.role_prompt_tta_selection_registry=configs/experiments/role_text_domain_extension_v1.json
        model.role_prompt_tta_visual_field_diagnosis=True
        model.role_prompt_tta_visual_field_registry=configs/experiments/role_visual_field_domain_extension_v1.json
        model.role_prompt_tta_visual_field_mode=joint_role_view
        model.role_prompt_tta_joint_profile_registry="${registry}"
        model.role_prompt_tta_joint_role_only="${role_only}"
        model.role_prompt_tta_stats_path="${out_dir}/screen.jsonl"
        model.role_prompt_tta_primary_variant=baseline
        model.role_prompt_tta_strict_integrity=True
        model.role_prompt_tta_integrity_tolerance="${INTEGRITY_TOLERANCE}"
        model.role_prompt_tta_save_npz=False
    )
    if [[ "${scope}" == "smoke" ]]; then
        options+=(test_dataloader.dataset.indices="${SMOKE_SAMPLES}")
    fi
    echo "[iSAID sequential ${MODE}] -> ${out_dir}"
    run_eval "${out_dir}" --cfg-options "${options[@]}"
}

screen_inputs() {
    find "${ROOT}/screen/isaid" -maxdepth 1 \
        -name 'screen.rank*.jsonl' -type f | sort
}

summarize_role() {
    local inputs=()
    local path
    while IFS= read -r path; do inputs+=("${path}"); done < <(screen_inputs)
    if [[ "${#inputs[@]}" -eq 0 ]]; then
        echo "No iSAID Role-only records found under ${ROOT}." >&2
        exit 2
    fi
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_isaid_sequential_role_view.py \
        --inputs "${inputs[@]}" \
        --out-dir "${ROOT}/summary" \
        --dataset isaid
}

summarize_joint() {
    local full_report="$1"
    local inputs=()
    local path
    while IFS= read -r path; do inputs+=("${path}"); done < <(screen_inputs)
    if [[ "${#inputs[@]}" -eq 0 ]]; then
        echo "No iSAID selected-Joint records found under ${ROOT}." >&2
        exit 2
    fi
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_joint_role_view.py \
        --inputs "${inputs[@]}" \
        --out-dir "${ROOT}/summary" \
        --expected-datasets isaid
    if [[ "${full_report}" == "True" ]]; then
        "${PYTHON_BIN}" tools/summarize_domain_extension_joint_profile.py \
            --root "${ROOT}" \
            --datasets isaid \
            --baseline-tolerance "${BASELINE_TOLERANCE}"
    fi
}

require_selected_registry() {
    if [[ -z "${SELECTED_REGISTRY}" || ! -f "${SELECTED_REGISTRY}" ]]; then
        echo "Set SELECTED_REGISTRY to the generated iSAID registry." >&2
        exit 2
    fi
}

case "${MODE}" in
    preflight) preflight ;;
    role-smoke)
        preflight
        collect smoke True "${SOURCE_REGISTRY}"
        summarize_role
        ;;
    role)
        preflight
        collect full True "${SOURCE_REGISTRY}"
        summarize_role
        ;;
    summarize-role) summarize_role ;;
    select)
        SELECTED_REGISTRY="${SELECTED_REGISTRY:-${ROOT}/summary/isaid_selected_joint_registry.json}"
        "${PYTHON_BIN}" tools/build_sequential_joint_registry.py \
            --summary "${ROOT}/summary/summary.json" \
            --source-registry "${SOURCE_REGISTRY}" \
            --dataset isaid \
            --output "${SELECTED_REGISTRY}"
        ;;
    joint-smoke)
        require_selected_registry
        preflight
        collect smoke False "${SELECTED_REGISTRY}"
        summarize_joint False
        ;;
    joint)
        require_selected_registry
        preflight
        collect full False "${SELECTED_REGISTRY}"
        summarize_joint True
        ;;
    summarize-joint) summarize_joint True ;;
    *)
        echo "Usage: $0 {preflight|role-smoke|role|summarize-role|select|joint-smoke|joint|summarize-joint}" >&2
        exit 2
        ;;
esac
