#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${ROOT:-logs/domain_extension_joint_profile_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NPROC="${NPROC:-4}"
DATASETS="${DATASETS:-uavid}"
DATA_HOME="${DATA_HOME:-/home/PengJunhao/workspace/data}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-1}"
INTEGRITY_TOLERANCE="${INTEGRITY_TOLERANCE:-1e-5}"
BASELINE_TOLERANCE="${BASELINE_TOLERANCE:-0.05}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

config_for() {
    case "$1" in
        voc20) echo "configs/cfg_voc20.py" ;;
        cityscapes) echo "configs/cfg_city_scapes.py" ;;
        isaid) echo "configs/cfg_isaid.py" ;;
        uavid) echo "configs/cfg_uavid.py" ;;
        *) echo "Unknown dataset: $1" >&2; return 2 ;;
    esac
}

root_override_for() {
    case "$1" in
        voc20) echo "${VOC20_ROOT:-}" ;;
        cityscapes) echo "${CITYSCAPES_ROOT:-}" ;;
        isaid) echo "${ISAID_ROOT:-}" ;;
        uavid) echo "${UAVID_ROOT:-}" ;;
    esac
}

root_candidates_for() {
    case "$1" in
        voc20)
            printf '%s\n' "${DATA_HOME}/VOC2012" \
                "${DATA_HOME}/VOCdevkit/VOC2012"
            ;;
        cityscapes)
            printf '%s\n' "${DATA_HOME}/CityScapes" \
                "${DATA_HOME}/cityscapes"
            ;;
        isaid) printf '%s\n' "${DATA_HOME}/iSAID" ;;
        uavid) printf '%s\n' "${DATA_HOME}/UAVid" ;;
    esac
}

has_layout() {
    local dataset="$1"
    local root="$2"
    case "${dataset}" in
        voc20)
            [[ -d "${root}/JPEGImages" \
                && -d "${root}/SegmentationClass" \
                && -f "${root}/ImageSets/Segmentation/val.txt" ]]
            ;;
        cityscapes)
            [[ -d "${root}/leftImg8bit/val" \
                && -d "${root}/gtFine/val" ]]
            ;;
        isaid)
            [[ -d "${root}/img_dir/val" \
                && -d "${root}/ann_dir/val" ]]
            ;;
        uavid)
            [[ -d "${root}/img_dir/test" \
                && -d "${root}/ann_dir/test" ]]
            ;;
    esac
}

data_root_for() {
    local dataset="$1"
    local override candidate
    override="$(root_override_for "${dataset}")"
    if [[ -n "${override}" ]]; then
        echo "${override}"
        return
    fi
    while IFS= read -r candidate; do
        if has_layout "${dataset}" "${candidate}"; then
            echo "${candidate}"
            return
        fi
    done < <(root_candidates_for "${dataset}")
    echo "Could not find ${dataset} under ${DATA_HOME}. Set its *_ROOT variable." >&2
    return 2
}

run_eval() {
    local config="$1"
    local out_dir="$2"
    local result_file="$3"
    shift 3
    mkdir -p "${out_dir}"
    if [[ "${NPROC}" -gt 1 ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            torchrun --standalone --nproc_per_node="${NPROC}" \
            eval.py "${config}" \
            --launcher pytorch \
            --work-dir "${out_dir}/work" \
            --result-file "${result_file}" \
            "$@"
    else
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            "${PYTHON_BIN}" eval.py "${config}" \
            --work-dir "${out_dir}/work" \
            --result-file "${result_file}" \
            "$@"
    fi
}

preflight() {
    "${PYTHON_BIN}" -m unittest discover \
        -s tests -p 'test_domain_extension_joint_profile.py'
    local dataset root
    for dataset in ${DATASETS}; do
        root="$(data_root_for "${dataset}")"
        if ! has_layout "${dataset}" "${root}"; then
            echo "Invalid ${dataset} layout: ${root}" >&2
            exit 2
        fi
        echo "[preflight] ${dataset}: ${root}"
    done
    if [[ ! -f weights/sam3/sam3.pt ]]; then
        echo "Missing weights/sam3/sam3.pt" >&2
        exit 2
    fi
}

collect_baseline() {
    local scope="$1"
    local dataset config data_root out_dir result_file
    for dataset in ${DATASETS}; do
        config="$(config_for "${dataset}")"
        data_root="$(data_root_for "${dataset}")"
        out_dir="${ROOT}/baseline/${dataset}"
        result_file="${out_dir}/results.json"
        if [[ -f "${result_file}" ]]; then
            echo "Refusing to append baseline result: ${result_file}" >&2
            exit 2
        fi
        echo "[domain-extension baseline] ${dataset} -> ${out_dir}"
        options=(test_dataloader.dataset.data_root="${data_root}")
        if [[ "${scope}" == "smoke" ]]; then
            options+=(test_dataloader.dataset.indices="${SMOKE_SAMPLES}")
        fi
        run_eval "${config}" "${out_dir}" "${result_file}" \
            --cfg-options "${options[@]}"
    done
}

collect_screen() {
    local scope="$1"
    local dataset config data_root out_dir existing
    for dataset in ${DATASETS}; do
        config="$(config_for "${dataset}")"
        data_root="$(data_root_for "${dataset}")"
        out_dir="${ROOT}/screen/${dataset}"
        mkdir -p "${out_dir}"
        existing="$(find "${out_dir}" -maxdepth 1 \
            -name 'screen.rank*.jsonl' -print -quit)"
        if [[ -n "${existing}" ]]; then
            echo "Refusing to append screen records: ${existing}" >&2
            exit 2
        fi
        echo "[domain-extension Role-View] ${dataset} -> ${out_dir}"
        options=(
            test_dataloader.dataset.data_root="${data_root}"
            model.use_role_prompt_tta=True
            model.dump_role_prompt_tta_stats=True
            model.role_prompt_tta_protocol=role_functional_text_screen_v1
            model.role_prompt_tta_dataset_name="${dataset}"
            model.role_prompt_tta_prompt_bank="configs/prompt_banks/role_functional_text_v3/${dataset}.json"
            model.role_prompt_tta_selection_registry="configs/experiments/role_text_domain_extension_v1.json"
            model.role_prompt_tta_visual_field_diagnosis=True
            model.role_prompt_tta_visual_field_registry="configs/experiments/role_visual_field_domain_extension_v1.json"
            model.role_prompt_tta_visual_field_mode=joint_role_view
            model.role_prompt_tta_joint_profile_registry="configs/experiments/joint_role_view_domain_extension_v1.json"
            model.role_prompt_tta_stats_path="${out_dir}/screen.jsonl"
            model.role_prompt_tta_primary_variant=baseline
            model.role_prompt_tta_strict_integrity=True
            model.role_prompt_tta_integrity_tolerance="${INTEGRITY_TOLERANCE}"
            model.role_prompt_tta_save_npz=False
        )
        if [[ "${scope}" == "smoke" ]]; then
            options+=(test_dataloader.dataset.indices="${SMOKE_SAMPLES}")
        fi
        run_eval "${config}" "${out_dir}" "${out_dir}/results.json" \
            --cfg-options "${options[@]}"
    done
}

summarize() {
    local inputs=()
    local path
    while IFS= read -r path; do
        inputs+=("${path}")
    done < <(find "${ROOT}/screen" -mindepth 2 -maxdepth 2 \
        -name 'screen.rank*.jsonl' -type f | sort)
    if [[ "${#inputs[@]}" -eq 0 ]]; then
        echo "No screen.rank*.jsonl files found under ${ROOT}/screen." >&2
        exit 2
    fi
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_joint_role_view.py \
        --inputs "${inputs[@]}" \
        --out-dir "${ROOT}/summary" \
        --expected-datasets ${DATASETS}
    "${PYTHON_BIN}" tools/summarize_domain_extension_joint_profile.py \
        --root "${ROOT}" \
        --datasets ${DATASETS} \
        --baseline-tolerance "${BASELINE_TOLERANCE}"
}

case "${MODE}" in
    preflight) preflight ;;
    smoke)
        preflight
        collect_baseline smoke
        collect_screen smoke
        summarize
        ;;
    baseline-all) preflight; collect_baseline full ;;
    screen-all) preflight; collect_screen full ;;
    uavid-main)
        DATASETS="uavid"
        preflight
        collect_screen full
        summarize
        ;;
    summarize) summarize ;;
    all)
        preflight
        collect_baseline full
        collect_screen full
        summarize
        ;;
    *)
        echo "Usage: $0 {preflight|smoke|baseline-all|screen-all|uavid-main|summarize|all}" >&2
        exit 2
        ;;
esac
