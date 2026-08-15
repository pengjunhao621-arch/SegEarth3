#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
if [[ "${MODE}" == class-role-alignment-* ]]; then
    ROOT="${ROOT:-logs/class_role_alignment_v1}"
elif [[ "${MODE}" == boundary-replay-* ]]; then
    ROOT="${ROOT:-logs/boundary_replay_v1}"
elif [[ "${MODE}" == pe-* ]]; then
    ROOT="${ROOT:-logs/pe_role_evidence_v1}"
elif [[ "${MODE}" == completion-* ]]; then
    ROOT="${ROOT:-logs/role_text_completion_v1}"
elif [[ "${MODE}" == pi-* ]]; then
    ROOT="${ROOT:-logs/pi_role_compatibility_v1}"
else
    ROOT="${ROOT:-logs/role_functional_text_screen_v1}"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
NPROC="${NPROC:-2}"
DATASETS="${DATASETS:-udd5 vdd vaihingen potsdam openearthmap loveda}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-1}"
INTEGRITY_TOLERANCE="${INTEGRITY_TOLERANCE:-1e-5}"
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

pi_slots_for() {
    # Fixed from the completed factorial diagnosis; semantic stays at anchor.
    case "$1" in
        udd5) echo "1 2" ;;
        vdd) echo "2 1" ;;
        vaihingen) echo "1 1" ;;
        potsdam) echo "1 2" ;;
        openearthmap) echo "2 2" ;;
        loveda) echo "1 1" ;;
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
    local experiment_mode="${2:-screen}"
    local dataset config out_dir existing presence_slot instance_slot
    validate_datasets
    "${PYTHON_BIN}" tools/preflight_role_functional_text_screen.py \
        --check-runtime-assets
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
        if [[ "${experiment_mode}" == "pe" ]]; then
            echo "[pe-role-evidence-v1] ${dataset} -> ${out_dir}"
        elif [[ "${experiment_mode}" == "boundary-replay" ]]; then
            echo "[boundary-replay-v1] ${dataset} -> ${out_dir}"
        elif [[ "${experiment_mode}" == "class-role-alignment" ]]; then
            echo "[class-role-alignment-v1] ${dataset} -> ${out_dir}"
        elif [[ "${experiment_mode}" == "pi" ]]; then
            echo "[pi-role-compatibility-v1] ${dataset} -> ${out_dir}"
        elif [[ "${experiment_mode}" == "completion" ]]; then
            echo "[role-text-completion-v1] ${dataset} -> ${out_dir}"
        else
            echo "[role-functional-text-v1] ${dataset} -> ${out_dir}"
        fi
        common_options=(
            model.role_prompt_tta_stats_path="${out_dir}/screen.jsonl"
            model.role_prompt_tta_primary_variant=baseline
            model.role_prompt_tta_integrity_tolerance="${INTEGRITY_TOLERANCE}"
            model.role_prompt_tta_save_npz=False
        )
        if [[ "${experiment_mode}" == "pe" ]]; then
            common_options+=(
                model.role_prompt_tta_prompt_bank="configs/prompt_banks/role_functional_text_v2/${dataset}.json"
                model.role_prompt_tta_pe_diagnosis=True
                model.role_prompt_tta_selection_registry="configs/experiments/role_text_selections_v1.json"
            )
        elif [[ "${experiment_mode}" == "boundary-replay" ]]; then
            common_options+=(
                model.role_prompt_tta_prompt_bank="configs/prompt_banks/role_functional_text_v2/${dataset}.json"
                model.role_prompt_tta_boundary_replay_diagnosis=True
                model.role_prompt_tta_selection_registry="configs/experiments/role_text_selections_v1.json"
            )
        elif [[ "${experiment_mode}" == "class-role-alignment" ]]; then
            common_options+=(
                model.role_prompt_tta_prompt_bank="configs/prompt_banks/role_functional_text_v2/${dataset}.json"
                model.role_prompt_tta_class_role_alignment=True
                model.role_prompt_tta_selection_registry="configs/experiments/role_text_selections_v1.json"
            )
        elif [[ "${experiment_mode}" == "pi" ]]; then
            read -r presence_slot instance_slot <<< "$(pi_slots_for "${dataset}")"
            common_options+=(
                model.role_prompt_tta_pi_diagnosis=True
                model.role_prompt_tta_pi_presence_slot="${presence_slot}"
                model.role_prompt_tta_pi_instance_slot="${instance_slot}"
            )
        elif [[ "${experiment_mode}" == "completion" ]]; then
            common_options+=(
                model.role_prompt_tta_prompt_bank="configs/prompt_banks/role_functional_text_v2/${dataset}.json"
                model.role_prompt_tta_completion_diagnosis=True
            )
        fi
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
    local experiment_mode="${1:-screen}"
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
    local summarizer="tools/summarize_role_functional_text_screen.py"
    if [[ "${experiment_mode}" == "pe" ]]; then
        summarizer="tools/summarize_pe_role_evidence.py"
    elif [[ "${experiment_mode}" == "boundary-replay" ]]; then
        summarizer="tools/summarize_boundary_replay.py"
    elif [[ "${experiment_mode}" == "class-role-alignment" ]]; then
        summarizer="tools/summarize_class_role_alignment.py"
    elif [[ "${experiment_mode}" == "pi" ]]; then
        summarizer="tools/summarize_pi_role_compatibility.py"
    fi
    "${PYTHON_BIN}" "${summarizer}" \
        --inputs "${inputs[@]}" \
        --out-dir "${ROOT}/summary" \
        --expected-datasets \
        udd5 vdd vaihingen potsdam openearthmap loveda \
        --allow-incomplete \
        --integrity-tolerance "${INTEGRITY_TOLERANCE}"
}

case "${MODE}" in
    preflight)
        "${PYTHON_BIN}" tools/preflight_role_functional_text_screen.py \
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
    pi-smoke)
        collect smoke pi
        summarize pi
        ;;
    pi-collect-all)
        collect full pi
        ;;
    pi-summarize)
        summarize pi
        ;;
    pi-all)
        collect full pi
        summarize pi
        ;;
    completion-smoke)
        collect smoke completion
        summarize completion
        ;;
    completion-collect-all)
        collect full completion
        ;;
    completion-summarize)
        summarize completion
        ;;
    completion-all)
        collect full completion
        summarize completion
        ;;
    pe-smoke)
        collect smoke pe
        summarize pe
        ;;
    pe-collect-all)
        collect full pe
        ;;
    pe-summarize)
        summarize pe
        ;;
    pe-all)
        collect full pe
        summarize pe
        ;;
    boundary-replay-smoke)
        collect smoke boundary-replay
        summarize boundary-replay
        ;;
    boundary-replay-collect-all)
        collect full boundary-replay
        ;;
    boundary-replay-summarize)
        summarize boundary-replay
        ;;
    boundary-replay-all)
        collect full boundary-replay
        summarize boundary-replay
        ;;
    class-role-alignment-smoke)
        collect smoke class-role-alignment
        summarize class-role-alignment
        ;;
    class-role-alignment-collect-all)
        collect full class-role-alignment
        ;;
    class-role-alignment-summarize)
        summarize class-role-alignment
        ;;
    class-role-alignment-all)
        collect full class-role-alignment
        summarize class-role-alignment
        ;;
    *)
        echo "Usage: bash $0 {preflight|smoke|collect-all|summarize|all|pi-smoke|pi-collect-all|pi-summarize|pi-all|completion-smoke|completion-collect-all|completion-summarize|completion-all|pe-smoke|pe-collect-all|pe-summarize|pe-all|boundary-replay-smoke|boundary-replay-collect-all|boundary-replay-summarize|boundary-replay-all|class-role-alignment-smoke|class-role-alignment-collect-all|class-role-alignment-summarize|class-role-alignment-all}" >&2
        exit 2
        ;;
esac
