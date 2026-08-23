#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${ROOT:-logs/joint_role_view_final_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0,1}"
NPROC="${NPROC:-2}"
BENCH_GPU="${BENCH_GPU:-0}"
DATASETS="${DATASETS:-udd5 vdd vaihingen potsdam openearthmap loveda}"
BENCH_PROFILES="${BENCH_PROFILES:-baseline role_only_residual view_only joint_residual joint_direct fast_joint_residual fast_joint_direct}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-1}"
BENCHMARK_SAMPLES="${BENCHMARK_SAMPLES:-20}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-2}"
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

run_distributed() {
    local config="$1" out_dir="$2"
    shift 2
    if [[ "${NPROC}" -gt 1 ]]; then
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            torchrun --standalone --nproc_per_node="${NPROC}" \
            eval.py "${config}" --launcher pytorch \
            --work-dir "${out_dir}/work" \
            --result-file "${out_dir}/results.xlsx" "$@"
    else
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            "${PYTHON_BIN}" eval.py "${config}" \
            --work-dir "${out_dir}/work" \
            --result-file "${out_dir}/results.xlsx" "$@"
    fi
}

preflight() {
    "${PYTHON_BIN}" -m unittest discover \
        -s tests -p 'test_joint_role_view_final.py'
    "${PYTHON_BIN}" tools/preflight_role_functional_text_screen.py \
        --check-runtime-assets
    if [[ "${SKIP_TILED_PREFLIGHT}" == "1" ]]; then
        return
    fi
    local dirs=()
    [[ " ${DATASETS} " == *" vaihingen "* ]] && dirs+=("${VAIHINGEN_IMG_DIR}")
    [[ " ${DATASETS} " == *" potsdam "* ]] && dirs+=("${POTSDAM_IMG_DIR}")
    if [[ "${#dirs[@]}" -gt 0 ]]; then
        mkdir -p "${ROOT}/preflight"
        "${PYTHON_BIN}" tools/verify_tiled_context.py "${dirs[@]}" \
            --output "${ROOT}/preflight/tiled_rgb_overlap.json"
    fi
}

audit() {
    local scope="$1" dataset config out_dir existing
    for dataset in ${DATASETS}; do
        config="$(config_for "${dataset}")"
        out_dir="${ROOT}/audit/${dataset}"
        mkdir -p "${out_dir}"
        existing="$(find "${out_dir}" -maxdepth 1 \
            -name 'screen.rank*.jsonl' -print -quit)"
        if [[ -n "${existing}" ]]; then
            echo "Refusing to append duplicate audit records: ${existing}" >&2
            exit 2
        fi
        echo "[joint-role-view-final-v1:audit] ${dataset} -> ${out_dir}"
        options=(
            model.role_prompt_tta_prompt_bank="configs/prompt_banks/role_functional_text_v2/${dataset}.json"
            model.role_prompt_tta_selection_registry="configs/experiments/role_text_selections_v1.json"
            model.role_prompt_tta_visual_field_diagnosis=True
            model.role_prompt_tta_visual_field_registry="configs/experiments/role_visual_field_v1.json"
            model.role_prompt_tta_visual_field_mode=joint_role_view_final
            model.role_prompt_tta_joint_final_registry="configs/experiments/joint_role_view_final_v1.json"
            model.role_prompt_tta_final_profile=audit
            model.role_prompt_tta_stats_path="${out_dir}/screen.jsonl"
            model.dump_role_prompt_tta_stats=True
            model.role_prompt_tta_integrity_tolerance="${INTEGRITY_TOLERANCE}"
            model.role_prompt_tta_save_npz=False
        )
        if [[ "${scope}" == "smoke" ]]; then
            run_distributed "${config}" "${out_dir}" --cfg-options \
                test_dataloader.dataset.indices="${SMOKE_SAMPLES}" \
                "${options[@]}"
        else
            run_distributed "${config}" "${out_dir}" \
                --cfg-options "${options[@]}"
        fi
    done
}

benchmark() {
    local scope="$1" dataset profile config out_dir samples warmup existing
    samples="${BENCHMARK_SAMPLES}"
    warmup="${BENCHMARK_WARMUP}"
    if [[ "${scope}" == "smoke" ]]; then
        samples="${SMOKE_SAMPLES}"
        warmup=0
    fi
    for dataset in ${DATASETS}; do
        config="$(config_for "${dataset}")"
        for profile in ${BENCH_PROFILES}; do
            out_dir="${ROOT}/benchmark/${dataset}/${profile}"
            mkdir -p "${out_dir}"
            existing="$(find "${out_dir}" -maxdepth 1 \
                -name 'runtime.rank*.jsonl' -print -quit)"
            if [[ -n "${existing}" ]]; then
                echo "Refusing to append benchmark records: ${existing}" >&2
                exit 2
            fi
            echo "[joint-role-view-final-v1:benchmark] ${dataset}/${profile}"
            common=(
                test_dataloader.dataset.indices="${samples}"
            )
            if [[ "${profile}" == "baseline" ]]; then
                options=(
                    model.use_role_prompt_tta=False
                    model.dump_role_prompt_tta_stats=False
                    model.role_prompt_tta_visual_field_diagnosis=False
                )
            else
                options=(
                    model.role_prompt_tta_prompt_bank="configs/prompt_banks/role_functional_text_v2/${dataset}.json"
                    model.role_prompt_tta_selection_registry="configs/experiments/role_text_selections_v1.json"
                    model.role_prompt_tta_visual_field_diagnosis=True
                    model.role_prompt_tta_visual_field_registry="configs/experiments/role_visual_field_v1.json"
                    model.role_prompt_tta_visual_field_mode=joint_role_view_final
                    model.role_prompt_tta_joint_final_registry="configs/experiments/joint_role_view_final_v1.json"
                    model.role_prompt_tta_final_profile="${profile}"
                    model.role_prompt_tta_stats_path="${out_dir}/unused.jsonl"
                    model.dump_role_prompt_tta_stats=False
                    model.role_prompt_tta_integrity_tolerance="${INTEGRITY_TOLERANCE}"
                    model.role_prompt_tta_save_npz=False
                )
            fi
            CUDA_VISIBLE_DEVICES="${BENCH_GPU}" \
                "${PYTHON_BIN}" eval.py "${config}" \
                --work-dir "${out_dir}/work" \
                --result-file "${out_dir}/results.xlsx" \
                --benchmark-jsonl "${out_dir}/runtime.jsonl" \
                --benchmark-profile "${profile}" \
                --benchmark-dataset "${dataset}" \
                --benchmark-warmup "${warmup}" \
                --cfg-options "${common[@]}" "${options[@]}"
        done
    done
}

summarize() {
    local audit_inputs=() benchmark_inputs=() path
    while IFS= read -r path; do audit_inputs+=("${path}"); done < <(
        find "${ROOT}/audit" -name 'screen.rank*.jsonl' -type f 2>/dev/null | sort)
    while IFS= read -r path; do benchmark_inputs+=("${path}"); done < <(
        find "${ROOT}/benchmark" -name 'runtime.rank*.jsonl' -type f 2>/dev/null | sort)
    if [[ "${#audit_inputs[@]}" -eq 0 && "${#benchmark_inputs[@]}" -eq 0 ]]; then
        echo "No final audit or benchmark records under ${ROOT}." >&2
        exit 2
    fi
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_joint_role_view_final.py \
        --audit-inputs "${audit_inputs[@]}" \
        --benchmark-inputs "${benchmark_inputs[@]}" \
        --out-dir "${ROOT}/summary"
}

case "${MODE}" in
    preflight) preflight ;;
    audit-smoke) preflight; audit smoke; summarize ;;
    audit-all) preflight; audit full; summarize ;;
    benchmark-smoke) preflight; benchmark smoke; summarize ;;
    benchmark) preflight; benchmark full; summarize ;;
    summarize) summarize ;;
    smoke) preflight; audit smoke; benchmark smoke; summarize ;;
    complete) preflight; audit full; benchmark full; summarize ;;
    *)
        echo "Usage: $0 {preflight|audit-smoke|audit-all|benchmark-smoke|benchmark|summarize|smoke|complete}" >&2
        exit 2
        ;;
esac
