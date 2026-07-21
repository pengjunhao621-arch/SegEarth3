#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/structure_aware_recalibration_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

DATASETS_SMALL=(udd5 vdd vaihingen)
DATASETS_MAIN=(udd5 vdd vaihingen potsdam openearthmap loveda)
VARIANTS=(final_context sem_inst_context sem_inst_max_context)

config_for() {
    case "$1" in
        udd5) echo "configs/cfg_udd5.py" ;;
        vdd) echo "configs/cfg_vdd.py" ;;
        vaihingen) echo "configs/cfg_vaihingen.py" ;;
        potsdam) echo "configs/cfg_potsdam.py" ;;
        openearthmap) echo "configs/cfg_openearthmap.py" ;;
        loveda) echo "configs/cfg_loveda.py" ;;
        isaid) echo "configs/cfg_iSAID.py" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

variant_settings() {
    local variant="$1"
    case "${variant}" in
        final_context)
            SAR_VARIANT="${SAR_VARIANT:-final_context}"
            SAR_ALPHA="${SAR_ALPHA:-0.20}"
            SAR_KERNEL="${SAR_KERNEL:-9}"
            ;;
        sem_inst_context)
            SAR_VARIANT="${SAR_VARIANT:-semantic_instance_context}"
            SAR_ALPHA="${SAR_ALPHA:-0.20}"
            SAR_KERNEL="${SAR_KERNEL:-9}"
            ;;
        sem_inst_max_context)
            SAR_VARIANT="${SAR_VARIANT:-semantic_instance_max_context}"
            SAR_ALPHA="${SAR_ALPHA:-0.15}"
            SAR_KERNEL="${SAR_KERNEL:-7}"
            ;;
        *)
            echo "Unknown variant: ${variant}" >&2
            exit 2
            ;;
    esac
}

run_eval() {
    local dataset="$1"
    local apply="${2:-false}"
    local suite_variant="${SAR_SUITE_VARIANT:-final_context}"
    variant_settings "${suite_variant}"

    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-31011}"
    local config
    config="$(config_for "${dataset}")"

    local mode_dir="diagnostic"
    local apply_cfg="False"
    if [[ "${apply}" == "true" || "${apply}" == "True" || "${apply}" == "1" ]]; then
        mode_dir="apply"
        apply_cfg="True"
    fi
    local out_dir="${ROOT}/${suite_variant}/${mode_dir}/${dataset}"
    mkdir -p "${out_dir}"

    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    torchrun \
        --nproc_per_node=2 \
        --master_port="${port}" \
        eval.py "${config}" \
        --launcher pytorch \
        --work-dir "${out_dir}" \
        --result-file "${out_dir}/results.xlsx" \
        --cfg-options \
            model.seed_dataset_name="${dataset}" \
            model.use_structure_aware_recalibration="${apply_cfg}" \
            model.dump_structure_aware_recalibration_stats=True \
            model.structure_aware_recalibration_stats_path="${out_dir}/structure_aware_recalibration.jsonl" \
            model.structure_recalibration_variant="${SAR_VARIANT}" \
            model.structure_recalibration_kernel="${SAR_KERNEL}" \
            model.structure_recalibration_alpha="${SAR_ALPHA}" \
            model.structure_recalibration_min_context_gain="${SAR_MIN_GAIN:-0.01}" \
            model.structure_recalibration_min_context_score="${SAR_MIN_SCORE:-0.0}" \
            model.structure_recalibration_max_base_margin="${SAR_MAX_BASE_MARGIN:-1.0}" \
            model.structure_recalibration_protect_bg="${SAR_PROTECT_BG:-True}" \
            model.structure_recalibration_include_roles="${SAR_INCLUDE_ROLES:-all}" \
            model.structure_recalibration_exclude_roles="${SAR_EXCLUDE_ROLES:-catch_all}" \
            model.structure_validity_topk="${SAR_TOPK:-3}" \
            model.structure_validity_high_conf_margin="${SAR_HIGH_CONF_MARGIN:-0.20}" \
            model.structure_validity_min_pair_pixels="${SAR_MIN_PAIR_PIXELS:-32}"
}

run_parallel() {
    local apply="$1"
    shift
    local datasets=("$@")
    local suite_variant="${SAR_SUITE_VARIANT:-final_context}"
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-31020}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" SAR_SUITE_VARIANT="${suite_variant}" \
            GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" "${apply}" \
            > "${ROOT}/launch_logs/${suite_variant}_${apply}_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))

        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" SAR_SUITE_VARIANT="${suite_variant}" \
                GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" "${apply}" \
                > "${ROOT}/launch_logs/${suite_variant}_${apply}_${dataset_b}.log" 2>&1 &
            pid_b=$!
            index=$((index + 1))
        fi

        wait "${pid_a}"
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}"
        fi
    done
}

summarize() {
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_structure_aware_recalibration.py \
        --inputs "${ROOT}"/*/*/*/structure_aware_recalibration_rank*.jsonl \
        --out-dir "${ROOT}/summary"
}

case "${MODE}" in
    eval)
        run_eval "${2:?dataset required}" "${3:-false}"
        ;;
    diagnostic-small)
        run_parallel false "${DATASETS_SMALL[@]}"
        ;;
    apply-small)
        run_parallel true "${DATASETS_SMALL[@]}"
        ;;
    diagnostic-suite)
        run_parallel false "${DATASETS_MAIN[@]}"
        ;;
    apply-suite)
        run_parallel true "${DATASETS_MAIN[@]}"
        ;;
    ablation-small)
        for variant in "${VARIANTS[@]}"; do
            SAR_SUITE_VARIANT="${variant}" bash "$0" apply-small
        done
        ;;
    ablation-suite)
        for variant in "${VARIANTS[@]}"; do
            SAR_SUITE_VARIANT="${variant}" bash "$0" apply-suite
        done
        ;;
    summarize)
        summarize
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_structure_aware_recalibration_v1.sh eval DATASET [true|false]
  bash tools/run_structure_aware_recalibration_v1.sh diagnostic-small
  bash tools/run_structure_aware_recalibration_v1.sh apply-small
  bash tools/run_structure_aware_recalibration_v1.sh diagnostic-suite
  bash tools/run_structure_aware_recalibration_v1.sh apply-suite
  bash tools/run_structure_aware_recalibration_v1.sh ablation-small
  bash tools/run_structure_aware_recalibration_v1.sh ablation-suite
  bash tools/run_structure_aware_recalibration_v1.sh summarize

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda, isaid

Suite variants:
  final_context:        local context of baseline final evidence
  sem_inst_context:     local context of mean semantic+instance evidence
  sem_inst_max_context: local context of max semantic/instance evidence

Environment overrides:
  ROOT, PYTHON_BIN, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE,
  SAR_SUITE_VARIANT, SAR_VARIANT, SAR_ALPHA, SAR_KERNEL,
  SAR_MIN_GAIN, SAR_MIN_SCORE, SAR_MAX_BASE_MARGIN, SAR_PROTECT_BG,
  SAR_INCLUDE_ROLES, SAR_EXCLUDE_ROLES, SAR_TOPK, SAR_HIGH_CONF_MARGIN,
  SAR_MIN_PAIR_PIXELS

Recommended first run:
  ROOT=logs/structure_aware_recalibration_v1 \
  PORT_BASE=31020 \
  bash tools/run_structure_aware_recalibration_v1.sh ablation-small

Outputs:
  VARIANT/apply/DATASET/results.txt
  VARIANT/apply/DATASET/structure_aware_recalibration_rank*.jsonl
  summary/structure_dataset_summary.csv
  summary/structure_role_summary.csv
  summary/structure_class_summary.csv
  summary/structure_pair_summary.csv
EOF
        ;;
esac
