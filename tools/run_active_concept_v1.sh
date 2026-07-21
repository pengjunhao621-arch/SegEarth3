#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/active_concept_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_active_concept.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_active_concept.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_active_concept.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_active_concept.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_active_concept.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_active_concept.py" ;;
        isaid) echo "configs/experiments/cfg_isaid_active_concept.py" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

run_eval() {
    local dataset="$1"
    local apply_variant="${2:-baseline}"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29731}"
    local config
    config="$(config_for "${dataset}")"
    local out_dir="${ROOT}/${apply_variant}/${dataset}"
    mkdir -p "${out_dir}"
    local use_pruning=False
    if [[ "${apply_variant}" != "baseline" ]]; then
        use_pruning=True
    fi
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
            model.dump_active_concept_stats=True \
            model.active_concept_stats_path="${out_dir}/active_concept.jsonl" \
            model.use_active_concept_pruning="${use_pruning}" \
            model.active_concept_apply_variant="${apply_variant}" \
            model.active_concept_variants="${ACTIVE_VARIANTS:-gt_oracle,presence_only,multi_evidence,conflict_preserving}" \
            model.active_concept_presence_thd="${ACTIVE_PRESENCE_THD:-0.20}" \
            model.active_concept_area_thd="${ACTIVE_AREA_THD:-0.001}" \
            model.active_concept_pred_area_thd="${ACTIVE_PRED_AREA_THD:-0.0005}" \
            model.active_concept_topk="${ACTIVE_TOPK:-3}" \
            model.active_concept_topk_area_thd="${ACTIVE_TOPK_AREA_THD:-0.0005}" \
            model.active_concept_conflict_mode="${ACTIVE_CONFLICT_MODE:-role_topk}"
}

run_parallel() {
    local apply_variant="$1"
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-29740}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS=0,1 PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" "${apply_variant}" \
            > "${ROOT}/launch_logs/${apply_variant}_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS=2,3 PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" "${apply_variant}" \
                > "${ROOT}/launch_logs/${apply_variant}_${dataset_b}.log" 2>&1 &
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
    "${PYTHON_BIN}" tools/summarize_active_concept_stats.py \
        --inputs "${ROOT}"/*/*/active_concept_rank*.jsonl \
        --output-dir "${ROOT}/summary"
}

case "${MODE}" in
    eval)
        run_eval "${2:?dataset required}" "${3:-baseline}"
        ;;
    diagnostic-suite)
        run_parallel baseline \
            udd5 vdd vaihingen potsdam openearthmap loveda
        ;;
    diagnostic-all)
        run_parallel baseline \
            udd5 vdd vaihingen potsdam openearthmap loveda isaid
        ;;
    prune-suite)
        run_parallel "${2:?variant required: presence_only|multi_evidence|conflict_preserving}" \
            udd5 vdd vaihingen potsdam openearthmap loveda
        ;;
    summarize)
        summarize
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_active_concept_v1.sh eval DATASET [baseline|presence_only|multi_evidence|conflict_preserving]
  bash tools/run_active_concept_v1.sh diagnostic-suite
  bash tools/run_active_concept_v1.sh diagnostic-all
  bash tools/run_active_concept_v1.sh prune-suite VARIANT
  bash tools/run_active_concept_v1.sh summarize

Environment overrides:
  ROOT, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  ACTIVE_VARIANTS, ACTIVE_PRESENCE_THD, ACTIVE_AREA_THD,
  ACTIVE_PRED_AREA_THD, ACTIVE_TOPK, ACTIVE_TOPK_AREA_THD,
  ACTIVE_CONFLICT_MODE
EOF
        exit 2
        ;;
esac
