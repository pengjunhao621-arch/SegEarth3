#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/concept_specificity_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_concept_specificity.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_concept_specificity.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_concept_specificity.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_concept_specificity.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_concept_specificity.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_concept_specificity.py" ;;
        isaid) echo "configs/experiments/cfg_isaid_concept_specificity.py" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

run_eval() {
    local dataset="$1"
    local ranker="${2:-baseline}"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29831}"
    local config
    config="$(config_for "${dataset}")"
    local out_dir="${ROOT}/${ranker}/${dataset}"
    mkdir -p "${out_dir}"
    local use_pruning=False
    if [[ "${ranker}" != "baseline" ]]; then
        use_pruning=True
    fi

    local cfg_options=(
        model.dump_concept_specificity_stats=True
        model.concept_specificity_stats_path="${out_dir}/concept_specificity.jsonl"
        model.use_concept_specificity_pruning="${use_pruning}"
        model.concept_specificity_apply_ranker="${ranker}"
        model.concept_specificity_max_side="${CONCEPT_MAX_SIDE:-512}"
        model.concept_specificity_keep_bg="${CONCEPT_KEEP_BG:-True}"
    )
    if [[ -n "${CONCEPT_SCORE_THD:-}" ]]; then
        cfg_options+=(
            model.concept_specificity_score_thd="${CONCEPT_SCORE_THD}"
        )
    fi
    if [[ -n "${CONCEPT_PRUNE_ACTIVE_COUNT:-}" ]]; then
        cfg_options+=(
            model.concept_specificity_prune_active_count="${CONCEPT_PRUNE_ACTIVE_COUNT}"
        )
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
        --cfg-options "${cfg_options[@]}"
}

run_parallel() {
    local ranker="$1"
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-29840}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS=0,1 PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" "${ranker}" \
            > "${ROOT}/launch_logs/${ranker}_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS=2,3 PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" "${ranker}" \
                > "${ROOT}/launch_logs/${ranker}_${dataset_b}.log" 2>&1 &
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
    "${PYTHON_BIN}" tools/summarize_concept_specificity_stats.py \
        --inputs "${ROOT}"/*/*/concept_specificity_rank*.jsonl \
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
        run_parallel "${2:?ranker required}" \
            udd5 vdd vaihingen potsdam openearthmap loveda
        ;;
    prune-all)
        run_parallel "${2:?ranker required}" \
            udd5 vdd vaihingen potsdam openearthmap loveda isaid
        ;;
    summarize)
        summarize
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_concept_specificity_v1.sh eval DATASET [baseline|presence_score|scene_contrast|region_specificity|head_prompt_specificity|specificity_combo|presence_specificity_combo]
  bash tools/run_concept_specificity_v1.sh diagnostic-suite
  bash tools/run_concept_specificity_v1.sh diagnostic-all
  bash tools/run_concept_specificity_v1.sh prune-suite RANKER
  bash tools/run_concept_specificity_v1.sh prune-all RANKER
  bash tools/run_concept_specificity_v1.sh summarize

Environment overrides:
  ROOT, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  CONCEPT_MAX_SIDE, CONCEPT_SCORE_THD, CONCEPT_KEEP_BG,
  CONCEPT_PRUNE_ACTIVE_COUNT
EOF
        exit 2
        ;;
esac
