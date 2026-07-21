#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/local_active_set_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

DATASETS_SMALL=(udd5 vdd vaihingen)
DATASETS_MAIN=(udd5 vdd vaihingen potsdam openearthmap loveda)

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
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-30231}"
    local config
    config="$(config_for "${dataset}")"
    local out_dir="${ROOT}/diagnostic/${dataset}"
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
            model.dump_local_active_set_stats=True \
            model.local_active_set_stats_path="${out_dir}/local_active_set.jsonl" \
            model.local_active_max_side="${LOCAL_ACTIVE_MAX_SIDE:-256}" \
            model.local_active_windows="${LOCAL_ACTIVE_WINDOWS:-17,33,65}" \
            model.local_active_topk="${LOCAL_ACTIVE_TOPK:-3}" \
            model.local_active_area_thresholds="${LOCAL_ACTIVE_AREA_THRESHOLDS:-0.001,0.005,0.01,0.02,0.05}" \
            model.local_active_sources="${LOCAL_ACTIVE_SOURCES:-topk,final,semantic,instance,agreement}" \
            model.local_active_min_pair_pixels="${LOCAL_ACTIVE_MIN_PAIR_PIXELS:-32}"
}

run_parallel() {
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-30240}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" \
            > "${ROOT}/launch_logs/local_active_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" \
                > "${ROOT}/launch_logs/local_active_${dataset_b}.log" 2>&1 &
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
    "${PYTHON_BIN}" tools/summarize_local_active_set_stats.py \
        --inputs "${ROOT}"/diagnostic/*/local_active_set_rank*.jsonl \
        --output-dir "${ROOT}/summary"
}

case "${MODE}" in
    eval)
        run_eval "${2:?dataset required}"
        ;;
    diagnostic-small)
        run_parallel diagnostic "${DATASETS_SMALL[@]}"
        ;;
    diagnostic-suite)
        run_parallel diagnostic "${DATASETS_MAIN[@]}"
        ;;
    summarize)
        summarize
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_local_active_set_v1.sh eval DATASET
  bash tools/run_local_active_set_v1.sh diagnostic-small
  bash tools/run_local_active_set_v1.sh diagnostic-suite
  bash tools/run_local_active_set_v1.sh summarize

Environment overrides:
  ROOT, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  LOCAL_ACTIVE_MAX_SIDE, LOCAL_ACTIVE_WINDOWS, LOCAL_ACTIVE_TOPK,
  LOCAL_ACTIVE_AREA_THRESHOLDS, LOCAL_ACTIVE_SOURCES,
  LOCAL_ACTIVE_MIN_PAIR_PIXELS
EOF
        exit 2
        ;;
esac
