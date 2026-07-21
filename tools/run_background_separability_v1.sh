#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/background_separability_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

DATASETS_SMALL=(udd5 vdd vaihingen)
DATASETS_MAIN=(udd5 vdd vaihingen potsdam openearthmap loveda)

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

run_eval() {
    local dataset="$1"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-30731}"
    local config
    config="$(config_for "${dataset}")"
    local out_dir="${ROOT}/${dataset}"
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
            model.dump_background_separability_stats=True \
            model.background_separability_stats_path="${out_dir}/background_separability.jsonl" \
            model.background_separability_score_thresholds="${BG_SEP_SCORE_THDS:-0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.30,0.50}" \
            model.background_separability_margin_thresholds="${BG_SEP_MARGIN_THDS:--0.10,-0.05,0.00,0.02,0.05,0.10,0.20,0.40}" \
            model.background_separability_reliability_thresholds="${BG_SEP_REL_THDS:--0.20,-0.10,-0.05,0.00,0.02,0.05,0.10,0.20,0.40}" \
            model.background_separability_local_thresholds="${BG_SEP_LOCAL_THDS:-0.40,0.50,0.60,0.70,0.80,0.90}"
}

run_parallel() {
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-30740}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" \
            > "${ROOT}/launch_logs/bgsep_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" \
                > "${ROOT}/launch_logs/bgsep_${dataset_b}.log" 2>&1 &
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
    "${PYTHON_BIN}" tools/summarize_background_separability.py \
        --inputs "${ROOT}"/*/background_separability_rank*.jsonl \
        --output-dir "${ROOT}/summary"
}

case "${MODE}" in
    eval)
        run_eval "${2:?dataset required}"
        ;;
    small)
        run_parallel "${DATASETS_SMALL[@]}"
        ;;
    suite)
        run_parallel "${DATASETS_MAIN[@]}"
        ;;
    summarize)
        summarize
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_background_separability_v1.sh eval DATASET
  bash tools/run_background_separability_v1.sh small
  bash tools/run_background_separability_v1.sh suite
  bash tools/run_background_separability_v1.sh summarize

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda, isaid

Environment overrides:
  ROOT, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  BG_SEP_SCORE_THDS, BG_SEP_MARGIN_THDS, BG_SEP_REL_THDS, BG_SEP_LOCAL_THDS
EOF
        exit 2
        ;;
esac
