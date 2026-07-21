#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/ontology_readout_oracle_v1}"
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
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

run_eval() {
    local dataset="$1"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-30531}"
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
            model.dump_ontology_readout_oracle_stats=True \
            model.ontology_readout_oracle_stats_path="${out_dir}/ontology_readout_oracle.jsonl" \
            model.ontology_readout_oracle_sources="${READOUT_SOURCES:-semantic,instance,raw_mask,presence_gated,pe_layer0}" \
            model.ontology_readout_oracle_max_side="${READOUT_MAX_SIDE:-256}" \
            model.ontology_readout_oracle_topk="${READOUT_TOPK:-3}" \
            model.ontology_readout_oracle_min_pair_pixels="${READOUT_MIN_PAIR_PIXELS:-16}" \
            model.internal_selection_raw_topk="${READOUT_RAW_TOPK:-10}" \
            model.internal_selection_seed_rule="${READOUT_PE_SEED_RULE:-final_score_margin_sem_inst_final_agree_local_core}"
}

run_parallel() {
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-30540}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" \
            > "${ROOT}/launch_logs/readout_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))

        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" \
                > "${ROOT}/launch_logs/readout_${dataset_b}.log" 2>&1 &
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
    "${PYTHON_BIN}" tools/summarize_ontology_readout_oracle.py \
        --inputs "${ROOT}"/diagnostic/*/ontology_readout_oracle_rank*.jsonl \
        --out-dir "${ROOT}/summary"
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
  bash tools/run_ontology_readout_oracle_v1.sh eval DATASET
  bash tools/run_ontology_readout_oracle_v1.sh diagnostic-small
  bash tools/run_ontology_readout_oracle_v1.sh diagnostic-suite
  bash tools/run_ontology_readout_oracle_v1.sh summarize

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda

Environment overrides:
  ROOT, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  READOUT_SOURCES, READOUT_MAX_SIDE, READOUT_TOPK,
  READOUT_MIN_PAIR_PIXELS, READOUT_RAW_TOPK, READOUT_PE_SEED_RULE

Default readouts:
  semantic, instance, raw_mask, presence_gated, pe_layer0

Notes:
  This is prediction-preserving. It evaluates baseline predictions and only
  dumps diagnostic JSONL/CSV for oracle analysis.
EOF
        exit 2
        ;;
esac
