#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/prompt_region_identity_v1}"
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
    local port="${PORT:-30431}"
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
            model.dump_prompt_winner_attribution_stats=True \
            model.prompt_winner_attribution_stats_path="${out_dir}/prompt_winner.jsonl" \
            model.prompt_winner_max_side="${PROMPT_WINNER_MAX_SIDE:-256}" \
            model.prompt_winner_sources="${PROMPT_WINNER_SOURCES:-final,semantic,instance}" \
            model.prompt_winner_min_pair_pixels="${PROMPT_WINNER_MIN_PAIR_PIXELS:-32}" \
            model.dump_region_prompt_identity_stats=True \
            model.region_prompt_identity_stats_path="${out_dir}/region_identity.jsonl" \
            model.region_prompt_identity_max_side="${REGION_IDENTITY_MAX_SIDE:-256}" \
            model.region_prompt_identity_bin_thd="${REGION_IDENTITY_BIN_THD:-0.5}" \
            model.region_prompt_identity_min_pixels="${REGION_IDENTITY_MIN_PIXELS:-16}" \
            model.region_prompt_identity_group_iou="${REGION_IDENTITY_GROUP_IOU:-0.5}" \
            model.region_prompt_identity_group_containment="${REGION_IDENTITY_GROUP_CONTAINMENT:-0.75}" \
            model.region_prompt_identity_max_regions="${REGION_IDENTITY_MAX_REGIONS:-256}" \
            model.region_prompt_identity_ring_kernel="${REGION_IDENTITY_RING_KERNEL:-15}" \
            model.region_prompt_identity_min_pair_pixels="${REGION_IDENTITY_MIN_PAIR_PIXELS:-16}" \
            model.raw_mask_oracle_topk="${REGION_IDENTITY_RAW_TOPK:-8}" \
            model.raw_mask_oracle_bin_thd="${REGION_IDENTITY_BIN_THD:-0.5}" \
            model.raw_mask_oracle_min_pixels="${REGION_IDENTITY_MIN_PIXELS:-16}"
}

run_parallel() {
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-30440}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" \
            > "${ROOT}/launch_logs/prompt_region_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" \
                > "${ROOT}/launch_logs/prompt_region_${dataset_b}.log" 2>&1 &
            pid_b=$!
            index=$((index + 1))
        fi
        wait "${pid_a}"
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}"
        fi
    done
}

summarize_prompt() {
    mkdir -p "${ROOT}/summary/prompt_winner"
    "${PYTHON_BIN}" tools/summarize_prompt_winner_attribution_stats.py \
        --inputs "${ROOT}"/diagnostic/*/prompt_winner_rank*.jsonl \
        --output-dir "${ROOT}/summary/prompt_winner"
}

summarize_region() {
    mkdir -p "${ROOT}/summary/region_identity"
    "${PYTHON_BIN}" tools/summarize_region_prompt_identity_stats.py \
        --inputs "${ROOT}"/diagnostic/*/region_identity_rank*.jsonl \
        --output-dir "${ROOT}/summary/region_identity"
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
    summarize-prompt)
        summarize_prompt
        ;;
    summarize-region)
        summarize_region
        ;;
    summarize)
        summarize_prompt
        summarize_region
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_prompt_region_identity_v1.sh eval DATASET
  bash tools/run_prompt_region_identity_v1.sh diagnostic-small
  bash tools/run_prompt_region_identity_v1.sh diagnostic-suite
  bash tools/run_prompt_region_identity_v1.sh summarize
  bash tools/run_prompt_region_identity_v1.sh summarize-prompt
  bash tools/run_prompt_region_identity_v1.sh summarize-region

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda

Environment overrides:
  ROOT, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  PROMPT_WINNER_MAX_SIDE, PROMPT_WINNER_SOURCES,
  PROMPT_WINNER_MIN_PAIR_PIXELS,
  REGION_IDENTITY_MAX_SIDE, REGION_IDENTITY_RAW_TOPK,
  REGION_IDENTITY_BIN_THD, REGION_IDENTITY_MIN_PIXELS,
  REGION_IDENTITY_GROUP_IOU, REGION_IDENTITY_GROUP_CONTAINMENT,
  REGION_IDENTITY_MAX_REGIONS, REGION_IDENTITY_RING_KERNEL,
  REGION_IDENTITY_MIN_PAIR_PIXELS
EOF
        exit 2
        ;;
esac
