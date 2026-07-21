#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/sam3_geometry_requery_v1}"
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
    local port="${PORT:-30931}"
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
            model.seed_dataset_name="${dataset}" \
            model.dump_sam3_geometry_requery_stats=True \
            model.sam3_geometry_requery_stats_path="${out_dir}/sam3_geometry_requery.jsonl" \
            model.sam3_geometry_requery_topk="${GEO_TOPK:-3}" \
            model.sam3_geometry_requery_max_pairs="${GEO_MAX_PAIRS:-4}" \
            model.sam3_geometry_requery_min_pair_pixels="${GEO_MIN_PAIR_PIXELS:-64}" \
            model.sam3_geometry_requery_min_region_pixels="${GEO_MIN_REGION_PIXELS:-32}" \
            model.sam3_geometry_requery_score_thd="${GEO_SCORE_THD:--1.0}" \
            model.sam3_geometry_requery_box_modes="${GEO_BOX_MODES:-target_pos,pred_neg,target_pos_pred_neg}" \
            model.sam3_geometry_requery_prompt_mode="${GEO_PROMPT_MODE:-best_query}" \
            model.sam3_geometry_requery_max_side="${GEO_MAX_SIDE:-1024}" \
            model.sam3_geometry_requery_min_box_size="${GEO_MIN_BOX_SIZE:-4}" \
            model.sam3_geometry_requery_context_scale="${GEO_CONTEXT_SCALE:-1.0}" \
            model.sam3_geometry_requery_include_bg="${GEO_INCLUDE_BG:-True}" \
            model.sam3_geometry_requery_empty_cache="${GEO_EMPTY_CACHE:-True}"
}

run_parallel() {
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-30940}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" \
            > "${ROOT}/launch_logs/geometry_requery_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" \
                > "${ROOT}/launch_logs/geometry_requery_${dataset_b}.log" 2>&1 &
            pid_b=$!
            index=$((index + 1))
        fi
        wait "${pid_a}"
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}"
        fi
    done
}

summarize_mode() {
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_sam3_geometry_requery_stats.py \
        --inputs "${ROOT}/diagnostic"/*/sam3_geometry_requery_rank*.jsonl \
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
        summarize_mode
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_sam3_geometry_requery_v1.sh eval DATASET
  bash tools/run_sam3_geometry_requery_v1.sh diagnostic-small
  bash tools/run_sam3_geometry_requery_v1.sh diagnostic-suite
  bash tools/run_sam3_geometry_requery_v1.sh summarize

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda, isaid

Environment overrides:
  ROOT, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  GEO_TOPK, GEO_MAX_PAIRS, GEO_MIN_PAIR_PIXELS, GEO_MIN_REGION_PIXELS,
  GEO_SCORE_THD, GEO_BOX_MODES, GEO_PROMPT_MODE, GEO_MAX_SIDE,
  GEO_MIN_BOX_SIZE, GEO_CONTEXT_SCALE, GEO_INCLUDE_BG, GEO_EMPTY_CACHE

Default diagnostic cost:
  top-k=3, max_pairs=4, modes=target_pos,pred_neg,target_pos_pred_neg,
  max_side=1024. Increase GEO_MAX_PAIRS or GEO_BOX_MODES only after the
  small-suite result shows a useful signal.
EOF
        exit 2
        ;;
esac
