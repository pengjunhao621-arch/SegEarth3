#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/scene_common_bias_v2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

DATASETS_SMALL=(udd5 vdd vaihingen)
DATASETS_MAIN=(udd5 vdd vaihingen potsdam openearthmap loveda)

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_scene_common_bias.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_scene_common_bias.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_scene_common_bias.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_scene_common_bias.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_scene_common_bias.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_scene_common_bias.py" ;;
        isaid) echo "configs/experiments/cfg_isaid_scene_common_bias.py" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

run_eval() {
    local dataset="$1"
    local branch="${2:-diagnostic}"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29931}"
    local config
    config="$(config_for "${dataset}")"
    local out_dir="${ROOT}/${branch}/${dataset}"
    mkdir -p "${out_dir}"

    local dump_scene=False
    local dump_residual=False
    if [[ "${branch}" == "diagnostic" ]]; then
        dump_scene=True
    elif [[ "${branch}" == "residual" ]]; then
        dump_residual=True
    elif [[ "${branch}" == "both" ]]; then
        dump_scene=True
        dump_residual=True
    else
        echo "Unknown branch: ${branch}" >&2
        exit 2
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
            model.dump_scene_common_bias_stats="${dump_scene}" \
            model.scene_common_bias_stats_path="${out_dir}/scene_common_bias.jsonl" \
            model.dump_candidate_residual_miou_stats="${dump_residual}" \
            model.candidate_residual_miou_stats_path="${out_dir}/candidate_residual_miou.jsonl" \
            model.internal_selection_sources="${INTERNAL_SOURCES:-final,semantic,instance,pe_all,vision}" \
            model.candidate_internal_topk="${BIAS_TOPK:-3}" \
            model.candidate_internal_dump_matched_stats=True \
            model.candidate_internal_matched_families="${BIAS_MATCHED_FAMILIES:-all}" \
            model.scene_common_layers="${SCENE_COMMON_LAYERS:-0,2}" \
            model.scene_common_variants="${SCENE_COMMON_VARIANTS:-global,robust,class_balanced,local,shared,controls}" \
            model.scene_common_strengths="${SCENE_COMMON_STRENGTHS:-0.25,0.50,0.75,1.00}" \
            model.scene_common_local_kernels="${SCENE_COMMON_LOCAL_KERNELS:-7,15}" \
            model.scene_common_compute_spectrum="${SCENE_COMMON_SPECTRUM:-True}" \
            model.scene_common_max_side="${SCENE_COMMON_MAX_SIDE:-256}" \
            model.candidate_residual_layers="${CANDIDATE_RESIDUAL_LAYERS:-0,2}" \
            model.candidate_residual_variants="${CANDIDATE_RESIDUAL_VARIANTS:-global,class_balanced,local,shared,controls}" \
            model.candidate_residual_strengths="${CANDIDATE_RESIDUAL_STRENGTHS:-0.00,0.25,0.50,0.75,1.00}" \
            model.candidate_residual_miou_scores="${CANDIDATE_RESIDUAL_MIOU_SCORES:-margin_slope,endpoint_margin_gain,candidate_gain,competitor_suppression}" \
            model.candidate_residual_miou_ranks="${CANDIDATE_RESIDUAL_MIOU_RANKS:-2}" \
            model.candidate_residual_miou_units="${CANDIDATE_RESIDUAL_MIOU_UNITS:-pixel,component}" \
            model.candidate_residual_miou_absolute_thresholds="${CANDIDATE_RESIDUAL_MIOU_THRESHOLDS:-0.00,0.02,0.05,0.10,0.20}" \
            model.candidate_residual_miou_coverages="${CANDIDATE_RESIDUAL_MIOU_COVERAGES:-0.001,0.0025,0.005,0.01,0.02}" \
            model.candidate_residual_miou_max_side="${CANDIDATE_RESIDUAL_MIOU_MAX_SIDE:-256}"
}

run_parallel() {
    local branch="$1"
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-29940}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" "${branch}" \
            > "${ROOT}/launch_logs/${branch}_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" "${branch}" \
                > "${ROOT}/launch_logs/${branch}_${dataset_b}.log" 2>&1 &
            pid_b=$!
            index=$((index + 1))
        fi
        wait "${pid_a}"
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}"
        fi
    done
}

summarize_scene() {
    mkdir -p "${ROOT}/summary/scene_common"
    "${PYTHON_BIN}" tools/summarize_scene_common_bias_stats.py \
        --inputs \
            "${ROOT}"/diagnostic/*/scene_common_bias_rank*.jsonl \
            "${ROOT}"/both/*/scene_common_bias_rank*.jsonl \
        --out-dir "${ROOT}/summary/scene_common"
    if [[ -s "${ROOT}/summary/scene_common/scene_common_image_summary.csv" ]]; then
        mkdir -p "${ROOT}/summary/scene_common_lodo"
        "${PYTHON_BIN}" tools/evaluate_scene_common_loco_selector.py \
            --input "${ROOT}/summary/scene_common/scene_common_image_summary.csv" \
            --out-dir "${ROOT}/summary/scene_common_lodo"
    fi
}

summarize_residual() {
    mkdir -p "${ROOT}/summary/residual_miou"
    "${PYTHON_BIN}" tools/summarize_candidate_residual_miou.py \
        --inputs \
            "${ROOT}"/residual/*/candidate_residual_miou_rank*.jsonl \
            "${ROOT}"/both/*/candidate_residual_miou_rank*.jsonl \
        --out-dir "${ROOT}/summary/residual_miou"
}

case "${MODE}" in
    eval)
        run_eval "${2:?dataset required}" "${3:-diagnostic}"
        ;;
    diagnostic-small)
        run_parallel diagnostic "${DATASETS_SMALL[@]}"
        ;;
    diagnostic-suite)
        run_parallel diagnostic "${DATASETS_MAIN[@]}"
        ;;
    residual-small)
        run_parallel residual "${DATASETS_SMALL[@]}"
        ;;
    residual-suite)
        run_parallel residual "${DATASETS_MAIN[@]}"
        ;;
    both-small)
        run_parallel both "${DATASETS_SMALL[@]}"
        ;;
    summarize-scene)
        summarize_scene
        ;;
    summarize-residual)
        summarize_residual
        ;;
    summarize)
        summarize_scene
        summarize_residual
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_scene_common_bias_v2.sh eval DATASET [diagnostic|residual|both]
  bash tools/run_scene_common_bias_v2.sh diagnostic-small
  bash tools/run_scene_common_bias_v2.sh diagnostic-suite
  bash tools/run_scene_common_bias_v2.sh residual-small
  bash tools/run_scene_common_bias_v2.sh residual-suite
  bash tools/run_scene_common_bias_v2.sh both-small
  bash tools/run_scene_common_bias_v2.sh summarize-scene
  bash tools/run_scene_common_bias_v2.sh summarize-residual
  bash tools/run_scene_common_bias_v2.sh summarize

Environment overrides:
  ROOT, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  INTERNAL_SOURCES, BIAS_TOPK, BIAS_MATCHED_FAMILIES,
  SCENE_COMMON_LAYERS,
  SCENE_COMMON_VARIANTS, SCENE_COMMON_STRENGTHS,
  SCENE_COMMON_LOCAL_KERNELS, SCENE_COMMON_SPECTRUM,
  SCENE_COMMON_MAX_SIDE, CANDIDATE_RESIDUAL_LAYERS,
  CANDIDATE_RESIDUAL_VARIANTS, CANDIDATE_RESIDUAL_STRENGTHS,
  CANDIDATE_RESIDUAL_MIOU_SCORES, CANDIDATE_RESIDUAL_MIOU_RANKS,
  CANDIDATE_RESIDUAL_MIOU_UNITS, CANDIDATE_RESIDUAL_MIOU_THRESHOLDS,
  CANDIDATE_RESIDUAL_MIOU_COVERAGES, CANDIDATE_RESIDUAL_MIOU_MAX_SIDE
EOF
        exit 2
        ;;
esac
