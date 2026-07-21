#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/self_prompted_concept_verification_v1}"
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
    local port="${PORT:-30631}"
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
            model.dump_self_prompted_concept_verification_stats=True \
            model.self_prompted_concept_verification_stats_path="${out_dir}/self_prompted_concept_verification.jsonl" \
            model.self_prompted_concept_verification_topk="${SPCV_TOPK:-3}" \
            model.self_prompted_concept_verification_max_pairs="${SPCV_MAX_PAIRS:-3}" \
            model.self_prompted_concept_verification_min_pair_pixels="${SPCV_MIN_PAIR_PIXELS:-96}" \
            model.self_prompted_concept_verification_min_region_pixels="${SPCV_MIN_REGION_PIXELS:-32}" \
            model.self_prompted_concept_verification_score_thd="${SPCV_SCORE_THD:--1.0}" \
            model.self_prompted_concept_verification_region_fraction="${SPCV_REGION_FRACTION:-0.35}" \
            model.self_prompted_concept_verification_max_base_gap="${SPCV_MAX_BASE_GAP:-1.0}" \
            model.self_prompted_concept_verification_min_candidate_score="${SPCV_MIN_CANDIDATE_SCORE:--1.0}" \
            model.self_prompted_concept_verification_prompt_mode="${SPCV_PROMPT_MODE:-best_query}" \
            model.self_prompted_concept_verification_max_side="${SPCV_MAX_SIDE:-1024}" \
            model.self_prompted_concept_verification_min_box_size="${SPCV_MIN_BOX_SIZE:-4}" \
            model.self_prompted_concept_verification_context_scale="${SPCV_CONTEXT_SCALE:-1.05}" \
            model.self_prompted_concept_verification_include_bg="${SPCV_INCLUDE_BG:-True}" \
            model.self_prompted_concept_verification_empty_cache="${SPCV_EMPTY_CACHE:-True}"
}

run_parallel() {
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-30640}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" \
            > "${ROOT}/launch_logs/spcv_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))

        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" \
                > "${ROOT}/launch_logs/spcv_${dataset_b}.log" 2>&1 &
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
    "${PYTHON_BIN}" tools/summarize_self_prompted_concept_verification.py \
        --inputs "${ROOT}"/diagnostic/*/self_prompted_concept_verification_rank*.jsonl \
        --out-dir "${ROOT}/summary" \
        --thresholds="${SPCV_THRESHOLDS:--0.30,-0.20,-0.10,-0.05,0.00,0.05,0.10,0.20,0.30}"
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
  bash tools/run_self_prompted_concept_verification_v1.sh eval DATASET
  bash tools/run_self_prompted_concept_verification_v1.sh diagnostic-small
  bash tools/run_self_prompted_concept_verification_v1.sh diagnostic-suite
  bash tools/run_self_prompted_concept_verification_v1.sh summarize

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda, isaid

Environment overrides:
  ROOT, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  SPCV_TOPK, SPCV_MAX_PAIRS, SPCV_MIN_PAIR_PIXELS,
  SPCV_MIN_REGION_PIXELS, SPCV_SCORE_THD, SPCV_REGION_FRACTION,
  SPCV_MAX_BASE_GAP, SPCV_MIN_CANDIDATE_SCORE, SPCV_PROMPT_MODE,
  SPCV_MAX_SIDE, SPCV_MIN_BOX_SIZE, SPCV_CONTEXT_SCALE,
  SPCV_INCLUDE_BG, SPCV_EMPTY_CACHE, SPCV_THRESHOLDS

Default diagnostic cost:
  top-k=3, max_pairs=3, max_side=1024. This re-queries two concepts for each
  selected top-k conflict region, so it is slower than a normal baseline run.

Outputs:
  diagnostic/DATASET/self_prompted_concept_verification_rank*.jsonl
  summary/spcv_dataset_summary.csv
  summary/spcv_role_pair_summary.csv
  summary/spcv_pair_summary.csv
  summary/spcv_feature_separability.csv
  summary/spcv_switch_curve.csv
  summary/spcv_pair_rows.csv
EOF
        exit 2
        ;;
esac
