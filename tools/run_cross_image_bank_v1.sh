#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/cross_image_bank_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_cross_image_bank.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_cross_image_bank.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_cross_image_bank.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_cross_image_bank.py" ;;
        openearthmap) echo "configs/experiments/cfg_openearthmap_cross_image_bank.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_cross_image_bank.py" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

run_eval() {
    local dataset="$1"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29921}"
    local config
    config="$(config_for "${dataset}")"
    local out_dir="${ROOT}/${dataset}"
    mkdir -p "${out_dir}"

    local cfg_options=(
        model.dump_cross_image_bank_stats=True
        model.cross_image_bank_stats_path="${out_dir}/cross_image_bank.jsonl"
        model.cross_image_bank_spaces="${BANK_SPACES:-vision,pe_layer_0,pe_layer_1,pe_layer_2}"
        model.cross_image_bank_seed_rule="${BANK_SEED_RULE:-final_score_margin_sem_inst_final_agree_local_core}"
        model.cross_image_bank_min_seed_pixels="${BANK_MIN_SEED_PIXELS:-4}"
        model.cross_image_bank_min_pair_pixels="${BANK_MIN_PAIR_PIXELS:-4}"
        model.cross_image_bank_feature_max_side="${BANK_FEATURE_MAX_SIDE:-1024}"
        model.cross_image_bank_reencode_missing_features="${BANK_REENCODE_MISSING:-True}"
    )

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
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-29930}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS=0,1 PORT=$((port_base + index)) \
            BANK_SPACES="${BANK_SPACES:-vision,pe_layer_0,pe_layer_1,pe_layer_2}" \
            BANK_SEED_RULE="${BANK_SEED_RULE:-final_score_margin_sem_inst_final_agree_local_core}" \
            BANK_MIN_SEED_PIXELS="${BANK_MIN_SEED_PIXELS:-4}" \
            BANK_MIN_PAIR_PIXELS="${BANK_MIN_PAIR_PIXELS:-4}" \
            BANK_FEATURE_MAX_SIDE="${BANK_FEATURE_MAX_SIDE:-1024}" \
            BANK_REENCODE_MISSING="${BANK_REENCODE_MISSING:-True}" \
            bash "$0" eval "${dataset_a}" \
            > "${ROOT}/launch_logs/${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS=2,3 PORT=$((port_base + index)) \
                BANK_SPACES="${BANK_SPACES:-vision,pe_layer_0,pe_layer_1,pe_layer_2}" \
                BANK_SEED_RULE="${BANK_SEED_RULE:-final_score_margin_sem_inst_final_agree_local_core}" \
                BANK_MIN_SEED_PIXELS="${BANK_MIN_SEED_PIXELS:-4}" \
                BANK_MIN_PAIR_PIXELS="${BANK_MIN_PAIR_PIXELS:-4}" \
                BANK_FEATURE_MAX_SIDE="${BANK_FEATURE_MAX_SIDE:-1024}" \
                BANK_REENCODE_MISSING="${BANK_REENCODE_MISSING:-True}" \
                bash "$0" eval "${dataset_b}" \
                > "${ROOT}/launch_logs/${dataset_b}.log" 2>&1 &
            pid_b=$!
            index=$((index + 1))
        fi
        local status_a=0
        local status_b=0
        wait "${pid_a}" || status_a=$?
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}" || status_b=$?
        fi
        if (( status_a != 0 || status_b != 0 )); then
            echo "At least one cross-image bank job failed: ${dataset_a}=${status_a}, ${dataset_b:-none}=${status_b}" >&2
            exit 1
        fi
    done
}

summarize() {
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_cross_image_bank.py \
        --inputs "${ROOT}"/*/cross_image_bank_rank*.jsonl \
        --output-dir "${ROOT}/summary"
}

build_bank() {
    local dataset="$1"
    local space="${2:-${BANK_APPLY_SPACE:-pe_layer_0}}"
    local variant="${3:-${BANK_APPLY_VARIANT:-image_centered}}"
    local out_dir="${ROOT}/banks"
    mkdir -p "${out_dir}"
    "${PYTHON_BIN}" tools/build_cross_image_bank.py \
        --inputs "${ROOT}/${dataset}/cross_image_bank_rank"*.jsonl \
        --spaces "${space}" \
        --variants "${variant}" \
        --output "${out_dir}/${dataset}_${space}_${variant}.json"
}

run_recompose() {
    local dataset="$1"
    local space="${2:-${BANK_APPLY_SPACE:-pe_layer_0}}"
    local variant="${3:-${BANK_APPLY_VARIANT:-image_centered}}"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29971}"
    local config
    config="$(config_for "${dataset}")"
    local bank_path="${ROOT}/banks/${dataset}_${space}_${variant}.json"
    if [[ ! -f "${bank_path}" ]]; then
        echo "Missing bank file: ${bank_path}" >&2
        echo "Run: ROOT=${ROOT} bash $0 build-bank ${dataset} ${space} ${variant}" >&2
        exit 2
    fi
    local tag="${space}_${variant}_m${BANK_MIN_MARGIN:-0.10}_g${BANK_MAX_GAP:-0.60}"
    local out_dir="${ROOT}/recompose/${tag}/${dataset}"
    mkdir -p "${out_dir}"
    local cfg_options=(
        model.use_cross_image_bank_recomposition=True
        model.dump_cross_image_bank_recomposition_stats=True
        model.cross_image_bank_recomposition_stats_path="${out_dir}/cross_image_bank_recompose.jsonl"
        model.cross_image_bank_file="${bank_path}"
        model.cross_image_bank_dataset_name="${dataset}"
        model.cross_image_bank_apply_space="${space}"
        model.cross_image_bank_apply_variant="${variant}"
        model.cross_image_bank_apply_seed_rule="${BANK_SEED_RULE:-final_score_margin_sem_inst_final_agree_local_core}"
        model.cross_image_bank_apply_topk="${BANK_TOPK:-3}"
        model.cross_image_bank_apply_min_bank_margin="${BANK_MIN_MARGIN:-0.10}"
        model.cross_image_bank_apply_max_base_gap="${BANK_MAX_GAP:-0.60}"
        model.cross_image_bank_apply_min_candidate_score="${BANK_MIN_CAND_SCORE:--1.0}"
        model.cross_image_bank_apply_min_candidate_affinity="${BANK_MIN_CAND_AFFINITY:--1.0}"
        model.cross_image_bank_apply_logit_boost="${BANK_LOGIT_BOOST:-0.0001}"
        model.cross_image_bank_apply_min_class_weight="${BANK_MIN_CLASS_WEIGHT:-32}"
        model.cross_image_bank_apply_exclude_current_image="${BANK_EXCLUDE_CURRENT:-True}"
        model.cross_image_bank_apply_protect_bg="${BANK_PROTECT_BG:-True}"
        model.cross_image_bank_apply_exclude_bg_candidate="${BANK_EXCLUDE_BG_CANDIDATE:-True}"
        model.cross_image_bank_apply_require_base_non_bg="${BANK_REQUIRE_BASE_NON_BG:-True}"
        model.cross_image_bank_feature_max_side="${BANK_FEATURE_MAX_SIDE:-1024}"
        model.cross_image_bank_reencode_missing_features="${BANK_REENCODE_MISSING:-True}"
    )

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

build_bank_parallel() {
    local space="${1:-${BANK_APPLY_SPACE:-pe_layer_0}}"
    local variant="${2:-${BANK_APPLY_VARIANT:-image_centered}}"
    shift || true
    shift || true
    local datasets=("$@")
    if (( ${#datasets[@]} == 0 )); then
        datasets=(udd5 vdd vaihingen potsdam openearthmap loveda)
    fi
    for dataset in "${datasets[@]}"; do
        build_bank "${dataset}" "${space}" "${variant}"
    done
}

run_recompose_parallel() {
    local space="${1:-${BANK_APPLY_SPACE:-pe_layer_0}}"
    local variant="${2:-${BANK_APPLY_VARIANT:-image_centered}}"
    shift || true
    shift || true
    local datasets=("$@")
    if (( ${#datasets[@]} == 0 )); then
        datasets=(udd5 vdd vaihingen potsdam openearthmap loveda)
    fi
    mkdir -p "${ROOT}/launch_logs/recompose"
    local index=0
    local port_base="${PORT_BASE:-29980}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS=0,1 PORT=$((port_base + index)) \
            BANK_APPLY_SPACE="${space}" \
            BANK_APPLY_VARIANT="${variant}" \
            BANK_MIN_MARGIN="${BANK_MIN_MARGIN:-0.10}" \
            BANK_MAX_GAP="${BANK_MAX_GAP:-0.60}" \
            BANK_TOPK="${BANK_TOPK:-3}" \
            BANK_MIN_CAND_SCORE="${BANK_MIN_CAND_SCORE:--1.0}" \
            BANK_MIN_CAND_AFFINITY="${BANK_MIN_CAND_AFFINITY:--1.0}" \
            BANK_MIN_CLASS_WEIGHT="${BANK_MIN_CLASS_WEIGHT:-32}" \
            BANK_EXCLUDE_CURRENT="${BANK_EXCLUDE_CURRENT:-True}" \
            BANK_PROTECT_BG="${BANK_PROTECT_BG:-True}" \
            BANK_EXCLUDE_BG_CANDIDATE="${BANK_EXCLUDE_BG_CANDIDATE:-True}" \
            BANK_REQUIRE_BASE_NON_BG="${BANK_REQUIRE_BASE_NON_BG:-True}" \
            BANK_FEATURE_MAX_SIDE="${BANK_FEATURE_MAX_SIDE:-1024}" \
            bash "$0" recompose "${dataset_a}" "${space}" "${variant}" \
            > "${ROOT}/launch_logs/recompose/${dataset_a}_${space}_${variant}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS=2,3 PORT=$((port_base + index)) \
                BANK_APPLY_SPACE="${space}" \
                BANK_APPLY_VARIANT="${variant}" \
                BANK_MIN_MARGIN="${BANK_MIN_MARGIN:-0.10}" \
                BANK_MAX_GAP="${BANK_MAX_GAP:-0.60}" \
                BANK_TOPK="${BANK_TOPK:-3}" \
                BANK_MIN_CAND_SCORE="${BANK_MIN_CAND_SCORE:--1.0}" \
                BANK_MIN_CAND_AFFINITY="${BANK_MIN_CAND_AFFINITY:--1.0}" \
                BANK_MIN_CLASS_WEIGHT="${BANK_MIN_CLASS_WEIGHT:-32}" \
                BANK_EXCLUDE_CURRENT="${BANK_EXCLUDE_CURRENT:-True}" \
                BANK_PROTECT_BG="${BANK_PROTECT_BG:-True}" \
                BANK_EXCLUDE_BG_CANDIDATE="${BANK_EXCLUDE_BG_CANDIDATE:-True}" \
                BANK_REQUIRE_BASE_NON_BG="${BANK_REQUIRE_BASE_NON_BG:-True}" \
                BANK_FEATURE_MAX_SIDE="${BANK_FEATURE_MAX_SIDE:-1024}" \
                bash "$0" recompose "${dataset_b}" "${space}" "${variant}" \
                > "${ROOT}/launch_logs/recompose/${dataset_b}_${space}_${variant}.log" 2>&1 &
            pid_b=$!
            index=$((index + 1))
        fi
        local status_a=0
        local status_b=0
        wait "${pid_a}" || status_a=$?
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}" || status_b=$?
        fi
        if (( status_a != 0 || status_b != 0 )); then
            echo "At least one recompose job failed: ${dataset_a}=${status_a}, ${dataset_b:-none}=${status_b}" >&2
            exit 1
        fi
    done
}

summarize_recompose() {
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_cross_image_bank_recomposition.py \
        --inputs "${ROOT}"/recompose/*/*/cross_image_bank_recompose_rank*.jsonl \
        --output-dir "${ROOT}/summary"
}

analyze_direction_safety() {
    local out_dir="${ROOT}/summary/direction_safety"
    mkdir -p "${out_dir}"
    "${PYTHON_BIN}" tools/analyze_cross_image_bank_direction_safety.py \
        --recompose-inputs "${ROOT}"/recompose/*/*/cross_image_bank_recompose_rank*.jsonl \
        --pair-summary "${ROOT}/summary/cross_image_bank_pair_summary.csv" \
        --class-coverage "${ROOT}/summary/cross_image_bank_class_coverage.csv" \
        --output-dir "${out_dir}" \
        --min-pixels "${DIRECTION_MIN_PIXELS:-1000}" \
        --min-train-selected-pixels "${DIRECTION_MIN_TRAIN_PIXELS:-10000}" \
        --min-train-precision "${DIRECTION_MIN_TRAIN_PRECISION:-0.0}"
}

case "${MODE}" in
    eval)
        run_eval "${2:?dataset required}"
        ;;
    diagnostic-suite)
        run_parallel udd5 vdd vaihingen potsdam openearthmap loveda
        ;;
    diagnostic-small)
        run_parallel udd5 vdd vaihingen
        ;;
    diagnostic-large)
        run_parallel potsdam openearthmap loveda
        ;;
    summarize)
        summarize
        ;;
    build-bank)
        build_bank "${2:?dataset required}" "${3:-${BANK_APPLY_SPACE:-pe_layer_0}}" "${4:-${BANK_APPLY_VARIANT:-image_centered}}"
        ;;
    build-bank-suite)
        build_bank_parallel "${2:-${BANK_APPLY_SPACE:-pe_layer_0}}" "${3:-${BANK_APPLY_VARIANT:-image_centered}}" \
            udd5 vdd vaihingen potsdam openearthmap loveda
        ;;
    recompose)
        run_recompose "${2:?dataset required}" "${3:-${BANK_APPLY_SPACE:-pe_layer_0}}" "${4:-${BANK_APPLY_VARIANT:-image_centered}}"
        ;;
    recompose-selected)
        run_recompose_parallel "${2:-${BANK_APPLY_SPACE:-pe_layer_0}}" "${3:-${BANK_APPLY_VARIANT:-image_centered}}" \
            vdd potsdam openearthmap vaihingen
        ;;
    recompose-suite)
        run_recompose_parallel "${2:-${BANK_APPLY_SPACE:-pe_layer_0}}" "${3:-${BANK_APPLY_VARIANT:-image_centered}}" \
            udd5 vdd vaihingen potsdam openearthmap loveda
        ;;
    summarize-recompose)
        summarize_recompose
        ;;
    direction-safety)
        analyze_direction_safety
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_cross_image_bank_v1.sh eval DATASET
  bash tools/run_cross_image_bank_v1.sh diagnostic-small
  bash tools/run_cross_image_bank_v1.sh diagnostic-large
  bash tools/run_cross_image_bank_v1.sh diagnostic-suite
  bash tools/run_cross_image_bank_v1.sh summarize
  bash tools/run_cross_image_bank_v1.sh build-bank DATASET [SPACE] [VARIANT]
  bash tools/run_cross_image_bank_v1.sh build-bank-suite [SPACE] [VARIANT]
  bash tools/run_cross_image_bank_v1.sh recompose DATASET [SPACE] [VARIANT]
  bash tools/run_cross_image_bank_v1.sh recompose-selected [SPACE] [VARIANT]
  bash tools/run_cross_image_bank_v1.sh recompose-suite [SPACE] [VARIANT]
  bash tools/run_cross_image_bank_v1.sh summarize-recompose
  bash tools/run_cross_image_bank_v1.sh direction-safety

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda

Environment overrides:
  ROOT, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  BANK_SPACES, BANK_SEED_RULE, BANK_MIN_SEED_PIXELS,
  BANK_MIN_PAIR_PIXELS, BANK_FEATURE_MAX_SIDE, BANK_REENCODE_MISSING,
  BANK_APPLY_SPACE, BANK_APPLY_VARIANT, BANK_MIN_MARGIN, BANK_MAX_GAP,
  BANK_TOPK, BANK_MIN_CAND_SCORE, BANK_MIN_CAND_AFFINITY,
  BANK_MIN_CLASS_WEIGHT, BANK_EXCLUDE_CURRENT, BANK_PROTECT_BG,
  BANK_EXCLUDE_BG_CANDIDATE, BANK_REQUIRE_BASE_NON_BG,
  DIRECTION_MIN_PIXELS, DIRECTION_MIN_TRAIN_PIXELS,
  DIRECTION_MIN_TRAIN_PRECISION
EOF
        exit 2
        ;;
esac
