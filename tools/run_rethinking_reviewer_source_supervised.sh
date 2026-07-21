#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/rethinking_reviewer_source_supervised}"
SOURCE_DATASET="${SOURCE_DATASET:-loveda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

VARIANTS=(output_only three_head internal_trajectory)
MAIN_DATASETS=(openearthmap loveda potsdam udd5 vaihingen vdd)

config_for() {
    case "$1" in
        openearthmap) echo "configs/experiments/cfg_openearthmap_rethinking_reviewer.py" ;;
        loveda) echo "configs/experiments/cfg_loveda_rethinking_reviewer.py" ;;
        potsdam) echo "configs/experiments/cfg_potsdam_rethinking_reviewer.py" ;;
        udd5) echo "configs/experiments/cfg_udd5_rethinking_reviewer.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_rethinking_reviewer.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_rethinking_reviewer.py" ;;
        isaid) echo "configs/experiments/cfg_isaid_rethinking_reviewer.py" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

source_split_for() {
    local source="$1"
    if [[ -n "${SOURCE_SPLIT:-}" ]]; then
        echo "${SOURCE_SPLIT}"
    elif [[ "${source}" == "loveda" ]]; then
        echo "val"
    else
        echo "train"
    fi
}

default_train_paths() {
    local source="$1"
    local split
    split="$(source_split_for "${source}")"
    case "$1" in
        loveda) echo "img_dir/${split} ann_dir/${split}" ;;
        isaid) echo "img_dir/${split} ann_dir/${split}" ;;
        openearthmap) echo "images/${split} labels/${split}" ;;
        potsdam) echo "img_dir/${split} ann_dir/${split}" ;;
        udd5) echo "train/src train/gt" ;;
        vaihingen) echo "img_dir/${split} ann_dir/${split}" ;;
        vdd) echo "img_dir/${split} ann_dir/${split}" ;;
        *) echo "Unknown source dataset: $1" >&2; exit 2 ;;
    esac
}

source_cache_key() {
    local source="$1"
    local split
    split="$(source_split_for "${source}")"
    if [[ "${split}" == "train" ]]; then
        echo "${source}"
    else
        echo "${source}_${split}"
    fi
}

target_datasets() {
    local source="$1"
    for dataset in "${MAIN_DATASETS[@]}"; do
        if [[ "${dataset}" != "${source}" ]]; then
            echo "${dataset}"
        fi
    done
    if [[ "${source}" != "isaid" && "${INCLUDE_ISAID_TARGET:-False}" == "True" ]]; then
        echo "isaid"
    fi
}

cache_source() {
    local source="${1:-${SOURCE_DATASET}}"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-31001}"
    local config
    config="$(config_for "${source}")"
    read -r default_img default_seg <<< "$(default_train_paths "${source}")"
    local img_path="${IMG_PATH:-${default_img}}"
    local seg_path="${SEG_PATH:-${default_seg}}"
    local cache_key
    cache_key="$(source_cache_key "${source}")"
    local out_dir="${ROOT}/cache/source/${cache_key}"
    mkdir -p "${out_dir}" "${ROOT}/work_dirs/cache_${cache_key}" "${ROOT}/launch_logs"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    torchrun \
        --nproc_per_node=2 \
        --master_port="${port}" \
        eval.py "${config}" \
        --launcher pytorch \
        --work-dir "${ROOT}/work_dirs/cache_${cache_key}" \
        --result-file "" \
        --cfg-options \
            model.dump_reviewer_cache=True \
            model.reviewer_cache_dir="${out_dir}" \
            model.reviewer_dataset_name="${source}" \
            model.reviewer_cache_hard_keep_margin="${HARD_KEEP_MARGIN:-0.15}" \
            model.reviewer_cache_samples_per_image="${CACHE_SAMPLES_PER_IMAGE:-4096}" \
            model.use_learned_reviewer=False \
            model.dump_learned_reviewer_stats=False \
            test_dataloader.dataset.data_prefix.img_path="${img_path}" \
            test_dataloader.dataset.data_prefix.seg_map_path="${seg_path}"
}

train_variant() {
    local variant="$1"
    local source="${2:-${SOURCE_DATASET}}"
    local gpu_id="${GPU_ID:-0}"
    local cache_key
    local split
    cache_key="$(source_cache_key "${source}")"
    split="$(source_split_for "${source}")"
    local out_dir="${ROOT}/checkpoints/${cache_key}/${variant}"
    local cache_dir="${ROOT}/cache/source/${cache_key}"
    mkdir -p "${out_dir}" "${ROOT}/launch_logs"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" tools/train_rethinking_reviewer.py \
        --train-cache "${cache_dir}" \
        --variant "${variant}" \
        --output-dir "${out_dir}" \
        --validation-fraction "${VALIDATION_FRACTION:-0.10}" \
        --epochs "${EPOCHS:-25}" \
        --batch-size "${BATCH_SIZE:-2048}" \
        --learning-rate "${LEARNING_RATE:-1e-4}" \
        --risk-weight "${RISK_WEIGHT:-0.50}" \
        --recoverable-weight "${RECOVERABLE_WEIGHT:-1.00}" \
        --preserve-weight "${PRESERVE_WEIGHT:-2.00}" \
        --gate-thresholds "${GATE_THRESHOLDS:-0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99,1.01}" \
        --action-margins "${ACTION_MARGINS:-0.00,0.05,0.10,0.20,0.30}" \
        --protocol-name source_supervised_candidate_validity \
        --source-dataset "${source}:${split}" \
        --seed "${SEED:-0}"
}

train_suite() {
    local source="${1:-${SOURCE_DATASET}}"
    local split
    local cache_key
    split="$(source_split_for "${source}")"
    cache_key="$(source_cache_key "${source}")"
    mkdir -p "${ROOT}/launch_logs"
    local pids=()
    local index=0
    for variant in "${VARIANTS[@]}"; do
        ROOT="${ROOT}" SOURCE_DATASET="${source}" GPU_ID="${index}" \
            SOURCE_SPLIT="${split}" \
            PYTHON_BIN="${PYTHON_BIN}" \
            bash "$0" train "${variant}" "${source}" \
            > "${ROOT}/launch_logs/train_${cache_key}_${variant}.log" 2>&1 &
        pids+=("$!")
        index=$((index + 1))
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
}

eval_variant() {
    local dataset="$1"
    local variant="$2"
    local source="${3:-${SOURCE_DATASET}}"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-31101}"
    local config
    config="$(config_for "${dataset}")"
    local cache_key
    cache_key="$(source_cache_key "${source}")"
    local checkpoint="${ROOT}/checkpoints/${cache_key}/${variant}/best.pth"
    local out_dir="${ROOT}/eval/${cache_key}/${variant}/${dataset}"
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
            model.dump_reviewer_cache=False \
            model.use_learned_reviewer=True \
            model.reviewer_checkpoint="${checkpoint}" \
            model.reviewer_variant="${variant}" \
            model.reviewer_dataset_name="${dataset}" \
            model.dump_learned_reviewer_stats=True \
            model.learned_reviewer_stats_path="${out_dir}/reviewer.jsonl"
}

eval_baseline() {
    local dataset="$1"
    local source="${2:-${SOURCE_DATASET}}"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-31201}"
    local config
    config="$(config_for "${dataset}")"
    local cache_key
    cache_key="$(source_cache_key "${source}")"
    local out_dir="${ROOT}/eval/${cache_key}/baseline/${dataset}"
    mkdir -p "${out_dir}"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    torchrun \
        --nproc_per_node=2 \
        --master_port="${port}" \
        eval.py "${config}" \
        --launcher pytorch \
        --work-dir "${out_dir}" \
        --result-file "${out_dir}/results.xlsx"
}

run_parallel_jobs() {
    local action="$1"
    local source="$2"
    shift 2
    local jobs=("$@")
    local split
    local cache_key
    split="$(source_split_for "${source}")"
    cache_key="$(source_cache_key "${source}")"
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-31240}"
    while (( index < ${#jobs[@]} )); do
        read -r dataset_a variant_a <<< "${jobs[index]}"
        if [[ "${action}" == "baseline" ]]; then
            ROOT="${ROOT}" SOURCE_DATASET="${source}" SOURCE_SPLIT="${split}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
                bash "$0" baseline "${dataset_a}" "${source}" \
                > "${ROOT}/launch_logs/baseline_${cache_key}_${dataset_a}.log" 2>&1 &
        else
            ROOT="${ROOT}" SOURCE_DATASET="${source}" SOURCE_SPLIT="${split}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_a}" "${variant_a}" "${source}" \
                > "${ROOT}/launch_logs/eval_${cache_key}_${variant_a}_${dataset_a}.log" 2>&1 &
        fi
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#jobs[@]} )); then
            read -r dataset_b variant_b <<< "${jobs[index]}"
            if [[ "${action}" == "baseline" ]]; then
                ROOT="${ROOT}" SOURCE_DATASET="${source}" SOURCE_SPLIT="${split}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                    bash "$0" baseline "${dataset_b}" "${source}" \
                    > "${ROOT}/launch_logs/baseline_${cache_key}_${dataset_b}.log" 2>&1 &
            else
                ROOT="${ROOT}" SOURCE_DATASET="${source}" SOURCE_SPLIT="${split}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                    bash "$0" eval "${dataset_b}" "${variant_b}" "${source}" \
                    > "${ROOT}/launch_logs/eval_${cache_key}_${variant_b}_${dataset_b}.log" 2>&1 &
            fi
            pid_b=$!
            index=$((index + 1))
        fi
        wait "${pid_a}"
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}"
        fi
    done
}

eval_target_suite() {
    local source="${1:-${SOURCE_DATASET}}"
    local jobs=()
    while read -r dataset; do
        for variant in "${VARIANTS[@]}"; do
            jobs+=("${dataset} ${variant}")
        done
    done < <(target_datasets "${source}")
    run_parallel_jobs eval "${source}" "${jobs[@]}"
}

baseline_target_suite() {
    local source="${1:-${SOURCE_DATASET}}"
    local jobs=()
    while read -r dataset; do
        jobs+=("${dataset} baseline")
    done < <(target_datasets "${source}")
    run_parallel_jobs baseline "${source}" "${jobs[@]}"
}

summarize_all() {
    local source="${1:-${SOURCE_DATASET}}"
    local cache_key
    cache_key="$(source_cache_key "${source}")"
    mkdir -p "${ROOT}/summary/${cache_key}"
    local datasets=()
    while read -r dataset; do
        datasets+=("${dataset}")
    done < <(target_datasets "${source}")
    "${PYTHON_BIN}" tools/summarize_rethinking_calibration.py \
        --checkpoints \
            "${ROOT}/checkpoints/${cache_key}/output_only/best.pth" \
            "${ROOT}/checkpoints/${cache_key}/three_head/best.pth" \
            "${ROOT}/checkpoints/${cache_key}/internal_trajectory/best.pth" \
        --output "${ROOT}/summary/${cache_key}/calibration_miou_summary.csv"
    "${PYTHON_BIN}" tools/summarize_rethinking_reviewer.py \
        --inputs "${ROOT}/eval/${cache_key}"/*/*/reviewer.rank*.jsonl \
        --output "${ROOT}/summary/${cache_key}/reviewer_change_summary.csv"
    "${PYTHON_BIN}" tools/collect_rethinking_results.py \
        --root "${ROOT}" \
        --eval-root "${ROOT}/eval/${cache_key}" \
        --datasets "${datasets[@]}" \
        --output "${ROOT}/summary/${cache_key}/exact_target_miou_summary.csv"
}

summarize_cache() {
    local source="${1:-${SOURCE_DATASET}}"
    local cache_key
    cache_key="$(source_cache_key "${source}")"
    mkdir -p "${ROOT}/summary/${cache_key}"
    "${PYTHON_BIN}" tools/summarize_rethinking_cache.py \
        --cache "${ROOT}/cache/source/${cache_key}" \
        --output "${ROOT}/summary/${cache_key}/cache_summary.csv"
}

case "${MODE}" in
    cache-source)
        cache_source "${2:-${SOURCE_DATASET}}"
        ;;
    train)
        train_variant \
            "${2:?variant required: output_only|three_head|internal_trajectory}" \
            "${3:-${SOURCE_DATASET}}"
        ;;
    train-suite)
        train_suite "${2:-${SOURCE_DATASET}}"
        ;;
    eval)
        eval_variant \
            "${2:?dataset required}" \
            "${3:?variant required}" \
            "${4:-${SOURCE_DATASET}}"
        ;;
    baseline)
        eval_baseline "${2:?dataset required}" "${3:-${SOURCE_DATASET}}"
        ;;
    eval-target-suite)
        eval_target_suite "${2:-${SOURCE_DATASET}}"
        ;;
    baseline-target-suite)
        baseline_target_suite "${2:-${SOURCE_DATASET}}"
        ;;
    summarize)
        summarize_all "${2:-${SOURCE_DATASET}}"
        ;;
    summarize-cache)
        summarize_cache "${2:-${SOURCE_DATASET}}"
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_rethinking_reviewer_source_supervised.sh cache-source [SOURCE]
  bash tools/run_rethinking_reviewer_source_supervised.sh train VARIANT [SOURCE]
  bash tools/run_rethinking_reviewer_source_supervised.sh train-suite [SOURCE]
  bash tools/run_rethinking_reviewer_source_supervised.sh baseline DATASET [SOURCE]
  bash tools/run_rethinking_reviewer_source_supervised.sh baseline-target-suite [SOURCE]
  bash tools/run_rethinking_reviewer_source_supervised.sh eval DATASET VARIANT [SOURCE]
  bash tools/run_rethinking_reviewer_source_supervised.sh eval-target-suite [SOURCE]
  bash tools/run_rethinking_reviewer_source_supervised.sh summarize [SOURCE]
  bash tools/run_rethinking_reviewer_source_supervised.sh summarize-cache [SOURCE]

Sources:
  loveda, isaid are the intended first-stage sources.

Environment overrides:
  ROOT, SOURCE_DATASET, SOURCE_SPLIT, INCLUDE_ISAID_TARGET, GPU_IDS, GPU_IDS_A,
  GPU_IDS_B, GPU_ID, PORT, PORT_BASE, PYTHON_BIN, IMG_PATH, SEG_PATH,
  CACHE_SAMPLES_PER_IMAGE, HARD_KEEP_MARGIN, VALIDATION_FRACTION,
  EPOCHS, BATCH_SIZE, LEARNING_RATE, RISK_WEIGHT, RECOVERABLE_WEIGHT,
  PRESERVE_WEIGHT, GATE_THRESHOLDS, ACTION_MARGINS, SEED

Protocol:
  The selected source dataset is used only for cache/train/calibration.
  eval-target-suite and baseline-target-suite automatically exclude it.
  SOURCE_SPLIT defaults to val for LoveDA source because this feasibility
  protocol is often run before LoveDA train files are available; other
  sources default to train. Set SOURCE_SPLIT explicitly to override this.
EOF
        exit 2
        ;;
esac
