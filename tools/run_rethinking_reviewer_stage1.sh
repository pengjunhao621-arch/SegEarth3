#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/rethinking_reviewer_stage1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

config_for() {
    case "$1" in
        udd5) echo "configs/experiments/cfg_udd5_rethinking_reviewer.py" ;;
        vdd) echo "configs/experiments/cfg_vdd_rethinking_reviewer.py" ;;
        vaihingen) echo "configs/experiments/cfg_vaihingen_rethinking_reviewer.py" ;;
        *) echo "Unknown dataset: $1" >&2; exit 2 ;;
    esac
}

default_train_paths() {
    case "$1" in
        udd5) echo "train/src train/gt" ;;
        vdd) echo "train/src train/gt" ;;
        vaihingen) echo "img_dir/train ann_dir/train" ;;
    esac
}

cache_dataset() {
    local dataset="$1"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29601}"
    local config
    config="$(config_for "${dataset}")"
    read -r default_img default_seg <<< "$(default_train_paths "${dataset}")"
    local img_path="${IMG_PATH:-${default_img}}"
    local seg_path="${SEG_PATH:-${default_seg}}"
    local out_dir="${ROOT}/cache/train/${dataset}"
    mkdir -p "${out_dir}" "${ROOT}/work_dirs/cache_${dataset}"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    torchrun \
        --nproc_per_node=2 \
        --master_port="${port}" \
        eval.py "${config}" \
        --launcher pytorch \
        --work-dir "${ROOT}/work_dirs/cache_${dataset}" \
        --result-file "" \
        --cfg-options \
            model.dump_reviewer_cache=True \
            model.reviewer_cache_dir="${out_dir}" \
            model.use_learned_reviewer=False \
            model.dump_learned_reviewer_stats=False \
            test_dataloader.dataset.data_prefix.img_path="${img_path}" \
            test_dataloader.dataset.data_prefix.seg_map_path="${seg_path}"
}

train_variant() {
    local variant="$1"
    local gpu_id="${GPU_ID:-0}"
    local out_dir="${ROOT}/checkpoints/${variant}"
    mkdir -p "${out_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" tools/train_rethinking_reviewer.py \
        --train-cache \
            "${ROOT}/cache/train/udd5" \
            "${ROOT}/cache/train/vdd" \
            "${ROOT}/cache/train/vaihingen" \
        --variant "${variant}" \
        --output-dir "${out_dir}" \
        --validation-fraction 0.10 \
        --epochs "${EPOCHS:-25}" \
        --batch-size "${BATCH_SIZE:-2048}" \
        --seed "${SEED:-0}"
}

eval_variant() {
    local dataset="$1"
    local variant="$2"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29701}"
    local config
    config="$(config_for "${dataset}")"
    local checkpoint="${ROOT}/checkpoints/${variant}/best.pth"
    local out_dir="${ROOT}/eval/${variant}/${dataset}"
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
            model.dump_learned_reviewer_stats=True \
            model.learned_reviewer_stats_path="${out_dir}/reviewer.jsonl"
}

eval_baseline() {
    local dataset="$1"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-29801}"
    local config
    config="$(config_for "${dataset}")"
    local out_dir="${ROOT}/eval/baseline/${dataset}"
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

eval_suite() {
    local jobs=(
        "udd5 output_only"
        "vdd output_only"
        "vaihingen output_only"
        "udd5 three_head"
        "vdd three_head"
        "vaihingen three_head"
        "udd5 internal_trajectory"
        "vdd internal_trajectory"
        "vaihingen internal_trajectory"
    )
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-29720}"
    while (( index < ${#jobs[@]} )); do
        read -r dataset_a variant_a <<< "${jobs[index]}"
        ROOT="${ROOT}" GPU_IDS=0,1 PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" "${variant_a}" \
            > "${ROOT}/launch_logs/eval_${variant_a}_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#jobs[@]} )); then
            read -r dataset_b variant_b <<< "${jobs[index]}"
            ROOT="${ROOT}" GPU_IDS=2,3 PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" "${variant_b}" \
                > "${ROOT}/launch_logs/eval_${variant_b}_${dataset_b}.log" 2>&1 &
            pid_b=$!
            index=$((index + 1))
        fi
        wait "${pid_a}"
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}"
        fi
    done
}

case "${MODE}" in
    cache)
        cache_dataset "${2:?dataset required: udd5|vdd|vaihingen}"
        ;;
    train)
        train_variant "${2:?variant required: output_only|three_head|internal_trajectory}"
        ;;
    eval)
        eval_variant \
            "${2:?dataset required}" \
            "${3:?variant required}"
        ;;
    baseline)
        eval_baseline "${2:?dataset required}"
        ;;
    eval-suite)
        eval_suite
        ;;
    summarize)
        mkdir -p "${ROOT}/summary"
        "${PYTHON_BIN}" tools/summarize_rethinking_reviewer.py \
            --inputs "${ROOT}/eval"/*/*/reviewer.rank*.jsonl \
            --output "${ROOT}/summary/reviewer_change_summary.csv"
        "${PYTHON_BIN}" tools/collect_rethinking_results.py \
            --root "${ROOT}" \
            --output "${ROOT}/summary/exact_miou_summary.csv"
        ;;
    summarize-cache)
        mkdir -p "${ROOT}/summary"
        "${PYTHON_BIN}" tools/summarize_rethinking_cache.py \
            --cache \
                "${ROOT}/cache/train/udd5" \
                "${ROOT}/cache/train/vdd" \
                "${ROOT}/cache/train/vaihingen" \
            --output "${ROOT}/summary/cache_summary.csv"
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_rethinking_reviewer_stage1.sh cache DATASET
  bash tools/run_rethinking_reviewer_stage1.sh train VARIANT
  bash tools/run_rethinking_reviewer_stage1.sh eval DATASET VARIANT
  bash tools/run_rethinking_reviewer_stage1.sh baseline DATASET
  bash tools/run_rethinking_reviewer_stage1.sh eval-suite
  bash tools/run_rethinking_reviewer_stage1.sh summarize
  bash tools/run_rethinking_reviewer_stage1.sh summarize-cache

Environment overrides:
  ROOT, GPU_IDS, GPU_ID, PORT, IMG_PATH, SEG_PATH, EPOCHS,
  BATCH_SIZE, SEED, PYTHON_BIN
EOF
        exit 2
        ;;
esac
