#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/evidence_enhancement_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"

DATASETS_SMALL=(udd5 vdd vaihingen)
DATASETS_MAIN=(udd5 vdd vaihingen potsdam openearthmap loveda)
VARIANTS=(semantic_prompt semantic_sqrt final_prompt)

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

variant_settings() {
    local variant="$1"
    case "${variant}" in
        semantic_prompt)
            EE_SOURCE="${EE_SOURCE:-semantic}"
            EE_PRESENCE="${EE_PRESENCE:-prompt}"
            EE_REDUCE="${EE_REDUCE:-max}"
            EE_COMBINE="${EE_COMBINE:-max}"
            ;;
        semantic_sqrt)
            EE_SOURCE="${EE_SOURCE:-semantic}"
            EE_PRESENCE="${EE_PRESENCE:-sqrt}"
            EE_REDUCE="${EE_REDUCE:-max}"
            EE_COMBINE="${EE_COMBINE:-max}"
            ;;
        final_prompt)
            EE_SOURCE="${EE_SOURCE:-final}"
            EE_PRESENCE="${EE_PRESENCE:-none}"
            EE_REDUCE="${EE_REDUCE:-max}"
            EE_COMBINE="${EE_COMBINE:-max}"
            ;;
        instance_prompt)
            EE_SOURCE="${EE_SOURCE:-instance}"
            EE_PRESENCE="${EE_PRESENCE:-prompt}"
            EE_REDUCE="${EE_REDUCE:-max}"
            EE_COMBINE="${EE_COMBINE:-max}"
            ;;
        *)
            echo "Unknown variant: ${variant}" >&2
            exit 2
            ;;
    esac
}

run_eval() {
    local dataset="$1"
    local apply="${2:-false}"
    local variant="${EE_VARIANT:-semantic_prompt}"
    variant_settings "${variant}"

    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-30731}"
    local config
    config="$(config_for "${dataset}")"

    local mode_dir="diagnostic"
    local apply_cfg="False"
    if [[ "${apply}" == "true" || "${apply}" == "True" || "${apply}" == "1" ]]; then
        mode_dir="apply"
        apply_cfg="True"
    fi
    local out_dir="${ROOT}/${variant}/${mode_dir}/${dataset}"
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
            model.use_evidence_enhancement="${apply_cfg}" \
            model.dump_evidence_enhancement_stats=True \
            model.evidence_enhancement_stats_path="${out_dir}/evidence_enhancement.jsonl" \
            model.evidence_enhancement_classes="${EE_CLASSES:-all}" \
            model.evidence_enhancement_source="${EE_SOURCE}" \
            model.evidence_enhancement_presence_mode="${EE_PRESENCE}" \
            model.evidence_enhancement_reduce="${EE_REDUCE}" \
            model.evidence_enhancement_combine="${EE_COMBINE}" \
            model.evidence_enhancement_alpha="${EE_ALPHA:-1.0}" \
            model.evidence_enhancement_max_prompts="${EE_MAX_PROMPTS:-4}" \
            model.evidence_enhancement_use_ontology_prompts="${EE_USE_ONTOLOGY:-True}" \
            model.evidence_enhancement_templates="${EE_TEMPLATES:-{class}|remote sensing {class}|aerial image {class}|satellite image {class}|{class} land cover|{class} region}"
}

run_parallel() {
    local apply="$1"
    shift
    local datasets=("$@")
    local variant="${EE_VARIANT:-semantic_prompt}"
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-30740}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" EE_VARIANT="${variant}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" "${apply}" \
            > "${ROOT}/launch_logs/${variant}_${apply}_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))

        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" EE_VARIANT="${variant}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" "${apply}" \
                > "${ROOT}/launch_logs/${variant}_${apply}_${dataset_b}.log" 2>&1 &
            pid_b=$!
            index=$((index + 1))
        fi

        wait "${pid_a}"
        if [[ -n "${pid_b}" ]]; then
            wait "${pid_b}"
        fi
    done
}

run_ablation() {
    local suite_mode="$1"
    local apply="$2"
    shift 2
    local datasets=("$@")
    local idx=0
    for variant in "${VARIANTS[@]}"; do
        EE_VARIANT="${variant}" PORT_BASE=$((${PORT_BASE:-30740} + idx * 20)) \
            bash "$0" "${suite_mode}" "${apply}"
        idx=$((idx + 1))
    done
}

summarize() {
    mkdir -p "${ROOT}/summary"
    "${PYTHON_BIN}" tools/summarize_evidence_enhancement.py \
        --inputs "${ROOT}"/*/*/*/evidence_enhancement_rank*.jsonl \
        --out-dir "${ROOT}/summary"
}

case "${MODE}" in
    eval)
        run_eval "${2:?dataset required}" "${3:-false}"
        ;;
    diagnostic-small)
        run_parallel false "${DATASETS_SMALL[@]}"
        ;;
    apply-small)
        run_parallel true "${DATASETS_SMALL[@]}"
        ;;
    diagnostic-suite)
        run_parallel false "${DATASETS_MAIN[@]}"
        ;;
    apply-suite)
        run_parallel true "${DATASETS_MAIN[@]}"
        ;;
    ablation-small)
        for variant in "${VARIANTS[@]}"; do
            EE_VARIANT="${variant}" bash "$0" apply-small
        done
        ;;
    ablation-suite)
        for variant in "${VARIANTS[@]}"; do
            EE_VARIANT="${variant}" bash "$0" apply-suite
        done
        ;;
    summarize)
        summarize
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_evidence_enhancement_v1.sh eval DATASET [true|false]
  bash tools/run_evidence_enhancement_v1.sh diagnostic-small
  bash tools/run_evidence_enhancement_v1.sh apply-small
  bash tools/run_evidence_enhancement_v1.sh diagnostic-suite
  bash tools/run_evidence_enhancement_v1.sh apply-suite
  bash tools/run_evidence_enhancement_v1.sh ablation-small
  bash tools/run_evidence_enhancement_v1.sh ablation-suite
  bash tools/run_evidence_enhancement_v1.sh summarize

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda, isaid

Variants:
  semantic_prompt: semantic map * prompt presence, max over ontology prompts
  semantic_sqrt:   semantic map * sqrt(prompt presence)
  final_prompt:    baseline-style final prompt map, max over ontology prompts

Environment overrides:
  ROOT, PYTHON_BIN, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE,
  EE_VARIANT, EE_CLASSES, EE_SOURCE, EE_PRESENCE, EE_REDUCE, EE_COMBINE,
  EE_ALPHA, EE_MAX_PROMPTS, EE_USE_ONTOLOGY, EE_TEMPLATES

Recommended first run:
  EE_CLASSES=auto EE_MAX_PROMPTS=4 bash tools/run_evidence_enhancement_v1.sh ablation-small

Outputs:
  VARIANT/apply/DATASET/results.txt
  VARIANT/apply/DATASET/evidence_enhancement_rank*.jsonl
  summary/evidence_enhancement_dataset_summary.csv
  summary/evidence_enhancement_class_summary.csv
  summary/evidence_enhancement_prompt_summary.csv
  summary/evidence_enhancement_image_summary.csv
EOF
        exit 2
        ;;
esac
