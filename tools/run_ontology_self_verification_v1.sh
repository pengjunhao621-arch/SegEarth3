#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-logs/ontology_self_verification_v1}"
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
    local apply="${2:-true}"
    local gpu_ids="${GPU_IDS:-0,1}"
    local port="${PORT:-30831}"
    local config
    config="$(config_for "${dataset}")"

    local mode_name="apply"
    local apply_value="True"
    if [[ "${apply}" != "true" ]]; then
        mode_name="diagnostic"
        apply_value="False"
    fi
    local out_dir="${ROOT}/${mode_name}/${dataset}"
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
            model.use_ontology_self_verification="${apply_value}" \
            model.dump_ontology_self_verification_stats=True \
            model.ontology_self_verification_stats_path="${out_dir}/ontology_self_verification.jsonl" \
            model.ontology_self_verification_spaces="${OSV_SPACES:-evidence,vision}" \
            model.ontology_self_verification_seed_rule="${OSV_SEED_RULE:-final_score_margin_any_head_agree_local_core}" \
            model.ontology_self_verification_topk="${OSV_TOPK:-3}" \
            model.ontology_self_verification_min_similarity_margin="${OSV_MIN_SIM_MARGIN:-0.35}" \
            model.ontology_self_verification_max_base_gap="${OSV_MAX_BASE_GAP:-0.35}" \
            model.ontology_self_verification_min_candidate_score="${OSV_MIN_CAND_SCORE:--1.0}" \
            model.ontology_self_verification_apply_scope="${OSV_APPLY_SCOPE:-non_bg_to_non_bg}" \
            model.ontology_self_verification_require_top1_seed="${OSV_REQUIRE_TOP1_SEED:-True}" \
            model.ontology_self_verification_include_roles="${OSV_INCLUDE_ROLES:-}" \
            model.ontology_self_verification_exclude_roles="${OSV_EXCLUDE_ROLES:-catch_all}" \
            model.ontology_self_verification_seed_exclude_roles="${OSV_SEED_EXCLUDE_ROLES:-catch_all}" \
            model.seed_final_score_thd="${OSV_SEED_SCORE_THD:-0.35}" \
            model.seed_margin_thd="${OSV_SEED_MARGIN_THD:-0.08}" \
            model.seed_local_consistency_thd="${OSV_SEED_LOCAL_THD:-0.70}" \
            model.seed_core_consistency_thd="${OSV_SEED_CORE_THD:-0.95}"
}

run_parallel() {
    local apply="$1"
    shift
    local datasets=("$@")
    mkdir -p "${ROOT}/launch_logs"
    local index=0
    local port_base="${PORT_BASE:-30840}"
    while (( index < ${#datasets[@]} )); do
        local dataset_a="${datasets[index]}"
        ROOT="${ROOT}" GPU_IDS="${GPU_IDS_A:-0,1}" PORT=$((port_base + index)) \
            bash "$0" eval "${dataset_a}" "${apply}" \
            > "${ROOT}/launch_logs/osv_${apply}_${dataset_a}.log" 2>&1 &
        local pid_a=$!
        index=$((index + 1))
        local pid_b=""
        if (( index < ${#datasets[@]} )); then
            local dataset_b="${datasets[index]}"
            ROOT="${ROOT}" GPU_IDS="${GPU_IDS_B:-2,3}" PORT=$((port_base + index)) \
                bash "$0" eval "${dataset_b}" "${apply}" \
                > "${ROOT}/launch_logs/osv_${apply}_${dataset_b}.log" 2>&1 &
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
    local mode_name="$1"
    mkdir -p "${ROOT}/summary/${mode_name}"
    "${PYTHON_BIN}" tools/summarize_ontology_self_verification.py \
        --inputs "${ROOT}/${mode_name}"/*/ontology_self_verification_rank*.jsonl \
        --output-dir "${ROOT}/summary/${mode_name}"
}

case "${MODE}" in
    eval)
        run_eval "${2:?dataset required}" "${3:-true}"
        ;;
    diagnostic-small)
        run_parallel false "${DATASETS_SMALL[@]}"
        ;;
    diagnostic-suite)
        run_parallel false "${DATASETS_MAIN[@]}"
        ;;
    apply-small)
        run_parallel true "${DATASETS_SMALL[@]}"
        ;;
    apply-suite)
        run_parallel true "${DATASETS_MAIN[@]}"
        ;;
    summarize-diagnostic)
        summarize_mode diagnostic
        ;;
    summarize-apply)
        summarize_mode apply
        ;;
    summarize)
        if compgen -G "${ROOT}/diagnostic/*/ontology_self_verification_rank*.jsonl" > /dev/null; then
            summarize_mode diagnostic
        fi
        if compgen -G "${ROOT}/apply/*/ontology_self_verification_rank*.jsonl" > /dev/null; then
            summarize_mode apply
        fi
        ;;
    *)
        cat <<'EOF'
Usage:
  bash tools/run_ontology_self_verification_v1.sh eval DATASET [true|false]
  bash tools/run_ontology_self_verification_v1.sh diagnostic-small
  bash tools/run_ontology_self_verification_v1.sh diagnostic-suite
  bash tools/run_ontology_self_verification_v1.sh apply-small
  bash tools/run_ontology_self_verification_v1.sh apply-suite
  bash tools/run_ontology_self_verification_v1.sh summarize

Datasets:
  udd5, vdd, vaihingen, potsdam, openearthmap, loveda, isaid

Environment overrides:
  ROOT, GPU_IDS_A, GPU_IDS_B, GPU_IDS, PORT, PORT_BASE, PYTHON_BIN,
  OSV_SPACES, OSV_SEED_RULE, OSV_TOPK, OSV_MIN_SIM_MARGIN,
  OSV_MAX_BASE_GAP, OSV_MIN_CAND_SCORE, OSV_APPLY_SCOPE,
  OSV_REQUIRE_TOP1_SEED, OSV_INCLUDE_ROLES, OSV_EXCLUDE_ROLES,
  OSV_SEED_EXCLUDE_ROLES, OSV_SEED_SCORE_THD, OSV_SEED_MARGIN_THD,
  OSV_SEED_LOCAL_THD, OSV_SEED_CORE_THD
EOF
        exit 2
        ;;
esac
