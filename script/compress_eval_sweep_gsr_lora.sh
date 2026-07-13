#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SOURCE_ADAPTER_DIR="${ADAPTER_DIR:-/root/GSR-lora/save/t5_base_gsr_router_train/checkpoint-12000}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/root/GSR-lora/save/t5_base_gsr_pure_svd_12000}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-t5-base}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
DATA_ROOT="${DATA_ROOT:-/root/GSR-lora/data/glue}"
THRESHOLD="${THRESHOLD:-0}"
MAX_PREFIX_RANK="${MAX_PREFIX_RANK:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

run_variant() {
    local tag="$1"
    local energy="$2"
    local min_rank="$3"
    local compressed_dir="${OUTPUT_PREFIX}_${tag}"
    local eval_dir="${compressed_dir}_eval"

    echo "[pure-svd] ${tag}: threshold=${THRESHOLD}, energy=${energy}, min_rank=${min_rank}"
    ADAPTER_DIR="${SOURCE_ADAPTER_DIR}" \
    OUTPUT_DIR="${compressed_dir}" \
    METHOD="svd" \
    ROUTER_MODE="full" \
    THRESHOLD="${THRESHOLD}" \
    ENERGY="${energy}" \
    MIN_RANK="${min_rank}" \
    MAX_PREFIX_RANK="${MAX_PREFIX_RANK}" \
    BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH}" \
    bash compress_gsr_lora.sh

    if [ "${RUN_EVAL}" = "1" ]; then
        echo "[eval] ${tag}: ${compressed_dir}"
        ADAPTER_DIR="${compressed_dir}" \
        OUTPUT_DIR="${eval_dir}" \
        MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
        DATA_ROOT="${DATA_ROOT}" \
        bash eval_compressed_gsr_lora.sh
    fi
}

run_variant "e099_min8" "0.99" "8"
run_variant "e0995_min10" "0.995" "10"
