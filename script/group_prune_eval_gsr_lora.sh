#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SOURCE_ADAPTER_DIR="${ADAPTER_DIR:-/root/GSR-lora/save/t5_base_gsr_router_train/checkpoint-12000}"
COMPRESSED_DIR="${OUTPUT_DIR:-/root/GSR-lora/save/t5_base_gsr_router_train_group_prune_12000}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-t5-base}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
DATA_ROOT="${DATA_ROOT:-/root/GSR-lora/data/glue}"
THRESHOLD="${THRESHOLD:-0.05}"
MIN_RANK="${MIN_RANK:-1}"
MAX_PREFIX_RANK="${MAX_PREFIX_RANK:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

echo "[group-prune] threshold=${THRESHOLD}, min_rank=${MIN_RANK}, max_prefix_rank=${MAX_PREFIX_RANK}"
ADAPTER_DIR="${SOURCE_ADAPTER_DIR}" \
OUTPUT_DIR="${COMPRESSED_DIR}" \
METHOD="group_prune" \
ROUTER_MODE="clamp" \
THRESHOLD="${THRESHOLD}" \
MIN_RANK="${MIN_RANK}" \
MAX_PREFIX_RANK="${MAX_PREFIX_RANK}" \
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH}" \
bash compress_gsr_lora.sh

if [ "${RUN_EVAL}" = "1" ]; then
    echo "[eval] ${COMPRESSED_DIR}"
    ADAPTER_DIR="${COMPRESSED_DIR}" \
    OUTPUT_DIR="${COMPRESSED_DIR}_eval" \
    MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
    DATA_ROOT="${DATA_ROOT}" \
    bash eval_compressed_gsr_lora.sh
fi
