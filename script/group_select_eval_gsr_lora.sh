#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SOURCE_ADAPTER_DIR="${ADAPTER_DIR:-/root/GSR-lora/save/t5_base_gsr_router_train/checkpoint-5000}"
COMPRESSED_DIR="${OUTPUT_DIR:-/root/GSR-lora/save/t5_base_gsr_router_train_group_select_5000}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-t5-base}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
DATA_ROOT="${DATA_ROOT:-/root/GSR-lora/data/glue}"
THRESHOLD="${THRESHOLD:-0.05}"
MIN_RANK="${MIN_RANK:-1}"
MAX_RETAINED_RANK="${MAX_RETAINED_RANK:-0}"
ROUTER_USAGE_POWER="${ROUTER_USAGE_POWER:-1.0}"
RUN_EVAL="${RUN_EVAL:-1}"

echo "[group-select] threshold=${THRESHOLD}, min_rank=${MIN_RANK}, max_retained_rank=${MAX_RETAINED_RANK}, router_usage_power=${ROUTER_USAGE_POWER}"
ADAPTER_DIR="${SOURCE_ADAPTER_DIR}" \
OUTPUT_DIR="${COMPRESSED_DIR}" \
METHOD="group_select" \
ROUTER_MODE="remap" \
THRESHOLD="${THRESHOLD}" \
MIN_RANK="${MIN_RANK}" \
MAX_RETAINED_RANK="${MAX_RETAINED_RANK}" \
ROUTER_USAGE_POWER="${ROUTER_USAGE_POWER}" \
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
