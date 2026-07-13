#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ADAPTER_DIR="${ADAPTER_DIR:-/root/GSR-lora/save/t5_base_gsr_router_train/checkpoint-12000}"
OUTPUT_DIR="${OUTPUT_DIR:-${ADAPTER_DIR%/}_expert_usage}"

python ../analyze_gsr_expert_usage.py \
    --adapter_dir "${ADAPTER_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --tasks cola mnli mrpc qnli qqp rte sst2 stsb
