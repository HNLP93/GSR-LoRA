#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SOURCE_ADAPTER_DIR="${ADAPTER_DIR:-/root/GSR-lora/save/t5_base_gsr_router_train/checkpoint-5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/GSR-lora/save/pruning_tradeoff_group_select_5000}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-t5-base}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"
DATA_ROOT="${DATA_ROOT:-/root/GSR-lora/data/glue}"

THRESHOLDS="${THRESHOLDS:-0.01 0.03 0.05 0.07 0.10}"
MIN_RANK="${MIN_RANK:-1}"
MAX_RETAINED_RANK="${MAX_RETAINED_RANK:-0}"
ROUTER_USAGE_POWER="${ROUTER_USAGE_POWER:-1.0}"
USE_HALF_VALIDATION="${USE_HALF_VALIDATION:-0}"
RUN_UNCOMPRESSED="${RUN_UNCOMPRESSED:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE="${FORCE:-0}"
ORIGINAL_RANK="${ORIGINAL_RANK:-16}"

SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_ROOT}/pruning_tradeoff_summary.json}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/pruning_tradeoff_summary.csv}"

mkdir -p "${OUTPUT_ROOT}"

threshold_tag() {
    local threshold="$1"
    echo "t${threshold//./p}"
}

collect_summary() {
    local variant="$1"
    local threshold="$2"
    local adapter_dir="$3"
    local eval_dir="$4"

    python - "$SUMMARY_JSON" "$SUMMARY_CSV" "$variant" "$threshold" "$adapter_dir" "$eval_dir" "$ORIGINAL_RANK" <<'PY'
import ast
import csv
import json
import os
import sys

summary_json, summary_csv, variant, threshold, adapter_dir, eval_dir, original_rank = sys.argv[1:8]
original_rank = float(original_rank)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_eval_log(path):
    metrics = {}
    if not os.path.exists(path):
        return metrics
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{") or "eval_average_metrics" not in line:
            continue
        try:
            payload = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(payload, dict):
            metrics.update(payload)
            break
    return metrics


compression = load_json(os.path.join(adapter_dir, "compression_report.json"))
compression_summary = compression.get("summary", {}) if compression else {}
eval_metrics = parse_eval_log(os.path.join(eval_dir, "eval.log"))

if compression_summary:
    avg_rank = float(compression_summary.get("avg_retained_rank", original_rank))
    reduction = float(compression_summary.get("rank_reduction_percent", 0.0))
    initial_rank = float(compression_summary.get("initial_avg_rank", original_rank))
    num_layers = int(compression_summary.get("num_lora_layers", 0))
else:
    avg_rank = original_rank
    reduction = 0.0
    initial_rank = original_rank
    num_layers = 0

record = {
    "variant": variant,
    "threshold": None if threshold in {"", "None", "none"} else float(threshold),
    "adapter_dir": os.path.abspath(adapter_dir),
    "eval_dir": os.path.abspath(eval_dir),
    "initial_avg_rank": initial_rank,
    "avg_retained_rank": avg_rank,
    "rank_reduction_percent": reduction,
    "num_lora_layers": num_layers,
    "eval_average_metrics": eval_metrics.get("eval_average_metrics"),
    "eval_loss": eval_metrics.get("eval_loss"),
}

for key, value in eval_metrics.items():
    if key.startswith("eval_") and (
        key.endswith("_acc")
        or key.endswith("_mcc")
        or key.endswith("_pearson_corrcoef")
        or key.endswith("_loss")
    ):
        record[key] = value

records = []
if os.path.exists(summary_json):
    with open(summary_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    records = loaded.get("records", loaded if isinstance(loaded, list) else [])

records = [
    item
    for item in records
    if not (item.get("variant") == record["variant"] and item.get("threshold") == record["threshold"])
]
records.append(record)
records.sort(key=lambda item: (-1.0 if item.get("threshold") is None else item.get("threshold", 0.0)))

payload = {
    "source_adapter_dir": os.path.abspath(os.environ.get("SOURCE_ADAPTER_DIR", "")),
    "records": records,
}
os.makedirs(os.path.dirname(summary_json), exist_ok=True)
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)

fieldnames = [
    "variant",
    "threshold",
    "avg_retained_rank",
    "rank_reduction_percent",
    "eval_average_metrics",
    "eval_loss",
    "adapter_dir",
    "eval_dir",
]
with open(summary_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for item in records:
        writer.writerow({key: item.get(key) for key in fieldnames})

print(json.dumps(record, sort_keys=True))
PY
}

run_eval_if_needed() {
    local adapter_dir="$1"
    local eval_dir="$2"
    if [ "${RUN_EVAL}" != "1" ]; then
        return
    fi
    if [ "${FORCE}" != "1" ] && [ -s "${eval_dir}/eval.log" ]; then
        echo "[skip-eval] ${eval_dir}/eval.log exists"
        return
    fi
    echo "[eval] ${adapter_dir}"
    ADAPTER_DIR="${adapter_dir}" \
    OUTPUT_DIR="${eval_dir}" \
    MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
    DATA_ROOT="${DATA_ROOT}" \
    USE_HALF_VALIDATION="${USE_HALF_VALIDATION}" \
    bash eval_compressed_gsr_lora.sh
}

export SOURCE_ADAPTER_DIR

echo "[sweep] source checkpoint: ${SOURCE_ADAPTER_DIR}"
echo "[sweep] output root: ${OUTPUT_ROOT}"
echo "[sweep] thresholds: ${THRESHOLDS}"

if [ "${RUN_UNCOMPRESSED}" = "1" ]; then
    uncompressed_eval_dir="${OUTPUT_ROOT}/uncompressed_eval"
    run_eval_if_needed "${SOURCE_ADAPTER_DIR}" "${uncompressed_eval_dir}"
    collect_summary "uncompressed" "None" "${SOURCE_ADAPTER_DIR}" "${uncompressed_eval_dir}"
fi

for threshold in ${THRESHOLDS}; do
    tag="$(threshold_tag "${threshold}")"
    compressed_dir="${OUTPUT_ROOT}/group_select_${tag}"
    eval_dir="${compressed_dir}_eval"

    if [ "${FORCE}" != "1" ] && [ -s "${compressed_dir}/compression_report.json" ]; then
        echo "[skip-compress] ${compressed_dir}/compression_report.json exists"
    else
        echo "[compress] group_select threshold=${threshold}"
        ADAPTER_DIR="${SOURCE_ADAPTER_DIR}" \
        OUTPUT_DIR="${compressed_dir}" \
        METHOD="group_select" \
        ROUTER_MODE="remap" \
        THRESHOLD="${threshold}" \
        MIN_RANK="${MIN_RANK}" \
        MAX_RETAINED_RANK="${MAX_RETAINED_RANK}" \
        ROUTER_USAGE_POWER="${ROUTER_USAGE_POWER}" \
        BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH}" \
        bash compress_gsr_lora.sh
    fi

    run_eval_if_needed "${compressed_dir}" "${eval_dir}"
    collect_summary "group_select_remap" "${threshold}" "${compressed_dir}" "${eval_dir}"
done

echo "[done] summary json: ${SUMMARY_JSON}"
echo "[done] summary csv: ${SUMMARY_CSV}"
echo "[next] plot with: python plot_pruning_tradeoff.py --summary ${SUMMARY_JSON}"
