TRAIN_DIR="${TRAIN_DIR:-../save/t5_base_gsr_router_train}"
if [ -z "${ADAPTER_DIR}" ]; then
    ADAPTER_DIR="$(python - "${TRAIN_DIR}" <<'PY'
import json
import os
import sys

train_dir = sys.argv[1]
state_path = os.path.join(train_dir, "trainer_state.json")
if os.path.exists(state_path):
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    best = state.get("best_model_checkpoint")
    if best:
        print(best)
        raise SystemExit

checkpoints = [
    os.path.join(train_dir, name)
    for name in os.listdir(train_dir)
    if name.startswith("checkpoint-") and os.path.isdir(os.path.join(train_dir, name))
]
if not checkpoints:
    raise SystemExit(f"No checkpoint found under {train_dir}")
checkpoints.sort(key=lambda path: int(path.rsplit("-", 1)[-1]))
print(checkpoints[-1])
PY
)"
fi
OUTPUT_DIR="${OUTPUT_DIR:-../save/t5_base_gsr_router_train_compressed}"
METHOD="${METHOD:-svd}"
THRESHOLD="${THRESHOLD:-0.05}"
ENERGY="${ENERGY:-0.97}"
MIN_RANK="${MIN_RANK:-1}"
MAX_PREFIX_RANK="${MAX_PREFIX_RANK:-0}"
MAX_RETAINED_RANK="${MAX_RETAINED_RANK:-0}"
ROUTER_USAGE_POWER="${ROUTER_USAGE_POWER:-1.0}"
RANDOM_SEED="${RANDOM_SEED:-42}"
RANDOM_KEEP_RANK="${RANDOM_KEEP_RANK:-0}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-t5-base}"
ROUTER_MODE="${ROUTER_MODE:-}"

EXTRA_ARGS=()
if [ -n "${ROUTER_MODE}" ]; then
    EXTRA_ARGS+=(--router_mode "${ROUTER_MODE}")
fi

python ../compress_gsr_lora.py \
    --adapter_dir "${ADAPTER_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --method "${METHOD}" \
    --threshold "${THRESHOLD}" \
    --energy "${ENERGY}" \
    --min_rank "${MIN_RANK}" \
    --max_prefix_rank "${MAX_PREFIX_RANK}" \
    --max_retained_rank "${MAX_RETAINED_RANK}" \
    --router_usage_power "${ROUTER_USAGE_POWER}" \
    --random_seed "${RANDOM_SEED}" \
    --random_keep_rank "${RANDOM_KEEP_RANK}" \
    --base_model_name_or_path "${BASE_MODEL_NAME_OR_PATH}" \
    "${EXTRA_ARGS[@]}"
