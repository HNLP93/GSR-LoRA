export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-t5-base}"
DATA_ROOT="${DATA_ROOT:-/root/GSR-lora/data/glue}"
ADAPTER_DIR="${ADAPTER_DIR:-../save/t5_base_gsr_router_train_compressed}"
OUTPUT_DIR="${OUTPUT_DIR:-../save/t5_base_gsr_router_train_compressed_eval}"
USE_HALF_VALIDATION="${USE_HALF_VALIDATION:-0}"

EXTRA_ARGS=()
if [ "${USE_HALF_VALIDATION}" = "1" ]; then
    EXTRA_ARGS+=(--use_half_validation True)
fi

mkdir -p "${OUTPUT_DIR}"

python ../eval_gsr_lora.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --data_root "${DATA_ROOT}" \
    --tasks cola mnli mrpc qnli qqp rte sst2 stsb \
    --max_length 128 \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 0 \
    --report_to none \
    "${EXTRA_ARGS[@]}" \
    | tee "${OUTPUT_DIR}/eval.log"
