#!/usr/bin/env bash

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-t5-base}"
DATA_ROOT="${DATA_ROOT:-/root/GSR-lora/data/glue}"
OUTPUT_DIR="${OUTPUT_DIR:-../save/t5_base_gsr_router_train}"
ROUTER_ENTROPY_LAMBDA="${ROUTER_ENTROPY_LAMBDA:-1e-3}"
ROUTER_RANK_FRACTION_LAMBDA="${ROUTER_RANK_FRACTION_LAMBDA:-${ROUTER_RANK_LAMBDA:-1e-4}}"
DATALOADER_EPOCHS="${DATALOADER_EPOCHS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
SEED="${SEED:-2023}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-True}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_ARGS=(
        --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
        --ignore_data_skip "${IGNORE_DATA_SKIP}"
    )
    echo "[train] resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"
    echo "[train] ignore_data_skip=${IGNORE_DATA_SKIP}"
fi

mkdir -p "${OUTPUT_DIR}"

python ../finetune.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_root "${DATA_ROOT}" \
    --tasks cola mnli mrpc qnli qqp rte sst2 stsb \
    --max_length 128 \
    --use_lora True \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --target_modules q k v o wi wo \
    --use_gsr True \
    --gsr_lambda 1e-5 \
    --gsr_power 1.0 \
    --gsr_epsilon 1e-8 \
    --cl_lambda 0.1 \
    --router_entropy_lambda "${ROUTER_ENTROPY_LAMBDA}" \
    --router_rank_fraction_lambda "${ROUTER_RANK_FRACTION_LAMBDA}" \
    --output_dir "${OUTPUT_DIR}" \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_average_metrics \
    --greater_is_better True \
    --dataloader_epochs "${DATALOADER_EPOCHS}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --logging_steps 100 \
    --dataloader_num_workers 0 \
    --seed "${SEED}" \
    "${RESUME_ARGS[@]}" \
    | tee "${OUTPUT_DIR}/train.log"
