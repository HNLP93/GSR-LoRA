from dataclasses import dataclass, field
import json
import os
from typing import List, Optional

import numpy as np
import torch
from transformers import HfArgumentParser, T5Config, T5ForConditionalGeneration, T5Tokenizer, TrainingArguments
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from data.multi_task_sample import AutoTask, TaskCollator
from metrics import accuracy, matthews_corrcoef, pearson_corrcoef
from rank import get_task_id, set_task_id
from trainer import MyTrainer


@dataclass
class EvalArguments:
    model_name_or_path: str = field(default="t5-large")
    adapter_dir: str = field(default=None)
    data_root: str = field(default="data/glue")
    tasks: Optional[List[str]] = field(default_factory=lambda: ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"])
    max_length: int = field(default=128)
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["q", "k", "v", "o", "wi", "wo"])
    lora_rank: int = field(default=16)
    original_lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_variant: str = field(default="gsr")
    use_half_validation: bool = field(default=False)


def lmap(f, x):
    return list(map(f, x))


def strip_base_prefix(module_name):
    for prefix in ("base_model.model.", "model."):
        if module_name.startswith(prefix):
            return module_name[len(prefix) :]
    return module_name


def module_name_from_lora_a_key(key):
    for suffix in (".lora_A.weight", ".lora_A.default.weight"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def infer_rank_pattern(state):
    rank_pattern = {}
    for key, value in state.items():
        module_name = module_name_from_lora_a_key(key)
        if module_name is None:
            continue
        rank_pattern[strip_base_prefix(module_name)] = int(value.shape[0])
    return rank_pattern


def load_lora_model(base_model, eval_args):
    adapter_config_path = os.path.join(eval_args.adapter_dir, "adapter_config.json")
    full_checkpoint_path = os.path.join(eval_args.adapter_dir, "pytorch_model.bin")
    if os.path.exists(adapter_config_path):
        return PeftModel.from_pretrained(base_model, eval_args.adapter_dir)

    if not os.path.exists(full_checkpoint_path):
        raise FileNotFoundError(
            f"Could not find adapter_config.json or pytorch_model.bin in {eval_args.adapter_dir}"
        )

    state = torch.load(full_checkpoint_path, map_location="cpu")
    rank_pattern = infer_rank_pattern(state)
    if not rank_pattern:
        raise RuntimeError(f"No LoRA weights found in {full_checkpoint_path}")

    alpha_pattern = {
        key: eval_args.lora_alpha * rank / eval_args.original_lora_rank
        for key, rank in rank_pattern.items()
    }
    lora_config = LoraConfig(
        r=max(rank_pattern.values()),
        lora_alpha=eval_args.lora_alpha,
        target_modules=eval_args.target_modules,
        lora_dropout=eval_args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        lora_variant=eval_args.lora_variant,
    )
    model = get_peft_model(base_model, lora_config)
    load_result = model.load_state_dict(state, strict=False)
    print(
        json.dumps(
            {
                "loaded_full_checkpoint": eval_args.adapter_dir,
                "missing_keys": len(load_result.missing_keys),
                "unexpected_keys": len(load_result.unexpected_keys),
            },
            sort_keys=True,
        )
    )
    return model


def summarize_loaded_lora(model):
    ranks = []
    b_norms = []
    for _, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None or "default" not in lora_a or "default" not in lora_b:
            continue
        ranks.append(int(lora_a["default"].weight.shape[0]))
        b_norms.append(float(lora_b["default"].weight.detach().float().norm().cpu()))
    if not ranks:
        return {"num_lora_layers": 0}
    return {
        "num_lora_layers": len(ranks),
        "avg_lora_rank": float(np.mean(ranks)),
        "min_lora_rank": int(np.min(ranks)),
        "max_lora_rank": int(np.max(ranks)),
        "zero_lora_b_layers": int(sum(norm == 0.0 for norm in b_norms)),
    }


def main():
    parser = HfArgumentParser((TrainingArguments, EvalArguments))
    training_args, eval_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    if eval_args.adapter_dir is None:
        raise ValueError("--adapter_dir is required")

    config = T5Config.from_pretrained(eval_args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(eval_args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(eval_args.model_name_or_path)
    model = load_lora_model(model, eval_args)
    model.eval()
    print(json.dumps({"loaded_adapter_summary": summarize_loaded_lora(model)}, sort_keys=True))

    eval_datasets = {
        task: AutoTask.get(
            task,
            seed=1189,
            data_root=eval_args.data_root,
            use_half_validation=eval_args.use_half_validation,
        ).get_dataset(split="validation")
        for task in eval_args.tasks
    }

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids
        pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_ids[label_ids == -100] = 0
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)

        task_name = eval_args.tasks[get_task_id()] if get_task_id() < len(eval_args.tasks) else None
        if task_name == "cola":
            return matthews_corrcoef(pred_str, label_str)
        if task_name == "stsb":
            pred_str = [float(pred) if pred.replace(".", "", 1).isdigit() else 0.0 for pred in pred_str]
            label_str = [float(label) for label in label_str]
            return pearson_corrcoef(pred_str, label_str)
        return accuracy(pred_str, label_str)

    trainer = MyTrainer(
        model=model,
        config=config,
        data_args=eval_args,
        args=training_args,
        eval_dataset=None,
        data_collator=TaskCollator(tokenizer, data_args=eval_args),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    all_metrics = {}
    for index, (task, dataset) in enumerate(eval_datasets.items()):
        set_task_id(index)
        metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=f"eval_{task}")
        all_metrics.update(metrics)

    task_scores = [value for key, value in all_metrics.items() if key.endswith("_acc") or key.endswith("_mcc") or key.endswith("_pearson_corrcoef")]
    losses = [value for key, value in all_metrics.items() if key.endswith("_loss")]
    all_metrics["eval_average_metrics"] = float(np.mean(task_scores)) if task_scores else 0.0
    all_metrics["eval_loss"] = float(np.mean(losses)) if losses else 0.0
    print(all_metrics)


if __name__ == "__main__":
    main()
