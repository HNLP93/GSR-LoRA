from data.multi_task_sample import AutoTask, TaskCollator, MultiTaskDataLoader
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, HfArgumentParser, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Optional, List, Union
from dataclasses import dataclass, field
from trainer import MyTrainer
import torch
import random
import numpy as np
import os
import datasets
from rank import *
from metrics import accuracy, pearson_corrcoef, matthews_corrcoef


def freeze_alora_rank_gates(model):
    for name, param in model.named_parameters():
        if "lora_rank_gate" in name:
            param.requires_grad = False


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
        if module_name is not None:
            rank_pattern[strip_base_prefix(module_name)] = int(value.shape[0])
    return rank_pattern


def load_trainable_lora_model(base_model, data_args):
    adapter_config_path = os.path.join(data_args.adapter_name_or_path, "adapter_config.json")
    full_checkpoint_path = os.path.join(data_args.adapter_name_or_path, "pytorch_model.bin")
    if os.path.exists(adapter_config_path):
        return PeftModel.from_pretrained(base_model, data_args.adapter_name_or_path, is_trainable=True)

    if not os.path.exists(full_checkpoint_path):
        raise FileNotFoundError(
            f"Could not find adapter_config.json or pytorch_model.bin in {data_args.adapter_name_or_path}"
        )

    state = torch.load(full_checkpoint_path, map_location="cpu")
    rank_pattern = infer_rank_pattern(state)
    if not rank_pattern:
        raise RuntimeError(f"No LoRA weights found in {full_checkpoint_path}")

    original_lora_rank = data_args.original_lora_rank or data_args.lora_rank
    alpha_pattern = {
        key: data_args.lora_alpha * rank / original_lora_rank
        for key, rank in rank_pattern.items()
    }
    lora_config = LoraConfig(
        r=max(rank_pattern.values()),
        lora_alpha=data_args.lora_alpha,
        target_modules=data_args.target_modules,
        lora_dropout=data_args.lora_dropout,
        lora_variant=data_args.lora_variant,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    model = get_peft_model(base_model, lora_config)
    load_result = model.load_state_dict(state, strict=False)
    print(
        {
            "loaded_full_checkpoint": data_args.adapter_name_or_path,
            "missing_keys": len(load_result.missing_keys),
            "unexpected_keys": len(load_result.unexpected_keys),
        }
    )
    return model

@dataclass
class MyArguments:
    model_name_or_path: Optional[str] = field(default=None)
    data_root: Optional[str] = field(default="data/glue")
    tasks: Optional[List[str]] = field(default_factory=lambda: ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2'])
    max_length: Optional[int] = field(default=128)
    use_lora: bool = field(default=True)
    adapter_name_or_path: Optional[str] = field(default=None)
    lora_rank: Optional[int] = field(default=16)
    original_lora_rank: Optional[int] = field(default=None)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_variant: Optional[str] = field(default="gsr")
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["query", "key", "value", "decoder"])
    dataloader_epochs: Optional[float] = field(default=1.0)
    epochs: Optional[float] = field(default=None)
    cl_lambda: Optional[float] = field(default=0.1)
    use_gsr: bool = field(default=False)
    gsr_lambda: Optional[float] = field(default=1e-5)
    gsr_power: Optional[float] = field(default=1.0)
    gsr_epsilon: Optional[float] = field(default=1e-8)
    router_entropy_lambda: Optional[float] = field(default=0.0)
    router_rank_fraction_lambda: Optional[float] = field(default=0.0)
    router_rank_lambda: Optional[float] = field(default=0.0)
    use_half_validation: bool = field(default=False)
    smooth_distribution: bool = field(default=True)
    sample_by_loss: bool = field(default=False)
    use_dyrank: bool = field(default=False)
    use_share_module: bool = field(default=False)

parser = HfArgumentParser((TrainingArguments, MyArguments))
training_args, data_args = parser.parse_args_into_dataclasses()
training_args.remove_unused_columns = False
training_args.save_safetensors = False

torch.manual_seed(training_args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(training_args.seed)
elif hasattr(torch, "npu") and torch.npu.is_available():
    torch.npu.manual_seed_all(training_args.seed)
np.random.seed(training_args.seed)
random.seed(training_args.seed)

#tasks = ['stsb']
tasks = data_args.tasks
print(tasks)
#['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']
dataset_class = AutoTask
train_datasets = [dataset_class.get(task, data_root=data_args.data_root).get_dataset(
    split="train") for task in tasks]#1238 1079
eval_datasets = ({task: dataset_class.get(
                    task,
                    seed=1189,
                    data_root=data_args.data_root,
                    use_half_validation=data_args.use_half_validation,
                ).get_dataset(
                split="validation") for task in tasks})

dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
print(train_datasets)
print({tasks[i]:dataset_sizes[i] for i in range(len(tasks))})
# train_datasets = datasets.concatenate_datasets(train_datasets)
# print(train_datasets)

# 加载 RoBERTa tokenizer
config = T5Config.from_pretrained(data_args.model_name_or_path)
tokenizer = T5Tokenizer.from_pretrained(data_args.model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(data_args.model_name_or_path)

if data_args.use_lora and data_args.adapter_name_or_path:
    model = load_trainable_lora_model(model, data_args)
    freeze_alora_rank_gates(model)
    print(f"Loaded trainable adapter from {data_args.adapter_name_or_path}")
    model.print_trainable_parameters()
elif data_args.use_lora:
    lora_config = LoraConfig(
        r=data_args.lora_rank,
        lora_alpha=data_args.lora_alpha,
        target_modules=data_args.target_modules,
        # 即论文图中右边两个模块输出的dropout
        lora_dropout=data_args.lora_dropout,
        lora_variant=data_args.lora_variant,
        bias="none",
        # 说明任务类型，从而影响模型的架构，loss，输出形式
        #task_type='CAUSAL_LM'
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)
    freeze_alora_rank_gates(model)
    print(data_args.target_modules)
    model.print_trainable_parameters() 
    #print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
def lmap(f, x) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def compute_metrics(eval_prediction):
    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    #print(predictions, label_ids)
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_ids[label_ids == -100] = 0
    #print(predictions, label_ids)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    #print(pred_str, label_str)
    pred_str = lmap(str.strip, pred_str)
    label_str = lmap(str.strip, label_str)

    task_name = tasks[get_task_id()] if get_task_id() < len(tasks) else None
    if task_name == 'cola': 
        #print(pred_str)hhbt
        acc = matthews_corrcoef(pred_str, label_str)
    elif task_name == 'stsb':
        pred_str = [float(pred) if pred.replace('.', '', 1).isdigit() else 0.0 for pred in pred_str]
        label_str = [float(label) for label in label_str]
        acc = pearson_corrcoef(pred_str, label_str)
    else:
        acc = accuracy(pred_str, label_str)
    return acc

        
my_trainer = MyTrainer(model=model, 
                       config=config,
                        data_args=data_args,
                        args=training_args,
                        train_dataset=train_datasets,
                        eval_dataset=eval_datasets,
                        data_collator=TaskCollator(tokenizer, data_args=data_args),
                        #data_collator=data_collator,
                        compute_metrics=compute_metrics,
                        tokenizer=tokenizer,
                        )

resume_from_checkpoint = training_args.resume_from_checkpoint
if isinstance(resume_from_checkpoint, str):
    normalized_resume_arg = resume_from_checkpoint.strip().lower()
    if normalized_resume_arg in ("true", "1", "yes"):
        resume_from_checkpoint = True
    elif normalized_resume_arg in ("false", "0", "no", "none", ""):
        resume_from_checkpoint = None

if resume_from_checkpoint:
    print(f"Resuming training from checkpoint: {resume_from_checkpoint}")

my_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
