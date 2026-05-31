import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from data.multi_task_sample import AutoTask, TaskCollator
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from rank import set_task_id


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
FULL_WEIGHTS_NAME = "pytorch_model.bin"
TOKENIZER_FILES = {
    "added_tokens.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}


@dataclass
class CollatorArgs:
    max_length: int


def get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def strip_base_prefix(module_name):
    for prefix in ("base_model.model.", "model."):
        if module_name.startswith(prefix):
            return module_name[len(prefix) :]
    return module_name


def canonical_adapter_key(key, adapter_name="default"):
    return key.replace(f".{adapter_name}", "")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_lora_config(args):
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        lora_variant="alora",
    )


def load_model(args, device):
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    config = T5Config.from_pretrained(args.model_name_or_path)
    base_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    config_path = os.path.join(args.adapter_dir, ADAPTER_CONFIG_NAME)
    full_path = os.path.join(args.adapter_dir, FULL_WEIGHTS_NAME)
    if os.path.exists(config_path):
        model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=False)
    else:
        model = get_peft_model(base_model, build_lora_config(args))
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Could not find {ADAPTER_CONFIG_NAME} or {FULL_WEIGHTS_NAME} in {args.adapter_dir}"
            )
        state = torch.load(full_path, map_location="cpu")
        load_result = model.load_state_dict(state, strict=False)
        print(
            json.dumps(
                {
                    "loaded_full_checkpoint": args.adapter_dir,
                    "missing_keys": len(load_result.missing_keys),
                    "unexpected_keys": len(load_result.unexpected_keys),
                },
                sort_keys=True,
            )
        )

    for module in model.modules():
        if hasattr(module, "lora_variant"):
            module.lora_variant = "alora"

    model.to(device)
    model.eval()
    return model, tokenizer, config


def collect_alora_modules(model):
    modules = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        rank_gate = getattr(module, "lora_rank_gate", None)
        if lora_a is None or lora_b is None or rank_gate is None:
            continue
        if "default" not in lora_a or "default" not in lora_b or "default" not in rank_gate:
            continue
        modules.append((name, module))
    return modules


def build_validation_batches(args, tokenizer, device):
    collator = TaskCollator(tokenizer, CollatorArgs(max_length=args.max_length))
    batches = []
    for task_id, task in enumerate(args.tasks):
        dataset = AutoTask.get(
            task,
            seed=args.seed,
            data_root=args.data_root,
            use_half_validation=args.use_half_validation,
        ).get_dataset(split="validation")
        sample_count = min(args.samples_per_task, len(dataset))
        if sample_count <= 0:
            continue
        batch = [dataset[i] for i in range(sample_count)]
        encoded = collator(batch)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        batches.append((task_id, task, encoded))
    if not batches:
        raise RuntimeError("No validation batches were built for ALoRA scoring.")
    return batches


@torch.no_grad()
def score_model(model, batches):
    losses = []
    for task_id, _, batch in batches:
        set_task_id(task_id)
        outputs = model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        losses.append(loss.detach().float())
    return -torch.stack(losses).mean().item()


def backup_gates(modules):
    return [(module, module.lora_rank_gate["default"].detach().clone()) for _, module in modules]


def restore_gates(gate_backup):
    for module, gate in gate_backup:
        module.lora_rank_gate["default"].data.copy_(gate.to(module.lora_rank_gate["default"].device))


def zero_all_gates(modules):
    for _, module in modules:
        module.lora_rank_gate["default"].data.zero_()


def score_alora_ranks(model, modules, batches, max_score_ranks=0):
    gate_backup = backup_gates(modules)
    base_score = score_model(model, batches)
    rank_scores = []
    scored = 0

    for module_index, (module_name, module) in enumerate(modules):
        rank = module.lora_A["default"].weight.shape[0]
        for rank_index in range(rank):
            if max_score_ranks and scored >= max_score_ranks:
                restore_gates(gate_backup)
                return base_score, rank_scores

            restore_gates(gate_backup)
            module.lora_rank_gate["default"].data[rank_index] = 0.0
            without_score = score_model(model, batches)

            zero_all_gates(modules)
            module.lora_rank_gate["default"].data[rank_index] = 1.0
            only_score = score_model(model, batches)

            importance = base_score - without_score + only_score
            rank_scores.append(
                {
                    "module": module_name,
                    "module_index": module_index,
                    "rank_index": rank_index,
                    "importance": float(importance),
                    "score_without_rank": float(without_score),
                    "score_only_rank": float(only_score),
                }
            )
            scored += 1

    restore_gates(gate_backup)
    return base_score, rank_scores


def allocate_ranks(modules, rank_scores, prune_count, min_rank):
    ranks_by_module = [module.lora_A["default"].weight.shape[0] for _, module in modules]
    keep_indices = [set(range(rank)) for rank in ranks_by_module]
    pruned_by_module = [set() for _ in modules]

    sorted_scores = sorted(rank_scores, key=lambda item: item["importance"])
    for item in sorted_scores:
        if sum(len(values) for values in pruned_by_module) >= prune_count:
            break
        module_index = item["module_index"]
        rank_index = item["rank_index"]
        if len(keep_indices[module_index]) <= min_rank:
            continue
        if rank_index not in keep_indices[module_index]:
            continue
        keep_indices[module_index].remove(rank_index)
        pruned_by_module[module_index].add(rank_index)

    actual_pruned = sum(len(values) for values in pruned_by_module)
    module_avg_scores = defaultdict(list)
    for item in rank_scores:
        module_avg_scores[item["module_index"]].append(item["importance"])
    avg_scores = {
        index: float(np.mean(values)) if values else float("-inf")
        for index, values in module_avg_scores.items()
    }

    unpruned_modules = [
        index for index, values in enumerate(pruned_by_module)
        if not values
    ]
    unpruned_modules.sort(key=lambda index: avg_scores.get(index, float("-inf")), reverse=True)
    added_by_module = [0 for _ in modules]
    if actual_pruned and unpruned_modules:
        base_add = actual_pruned // len(unpruned_modules)
        remainder = actual_pruned % len(unpruned_modules)
        for order, module_index in enumerate(unpruned_modules):
            added_by_module[module_index] = base_add + (1 if order < remainder else 0)

    allocation = []
    for index, (name, _) in enumerate(modules):
        kept = sorted(keep_indices[index])
        allocation.append(
            {
                "module": name,
                "module_index": index,
                "original_rank": ranks_by_module[index],
                "kept_indices": kept,
                "pruned_indices": sorted(pruned_by_module[index]),
                "added_rank": added_by_module[index],
                "final_rank": len(kept) + added_by_module[index],
                "avg_importance": avg_scores.get(index, 0.0),
            }
        )
    return allocation


def init_added_a(num_rows, in_features, dtype, device):
    weight = torch.empty(num_rows, in_features, dtype=torch.float32, device=device)
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    return weight.to(dtype=dtype)


def build_allocated_state(modules, allocation, lora_alpha):
    state = {}
    rank_pattern = {}
    alpha_pattern = {}
    for item, (module_name, module) in zip(allocation, modules):
        lora_a = module.lora_A["default"].weight.detach().cpu()
        lora_b = module.lora_B["default"].weight.detach().cpu()
        rank_gate = module.lora_rank_gate["default"].detach().cpu()

        kept = item["kept_indices"]
        a_new = lora_a[kept, :].clone()
        b_new = lora_b[:, kept].clone()
        gate_new = rank_gate[kept].clone()

        if item["added_rank"] > 0:
            a_extra = init_added_a(item["added_rank"], lora_a.shape[1], lora_a.dtype, lora_a.device).cpu()
            b_extra = torch.zeros(lora_b.shape[0], item["added_rank"], dtype=lora_b.dtype)
            gate_extra = torch.ones(item["added_rank"], dtype=rank_gate.dtype)
            a_new = torch.cat([a_new, a_extra], dim=0)
            b_new = torch.cat([b_new, b_extra], dim=1)
            gate_new = torch.cat([gate_new, gate_extra], dim=0)

        adapter_prefix = canonical_adapter_key(module_name)
        state[f"{adapter_prefix}.lora_A.weight"] = a_new.contiguous()
        state[f"{adapter_prefix}.lora_B.weight"] = b_new.contiguous()
        state[f"{adapter_prefix}.lora_rank_gate"] = gate_new.contiguous()

        rank_key = strip_base_prefix(module_name)
        rank_pattern[rank_key] = item["final_rank"]
        alpha_pattern[rank_key] = lora_alpha * item["final_rank"] / item["original_rank"]
    return state, rank_pattern, alpha_pattern


def build_output_config(args, rank_pattern, alpha_pattern):
    config_path = os.path.join(args.adapter_dir, ADAPTER_CONFIG_NAME)
    if os.path.exists(config_path):
        config = load_json(config_path)
    else:
        config = {
            "base_model_name_or_path": args.model_name_or_path,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": False,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "loftq_config": {},
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "megatron_config": None,
            "megatron_core": "megatron.core",
            "modules_to_save": None,
            "peft_type": "LORA",
            "revision": None,
            "target_modules": args.target_modules,
            "task_type": "SEQ_2_SEQ_LM",
            "use_dora": False,
            "use_rslora": False,
        }
    config["r"] = max(rank_pattern.values())
    config["rank_pattern"] = rank_pattern
    config["alpha_pattern"] = alpha_pattern
    config["lora_variant"] = "alora"
    config["inference_mode"] = False
    return config


def summarize_allocation(allocation):
    initial_rank = [item["original_rank"] for item in allocation]
    final_rank = [item["final_rank"] for item in allocation]
    pruned = [len(item["pruned_indices"]) for item in allocation]
    added = [item["added_rank"] for item in allocation]
    return {
        "num_lora_layers": len(allocation),
        "initial_total_rank": int(sum(initial_rank)),
        "final_total_rank": int(sum(final_rank)),
        "initial_avg_rank": float(np.mean(initial_rank)),
        "final_avg_rank": float(np.mean(final_rank)),
        "total_pruned_rank": int(sum(pruned)),
        "total_added_rank": int(sum(added)),
        "min_final_rank": int(min(final_rank)),
        "max_final_rank": int(max(final_rank)),
    }


def main():
    parser = argparse.ArgumentParser(description="Allocate LoRA ranks with the ALoRA AB-LoRA baseline.")
    parser.add_argument("--model_name_or_path", default="t5-base")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default="data/glue")
    parser.add_argument("--tasks", nargs="+", default=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"])
    parser.add_argument("--target_modules", nargs="+", default=["q", "k", "v", "o", "wi", "wo"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--samples_per_task", type=int, default=4)
    parser.add_argument("--use_half_validation", action="store_true")
    parser.add_argument("--seed", type=int, default=1189)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--prune_count", type=int, default=0)
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--max_score_ranks", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    model, tokenizer, _ = load_model(args, device)
    modules = collect_alora_modules(model)
    if not modules:
        raise RuntimeError("No ALoRA LoRA modules found. Train or load with lora_variant=alora first.")

    prune_count = args.prune_count if args.prune_count > 0 else len(modules)
    batches = build_validation_batches(args, tokenizer, device)
    base_score, rank_scores = score_alora_ranks(model, modules, batches, max_score_ranks=args.max_score_ranks)
    allocation = allocate_ranks(modules, rank_scores, prune_count=prune_count, min_rank=args.min_rank)
    adapter_state, rank_pattern, alpha_pattern = build_allocated_state(modules, allocation, args.lora_alpha)
    config = build_output_config(args, rank_pattern, alpha_pattern)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(adapter_state, os.path.join(args.output_dir, ADAPTER_WEIGHTS_NAME))
    save_json(os.path.join(args.output_dir, ADAPTER_CONFIG_NAME), config)
    for filename in TOKENIZER_FILES:
        src = os.path.join(args.adapter_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, filename))

    summary = summarize_allocation(allocation)
    summary.update(
        {
            "adapter_dir": os.path.abspath(args.adapter_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "base_score": base_score,
            "scored_ranks": len(rank_scores),
            "requested_prune_count": prune_count,
            "samples_per_task": args.samples_per_task,
        }
    )
    report = {
        "summary": summary,
        "allocation": allocation,
        "rank_scores": rank_scores,
    }
    save_json(os.path.join(args.output_dir, "alora_allocation_report.json"), report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
