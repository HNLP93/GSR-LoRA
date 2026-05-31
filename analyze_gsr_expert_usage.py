import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import torch

from compress_gsr_lora import (
    load_checkpoint_state,
    module_name_from_lora_a_key,
    strip_base_prefix,
)


DEFAULT_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def find_state_value(state, candidates):
    for key in candidates:
        if key in state:
            return state[key]
    return None


def get_router_tensors(state, module_name, adapter_name="default"):
    gate_weight = find_state_value(
        state,
        [
            f"{module_name}.lora_gate.{adapter_name}.0.weight",
            f"{module_name}.lora_gate.0.weight",
        ],
    )
    gate_bias = find_state_value(
        state,
        [
            f"{module_name}.lora_gate.{adapter_name}.0.bias",
            f"{module_name}.lora_gate.0.bias",
        ],
    )
    task_embedding = find_state_value(
        state,
        [
            f"{module_name}.lora_task_embedding.{adapter_name}.weight",
            f"{module_name}.lora_task_embedding.weight",
        ],
    )
    return gate_weight, gate_bias, task_embedding


def infer_module_rank(state, module_name, adapter_name="default"):
    candidates = [
        f"{module_name}.lora_A.{adapter_name}.weight",
        f"{module_name}.lora_A.weight",
    ]
    lora_a = find_state_value(state, candidates)
    if lora_a is None:
        return None
    return int(lora_a.shape[0])


def softmax_entropy(logits):
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    if probs.shape[-1] <= 1:
        return torch.zeros_like(entropy)
    return entropy / torch.log(torch.tensor(float(probs.shape[-1])))


def module_type(module_name):
    return module_name.split(".")[-1]


def summarize_records(records, tasks):
    by_task = defaultdict(list)
    by_module_type = defaultdict(list)
    hist = Counter()

    for record in records:
        by_task[record["task"]].append(record)
        by_module_type[record["module_type"]].append(record)
        hist[str(record["selected_rank"])] += 1

    task_summary = {}
    for task in tasks:
        items = by_task.get(task, [])
        task_summary[task] = {
            "num_layers": len(items),
            "avg_selected_expert_id": mean([item["selected_expert_id"] for item in items]),
            "avg_selected_rank": mean([item["selected_rank"] for item in items]),
            "min_selected_rank": min([item["selected_rank"] for item in items]) if items else 0,
            "max_selected_rank": max([item["selected_rank"] for item in items]) if items else 0,
            "avg_rank_fraction": mean([item["rank_fraction"] for item in items]),
            "unique_selected_ranks": len(set(item["selected_rank"] for item in items)),
            "avg_router_entropy": mean([item["router_entropy"] for item in items]),
        }

    module_type_summary = {}
    for name, items in sorted(by_module_type.items()):
        module_type_summary[name] = {
            "num_records": len(items),
            "avg_selected_expert_id": mean([item["selected_expert_id"] for item in items]),
            "avg_selected_rank": mean([item["selected_rank"] for item in items]),
            "avg_rank_fraction": mean([item["rank_fraction"] for item in items]),
            "unique_selected_ranks": len(set(item["selected_rank"] for item in items)),
            "avg_router_entropy": mean([item["router_entropy"] for item in items]),
        }

    selected_ranks = [item["selected_rank"] for item in records]
    rank_fractions = [item["rank_fraction"] for item in records]
    summary = {
        "num_records": len(records),
        "num_lora_layers": len(set(item["module"] for item in records)),
        "num_tasks": len(tasks),
        "avg_selected_expert_id": mean([item["selected_expert_id"] for item in records]),
        "avg_selected_rank": mean(selected_ranks),
        "min_selected_rank": min(selected_ranks) if selected_ranks else 0,
        "max_selected_rank": max(selected_ranks) if selected_ranks else 0,
        "avg_rank_fraction": mean(rank_fractions),
        "unique_selected_ranks": len(set(selected_ranks)),
        "selected_rank_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        "task_summary": task_summary,
        "module_type_summary": module_type_summary,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze GSR-LoRA prefix expert/router usage from a checkpoint.")
    parser.add_argument("--adapter_dir", required=True, help="Checkpoint or adapter directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for expert usage reports.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--adapter_name", default="default")
    args = parser.parse_args()

    state = load_checkpoint_state(args.adapter_dir)
    lora_a_keys = sorted(key for key in state if module_name_from_lora_a_key(key) is not None)
    if not lora_a_keys:
        raise RuntimeError(f"No LoRA weights found in {args.adapter_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    skipped_modules = []
    for a_key in lora_a_keys:
        raw_module_name = module_name_from_lora_a_key(a_key)
        module = strip_base_prefix(raw_module_name)
        rank = infer_module_rank(state, raw_module_name, args.adapter_name)
        gate_weight, gate_bias, task_embedding = get_router_tensors(state, raw_module_name, args.adapter_name)
        if rank is None or gate_weight is None or task_embedding is None:
            skipped_modules.append(module)
            continue

        gate_weight = gate_weight.float().cpu()
        task_embedding = task_embedding.float().cpu()
        if gate_bias is not None:
            gate_bias = gate_bias.float().cpu()

        num_router_experts = int(gate_weight.shape[0])
        num_tasks_available = int(task_embedding.shape[0])
        for task_id, task in enumerate(args.tasks):
            if task_id >= num_tasks_available:
                continue
            logits = task_embedding[task_id] @ gate_weight.t()
            if gate_bias is not None:
                logits = logits + gate_bias
            selected_expert_id = int(torch.argmax(logits).item())
            selected_rank = min(selected_expert_id + 1, rank)
            entropy = float(softmax_entropy(logits.unsqueeze(0))[0].item())
            records.append(
                {
                    "task": task,
                    "task_id": task_id,
                    "module": module,
                    "module_type": module_type(module),
                    "rank": rank,
                    "num_router_experts": num_router_experts,
                    "selected_expert_id": selected_expert_id,
                    "selected_rank": selected_rank,
                    "rank_fraction": selected_rank / rank if rank else 0.0,
                    "router_entropy": entropy,
                }
            )

    if not records:
        raise RuntimeError(
            "No router usage records were produced. This checkpoint may not contain lora_gate/lora_task_embedding."
        )

    summary = summarize_records(records, args.tasks)
    summary.update(
        {
            "adapter_dir": os.path.abspath(args.adapter_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "skipped_modules": len(skipped_modules),
        }
    )

    report = {
        "summary": summary,
        "records": records,
        "skipped_modules": skipped_modules,
    }

    json_path = os.path.join(args.output_dir, "gsr_expert_usage_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    csv_path = os.path.join(args.output_dir, "gsr_expert_usage.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "task",
            "task_id",
            "module",
            "module_type",
            "rank",
            "num_router_experts",
            "selected_expert_id",
            "selected_rank",
            "rank_fraction",
            "router_entropy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
