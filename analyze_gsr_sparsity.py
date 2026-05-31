import argparse
import csv
import json
import os
from collections import defaultdict

import torch

from compress_gsr_lora import (
    load_checkpoint_state,
    lora_b_key_from_lora_a_key,
    module_name_from_lora_a_key,
    strip_base_prefix,
)


def compute_scores(A, B, epsilon):
    A = A.float().cpu()
    B = B.float().cpu()
    return torch.sqrt(A.pow(2).sum(dim=1) + B.pow(2).sum(dim=0) + epsilon)


def prefix_rank_from_mask(mask, min_rank):
    active = torch.nonzero(mask, as_tuple=False).flatten()
    if active.numel() == 0:
        return min_rank
    return max(min_rank, int(active.max().item() + 1))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_layers(layers):
    by_module = defaultdict(list)
    by_rank_position = defaultdict(list)
    for layer in layers:
        module_type = layer["module"].split(".")[-1]
        by_module[module_type].append(layer)
        for idx, value in enumerate(layer["normalized_scores"], start=1):
            by_rank_position[idx].append(value)

    module_summary = {}
    for module_type, items in sorted(by_module.items()):
        module_summary[module_type] = {
            "avg_active_groups": mean([item["active_groups"] for item in items]),
            "avg_prefix_rank": mean([item["prefix_rank"] for item in items]),
            "avg_sparsity_percent": mean([item["sparsity_percent"] for item in items]),
            "num_layers": len(items),
        }

    rank_position_summary = {
        str(idx): mean(values) for idx, values in sorted(by_rank_position.items())
    }

    return module_summary, rank_position_summary


def main():
    parser = argparse.ArgumentParser(description="Analyze Group-Lasso rank-group sparsity before SVD compression.")
    parser.add_argument("--adapter_dir", required=True, help="Checkpoint or adapter directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for sparsity reports.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Normalized score threshold for active groups.")
    parser.add_argument("--min_rank", type=int, default=1, help="Minimum prefix rank used for reporting.")
    parser.add_argument("--epsilon", type=float, default=1e-12)
    args = parser.parse_args()

    state = load_checkpoint_state(args.adapter_dir)
    lora_a_keys = sorted(key for key in state if module_name_from_lora_a_key(key) is not None)
    if not lora_a_keys:
        raise RuntimeError(f"No LoRA A/B weights found in {args.adapter_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    layers = []
    for a_key in lora_a_keys:
        b_key = lora_b_key_from_lora_a_key(a_key)
        if b_key not in state:
            continue

        module = strip_base_prefix(module_name_from_lora_a_key(a_key))
        scores = compute_scores(state[a_key], state[b_key], args.epsilon)
        max_score = scores.max()
        if max_score > 0:
            normalized_scores = scores / max_score
        else:
            normalized_scores = torch.zeros_like(scores)

        mask = normalized_scores >= args.threshold
        active_groups = int(mask.sum().item())
        original_rank = int(scores.numel())
        prefix_rank = prefix_rank_from_mask(mask, args.min_rank)
        sparsity_percent = 100.0 * (1.0 - active_groups / original_rank)

        layers.append(
            {
                "module": module,
                "original_rank": original_rank,
                "active_groups": active_groups,
                "inactive_groups": original_rank - active_groups,
                "prefix_rank": prefix_rank,
                "sparsity_percent": sparsity_percent,
                "scores": [float(x) for x in scores.tolist()],
                "normalized_scores": [float(x) for x in normalized_scores.tolist()],
                "sparsity_mask": [int(x) for x in mask.tolist()],
            }
        )

    module_summary, rank_position_summary = summarize_layers(layers)
    original_rank_avg = mean([item["original_rank"] for item in layers])
    active_avg = mean([item["active_groups"] for item in layers])
    prefix_avg = mean([item["prefix_rank"] for item in layers])
    sparsity_avg = mean([item["sparsity_percent"] for item in layers])

    report = {
        "summary": {
            "adapter_dir": os.path.abspath(args.adapter_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "threshold": args.threshold,
            "num_lora_layers": len(layers),
            "initial_avg_rank": original_rank_avg,
            "avg_active_groups": active_avg,
            "avg_inactive_groups": original_rank_avg - active_avg,
            "avg_prefix_rank": prefix_avg,
            "avg_group_sparsity_percent": sparsity_avg,
            "module_type_summary": module_summary,
            "rank_position_mean_normalized_score": rank_position_summary,
        },
        "layers": layers,
    }

    json_path = os.path.join(args.output_dir, "gsr_sparsity_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    csv_path = os.path.join(args.output_dir, "gsr_sparsity_layers.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "module",
                "original_rank",
                "active_groups",
                "inactive_groups",
                "prefix_rank",
                "sparsity_percent",
                "sparsity_mask",
            ],
        )
        writer.writeheader()
        for item in layers:
            writer.writerow(
                {
                    "module": item["module"],
                    "original_rank": item["original_rank"],
                    "active_groups": item["active_groups"],
                    "inactive_groups": item["inactive_groups"],
                    "prefix_rank": item["prefix_rank"],
                    "sparsity_percent": item["sparsity_percent"],
                    "sparsity_mask": "".join(str(x) for x in item["sparsity_mask"]),
                }
            )

    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
