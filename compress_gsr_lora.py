import argparse
import json
import math
import os
import shutil
from collections import defaultdict

import torch


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
FULL_WEIGHTS_NAME = "pytorch_model.bin"
TOKENIZER_FILES = {
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}


def load_adapter_state(adapter_dir):
    bin_path = os.path.join(adapter_dir, ADAPTER_WEIGHTS_NAME)
    safe_path = os.path.join(adapter_dir, ADAPTER_SAFE_WEIGHTS_NAME)
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    if os.path.exists(safe_path):
        from safetensors.torch import load_file

        return load_file(safe_path, device="cpu")
    raise FileNotFoundError(f"Could not find {ADAPTER_WEIGHTS_NAME} or {ADAPTER_SAFE_WEIGHTS_NAME} in {adapter_dir}")


def load_checkpoint_state(adapter_dir):
    try:
        return load_adapter_state(adapter_dir)
    except FileNotFoundError:
        full_path = os.path.join(adapter_dir, FULL_WEIGHTS_NAME)
        if os.path.exists(full_path):
            state = torch.load(full_path, map_location="cpu")
            return {key: value for key, value in state.items() if "lora_" in key}
        raise


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


def lora_b_key_from_lora_a_key(key):
    if key.endswith(".lora_A.weight"):
        return key[: -len(".lora_A.weight")] + ".lora_B.weight"
    if key.endswith(".lora_A.default.weight"):
        return key[: -len(".lora_A.default.weight")] + ".lora_B.default.weight"
    return None


def canonical_adapter_key(key, adapter_name="default"):
    # PEFT adapter_model.bin is saved without the adapter-name segment.
    # PeftModel.from_pretrained inserts it again while loading.
    return key.replace(f".{adapter_name}", "")


def forward_rank_factors(rank, max_rank=None, dtype=torch.float32):
    # The project LoRA layer applies k / r_max factors inside forward().
    max_rank = max_rank or rank
    return torch.arange(1, rank + 1, dtype=dtype) / max_rank


def choose_prefix_rank(scores, threshold, min_rank, max_prefix_rank):
    if scores.numel() == 0:
        return min_rank
    max_score = scores.max()
    if max_score <= 0:
        prefix_rank = min_rank
    else:
        normalized = scores / max_score
        active = torch.nonzero(normalized >= threshold, as_tuple=False).flatten()
        prefix_rank = int(active.max().item() + 1) if active.numel() else min_rank
    if max_prefix_rank and max_prefix_rank > 0:
        prefix_rank = min(prefix_rank, max_prefix_rank)
    return max(min_rank, min(prefix_rank, scores.numel()))


def selected_expert_ids_from_router(state, module_name, num_tasks, adapter_name="default"):
    gate_weight, gate_bias, task_embedding = get_router_tensors(state, module_name, adapter_name=adapter_name)
    if gate_weight is None or task_embedding is None:
        return None

    task_embedding = task_embedding[:num_tasks]
    logits = task_embedding @ gate_weight.t()
    if gate_bias is not None:
        logits = logits + gate_bias
    return torch.argmax(logits, dim=1).cpu()


def router_usage_by_rank(state, module_name, rank, num_tasks, adapter_name="default"):
    selected = selected_expert_ids_from_router(state, module_name, num_tasks, adapter_name=adapter_name)
    if selected is None or selected.numel() == 0:
        return torch.ones(rank, dtype=torch.float32), None

    selected = selected.clamp(min=0, max=rank - 1)
    rank_ids = torch.arange(rank, dtype=torch.long).unsqueeze(0)
    usage = (selected.unsqueeze(1) >= rank_ids).float().mean(dim=0)
    return usage, [int(index) for index in selected.tolist()]


def choose_group_indices(selection_scores, threshold, min_rank, max_retained_rank):
    rank = selection_scores.numel()
    if rank == 0:
        return torch.empty(0, dtype=torch.long)

    max_score = selection_scores.max()
    if max_score > 0:
        normalized = selection_scores / max_score
        keep = torch.nonzero(normalized >= threshold, as_tuple=False).flatten()
    else:
        normalized = torch.zeros_like(selection_scores)
        keep = torch.empty(0, dtype=torch.long)

    min_rank = max(1, min(min_rank, rank))
    if keep.numel() < min_rank:
        keep = torch.topk(selection_scores, k=min_rank).indices

    if max_retained_rank and max_retained_rank > 0 and keep.numel() > max_retained_rank:
        max_retained_rank = max(min_rank, min(max_retained_rank, rank))
        candidate_scores = selection_scores[keep]
        keep = keep[torch.topk(candidate_scores, k=max_retained_rank).indices]

    return torch.sort(keep.unique()).values


def low_rank_product_svd(B, A):
    # Efficient SVD for B @ A where rank is tiny.
    # B: [out, r], A: [r, in]
    q_b, r_b = torch.linalg.qr(B, mode="reduced")
    q_a, r_a = torch.linalg.qr(A.t(), mode="reduced")
    core = r_b @ r_a.t()
    u_core, singular_values, vh_core = torch.linalg.svd(core, full_matrices=False)
    u = q_b @ u_core
    vh = vh_core @ q_a.t()
    return u, singular_values, vh


def choose_energy_rank(singular_values, energy, min_rank):
    if singular_values.numel() == 0:
        return min_rank
    available_rank = singular_values.numel()
    min_rank = max(1, min(min_rank, available_rank))
    total = singular_values.pow(2).sum()
    if total <= 0:
        return min_rank
    cumulative = torch.cumsum(singular_values.pow(2), dim=0) / total
    rank = int(torch.searchsorted(cumulative, torch.tensor(energy, dtype=cumulative.dtype)).item() + 1)
    return max(min_rank, min(rank, available_rank))


def compress_pair(A, B, threshold, energy, min_rank, max_prefix_rank, use_forward_rank_scaling):
    original_dtype = A.dtype
    A = A.float().cpu()
    B = B.float().cpu()
    original_rank = A.shape[0]

    scores = torch.sqrt(A.pow(2).sum(dim=1) + B.pow(2).sum(dim=0) + 1e-12)
    prefix_rank = choose_prefix_rank(scores, threshold, min_rank, max_prefix_rank)
    A_prefix = A[:prefix_rank, :].contiguous()
    B_prefix = B[:, :prefix_rank].contiguous()

    if use_forward_rank_scaling:
        factors = forward_rank_factors(prefix_rank, max_rank=original_rank, dtype=A_prefix.dtype)
        A_effective = A_prefix * factors.unsqueeze(1)
        B_effective = B_prefix * factors.unsqueeze(0)
    else:
        factors = torch.ones(prefix_rank, dtype=A_prefix.dtype)
        A_effective = A_prefix
        B_effective = B_prefix

    u, singular_values, vh = low_rank_product_svd(B_effective, A_effective)
    retained_rank = choose_energy_rank(singular_values, energy, min_rank)

    singular_values = singular_values[:retained_rank]
    sqrt_s = torch.sqrt(singular_values)
    B_new_effective = u[:, :retained_rank] * sqrt_s.unsqueeze(0)
    A_new_effective = sqrt_s.unsqueeze(1) * vh[:retained_rank, :]

    if use_forward_rank_scaling:
        retained_factors = forward_rank_factors(retained_rank, max_rank=retained_rank, dtype=A_new_effective.dtype)
        B_new = B_new_effective / retained_factors.unsqueeze(0)
        A_new = A_new_effective / retained_factors.unsqueeze(1)
    else:
        B_new = B_new_effective
        A_new = A_new_effective

    return {
        "A": A_new.to(original_dtype).contiguous(),
        "B": B_new.to(original_dtype).contiguous(),
        "scores": scores,
        "prefix_rank": prefix_rank,
        "retained_rank": retained_rank,
        "original_rank": original_rank,
        "singular_values": singular_values,
    }


def prune_pair(A, B, threshold, min_rank, max_prefix_rank, use_forward_rank_scaling):
    original_dtype = A.dtype
    A = A.float().cpu()
    B = B.float().cpu()
    original_rank = A.shape[0]

    scores = torch.sqrt(A.pow(2).sum(dim=1) + B.pow(2).sum(dim=0) + 1e-12)
    retained_rank = choose_prefix_rank(scores, threshold, min_rank, max_prefix_rank)
    A_new = A[:retained_rank, :].contiguous()
    B_new = B[:, :retained_rank].contiguous()

    if use_forward_rank_scaling and retained_rank != original_rank:
        compensation = retained_rank / original_rank
        A_new = A_new * compensation
        B_new = B_new * compensation

    return {
        "A": A_new.to(original_dtype).contiguous(),
        "B": B_new.to(original_dtype).contiguous(),
        "scores": scores,
        "prefix_rank": retained_rank,
        "retained_rank": retained_rank,
        "original_rank": original_rank,
        "singular_values": torch.empty(0),
    }


def select_pair(
    A,
    B,
    state,
    module_name,
    threshold,
    min_rank,
    max_retained_rank,
    use_forward_rank_scaling,
    num_tasks,
    router_usage_power,
):
    original_dtype = A.dtype
    A = A.float().cpu()
    B = B.float().cpu()
    original_rank = A.shape[0]

    scores = torch.sqrt(A.pow(2).sum(dim=1) + B.pow(2).sum(dim=0) + 1e-12)
    router_usage, original_task_map = router_usage_by_rank(
        state,
        module_name,
        original_rank,
        num_tasks,
    )
    selection_scores = scores * router_usage.pow(router_usage_power)
    keep_indices = choose_group_indices(selection_scores, threshold, min_rank, max_retained_rank)
    retained_rank = int(keep_indices.numel())
    if retained_rank <= 0:
        keep_indices = torch.tensor([int(torch.argmax(selection_scores).item())], dtype=torch.long)
        retained_rank = 1

    A_new = A[keep_indices, :].contiguous()
    B_new = B[:, keep_indices].contiguous()

    if use_forward_rank_scaling:
        old_factors = (keep_indices.float() + 1.0) / original_rank
        new_factors = forward_rank_factors(retained_rank, max_rank=retained_rank, dtype=A_new.dtype)
        compensation = old_factors / new_factors
        A_new = A_new * compensation.unsqueeze(1)
        B_new = B_new * compensation.unsqueeze(0)

    prefix_rank = int(keep_indices.max().item() + 1)

    return {
        "A": A_new.to(original_dtype).contiguous(),
        "B": B_new.to(original_dtype).contiguous(),
        "scores": scores,
        "selection_scores": selection_scores,
        "router_usage": router_usage,
        "prefix_rank": prefix_rank,
        "retained_rank": retained_rank,
        "original_rank": original_rank,
        "keep_indices": keep_indices,
        "original_router_task_map": original_task_map,
        "singular_values": torch.empty(0),
    }


def random_select_pair(
    A,
    B,
    state,
    module_name,
    threshold,
    min_rank,
    max_retained_rank,
    use_forward_rank_scaling,
    num_tasks,
    router_usage_power,
    generator,
    random_keep_rank=0,
):
    original_dtype = A.dtype
    A = A.float().cpu()
    B = B.float().cpu()
    original_rank = A.shape[0]

    scores = torch.sqrt(A.pow(2).sum(dim=1) + B.pow(2).sum(dim=0) + 1e-12)
    router_usage, original_task_map = router_usage_by_rank(
        state,
        module_name,
        original_rank,
        num_tasks,
    )
    selection_scores = scores * router_usage.pow(router_usage_power)

    if random_keep_rank and random_keep_rank > 0:
        retained_rank = random_keep_rank
    else:
        reference_keep = choose_group_indices(selection_scores, threshold, min_rank, max_retained_rank)
        retained_rank = int(reference_keep.numel())
    retained_rank = max(1, min(retained_rank, original_rank))

    keep_indices = torch.randperm(original_rank, generator=generator)[:retained_rank]
    keep_indices = torch.sort(keep_indices).values

    A_new = A[keep_indices, :].contiguous()
    B_new = B[:, keep_indices].contiguous()

    if use_forward_rank_scaling:
        old_factors = (keep_indices.float() + 1.0) / original_rank
        new_factors = forward_rank_factors(retained_rank, max_rank=retained_rank, dtype=A_new.dtype)
        compensation = old_factors / new_factors
        A_new = A_new * compensation.unsqueeze(1)
        B_new = B_new * compensation.unsqueeze(0)

    prefix_rank = int(keep_indices.max().item() + 1)

    return {
        "A": A_new.to(original_dtype).contiguous(),
        "B": B_new.to(original_dtype).contiguous(),
        "scores": scores,
        "selection_scores": selection_scores,
        "router_usage": router_usage,
        "prefix_rank": prefix_rank,
        "retained_rank": retained_rank,
        "original_rank": original_rank,
        "keep_indices": keep_indices,
        "original_router_task_map": original_task_map,
        "singular_values": torch.empty(0),
    }


def summarize_by_module_type(layer_reports):
    groups = defaultdict(list)
    for item in layer_reports:
        module_type = item["module"].split(".")[-1]
        groups[module_type].append(item["retained_rank"])
    return {key: sum(values) / len(values) for key, values in sorted(groups.items())}


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def infer_target_modules(lora_a_keys):
    modules = sorted({strip_base_prefix(module_name_from_lora_a_key(key)).split(".")[-1] for key in lora_a_keys})
    return modules


def build_fallback_config(args, state):
    lora_a_keys = sorted(key for key in state if module_name_from_lora_a_key(key) is not None)
    if not lora_a_keys:
        raise RuntimeError("No LoRA weights found for fallback config generation.")
    first_a = state[lora_a_keys[0]]
    return {
        "base_model_name_or_path": args.base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
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
        "r": int(first_a.shape[0]),
        "rank_pattern": {},
        "alpha_pattern": {},
        "revision": None,
        "target_modules": infer_target_modules(lora_a_keys),
        "task_type": args.task_type,
        "use_dora": False,
        "use_rslora": False,
    }


def find_state_value(state, candidates):
    for key in candidates:
        if key in state:
            return state[key]
    return None


def get_router_tensors(state, module_name, adapter_name="default"):
    gate_weight = find_state_value(
        state,
        [
            f"{module_name}.lora_gate.0.weight",
            f"{module_name}.lora_gate.{adapter_name}.0.weight",
        ],
    )
    gate_bias = find_state_value(
        state,
        [
            f"{module_name}.lora_gate.0.bias",
            f"{module_name}.lora_gate.{adapter_name}.0.bias",
        ],
    )
    task_embedding = find_state_value(
        state,
        [
            f"{module_name}.lora_task_embedding.weight",
            f"{module_name}.lora_task_embedding.{adapter_name}.weight",
        ],
    )
    if gate_weight is None or task_embedding is None:
        return None, None, None
    return gate_weight.float().cpu(), None if gate_bias is None else gate_bias.float().cpu(), task_embedding.float().cpu()


def compute_clamped_task_map(state, module_name, retained_rank, num_tasks, adapter_name="default"):
    gate_weight, gate_bias, task_embedding = get_router_tensors(state, module_name, adapter_name=adapter_name)
    if gate_weight is None or task_embedding is None:
        return None

    task_embedding = task_embedding[:num_tasks]
    logits = task_embedding @ gate_weight.t()
    if gate_bias is not None:
        logits = logits + gate_bias
    selected = torch.argmax(logits, dim=1).clamp(max=retained_rank - 1)
    return [int(index) for index in selected.tolist()]


def save_full_rank_router(compressed_state, module_name, retained_rank, in_features, num_tasks):
    gate_weight = torch.zeros(retained_rank, in_features)
    gate_bias = torch.zeros(retained_rank)
    gate_bias[-1] = 20.0
    task_embedding = torch.zeros(num_tasks, in_features)

    compressed_state[f"{module_name}.lora_gate.0.weight"] = gate_weight.contiguous()
    compressed_state[f"{module_name}.lora_gate.0.bias"] = gate_bias.contiguous()
    compressed_state[f"{module_name}.lora_task_embedding.weight"] = task_embedding.contiguous()
    return 3


def save_clamped_router(compressed_state, state, module_name, retained_rank, in_features, num_tasks, adapter_name="default"):
    task_map = compute_clamped_task_map(
        state,
        module_name,
        retained_rank,
        num_tasks,
        adapter_name=adapter_name,
    )
    if task_map is None:
        save_full_rank_router(compressed_state, module_name, retained_rank, in_features, num_tasks)
        return 3, None

    gate_weight = torch.zeros(retained_rank, in_features)
    gate_bias = torch.zeros(retained_rank)
    task_embedding = torch.zeros(num_tasks, in_features)
    for task_id, expert_id in enumerate(task_map):
        if task_id >= in_features:
            break
        task_embedding[task_id, task_id] = 1.0
        gate_weight[expert_id, task_id] = 20.0

    compressed_state[f"{module_name}.lora_gate.0.weight"] = gate_weight.contiguous()
    compressed_state[f"{module_name}.lora_gate.0.bias"] = gate_bias.contiguous()
    compressed_state[f"{module_name}.lora_task_embedding.weight"] = task_embedding.contiguous()
    return 3, task_map


def compute_remapped_task_map(state, module_name, keep_indices, num_tasks, adapter_name="default"):
    selected = selected_expert_ids_from_router(state, module_name, num_tasks, adapter_name=adapter_name)
    if selected is None:
        return None, None

    keep_indices = torch.as_tensor(keep_indices, dtype=torch.long).cpu()
    remapped = []
    original = []
    for expert_id in selected.tolist():
        old_expert_id = int(expert_id)
        original.append(old_expert_id)
        kept_in_prefix = int((keep_indices <= old_expert_id).sum().item())
        remapped.append(max(0, kept_in_prefix - 1))
    return remapped, original


def save_remapped_router(compressed_state, state, module_name, keep_indices, in_features, num_tasks, adapter_name="default"):
    retained_rank = len(keep_indices)
    task_map, original_task_map = compute_remapped_task_map(
        state,
        module_name,
        keep_indices,
        num_tasks,
        adapter_name=adapter_name,
    )
    if task_map is None:
        save_full_rank_router(compressed_state, module_name, retained_rank, in_features, num_tasks)
        return 3, None, None

    gate_weight = torch.zeros(retained_rank, in_features)
    gate_bias = torch.zeros(retained_rank)
    task_embedding = torch.zeros(num_tasks, in_features)
    for task_id, expert_id in enumerate(task_map):
        if task_id >= in_features:
            break
        task_embedding[task_id, task_id] = 1.0
        gate_weight[expert_id, task_id] = 20.0

    compressed_state[f"{module_name}.lora_gate.0.weight"] = gate_weight.contiguous()
    compressed_state[f"{module_name}.lora_gate.0.bias"] = gate_bias.contiguous()
    compressed_state[f"{module_name}.lora_task_embedding.weight"] = task_embedding.contiguous()
    return 3, task_map, original_task_map


def copy_router_state(state, compressed_state, module_name, retained_rank, adapter_name="default"):
    router_tensors = [
        ("lora_gate", "0.weight", True),
        ("lora_gate", "0.bias", True),
        ("lora_task_embedding", "weight", False),
    ]
    copied = 0
    for adapter_module, tail, truncate_to_rank in router_tensors:
        candidates = [
            f"{module_name}.{adapter_module}.{tail}",
            f"{module_name}.{adapter_module}.{adapter_name}.{tail}",
        ]
        for src_key in candidates:
            if src_key not in state:
                continue
            value = state[src_key]
            if truncate_to_rank and value.ndim > 0:
                value = value[:retained_rank]
            compressed_state[canonical_adapter_key(src_key, adapter_name)] = value.contiguous()
            copied += 1
            break
    return copied


def main():
    parser = argparse.ArgumentParser(description="Compress GSR-LoRA adapters with sparsity-guided rank pruning.")
    parser.add_argument("--adapter_dir", required=True, help="Checkpoint or adapter directory containing adapter_config.json.")
    parser.add_argument("--output_dir", required=True, help="Directory for the compact adapter.")
    parser.add_argument(
        "--method",
        choices=["svd", "group_prune", "group_select", "random_select"],
        default="svd",
        help=(
            "svd rotates the retained prefix into a new low-rank basis; "
            "group_prune keeps an original prefix; group_select keeps non-prefix original rank groups; "
            "random_select keeps random non-prefix rank groups with the same per-layer budget as group_select."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.05, help="Normalized rank-score threshold for prefix rank.")
    parser.add_argument("--energy", type=float, default=0.97, help="SVD cumulative energy retention threshold.")
    parser.add_argument("--min_rank", type=int, default=1, help="Minimum retained rank per LoRA layer.")
    parser.add_argument("--max_prefix_rank", type=int, default=0, help="Maximum prefix candidate rank. Use 0 to disable.")
    parser.add_argument("--max_retained_rank", type=int, default=0, help="Maximum non-prefix retained rank for group_select. Use 0 to disable.")
    parser.add_argument("--router_usage_power", type=float, default=1.0, help="Power applied to router usage in group_select scoring. Use 0 for group scores only.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed used by random_select.")
    parser.add_argument("--random_keep_rank", type=int, default=0, help="Fixed retained rank per layer for random_select. Use 0 to match group_select's per-layer budget.")
    parser.add_argument("--base_model_name_or_path", default="t5-base", help="Used only when adapter_config.json is absent.")
    parser.add_argument("--task_type", default="SEQ_2_SEQ_LM", help="Used only when adapter_config.json is absent.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Used only when adapter_config.json is absent.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Used only when adapter_config.json is absent.")
    parser.add_argument(
        "--disable_forward_rank_scaling",
        action="store_true",
        help="Disable compensation for this project's forward rank scaling factors.",
    )
    parser.add_argument(
        "--preserve_router",
        action="store_true",
        help="Deprecated alias for --router_mode copy.",
    )
    parser.add_argument(
        "--router_mode",
        choices=["full", "copy", "clamp", "remap"],
        default=None,
        help=(
            "full routes every task to the full retained rank; copy truncates the saved router; "
            "clamp precomputes and clamps the saved task map; remap aligns old prefix choices "
            "to non-prefix kept rank groups."
        ),
    )
    parser.add_argument("--num_tasks", type=int, default=8, help="Number of task embeddings for full-rank routing.")
    args = parser.parse_args()

    router_mode = args.router_mode
    if router_mode is None:
        if args.preserve_router:
            router_mode = "copy"
        else:
            if args.method == "group_prune":
                router_mode = "clamp"
            elif args.method in {"group_select", "random_select"}:
                router_mode = "remap"
            else:
                router_mode = "full"

    config_path = os.path.join(args.adapter_dir, ADAPTER_CONFIG_NAME)
    state = load_checkpoint_state(args.adapter_dir)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = build_fallback_config(args, state)

    os.makedirs(args.output_dir, exist_ok=True)
    compressed_state = {}
    rank_pattern = {}
    alpha_pattern = {}
    layer_reports = []
    copied_router_tensors = 0
    random_generator = torch.Generator()
    random_generator.manual_seed(args.random_seed)

    lora_alpha = float(config.get("lora_alpha", 1.0))
    lora_a_keys = sorted(key for key in state if module_name_from_lora_a_key(key) is not None)

    for a_key in lora_a_keys:
        b_key = lora_b_key_from_lora_a_key(a_key)
        if b_key not in state:
            continue

        module_name = module_name_from_lora_a_key(a_key)
        rank_key = strip_base_prefix(module_name)
        if args.method == "group_prune":
            result = prune_pair(
                state[a_key],
                state[b_key],
                threshold=args.threshold,
                min_rank=args.min_rank,
                max_prefix_rank=args.max_prefix_rank,
                use_forward_rank_scaling=not args.disable_forward_rank_scaling,
            )
        elif args.method == "group_select":
            max_retained_rank = args.max_retained_rank if args.max_retained_rank > 0 else args.max_prefix_rank
            result = select_pair(
                state[a_key],
                state[b_key],
                state=state,
                module_name=module_name,
                threshold=args.threshold,
                min_rank=args.min_rank,
                max_retained_rank=max_retained_rank,
                use_forward_rank_scaling=not args.disable_forward_rank_scaling,
                num_tasks=args.num_tasks,
                router_usage_power=args.router_usage_power,
            )
        elif args.method == "random_select":
            max_retained_rank = args.max_retained_rank if args.max_retained_rank > 0 else args.max_prefix_rank
            result = random_select_pair(
                state[a_key],
                state[b_key],
                state=state,
                module_name=module_name,
                threshold=args.threshold,
                min_rank=args.min_rank,
                max_retained_rank=max_retained_rank,
                use_forward_rank_scaling=not args.disable_forward_rank_scaling,
                num_tasks=args.num_tasks,
                router_usage_power=args.router_usage_power,
                generator=random_generator,
                random_keep_rank=args.random_keep_rank,
            )
        else:
            result = compress_pair(
                state[a_key],
                state[b_key],
                threshold=args.threshold,
                energy=args.energy,
                min_rank=args.min_rank,
                max_prefix_rank=args.max_prefix_rank,
                use_forward_rank_scaling=not args.disable_forward_rank_scaling,
            )

        compressed_state[canonical_adapter_key(a_key)] = result["A"]
        compressed_state[canonical_adapter_key(b_key)] = result["B"]
        router_task_map = None
        router_original_task_map = None
        if router_mode == "copy":
            copied_router_tensors += copy_router_state(
                state,
                compressed_state,
                module_name,
                result["retained_rank"],
            )
        elif router_mode == "clamp":
            copied, router_task_map = save_clamped_router(
                compressed_state,
                state,
                module_name,
                result["retained_rank"],
                result["A"].shape[1],
                args.num_tasks,
            )
            copied_router_tensors += copied
        elif router_mode == "remap":
            keep_indices = result.get("keep_indices")
            if keep_indices is None:
                raise ValueError("--router_mode remap requires --method group_select or random_select")
            copied, router_task_map, router_original_task_map = save_remapped_router(
                compressed_state,
                state,
                module_name,
                keep_indices,
                result["A"].shape[1],
                args.num_tasks,
            )
            copied_router_tensors += copied
        else:
            copied_router_tensors += save_full_rank_router(
                compressed_state,
                module_name,
                result["retained_rank"],
                result["A"].shape[1],
                args.num_tasks,
            )
        rank_pattern[rank_key] = result["retained_rank"]
        alpha_pattern[rank_key] = lora_alpha * result["retained_rank"] / result["original_rank"]

        layer_reports.append(
            {
                "module": rank_key,
                "original_rank": result["original_rank"],
                "prefix_rank": result["prefix_rank"],
                "retained_rank": result["retained_rank"],
                "rank_reduction_percent": 100.0 * (1.0 - result["retained_rank"] / result["original_rank"]),
                "scores": [float(x) for x in result["scores"].tolist()],
                "selection_scores": [
                    float(x) for x in result.get("selection_scores", result["scores"]).tolist()
                ],
                "router_usage": [
                    float(x) for x in result.get("router_usage", torch.ones_like(result["scores"])).tolist()
                ],
                "keep_indices": [
                    int(x) for x in result.get("keep_indices", torch.arange(result["retained_rank"])).tolist()
                ],
                "keep_ranks": [
                    int(x) + 1 for x in result.get("keep_indices", torch.arange(result["retained_rank"])).tolist()
                ],
                "singular_values": [float(x) for x in result["singular_values"].tolist()],
                "router_original_task_map": router_original_task_map,
                "router_task_map": router_task_map,
            }
        )

    if not layer_reports:
        raise RuntimeError(f"No LoRA A/B pairs found in {args.adapter_dir}")

    config["r"] = max(item["retained_rank"] for item in layer_reports)
    config["rank_pattern"] = rank_pattern
    config["alpha_pattern"] = alpha_pattern
    config["inference_mode"] = True

    torch.save(compressed_state, os.path.join(args.output_dir, ADAPTER_WEIGHTS_NAME))
    save_json(os.path.join(args.output_dir, ADAPTER_CONFIG_NAME), config)

    for filename in TOKENIZER_FILES:
        src = os.path.join(args.adapter_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, filename))

    original_ranks = [item["original_rank"] for item in layer_reports]
    retained_ranks = [item["retained_rank"] for item in layer_reports]
    summary = {
        "adapter_dir": os.path.abspath(args.adapter_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "num_lora_layers": len(layer_reports),
        "initial_avg_rank": sum(original_ranks) / len(original_ranks),
        "avg_retained_rank": sum(retained_ranks) / len(retained_ranks),
        "rank_reduction_percent": 100.0
        * (1.0 - (sum(retained_ranks) / len(retained_ranks)) / (sum(original_ranks) / len(original_ranks))),
        "threshold": args.threshold,
        "energy": args.energy,
        "method": args.method,
        "svd_input": (
            "full_rank"
            if args.method == "svd" and args.threshold <= 0
            else "gsr_prefix"
            if args.method == "svd"
            else "not_applicable"
        ),
        "min_rank": args.min_rank,
        "max_prefix_rank": args.max_prefix_rank,
        "max_retained_rank": args.max_retained_rank,
        "router_usage_power": args.router_usage_power,
        "random_seed": args.random_seed,
        "random_keep_rank": args.random_keep_rank,
        "preserve_router": args.preserve_router,
        "router_mode": router_mode,
        "module_type_avg_rank": summarize_by_module_type(layer_reports),
        "num_saved_tensors": len(compressed_state),
        "copied_router_tensors": copied_router_tensors,
    }
    report = {"summary": summary, "layers": layer_reports}
    save_json(os.path.join(args.output_dir, "compression_report.json"), report)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
