import argparse
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "figures", "analysis")
MODULE_ORDER = ["q", "k", "v", "o", "wi", "wo"]

COLORS = {
    "ink": "#202124",
    "muted": "#53627A",
    "grid": "#D8E1EC",
    "axis": "#C8D3E0",
    "blue": "#2563EB",
    "blue_light": "#BFD9FF",
    "orange": "#FB7C2B",
    "green": "#2EAD62",
    "teal": "#14B8A6",
    "purple": "#7C3AED",
    "red": "#E11D48",
    "gray": "#6B7280",
}

MODULE_COLORS = {
    "q": "#2563EB",
    "k": "#3B82F6",
    "v": "#60A5FA",
    "o": "#14B8A6",
    "wi": "#10B981",
    "wo": "#059669",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def beautify(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def save(fig, output_dir, name, dpi):
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, f"{name}.png")
    pdf = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


def load_layers(path):
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    layers = []
    for item in report.get("layers", []):
        module = item.get("module")
        if module is None or "retained_rank" not in item:
            continue
        layers.append(
            {
                "module": module,
                "stack": infer_stack(module),
                "block": infer_block(module),
                "module_type": module_type(module),
                "original_rank": int(item.get("original_rank", 16)),
                "retained_rank": float(item["retained_rank"]),
                "prefix_rank": float(item.get("prefix_rank", item["retained_rank"])),
                "rank_reduction_percent": float(item.get("rank_reduction_percent", 0.0)),
                "scores": np.asarray(item.get("scores", []), dtype=np.float64),
                "selection_scores": np.asarray(item.get("selection_scores", item.get("scores", [])), dtype=np.float64),
                "router_usage": np.asarray(item.get("router_usage", []), dtype=np.float64),
                "keep_indices": [int(x) for x in item.get("keep_indices", [])],
                "keep_ranks": [int(x) for x in item.get("keep_ranks", [])],
            }
        )
    if not layers:
        raise RuntimeError(f"No layer records found in {path}")
    return report, layers


def infer_stack(module_name):
    match = re.search(r"(?:^|\.)(encoder|decoder)\.block\.", module_name)
    return match.group(1) if match else "unknown"


def infer_block(module_name):
    match = re.search(r"(?:^|\.)(?:encoder|decoder)\.block\.(\d+)(?:\.|$)", module_name)
    return int(match.group(1)) if match else None


def module_type(module_name):
    return str(module_name).split(".")[-1]


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def values_by_module(layers, field):
    grouped = defaultdict(list)
    for layer in layers:
        grouped[layer["module_type"]].append(layer[field])
    return {module: mean(grouped[module]) for module in MODULE_ORDER if module in grouped}


def curves_by_block(layers, field="retained_rank"):
    by_stack_block = defaultdict(list)
    by_block = defaultdict(list)
    for layer in layers:
        block = layer["block"]
        if block is None:
            continue
        if layer["stack"] in {"encoder", "decoder"}:
            by_stack_block[(layer["stack"], block)].append(layer[field])
        by_block[block].append(layer[field])

    blocks = list(range(max(by_block) + 1))
    curves = {"blocks": blocks, "overall": [mean(by_block[i]) for i in blocks]}
    for stack in ["encoder", "decoder"]:
        curves[stack] = [mean(by_stack_block[(stack, i)]) for i in blocks]
    return curves


def hist_by_rank(layers, field):
    counts = Counter(int(round(layer[field])) for layer in layers)
    xs = np.arange(1, 17)
    ys = np.asarray([counts.get(int(x), 0) for x in xs], dtype=np.float64)
    return xs, ys


def effective_strength(layers):
    raw_share = np.zeros(16, dtype=np.float64)
    scaled_share = np.zeros(16, dtype=np.float64)
    usage = np.zeros(16, dtype=np.float64)
    used_strength_layers = 0
    used_usage_layers = 0

    for layer in layers:
        original_rank = layer["original_rank"]
        scores = layer["selection_scores"]
        keep_indices = [idx for idx in layer["keep_indices"] if 0 <= idx < min(original_rank, len(scores), 16)]
        if original_rank <= 0 or len(scores) == 0 or not keep_indices:
            continue

        raw = np.zeros(16, dtype=np.float64)
        scaled = np.zeros(16, dtype=np.float64)
        for idx in keep_indices:
            score = max(float(scores[idx]), 0.0)
            rank_factor = (idx + 1) / original_rank
            raw[idx] = score
            scaled[idx] = score * (rank_factor ** 2)

        if raw.sum() > 0 and scaled.sum() > 0:
            raw_share += raw / raw.sum()
            scaled_share += scaled / scaled.sum()
            used_strength_layers += 1

        router_usage = layer["router_usage"]
        if len(router_usage):
            usable = min(len(router_usage), 16)
            usage[:usable] += router_usage[:usable]
            used_usage_layers += 1

    if used_strength_layers:
        raw_share = raw_share / used_strength_layers * 100.0
        scaled_share = scaled_share / used_strength_layers * 100.0
    if used_usage_layers:
        usage = usage / used_usage_layers * 100.0
    return np.arange(1, 17), raw_share, scaled_share, usage


def heatmap_matrix(layers, stack, field="retained_rank"):
    stack_layers = [layer for layer in layers if layer["stack"] == stack and layer["block"] is not None]
    max_block = max(layer["block"] for layer in stack_layers)
    matrix = np.full((len(MODULE_ORDER), max_block + 1), np.nan, dtype=np.float64)
    for mi, module in enumerate(MODULE_ORDER):
        for block in range(max_block + 1):
            vals = [layer[field] for layer in stack_layers if layer["module_type"] == module and layer["block"] == block]
            if vals:
                matrix[mi, block] = mean(vals)
    return matrix


def plot_module_bars(ax, layers):
    retained = values_by_module(layers, "retained_rank")
    prefix = values_by_module(layers, "prefix_rank")
    modules = list(retained.keys())
    y = np.arange(len(modules))
    h = 0.36
    ax.barh(y - h / 2, [retained[m] for m in modules], height=h, color=COLORS["blue"], label="Retained rank")
    ax.barh(y + h / 2, [prefix[m] for m in modules], height=h, color=COLORS["blue_light"], label="Prefix span")
    ax.set_yticks(y)
    ax.set_yticklabels(modules)
    ax.invert_yaxis()
    ax.set_xlim(0, 16.5)
    ax.set_xlabel("Average rank")
    ax.set_title("Module-wise Rank Allocation", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower right")
    beautify(ax, grid_axis="x")


def plot_layer_curves(ax, layers):
    curves = curves_by_block(layers, field="retained_rank")
    blocks = curves["blocks"]
    ax.plot(blocks, curves["encoder"], color=COLORS["orange"], marker="s", linewidth=2.0, label="Encoder")
    ax.plot(blocks, curves["decoder"], color=COLORS["green"], marker="^", linewidth=2.0, label="Decoder")
    ax.plot(blocks, curves["overall"], color=COLORS["blue"], marker="o", linewidth=2.2, label="Overall")
    ax.axhline(mean(layer["retained_rank"] for layer in layers), color=COLORS["muted"], linestyle="--", linewidth=1.0)
    ax.set_xticks(blocks)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Average retained rank")
    ax.set_title("Layer-wise Retained Rank", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right")
    beautify(ax)


def plot_rank_hist_pair(ax, layers):
    xs, retained = hist_by_rank(layers, "retained_rank")
    _, prefix = hist_by_rank(layers, "prefix_rank")
    width = 0.38
    ax.bar(xs - width / 2, retained, width=width, color=COLORS["blue"], label="Retained rank")
    ax.bar(xs + width / 2, prefix, width=width, color=COLORS["blue_light"], label="Prefix span")
    ax.axvline(mean(layer["retained_rank"] for layer in layers), color=COLORS["blue"], linestyle="--", linewidth=1.0)
    ax.axvline(mean(layer["prefix_rank"] for layer in layers), color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xlabel("Rank value")
    ax.set_ylabel("Number of layers")
    ax.set_title("Retained Rank vs. Prefix Span", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right")
    beautify(ax)


def plot_effective_strength(ax, layers):
    xs, raw_share, scaled_share, _ = effective_strength(layers)
    width = 0.38
    ax.bar(xs - width / 2, raw_share, width=width, color=COLORS["blue_light"], label="Selection score")
    ax.bar(xs + width / 2, scaled_share, width=width, color=COLORS["blue"], label="Scaled strength")
    ax.set_xticks(xs)
    ax.set_xlabel("Original rank position")
    ax.set_ylabel("Average contribution share (%)")
    ax.set_title("Effective Contribution Strength", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right")
    beautify(ax)


def plot_retained_prefix_scatter(ax, layers):
    for module in MODULE_ORDER:
        xs = [layer["retained_rank"] for layer in layers if layer["module_type"] == module]
        ys = [layer["prefix_rank"] for layer in layers if layer["module_type"] == module]
        if xs:
            ax.scatter(xs, ys, s=34, alpha=0.72, color=MODULE_COLORS[module], label=module, edgecolors="white", linewidths=0.35)
    ax.plot([1, 16], [1, 16], color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0.5, 16.5)
    ax.set_xticks(range(1, 17, 3))
    ax.set_yticks(range(1, 17, 3))
    ax.set_xlabel("Retained rank")
    ax.set_ylabel("Prefix span")
    ax.set_title("Non-prefix Group Selection", fontweight="bold", loc="left")
    ax.legend(ncol=3, frameon=False, loc="lower right")
    beautify(ax, grid_axis="both")


def plot_gap_hist(ax, layers):
    gaps = [int(round(layer["prefix_rank"] - layer["retained_rank"])) for layer in layers]
    max_gap = max(gaps)
    xs = np.arange(0, max_gap + 1)
    counts = Counter(gaps)
    ys = [counts.get(int(x), 0) for x in xs]
    ax.bar(xs, ys, color=COLORS["purple"], alpha=0.82)
    ax.set_xticks(xs)
    ax.set_xlabel("Prefix span - retained rank")
    ax.set_ylabel("Number of layers")
    ax.set_title("Skipped Rank Groups", fontweight="bold", loc="left")
    beautify(ax)


def plot_router_usage(ax, layers):
    xs, _, scaled_share, usage = effective_strength(layers)
    ax.plot(xs, usage, marker="o", color=COLORS["green"], linewidth=2.0, label="Router usage")
    ax.bar(xs, scaled_share, color=COLORS["blue_light"], alpha=0.75, label="Scaled strength")
    ax.set_xticks(xs)
    ax.set_xlabel("Original rank position")
    ax.set_ylabel("Average share / usage (%)")
    ax.set_title("Router Usage and Effective Strength", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right")
    beautify(ax)


def plot_prefix_depth_distribution(ax, layers):
    xs, counts = hist_by_rank(layers, "prefix_rank")
    mean_prefix = mean(layer["prefix_rank"] for layer in layers)
    colors = [COLORS["blue_light"] if x < mean_prefix else COLORS["blue"] for x in xs]
    bars = ax.bar(xs, counts, color=colors, width=0.75)
    ax.axvline(mean_prefix, color=COLORS["blue"], linestyle="--", linewidth=1.2)
    ax.text(mean_prefix + 0.2, max(counts) * 0.88, f"mean = {mean_prefix:.2f}", color=COLORS["blue"], fontsize=10)
    for bar, count in zip(bars, counts):
        if count:
            ax.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(int(count)), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xlabel("Selected prefix depth")
    ax.set_ylabel("Number of layers")
    ax.set_title("How Deep a Prefix Is Needed?", fontweight="bold", loc="left")
    beautify(ax)


def plot_prefix_depth_by_module(ax, layers):
    prefix = values_by_module(layers, "prefix_rank")
    retained = values_by_module(layers, "retained_rank")
    modules = list(prefix.keys())
    x = np.arange(len(modules))
    width = 0.36
    ax.bar(x - width / 2, [prefix[m] for m in modules], width=width, color=COLORS["blue_light"], label="Prefix depth")
    ax.bar(x + width / 2, [retained[m] for m in modules], width=width, color=COLORS["blue"], label="Retained rank")
    for xi, module in zip(x, modules):
        ax.text(xi - width / 2, prefix[module] + 0.25, f"{prefix[module]:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(xi + width / 2, retained[module] + 0.25, f"{retained[module]:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(modules)
    ax.set_ylim(0, 16.8)
    ax.set_ylabel("Average rank")
    ax.set_title("Prefix Depth by Module Type", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right")
    beautify(ax)


def plot_rank_savings_by_module(ax, layers):
    grouped = defaultdict(list)
    for layer in layers:
        grouped[layer["module_type"]].append(1.0 - layer["retained_rank"] / layer["original_rank"])
    modules = [m for m in MODULE_ORDER if m in grouped]
    savings = [mean(grouped[m]) * 100.0 for m in modules]
    colors = [MODULE_COLORS[m] for m in modules]
    bars = ax.bar(modules, savings, color=colors)
    for bar, value in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(savings) * 1.2)
    ax.set_ylabel("Average rank saving (%)")
    ax.set_title("Where Does Compression Come From?", fontweight="bold", loc="left")
    beautify(ax)


def plot_budget_flow(ax, layers):
    total_original = sum(layer["original_rank"] for layer in layers)
    total_prefix = sum(layer["prefix_rank"] for layer in layers)
    total_retained = sum(layer["retained_rank"] for layer in layers)
    labels = ["Original\nbudget", "Selected\nprefix depth", "Retained\nrank"]
    values = [total_original, total_prefix, total_retained]
    colors = [COLORS["gray"], COLORS["blue_light"], COLORS["blue"]]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    for bar, value in zip(bars, values):
        pct = value / total_original * 100.0
        ax.text(bar.get_x() + bar.get_width() / 2, value + total_original * 0.02, f"{int(value)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Total rank groups across layers")
    ax.set_title("Rank Budget Funnel", fontweight="bold", loc="left")
    ax.set_ylim(0, total_original * 1.18)
    beautify(ax)


def plot_prefix_depth_heatmap(ax, layers):
    matrix = heatmap_matrix(layers, "encoder", field="prefix_rank")
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=16)
    ax.set_yticks(np.arange(len(MODULE_ORDER)))
    ax.set_yticklabels(MODULE_ORDER)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xlabel("Encoder block index")
    ax.set_title("Encoder Prefix Depth Map", fontweight="bold", loc="left")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=7.0, color=COLORS["ink"])
    return im


def draw_heatmap(ax, matrix, title):
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=16)
    ax.set_yticks(np.arange(len(MODULE_ORDER)))
    ax.set_yticklabels(MODULE_ORDER)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xlabel("Block index")
    ax.set_title(title, fontweight="bold", loc="left")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7.5, color=COLORS["ink"])
    return im


def save_summary(report, layers, output_dir):
    summary = dict(report.get("summary", {}))
    summary.update(
        {
            "num_layers_loaded": len(layers),
            "mean_retained_rank": mean(layer["retained_rank"] for layer in layers),
            "mean_prefix_rank": mean(layer["prefix_rank"] for layer in layers),
            "mean_prefix_retained_gap": mean(layer["prefix_rank"] - layer["retained_rank"] for layer in layers),
            "module_retained_rank": values_by_module(layers, "retained_rank"),
            "module_prefix_span": values_by_module(layers, "prefix_rank"),
        }
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "analysis_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(path)


def main():
    parser = argparse.ArgumentParser(description="Generate GSR-LoRA experiment analysis figures.")
    parser.add_argument("--compression-report", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    setup_style()
    report, layers = load_layers(args.compression_report)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    plot_module_bars(axes[0, 0], layers)
    plot_layer_curves(axes[0, 1], layers)
    plot_rank_hist_pair(axes[1, 0], layers)
    plot_effective_strength(axes[1, 1], layers)
    fig.suptitle("Rank Allocation Analysis of GSR-LoRA", fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, args.output_dir, "rank_allocation_analysis", args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    plot_retained_prefix_scatter(axes[0, 0], layers)
    plot_gap_hist(axes[0, 1], layers)
    plot_router_usage(axes[1, 0], layers)
    plot_effective_strength(axes[1, 1], layers)
    fig.suptitle("Group Selection Mechanism Analysis", fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, args.output_dir, "group_selection_analysis", args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6))
    plot_prefix_depth_distribution(axes[0, 0], layers)
    plot_prefix_depth_by_module(axes[0, 1], layers)
    plot_rank_savings_by_module(axes[1, 0], layers)
    plot_budget_flow(axes[1, 1], layers)
    fig.suptitle("Prefix Budget Analysis", fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, args.output_dir, "prefix_budget_analysis", args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.6, 4.0))
    im = plot_prefix_depth_heatmap(ax, layers)
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.20, 0.018, 0.62])
    fig.colorbar(im, cax=cax, label="Prefix depth")
    save(fig, args.output_dir, "prefix_depth_heatmap", args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 6.8), sharex=True)
    draw_heatmap(axes[0], heatmap_matrix(layers, "encoder"), "Encoder Retained Rank Heatmap")
    im1 = draw_heatmap(axes[1], heatmap_matrix(layers, "decoder"), "Decoder Retained Rank Heatmap")
    fig.subplots_adjust(right=0.88, hspace=0.32)
    cax = fig.add_axes([0.90, 0.16, 0.018, 0.68])
    fig.colorbar(im1, cax=cax, label="Average retained rank")
    save(fig, args.output_dir, "rank_heatmap", args.dpi)
    plt.close(fig)

    save_summary(report, layers, args.output_dir)


if __name__ == "__main__":
    main()
