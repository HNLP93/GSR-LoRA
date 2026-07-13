import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict, OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "figures", "paper_redraw")
MODULE_ORDER = ["q", "k", "v", "o", "wi", "wo"]


BASELINES = OrderedDict(
    [
        ("Full Fine-tuning", {"params": 28.00, "score": 83.8, "kind": "full"}),
        ("MoRE", {"params": 0.78, "score": 83.8, "kind": "baseline"}),
        ("MoDE", {"params": 0.78, "score": 83.5, "kind": "baseline"}),
        ("Fixed-rank LoRA", {"params": 0.78, "score": 83.4, "kind": "baseline"}),
        ("ALoRA", {"params": 0.78, "score": 83.4, "kind": "baseline"}),
        ("MALoRA", {"params": 0.78, "score": 83.0, "kind": "baseline"}),
        ("AutoLoRA", {"params": 0.78, "score": 82.9, "kind": "baseline"}),
        ("SoRA", {"params": 0.36, "score": 83.2, "kind": "sora"}),
    ]
)


COLORS = {
    "ink": "#202124",
    "muted": "#53627A",
    "grid": "#D8E1EC",
    "axis": "#C8D3E0",
    "blue": "#2563EB",
    "blue2": "#60A5FA",
    "orange": "#FB7C2B",
    "green": "#2EAD62",
    "teal": "#14B8A6",
    "gray": "#404040",
    "light_blue": "#BFD9FF",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
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
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=1.0, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure(fig, output_dir, name, dpi):
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{name}.png")
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


def load_report(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_module_ranks(items):
    ranks = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected module rank as name=value, got: {item}")
        name, value = item.split("=", 1)
        ranks[name.strip()] = float(value)
    return ranks


def module_type(module_name):
    return str(module_name).split(".")[-1]


def infer_stack_and_block(module_name):
    match = re.search(r"(?:^|\.)(encoder|decoder)\.block\.(\d+)(?:\.|$)", module_name)
    if match:
        return match.group(1), int(match.group(2))
    match = re.search(r"(?:^|\.)block\.(\d+)(?:\.|$)", module_name)
    if match:
        return "overall", int(match.group(1))
    return None, None


def layers_from_report(report):
    if not report:
        return []
    layers = report.get("layers", [])
    clean_layers = []
    for item in layers:
        if "module" not in item or "retained_rank" not in item:
            continue
        clean_layers.append(
            {
                "module": item["module"],
                "retained_rank": float(item["retained_rank"]),
                "prefix_rank": float(item.get("prefix_rank", item["retained_rank"])),
                "original_rank": int(item.get("original_rank", len(item.get("scores", [])) or 16)),
                "scores": item.get("scores", []),
                "selection_scores": item.get("selection_scores", item.get("scores", [])),
                "keep_indices": item.get("keep_indices", []),
            }
        )
    return clean_layers


def module_ranks_from_layers(layers):
    values = defaultdict(list)
    for item in layers:
        values[module_type(item["module"])].append(item["retained_rank"])
    return {name: float(np.mean(vals)) for name, vals in values.items()}


def layer_curves_from_layers(layers):
    by_stack_block = defaultdict(list)
    by_block = defaultdict(list)
    for item in layers:
        stack, block = infer_stack_and_block(item["module"])
        if block is None:
            continue
        if stack in {"encoder", "decoder"}:
            by_stack_block[(stack, block)].append(item["retained_rank"])
        by_block[block].append(item["retained_rank"])

    if not by_block:
        return None

    max_block = max(by_block)
    blocks = list(range(max_block + 1))
    curves = {
        "blocks": blocks,
        "overall": [float(np.mean(by_block[idx])) if by_block[idx] else np.nan for idx in blocks],
    }
    for stack in ["encoder", "decoder"]:
        values = []
        found = False
        for idx in blocks:
            ranks = by_stack_block[(stack, idx)]
            if ranks:
                found = True
                values.append(float(np.mean(ranks)))
            else:
                values.append(np.nan)
        if found:
            curves[stack] = values
    return curves


def rank_hist_from_layers(layers, field="retained_rank"):
    counts = Counter(int(round(item[field])) for item in layers)
    ranks = list(range(1, 17))
    values = [counts.get(rank, 0) for rank in ranks]
    mean_rank = float(np.mean([item[field] for item in layers])) if layers else 0.0
    return ranks, values, mean_rank


def effective_strength_by_rank(layers):
    raw_share = np.zeros(16, dtype=np.float64)
    scaled_share = np.zeros(16, dtype=np.float64)
    used_layers = 0

    for item in layers:
        original_rank = int(item.get("original_rank", 0))
        selection_scores = np.asarray(item.get("selection_scores") or item.get("scores") or [], dtype=np.float64)
        keep_indices = [int(idx) for idx in item.get("keep_indices", [])]
        if original_rank <= 0 or selection_scores.size == 0 or not keep_indices:
            continue

        keep_indices = [idx for idx in keep_indices if 0 <= idx < min(original_rank, selection_scores.size, 16)]
        if not keep_indices:
            continue

        raw = np.zeros(16, dtype=np.float64)
        scaled = np.zeros(16, dtype=np.float64)
        for idx in keep_indices:
            rank_factor = (idx + 1) / original_rank
            score = max(float(selection_scores[idx]), 0.0)
            raw[idx] = score
            scaled[idx] = score * (rank_factor ** 2)

        raw_sum = raw.sum()
        scaled_sum = scaled.sum()
        if raw_sum <= 0 or scaled_sum <= 0:
            continue

        raw_share += raw / raw_sum
        scaled_share += scaled / scaled_sum
        used_layers += 1

    ranks = np.arange(1, 17)
    if used_layers:
        raw_share = raw_share / used_layers * 100.0
        scaled_share = scaled_share / used_layers * 100.0
    return ranks, raw_share, scaled_share, used_layers


def plot_tradeoff(ax, ours_params, ours_score):
    label_positions = {
        "Full Fine-tuning": (18.0, 83.92, "left"),
        "MoRE": (1.50, 83.86, "left"),
        "MoDE": (1.50, 83.64, "left"),
        "Fixed-rank LoRA": (1.50, 83.46, "left"),
        "ALoRA": (1.50, 83.31, "left"),
        "MALoRA": (1.50, 83.06, "left"),
        "AutoLoRA": (1.50, 82.84, "left"),
        "SoRA": (0.31, 83.28, "right"),
    }
    for name, item in BASELINES.items():
        if item["kind"] == "full":
            color, marker, size = COLORS["gray"], "o", 70
        elif item["kind"] == "sora":
            color, marker, size = COLORS["blue"], "o", 70
        else:
            color, marker, size = COLORS["blue"], "o", 70
        ax.scatter(item["params"], item["score"], s=size, c=color, marker=marker, zorder=3)
        label_x, label_y, ha = label_positions[name]
        label = f"{name}\n({item['params']:.2f}, {item['score']:.1f})"
        ax.annotate(
            label,
            (item["params"], item["score"]),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=9.5,
            fontweight="bold" if name in {"MoRE", "SoRA", "Full Fine-tuning"} else "normal",
            color=COLORS["ink"],
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", color="#8B8B8B", lw=0.85, shrinkA=2, shrinkB=4)
            if name not in {"Full Fine-tuning"}
            else None,
        )

    ax.scatter(ours_params, ours_score, s=260, marker="*", c="#FF5A1F", edgecolor="white", linewidth=0.9, zorder=5)
    ax.annotate(
        f"Ours\n({ours_params:.2f}, {ours_score:.2f})",
        (ours_params, ours_score),
        xytext=(24, 6),
        textcoords="offset points",
        color="#F05A16",
        fontsize=14,
        fontweight="bold",
        va="center",
    )

    ax.set_xscale("log")
    ax.set_xlim(0.09, 100)
    ax.set_ylim(82.5, max(84.75, ours_score + 0.2))
    ax.set_xlabel("Trainable Parameters (M)")
    ax.set_ylabel("Average Score")
    ax.set_title("Performance-Parameter Trade-off on GLUE", fontweight="bold", pad=16)
    beautify(ax, grid_axis="both")


def plot_module_rank(ax, module_ranks):
    modules = [name for name in MODULE_ORDER if name in module_ranks]
    if not modules:
        raise ValueError("No module-rank data available.")
    values = [module_ranks[name] for name in modules]
    y = np.arange(len(modules))
    palette = ["#2563EB", "#3B82F6", "#60A5FA", "#14B8A6", "#10B981", "#059669"]
    bars = ax.barh(y, values, color=palette[: len(modules)], height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(modules)
    ax.invert_yaxis()
    ax.set_xlim(0, 16.8)
    ax.set_xlabel("Average kept rank")
    ax.set_title("Average Kept Rank by Module Type", fontweight="bold", loc="left", pad=12)
    for bar, value in zip(bars, values):
        ax.text(value + 0.12, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=13, color="#344054")
    beautify(ax, grid_axis="x")


def plot_rank_hist(ax, ranks, counts, mean_rank, title="Distribution of Effective Layerwise Rank", xlabel="Kept rank per layer"):
    colors = [COLORS["light_blue"] if rank < round(mean_rank) else COLORS["blue2"] for rank in ranks]
    peak_idx = int(np.argmax(counts)) if counts else 0
    if counts:
        colors[peak_idx] = COLORS["blue"]
    bars = ax.bar(ranks, counts, color=colors, width=0.72)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, count + 0.8, str(count), ha="center", va="bottom", fontsize=12, color="#344054")
    ax.axvline(mean_rank, color=COLORS["blue"], linestyle="--", linewidth=1.6)
    ymax = max(counts) if counts else 1
    ax.text(mean_rank + 0.12, ymax * 0.9, f"mean = {mean_rank:.2f}", color=COLORS["blue"], fontsize=13)
    ax.set_xlim(0.25, 16.75)
    ax.set_xticks(range(1, 17))
    ax.set_ylim(0, ymax * 1.15 + 2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of layers")
    ax.set_title(title, fontweight="bold", loc="left", pad=12)
    beautify(ax)


def plot_effective_strength(ax, ranks, raw_share, scaled_share):
    width = 0.38
    x = np.asarray(ranks)
    bars_raw = ax.bar(
        x - width / 2,
        raw_share,
        width=width,
        color=COLORS["light_blue"],
        label="Selection score",
    )
    bars_scaled = ax.bar(
        x + width / 2,
        scaled_share,
        width=width,
        color=COLORS["blue"],
        label="Scaled effective strength",
    )
    ax.set_xlim(0.25, 16.75)
    ax.set_xticks(range(1, 17))
    ax.set_xlabel("Original rank position")
    ax.set_ylabel("Average contribution share per layer (%)")
    ax.set_title("Distribution of Effective Contribution Strength", fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left")
    ymax = max(float(np.max(raw_share)), float(np.max(scaled_share)), 1.0)
    ax.set_ylim(0, ymax * 1.18)
    for bars, values in [(bars_scaled, scaled_share)]:
        for bar, value in zip(bars, values):
            if value >= ymax * 0.08:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.015,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#344054",
                )
    beautify(ax)


def plot_layer_rank(ax, curves, mean_rank=None):
    blocks = curves["blocks"]
    plotted = []
    if "encoder" in curves:
        ax.plot(blocks, curves["encoder"], color=COLORS["orange"], marker="s", markersize=5, linewidth=2.2, label="Encoder")
        plotted.extend([v for v in curves["encoder"] if not np.isnan(v)])
    if "decoder" in curves:
        ax.plot(blocks, curves["decoder"], color=COLORS["green"], marker="^", markersize=6, linewidth=2.2, label="Decoder")
        plotted.extend([v for v in curves["decoder"] if not np.isnan(v)])
    ax.plot(blocks, curves["overall"], color=COLORS["blue"], marker="o", markersize=6, linewidth=2.6, label="Overall")
    plotted.extend([v for v in curves["overall"] if not np.isnan(v)])
    if mean_rank is None:
        mean_rank = float(np.mean(plotted)) if plotted else 0.0
    ax.axhline(mean_rank, color=COLORS["muted"], linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(
        blocks[-1] - 0.15,
        mean_rank + 0.08,
        f"Mean = {mean_rank:.2f}",
        color=COLORS["muted"],
        fontsize=12,
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
    )
    ax.set_xticks(blocks)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Average retained rank")
    ax.set_title("Layer-wise Retained Rank Across Transformer Blocks", fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ymin = max(0, min(plotted) - 0.7) if plotted else 0
    ymax = max(plotted) + 0.7 if plotted else 1
    ax.set_ylim(ymin, ymax)
    beautify(ax)


def main():
    parser = argparse.ArgumentParser(description="Redraw paper figures from GSR-LoRA compression reports.")
    parser.add_argument("--compression-report", default=None, help="Path to compression_report.json.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--module-ranks", nargs="*", default=None, help="Fallback module ranks, e.g. q=9.0278 k=4.1111.")
    parser.add_argument("--ours-score", type=float, default=84.66151134001889)
    parser.add_argument("--ours-rank", type=float, default=5.458333333333333)
    parser.add_argument("--fixed-lora-params", type=float, default=0.78)
    parser.add_argument("--ours-params", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    setup_style()
    report = load_report(args.compression_report)
    layers = layers_from_report(report)
    if report and not layers:
        raise RuntimeError(f"No usable layer records found in {args.compression_report}")

    report_summary = report.get("summary", {}) if report else {}
    module_ranks = {}
    if layers:
        module_ranks = module_ranks_from_layers(layers)
    elif "module_type_avg_rank" in report_summary:
        module_ranks = {name: float(value) for name, value in report_summary["module_type_avg_rank"].items()}
    module_ranks.update(parse_module_ranks(args.module_ranks))

    ours_rank = report_summary.get("avg_retained_rank", args.ours_rank)
    ours_params = args.ours_params
    if ours_params is None:
        ours_params = args.fixed_lora_params * float(ours_rank) / 16.0

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    plot_tradeoff(ax, ours_params=ours_params, ours_score=args.ours_score)
    fig.tight_layout()
    save_figure(fig, args.output_dir, "glue_tradeoff", args.dpi)
    plt.close(fig)

    if module_ranks:
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        plot_module_rank(ax, module_ranks)
        fig.tight_layout()
        save_figure(fig, args.output_dir, "fig_module_rank", args.dpi)
        plt.close(fig)
    else:
        print("MISSING: module-rank data. Provide --compression-report or --module-ranks q=... k=... v=... o=... wi=... wo=...")

    if layers:
        ranks, counts, mean_rank = rank_hist_from_layers(layers, field="retained_rank")
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        plot_rank_hist(ax, ranks, counts, mean_rank)
        fig.tight_layout()
        save_figure(fig, args.output_dir, "fig_rank_hist", args.dpi)
        plt.close(fig)

        prefix_ranks, prefix_counts, prefix_mean = rank_hist_from_layers(layers, field="prefix_rank")
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        plot_rank_hist(
            ax,
            prefix_ranks,
            prefix_counts,
            prefix_mean,
            title="Distribution of Selected Prefix Span",
            xlabel="Selected prefix span per layer",
        )
        fig.tight_layout()
        save_figure(fig, args.output_dir, "fig_prefix_rank_hist", args.dpi)
        plt.close(fig)

        strength_ranks, raw_share, scaled_share, used_layers = effective_strength_by_rank(layers)
        if used_layers:
            fig, ax = plt.subplots(figsize=(8.8, 5.2))
            plot_effective_strength(ax, strength_ranks, raw_share, scaled_share)
            fig.tight_layout()
            save_figure(fig, args.output_dir, "fig_effective_strength", args.dpi)
            plt.close(fig)
        else:
            print("MISSING: selection_scores/keep_indices data for effective-strength plot.")

        curves = layer_curves_from_layers(layers)
        if curves is None:
            print("MISSING: could not infer encoder/decoder block indices from layer module names.")
        else:
            fig, ax = plt.subplots(figsize=(10.8, 5.6))
            plot_layer_rank(ax, curves, mean_rank=mean_rank)
            fig.tight_layout()
            save_figure(fig, args.output_dir, "fig_layer_rank", args.dpi)
            plt.close(fig)
    else:
        print("MISSING: per-layer retained_rank records. Provide compression_report.json to draw fig_rank_hist and fig_layer_rank.")


if __name__ == "__main__":
    main()
