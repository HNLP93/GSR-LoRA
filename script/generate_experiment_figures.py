import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "figures")


METHODS = OrderedDict(
    [
        ("Full FT", {"avg": 83.8, "params": 28.0, "rank": None, "red": None}),
        ("LoRA", {"avg": 83.4, "params": 0.78, "rank": 16.00, "red": 0.0}),
        ("MoRE", {"avg": 83.8, "params": 0.78, "rank": 16.00, "red": 0.0}),
        ("MoDE", {"avg": 83.5, "params": 0.78, "rank": 16.00, "red": 0.0}),
        ("MALoRA", {"avg": 83.0, "params": 0.78, "rank": 16.00, "red": 0.0}),
        ("ALoRA", {"avg": 83.4, "params": 0.78, "rank": 16.00, "red": 0.0}),
        ("AutoLoRA", {"avg": 82.9, "params": 0.78, "rank": 16.00, "red": 0.0}),
        ("SoRA", {"avg": 83.2, "params": 0.36, "rank": 8.70, "red": 54.3}),
        ("GSR-LoRA", {"avg": 84.6615, "params": 0.27, "rank": 5.4583, "red": 65.8854}),
    ]
)


TASKS = ["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI", "QNLI", "RTE"]
TASK_SCORES = {
    "LoRA": [55.4, 93.3, 88.2, 90.2, 89.1, 86.3, 93.1, 71.4],
    "SoRA": [52.9, 93.9, 89.9, 88.7, 89.4, 86.6, 91.5, 72.6],
    "GSR-LoRA": [60.4103, 94.1514, 87.2549, 90.2470, 90.2943, 86.2557, 93.2272, 75.4513],
}


MODULE_RANKS = OrderedDict(
    [
        ("q", 9.0278),
        ("k", 4.1111),
        ("v", 4.7500),
        ("o", 3.5278),
        ("wi", 7.6250),
        ("wo", 3.9167),
    ]
)


CHECKPOINTS = OrderedDict(
    [
        ("Unpruned\nckpt-5000", {"avg": 84.7444, "rank": 16.0000, "red": 0.0}),
        ("Group-select\nckpt-3000", {"avg": 84.6882, "rank": 6.1146, "red": 61.7839}),
        ("Group-select\nckpt-5000", {"avg": 84.6615, "rank": 5.4583, "red": 65.8854}),
    ]
)


COLORS = {
    "ink": "#1F2937",
    "muted": "#667085",
    "grid": "#D0D5DD",
    "blue": "#4C78A8",
    "blue_light": "#DCEBFF",
    "green": "#2F855A",
    "green_light": "#DDF3E4",
    "orange": "#D9822B",
    "orange_light": "#FDE8D2",
    "purple": "#7E57C2",
    "purple_light": "#EDE7F6",
    "gray": "#8A94A6",
    "red": "#C44E52",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
        }
    )


def beautify(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], alpha=0.45, linewidth=0.7)
    ax.set_axisbelow(True)


def plot_tradeoff(ax):
    label_offsets = {
        "Full FT": (0, 9),
        "LoRA": (-14, -16),
        "MoRE": (5, 10),
        "MoDE": (7, -6),
        "MALoRA": (8, -14),
        "ALoRA": (-35, 8),
        "AutoLoRA": (-38, -8),
        "SoRA": (-20, -18),
        "GSR-LoRA": (8, 9),
    }
    for name, item in METHODS.items():
        if name == "GSR-LoRA":
            color, marker, size, zorder = COLORS["green"], "*", 210, 5
        elif name == "Full FT":
            color, marker, size, zorder = COLORS["gray"], "s", 70, 3
        elif name == "SoRA":
            color, marker, size, zorder = COLORS["orange"], "D", 85, 4
        else:
            color, marker, size, zorder = COLORS["blue"], "o", 60, 3
        ax.scatter(item["params"], item["avg"], s=size, c=color, marker=marker, edgecolors="white", linewidths=0.8, zorder=zorder)
        dx, dy = label_offsets[name]
        ax.annotate(
            name,
            (item["params"], item["avg"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=7.5,
            color=COLORS["ink"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Adapter / trainable parameters (M, log scale)")
    ax.set_ylabel("Average GLUE score")
    ax.set_title("Performance-parameter trade-off", weight="bold")
    ax.set_ylim(82.55, 84.95)
    ax.set_xlim(0.2, 40)
    beautify(ax, grid_axis="both")


def plot_task_scores(ax):
    x = np.arange(len(TASKS))
    width = 0.24
    series = [("LoRA", COLORS["blue"]), ("SoRA", COLORS["orange"]), ("GSR-LoRA", COLORS["green"])]
    for idx, (name, color) in enumerate(series):
        ax.bar(x + (idx - 1) * width, TASK_SCORES[name], width=width, label=name, color=color, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Task-level comparison", weight="bold")
    ax.set_ylim(50, 98)
    ax.legend(ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.01))
    beautify(ax)


def plot_compression(ax):
    names = ["LoRA", "SoRA", "GSR-LoRA"]
    ranks = [METHODS[name]["rank"] for name in names]
    reds = [METHODS[name]["red"] for name in names]
    params = [METHODS[name]["params"] for name in names]
    x = np.arange(len(names))
    width = 0.28
    ax2 = ax.twinx()
    bars_rank = ax.bar(x - width / 2, ranks, width=width, label="Avg rank", color=COLORS["blue"], alpha=0.88)
    bars_param = ax.bar(x + width / 2, params, width=width, label="Params (M)", color=COLORS["purple"], alpha=0.82)
    line = ax2.plot(x, reds, color=COLORS["green"], marker="o", linewidth=2.0, label="Rank reduction")[0]
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Avg rank / Params (M)")
    ax2.set_ylabel("Rank reduction (%)")
    ax.set_ylim(0, 18)
    ax2.set_ylim(0, 75)
    ax.set_title("Compression statistics", weight="bold")
    for bar, value in zip(bars_rank, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5)
    for bar, value in zip(bars_param, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5)
    for xi, value in zip(x, reds):
        ax2.text(xi, value + 2.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=7.5, color=COLORS["green"])
    handles = [bars_rank, bars_param, line]
    labels = ["Avg rank", "Params (M)", "Rank reduction"]
    ax.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.03))
    beautify(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["grid"])
    ax2.tick_params(axis="y", colors=COLORS["muted"])


def plot_module_ranks(ax):
    modules = list(MODULE_RANKS.keys())
    ranks = list(MODULE_RANKS.values())
    x = np.arange(len(modules))
    colors = [COLORS["green"] if m in ("q", "wi") else COLORS["blue"] for m in modules]
    bars = ax.bar(x, ranks, color=colors, alpha=0.9)
    ax.axhline(16, color=COLORS["red"], linestyle="--", linewidth=1.2, label="Initial rank = 16")
    ax.set_xticks(x)
    ax.set_xticklabels(modules)
    ax.set_ylabel("Average retained rank")
    ax.set_title("Module-wise retained rank of GSR-LoRA", weight="bold")
    ax.set_ylim(0, 17.5)
    for bar, value in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=7.8)
    ax.legend(frameon=False, loc="upper right")
    beautify(ax)


def plot_checkpoint_tradeoff(ax):
    x = [item["rank"] for item in CHECKPOINTS.values()]
    y = [item["avg"] for item in CHECKPOINTS.values()]
    names = list(CHECKPOINTS.keys())
    colors = [COLORS["gray"], COLORS["orange"], COLORS["green"]]
    ax.plot(x, y, color=COLORS["muted"], linewidth=1.2, alpha=0.7)
    for xi, yi, name, color in zip(x, y, names, colors):
        ax.scatter(xi, yi, s=90, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(name, (xi, yi), xytext=(8, 5), textcoords="offset points", fontsize=7.5, color=COLORS["ink"])
    ax.invert_xaxis()
    ax.set_xlabel("Average retained rank (lower is smaller)")
    ax.set_ylabel("Average GLUE score")
    ax.set_title("Checkpoint pruning trade-off", weight="bold")
    ax.set_ylim(84.60, 84.78)
    ax.set_xlim(17, 4.5)
    beautify(ax, grid_axis="both")


def save(fig, name):
    png = os.path.join(OUT_DIR, f"{name}.png")
    pdf = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(pdf)
    print(png)


def make_all():
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4))
    plot_tradeoff(axes[0, 0])
    plot_task_scores(axes[0, 1])
    plot_compression(axes[1, 0])
    plot_module_ranks(axes[1, 1])
    fig.suptitle("Experimental Results of GSR-LoRA on GLUE", y=1.02, fontsize=15, weight="bold", color=COLORS["ink"])
    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_style()

    fig = make_all()
    save(fig, "experiment_results")
    plt.close(fig)

    single_specs = [
        ("glue_tradeoff", plot_tradeoff, (6.2, 4.2)),
        ("task_scores", plot_task_scores, (7.4, 4.0)),
        ("compression_stats", plot_compression, (6.2, 4.0)),
        ("module_retained_rank", plot_module_ranks, (6.2, 4.0)),
        ("checkpoint_pruning_tradeoff", plot_checkpoint_tradeoff, (6.2, 4.0)),
    ]
    for name, fn, size in single_specs:
        fig, ax = plt.subplots(figsize=size)
        fn(ax)
        fig.tight_layout()
        save(fig, name)
        plt.close(fig)


if __name__ == "__main__":
    main()
