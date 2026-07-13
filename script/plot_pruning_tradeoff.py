import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "ink": "#202124",
    "muted": "#53627A",
    "grid": "#D8E1EC",
    "axis": "#C8D3E0",
    "blue": "#2563EB",
    "blue_light": "#BFD9FF",
    "orange": "#FB7C2B",
    "green": "#2EAD62",
    "gray": "#6B7280",
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


def beautify(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", payload if isinstance(payload, list) else [])
    records = [
        item
        for item in records
        if item.get("eval_average_metrics") is not None
        and item.get("avg_retained_rank") is not None
        and item.get("rank_reduction_percent") is not None
    ]
    if not records:
        raise RuntimeError(f"No evaluated records found in {path}")
    return sorted(records, key=lambda item: item.get("rank_reduction_percent", 0.0))


def threshold_label(record):
    threshold = record.get("threshold")
    if threshold is None:
        return "unpruned"
    return f"t={threshold:g}"


def series(records):
    x_reduction = np.asarray([float(item["rank_reduction_percent"]) for item in records])
    x_rank = np.asarray([float(item["avg_retained_rank"]) for item in records])
    y_score = np.asarray([float(item["eval_average_metrics"]) for item in records])
    y_loss = np.asarray([float(item["eval_loss"]) if item.get("eval_loss") is not None else np.nan for item in records])
    labels = [threshold_label(item) for item in records]
    return x_reduction, x_rank, y_score, y_loss, labels


def plot_score_vs_reduction(ax, records):
    x_reduction, _, y_score, _, labels = series(records)
    ax.plot(x_reduction, y_score, color=COLORS["blue"], marker="o", linewidth=2.0)
    for x, y, label in zip(x_reduction, y_score, labels):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Rank reduction (%)")
    ax.set_ylabel("Average GLUE score")
    ax.set_title("Pruning Strength vs. Performance", fontweight="bold", loc="left")
    beautify(ax)


def plot_score_vs_rank(ax, records):
    _, x_rank, y_score, _, labels = series(records)
    ax.plot(x_rank, y_score, color=COLORS["green"], marker="o", linewidth=2.0)
    for x, y, label in zip(x_rank, y_score, labels):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.invert_xaxis()
    ax.set_xlabel("Average retained rank")
    ax.set_ylabel("Average GLUE score")
    ax.set_title("Rank Budget vs. Performance", fontweight="bold", loc="left")
    beautify(ax)


def plot_rank_and_loss(ax, records):
    x_reduction, x_rank, _, y_loss, labels = series(records)
    ax2 = ax.twinx()
    bars = ax.bar(x_reduction, x_rank, width=3.0, color=COLORS["blue_light"], label="Avg retained rank")
    ax2.plot(x_reduction, y_loss, color=COLORS["orange"], marker="s", linewidth=1.8, label="Eval loss")
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, label, ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Rank reduction (%)")
    ax.set_ylabel("Average retained rank")
    ax2.set_ylabel("Eval loss")
    ax.set_title("Compression and Loss", fontweight="bold", loc="left")
    beautify(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["axis"])
    ax2.tick_params(axis="y", colors=COLORS["muted"])
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")


def plot_task_sensitivity(ax, records):
    task_keys = [
        "eval_cola_mcc",
        "eval_mnli_acc",
        "eval_mrpc_acc",
        "eval_qnli_acc",
        "eval_qqp_acc",
        "eval_rte_acc",
        "eval_sst2_acc",
        "eval_stsb_pearson_corrcoef",
    ]
    task_labels = ["CoLA", "MNLI", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B"]
    baseline = next((item for item in records if item.get("threshold") is None), None)
    strongest = max(records, key=lambda item: item.get("rank_reduction_percent", 0.0))
    if baseline is None:
        baseline = min(records, key=lambda item: item.get("rank_reduction_percent", 0.0))
    deltas = []
    labels = []
    for key, label in zip(task_keys, task_labels):
        if baseline.get(key) is None or strongest.get(key) is None:
            continue
        deltas.append(float(strongest[key]) - float(baseline[key]))
        labels.append(label)
    if not deltas:
        ax.text(0.5, 0.5, "Task-level metrics not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    colors = [COLORS["green"] if value >= 0 else COLORS["orange"] for value in deltas]
    ax.bar(labels, deltas, color=colors)
    ax.axhline(0, color=COLORS["gray"], linewidth=1.0)
    ax.set_ylabel("Score change")
    ax.set_title(f"Task Sensitivity at {threshold_label(strongest)}", fontweight="bold", loc="left")
    ax.tick_params(axis="x", rotation=30)
    beautify(ax)


def save(fig, output_dir, name, dpi):
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, f"{name}.png")
    pdf = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


def main():
    parser = argparse.ArgumentParser(description="Plot pruning strength/performance trade-off from sweep summary.")
    parser.add_argument("--summary", required=True, help="Path to pruning_tradeoff_summary.json.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    setup_style()
    records = load_records(args.summary)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.summary))

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    plot_score_vs_reduction(axes[0, 0], records)
    plot_score_vs_rank(axes[0, 1], records)
    plot_rank_and_loss(axes[1, 0], records)
    plot_task_sensitivity(axes[1, 1], records)
    fig.suptitle("Pruning-Performance Trade-off of GSR-LoRA", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, output_dir, "pruning_tradeoff_analysis", args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    plot_score_vs_reduction(ax, records)
    fig.tight_layout()
    save(fig, output_dir, "pruning_score_vs_reduction", args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
