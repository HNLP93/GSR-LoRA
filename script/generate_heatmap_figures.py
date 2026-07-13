import argparse
import json
import os
import re
import shutil
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "figures", "heatmaps_current")
MODULE_ORDER = ["q", "k", "v", "o", "wi", "wo"]
STACK_ORDER = ["encoder", "decoder"]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_dir, name, dpi):
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{name}.png")
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)
    return png_path, pdf_path


def parse_layer(item):
    module_name = item.get("module", "")
    match = re.search(r"(encoder|decoder)\.block\.(\d+)", module_name)
    if not match:
        return None
    module_type = module_name.split(".")[-1]
    if module_type not in MODULE_ORDER:
        return None
    return {
        "stack": match.group(1),
        "block": int(match.group(2)),
        "module_type": module_type,
        "retained_rank": float(item["retained_rank"]),
        "prefix_rank": float(item.get("prefix_rank", item["retained_rank"])),
        "rank_reduction_percent": float(item.get("rank_reduction_percent", 0.0)),
    }


def load_layers(path):
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    layers = []
    for item in report.get("layers", []):
        parsed = parse_layer(item)
        if parsed is not None:
            layers.append(parsed)
    if not layers:
        raise RuntimeError(f"No usable layer records found in {path}")
    return layers


def matrix_for(layers, stack, field):
    stack_layers = [x for x in layers if x["stack"] == stack]
    max_block = max(x["block"] for x in stack_layers)
    matrix = np.full((len(MODULE_ORDER), max_block + 1), np.nan, dtype=np.float64)
    buckets = defaultdict(list)
    for item in stack_layers:
        buckets[(item["module_type"], item["block"])].append(float(item[field]))
    for row, module in enumerate(MODULE_ORDER):
        for block in range(max_block + 1):
            values = buckets.get((module, block), [])
            if values:
                matrix[row, block] = float(np.mean(values))
    return matrix


def gap_matrix(layers, stack):
    retained = matrix_for(layers, stack, "retained_rank")
    prefix = matrix_for(layers, stack, "prefix_rank")
    return prefix - retained


def stack_gap_matrix(layers):
    encoder = matrix_for(layers, "encoder", "retained_rank")
    decoder = matrix_for(layers, "decoder", "retained_rank")
    return encoder - decoder


def fmt_value(value):
    if np.isnan(value):
        return ""
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"


def annotate_cells(ax, matrix, vmin, vmax, fontsize=5.2):
    midpoint = (vmin + vmax) / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                continue
            color = "white" if value > midpoint else "#202124"
            ax.text(col, row, fmt_value(value), ha="center", va="center", fontsize=fontsize, color=color)


def draw_single_heatmap(ax, matrix, title, cmap, vmin, vmax, xlabel="Block index", annotate=True):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontweight="bold", loc="left", pad=3)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(i) for i in range(matrix.shape[1])])
    ax.set_yticks(np.arange(len(MODULE_ORDER)))
    ax.set_yticklabels(MODULE_ORDER)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Module")
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODULE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_color("#C8D3E0")
        spine.set_linewidth(0.8)
    if annotate:
        annotate_cells(ax, matrix, vmin, vmax)
    return im


def draw_stack_pair(layers, field, name, title, cbar_label, cmap, vmin, vmax, output_dir, dpi, annotate=True):
    fig = plt.figure(figsize=(8.8, 3.65))
    axes = [
        fig.add_axes([0.075, 0.56, 0.79, 0.29]),
        fig.add_axes([0.075, 0.15, 0.79, 0.29]),
    ]
    cax = fig.add_axes([0.885, 0.15, 0.022, 0.70])
    images = []
    for ax, stack in zip(axes, STACK_ORDER):
        matrix = matrix_for(layers, stack, field)
        images.append(
            draw_single_heatmap(
                ax,
                matrix,
                f"{stack.capitalize()}",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                xlabel="Block index" if stack == "decoder" else "",
                annotate=annotate,
            )
        )
        if stack == "encoder":
            ax.tick_params(labelbottom=False)
    fig.suptitle(title.capitalize(), x=0.075, y=0.965, ha="left", fontsize=10.2, fontweight="bold")
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label(cbar_label)
    return save_figure(fig, output_dir, name, dpi)


def draw_gap_pair(layers, output_dir, dpi):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True)
    vmax = max(float(np.nanmax(gap_matrix(layers, s))) for s in STACK_ORDER)
    vmax = max(1.0, vmax)
    images = []
    for ax, stack in zip(axes, STACK_ORDER):
        matrix = gap_matrix(layers, stack)
        images.append(
            draw_single_heatmap(
                ax,
                matrix,
                f"{stack.capitalize()} prefix-retained gap",
                cmap="Oranges",
                vmin=0.0,
                vmax=vmax,
                xlabel="Block index" if stack == "decoder" else "",
                annotate=True,
            )
        )
        if stack == "encoder":
            ax.tick_params(labelbottom=False)
    cbar = fig.colorbar(images[-1], ax=axes, fraction=0.028, pad=0.025)
    cbar.set_label("Prefix span - retained rank")
    fig.subplots_adjust(left=0.075, right=0.925, top=0.94, bottom=0.12, hspace=0.36)
    return save_figure(fig, output_dir, "fig_heatmap_prefix_retained_gap", dpi)


def draw_stack_gap(layers, output_dir, dpi):
    matrix = stack_gap_matrix(layers)
    max_abs = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 2.25))
    im = draw_single_heatmap(
        ax,
        matrix,
        "Encoder-decoder retained-rank gap",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        xlabel="Block index",
        annotate=True,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.025)
    cbar.set_label("Encoder - decoder")
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_heatmap_encoder_decoder_gap", dpi)


def copy_pngs(paths, target_dir):
    if not target_dir:
        return
    os.makedirs(target_dir, exist_ok=True)
    for path in paths:
        if path.endswith(".png"):
            shutil.copy2(path, os.path.join(target_dir, os.path.basename(path)))


def main():
    parser = argparse.ArgumentParser(description="Generate compact heatmap figures from a GSR-LoRA compression report.")
    parser.add_argument("--compression-report", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--copy-png-to", default=None)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    setup_style()
    layers = load_layers(args.compression_report)

    outputs = []
    outputs.extend(
        draw_stack_pair(
            layers,
            "retained_rank",
            "fig_heatmap_retained_rank",
            "retained rank",
            "Average retained rank",
            "YlGnBu",
            1.0,
            16.0,
            args.output_dir,
            args.dpi,
            annotate=True,
        )
    )
    plt.close("all")
    outputs.extend(
        draw_stack_pair(
            layers,
            "prefix_rank",
            "fig_heatmap_prefix_span",
            "selected prefix span",
            "Average prefix span",
            "Blues",
            1.0,
            16.0,
            args.output_dir,
            args.dpi,
            annotate=True,
        )
    )
    plt.close("all")
    outputs.extend(draw_gap_pair(layers, args.output_dir, args.dpi))
    plt.close("all")
    outputs.extend(draw_stack_gap(layers, args.output_dir, args.dpi))
    plt.close("all")

    copy_pngs(outputs, args.copy_png_to)


if __name__ == "__main__":
    main()
