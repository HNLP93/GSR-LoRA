import argparse
import json
import os
import re
import shutil
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3D projection.


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "figures", "three_d_analysis")
MODULE_ORDER = ["q", "k", "v", "o", "wi", "wo"]
STACK_ORDER = ["encoder", "decoder"]


COLORS = {
    "ink": "#202124",
    "muted": "#526173",
    "axis": "#C8D3E0",
    "grid": "#DEE7F0",
    "encoder": "#FB7C2B",
    "decoder": "#2EAD62",
    "overall": "#2563EB",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
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


def parse_layer_record(item):
    match = re.search(r"(encoder|decoder)\.block\.(\d+)", item.get("module", ""))
    if not match:
        return None
    module = str(item["module"]).split(".")[-1]
    if module not in MODULE_ORDER:
        return None
    return {
        "stack": match.group(1),
        "block": int(match.group(2)),
        "module": module,
        "retained_rank": float(item["retained_rank"]),
        "prefix_rank": float(item.get("prefix_rank", item["retained_rank"])),
        "rank_reduction_percent": float(item.get("rank_reduction_percent", 0.0)),
    }


def load_layers(path):
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    layers = []
    for item in report.get("layers", []):
        parsed = parse_layer_record(item)
        if parsed is not None:
            layers.append(parsed)
    if not layers:
        raise RuntimeError(f"No usable layer records found in {path}")
    return layers


def mean_matrix(layers, stack):
    by_key = defaultdict(list)
    for item in layers:
        if item["stack"] == stack:
            by_key[(item["block"], item["module"])].append(item["retained_rank"])

    blocks = sorted({item["block"] for item in layers if item["stack"] == stack})
    matrix = np.full((len(MODULE_ORDER), len(blocks)), np.nan, dtype=np.float64)
    for module_idx, module in enumerate(MODULE_ORDER):
        for block_idx, block in enumerate(blocks):
            values = by_key.get((block, module), [])
            if values:
                matrix[module_idx, block_idx] = float(np.mean(values))
    return blocks, matrix


def stack_module_matrix(layers):
    matrix = np.full((len(STACK_ORDER), len(MODULE_ORDER)), np.nan, dtype=np.float64)
    for stack_idx, stack in enumerate(STACK_ORDER):
        for module_idx, module in enumerate(MODULE_ORDER):
            values = [
                item["retained_rank"]
                for item in layers
                if item["stack"] == stack and item["module"] == module
            ]
            if values:
                matrix[stack_idx, module_idx] = float(np.mean(values))
    return matrix


def style_3d_axes(ax):
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.xaxis._axinfo["grid"]["color"] = COLORS["grid"]
    ax.yaxis._axinfo["grid"]["color"] = COLORS["grid"]
    ax.zaxis._axinfo["grid"]["color"] = COLORS["grid"]
    ax.xaxis._axinfo["axisline"]["color"] = COLORS["axis"]
    ax.yaxis._axinfo["axisline"]["color"] = COLORS["axis"]
    ax.zaxis._axinfo["axisline"]["color"] = COLORS["axis"]
    ax.tick_params(colors=COLORS["ink"], pad=1)
    if hasattr(ax, "dist"):
        ax.dist = 8.5


def set_box_aspect_if_available(ax, aspect):
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(aspect)


def plot_block_module_bars(layers, stack, output_dir, dpi):
    blocks, matrix = mean_matrix(layers, stack)
    valid_values = matrix[~np.isnan(matrix)]
    norm = Normalize(vmin=max(1.0, float(np.nanmin(matrix))), vmax=float(np.nanmax(matrix)))
    cmap = cm.get_cmap("viridis")

    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_axes([0.00, 0.04, 0.82, 0.88], projection="3d")
    xpos, ypos = np.meshgrid(np.arange(len(blocks)), np.arange(len(MODULE_ORDER)))
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos, dtype=np.float64)
    dz = matrix.ravel()
    mask = ~np.isnan(dz)

    colors = cmap(norm(dz[mask]))
    ax.bar3d(
        xpos[mask] - 0.34,
        ypos[mask] - 0.34,
        zpos[mask],
        0.68,
        0.68,
        dz[mask],
        color=colors,
        edgecolor="white",
        linewidth=0.35,
        shade=True,
        alpha=0.96,
    )

    ax.set_xticks(np.arange(len(blocks)))
    ax.set_xticklabels(blocks)
    ax.set_yticks(np.arange(len(MODULE_ORDER)))
    ax.set_yticklabels(MODULE_ORDER)
    ax.set_zlim(0, max(10.5, float(np.nanmax(matrix)) + 1.0))
    ax.set_xlabel("Block index", labelpad=4)
    ax.set_ylabel("Module type", labelpad=4)
    ax.set_zlabel("Average retained rank", labelpad=4)
    ax.set_title(f"{stack.capitalize()}: block-module retained rank", fontweight="bold", pad=6)
    ax.view_init(elev=26, azim=-58)
    set_box_aspect_if_available(ax, (12, 6, 4.8))
    style_3d_axes(ax)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(valid_values)
    cax = fig.add_axes([0.84, 0.26, 0.025, 0.48])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Avg. retained rank")

    return save_figure(fig, output_dir, f"fig3d_{stack}_block_module_rank", dpi)


def plot_stack_module_bars(layers, output_dir, dpi):
    matrix = stack_module_matrix(layers)
    norm = Normalize(vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix)))
    cmap = cm.get_cmap("plasma")

    fig = plt.figure(figsize=(5.8, 4.2))
    ax = fig.add_axes([0.00, 0.03, 0.96, 0.88], projection="3d")
    xpos, ypos = np.meshgrid(np.arange(len(MODULE_ORDER)), np.arange(len(STACK_ORDER)))
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    dz = matrix.ravel()

    colors = cmap(norm(dz))
    ax.bar3d(
        xpos - 0.32,
        ypos - 0.28,
        np.zeros_like(dz),
        0.64,
        0.56,
        dz,
        color=colors,
        edgecolor="white",
        linewidth=0.45,
        shade=True,
        alpha=0.96,
    )

    ax.set_xticks(np.arange(len(MODULE_ORDER)))
    ax.set_xticklabels(MODULE_ORDER)
    ax.set_yticks(np.arange(len(STACK_ORDER)))
    ax.set_yticklabels(["Encoder", "Decoder"])
    ax.set_zlim(0, max(10.5, float(np.nanmax(matrix)) + 1.0))
    ax.set_xlabel("Module type", labelpad=4)
    ax.set_ylabel("Stack", labelpad=4)
    ax.set_zlabel("Average retained rank", labelpad=4)
    ax.set_title("Stack-module retained rank", fontweight="bold", pad=6)
    ax.view_init(elev=24, azim=-48)
    set_box_aspect_if_available(ax, (6.0, 2.4, 4.8))
    style_3d_axes(ax)

    return save_figure(fig, output_dir, "fig3d_stack_module_rank", dpi)


def plot_prefix_retained_scatter(layers, output_dir, dpi):
    fig = plt.figure(figsize=(6.2, 4.6))
    ax = fig.add_axes([0.00, 0.03, 0.94, 0.90], projection="3d")

    for stack in STACK_ORDER:
        values = [item for item in layers if item["stack"] == stack]
        xs = np.array([item["prefix_rank"] for item in values], dtype=np.float64)
        ys = np.array([item["retained_rank"] for item in values], dtype=np.float64)
        zs = np.array([item["block"] for item in values], dtype=np.float64)
        ax.scatter(
            xs,
            ys,
            zs,
            s=28,
            alpha=0.78,
            color=COLORS[stack],
            edgecolor="white",
            linewidth=0.3,
            label=stack.capitalize(),
        )

    ax.plot([1, 16], [1, 16], [0, 0], color=COLORS["muted"], linestyle="--", linewidth=1.1, alpha=0.8)
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0.5, 16.5)
    ax.set_zlim(-0.5, 11.5)
    ax.set_xlabel("Selected prefix span", labelpad=4)
    ax.set_ylabel("Actual retained rank", labelpad=4)
    ax.set_zlabel("Block index", labelpad=4)
    ax.set_title("Prefix span vs. actual retained rank", fontweight="bold", pad=6)
    ax.view_init(elev=23, azim=-52)
    set_box_aspect_if_available(ax, (6, 6, 4.5))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.96))
    style_3d_axes(ax)

    return save_figure(fig, output_dir, "fig3d_prefix_retained_scatter", dpi)


def copy_selected(outputs, target_dir):
    if not target_dir:
        return
    os.makedirs(target_dir, exist_ok=True)
    for path in outputs:
        if path.endswith(".png"):
            shutil.copy2(path, os.path.join(target_dir, os.path.basename(path)))


def main():
    parser = argparse.ArgumentParser(description="Generate 3D analysis figures from a GSR-LoRA compression report.")
    parser.add_argument("--compression-report", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--copy-png-to", default=None, help="Optional directory to receive PNG copies.")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    setup_style()
    layers = load_layers(args.compression_report)

    outputs = []
    for stack in STACK_ORDER:
        outputs.extend(plot_block_module_bars(layers, stack, args.output_dir, args.dpi))
        plt.close("all")
    outputs.extend(plot_stack_module_bars(layers, args.output_dir, args.dpi))
    plt.close("all")
    outputs.extend(plot_prefix_retained_scatter(layers, args.output_dir, args.dpi))
    plt.close("all")

    copy_selected(outputs, args.copy_png_to)


if __name__ == "__main__":
    main()
