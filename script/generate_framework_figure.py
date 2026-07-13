import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "figures")


COLORS = {
    "blue": "#DCEBFF",
    "blue_edge": "#4C78A8",
    "green": "#E0F2E9",
    "green_edge": "#3B8A5B",
    "orange": "#FDE8D2",
    "orange_edge": "#D9822B",
    "purple": "#EDE7F6",
    "purple_edge": "#7E57C2",
    "gray": "#F2F4F7",
    "gray_edge": "#8792A2",
    "red": "#FBE3E4",
    "red_edge": "#C44E52",
    "ink": "#1F2937",
    "muted": "#667085",
}


def box(ax, xy, w, h, text, face, edge, fontsize=9, weight="normal", radius=0.025):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.35,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight=weight,
        linespacing=1.18,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color="#344054", lw=1.35, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=4,
    )
    ax.add_patch(arr)
    return arr


def stage_label(ax, x, y, title, subtitle):
    ax.text(x, y, title, ha="left", va="top", fontsize=13, weight="bold", color=COLORS["ink"])
    ax.text(x, y - 0.035, subtitle, ha="left", va="top", fontsize=8.2, color=COLORS["muted"])


def rank_groups(ax, x, y, w, h, kept=False):
    n = 8
    gap = 0.006
    gw = (w - gap * (n - 1)) / n
    active = [1, 3, 4, 6, 7] if kept else list(range(n))
    weak = [0, 2, 5] if not kept else []
    for i in range(n):
        if kept and i not in active:
            face = "#FFFFFF"
            edge = "#D0D5DD"
            alpha = 0.35
        elif i in weak:
            face = "#FFF6ED"
            edge = COLORS["orange_edge"]
            alpha = 0.75
        else:
            face = "#E0F2E9" if kept else "#DCEBFF"
            edge = COLORS["green_edge"] if kept else COLORS["blue_edge"]
            alpha = 1.0
        rect = Rectangle((x + i * (gw + gap), y), gw, h, facecolor=face, edgecolor=edge, linewidth=1.0, alpha=alpha)
        ax.add_patch(rect)
        ax.text(
            x + i * (gw + gap) + gw / 2,
            y + h / 2,
            f"g{i + 1}",
            ha="center",
            va="center",
            fontsize=7.5,
            color=COLORS["ink"],
        )


def make_figure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
        }
    )

    fig, ax = plt.subplots(figsize=(13.6, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "GSR-LoRA: Routing-aware Structured Rank Compression",
        ha="center",
        va="top",
        fontsize=16,
        weight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.925,
        "Task-dependent rank routing, Group Lasso sparsification, non-prefix group selection, and router remapping",
        ha="center",
        va="top",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    # Stage containers
    train_panel = FancyBboxPatch(
        (0.035, 0.13),
        0.59,
        0.75,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        facecolor="#FAFBFF",
        edgecolor="#CBD5E1",
        linewidth=1.3,
        zorder=0,
    )
    comp_panel = FancyBboxPatch(
        (0.655, 0.13),
        0.31,
        0.75,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        facecolor="#FCFCFD",
        edgecolor="#CBD5E1",
        linewidth=1.3,
        zorder=0,
    )
    ax.add_patch(train_panel)
    ax.add_patch(comp_panel)

    stage_label(ax, 0.06, 0.845, "Training stage", "Learn task-adaptive rank usage and structured redundancy")
    stage_label(ax, 0.68, 0.845, "Compression stage", "Turn learned redundancy into compact LoRA adapters")

    # Training stage left column
    box(
        ax,
        (0.075, 0.68),
        0.15,
        0.095,
        "Multi-task data\n+ task id t",
        COLORS["gray"],
        COLORS["gray_edge"],
        fontsize=8.8,
        weight="bold",
    )
    box(
        ax,
        (0.075, 0.50),
        0.15,
        0.11,
        "Frozen backbone\nT5 blocks",
        COLORS["blue"],
        COLORS["blue_edge"],
        fontsize=8.8,
        weight="bold",
    )
    ax.text(0.15, 0.455, "adapt q, k, v, o, wi, wo", ha="center", va="center", fontsize=8, color=COLORS["muted"])
    arrow(ax, (0.15, 0.68), (0.15, 0.61))

    # Router path
    box(
        ax,
        (0.285, 0.69),
        0.16,
        0.085,
        "Task embedding\n$e_t$",
        COLORS["purple"],
        COLORS["purple_edge"],
        fontsize=8.8,
        weight="bold",
    )
    box(
        ax,
        (0.47, 0.69),
        0.13,
        0.085,
        "Router\n$G^{(l)}(e_t)$",
        COLORS["purple"],
        COLORS["purple_edge"],
        fontsize=8.8,
        weight="bold",
    )
    arrow(ax, (0.225, 0.727), (0.285, 0.727))
    arrow(ax, (0.445, 0.727), (0.47, 0.727))

    # LoRA rank groups
    box(
        ax,
        (0.28, 0.45),
        0.32,
        0.15,
        "Shared LoRA rank groups\n$\\Delta W = \\sum_k B_{:,k} A_{k,:}$",
        "#FFFFFF",
        COLORS["blue_edge"],
        fontsize=9,
        weight="bold",
    )
    rank_groups(ax, 0.305, 0.475, 0.27, 0.045, kept=False)
    arrow(ax, (0.225, 0.555), (0.28, 0.535))
    arrow(ax, (0.535, 0.69), (0.535, 0.60))
    ax.text(0.57, 0.642, "effective rank\nmask per task", ha="center", va="center", fontsize=7.7, color=COLORS["muted"])

    # Group lasso and loss
    box(
        ax,
        (0.28, 0.285),
        0.145,
        0.085,
        "Group Lasso\n$\\sum_k \\|g_k\\|_2$",
        COLORS["orange"],
        COLORS["orange_edge"],
        fontsize=8.6,
        weight="bold",
    )
    box(
        ax,
        (0.455, 0.285),
        0.145,
        0.085,
        "Router regularizer\nentropy / rank cost",
        COLORS["orange"],
        COLORS["orange_edge"],
        fontsize=8.4,
        weight="bold",
    )
    box(
        ax,
        (0.195, 0.18),
        0.29,
        0.075,
        "$L = L_{task} + \\lambda_{gl}L_{gl} + \\lambda_{route}L_{route}$",
        COLORS["green"],
        COLORS["green_edge"],
        fontsize=9,
        weight="bold",
    )
    arrow(ax, (0.36, 0.45), (0.36, 0.37))
    arrow(ax, (0.525, 0.45), (0.525, 0.37))
    arrow(ax, (0.35, 0.285), (0.33, 0.255))
    arrow(ax, (0.525, 0.285), (0.43, 0.255))

    # Bridge
    arrow(ax, (0.61, 0.52), (0.66, 0.52), lw=1.8)
    ax.text(0.635, 0.555, "trained\nadapter", ha="center", va="center", fontsize=8, color=COLORS["muted"])

    # Compression stage
    box(
        ax,
        (0.69, 0.66),
        0.22,
        0.1,
        "Rank-group scoring\n$q_k = s_k \\cdot u_k^{\\gamma}$",
        COLORS["gray"],
        COLORS["gray_edge"],
        fontsize=9,
        weight="bold",
    )
    ax.text(
        0.8,
        0.625,
        "parameter importance $s_k$ + router usage $u_k$",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["muted"],
    )
    box(
        ax,
        (0.69, 0.49),
        0.22,
        0.1,
        "Non-prefix group selection\n$\\mathcal{K}=\\{k:q_k/\\max q \\geq \\tau\\}$",
        COLORS["green"],
        COLORS["green_edge"],
        fontsize=8.7,
        weight="bold",
    )
    rank_groups(ax, 0.715, 0.445, 0.17, 0.04, kept=True)
    box(
        ax,
        (0.69, 0.29),
        0.22,
        0.1,
        "Router remapping\nold rank space $\\rightarrow$ compressed rank space",
        COLORS["purple"],
        COLORS["purple_edge"],
        fontsize=8.5,
        weight="bold",
    )
    box(
        ax,
        (0.69, 0.17),
        0.22,
        0.075,
        "Compact adapter\n$A'=A_{\\mathcal{K},:},\\;B'=B_{:,\\mathcal{K}}$",
        COLORS["blue"],
        COLORS["blue_edge"],
        fontsize=8.8,
        weight="bold",
    )
    arrow(ax, (0.80, 0.66), (0.80, 0.59))
    arrow(ax, (0.80, 0.49), (0.80, 0.39))
    arrow(ax, (0.80, 0.29), (0.80, 0.245))

    # Small legend
    legend_y = 0.075
    legend_items = [
        ("Trainable LoRA/router", COLORS["purple"], COLORS["purple_edge"]),
        ("Structured sparsity", COLORS["orange"], COLORS["orange_edge"]),
        ("Compressed adapter", COLORS["green"], COLORS["green_edge"]),
    ]
    x = 0.225
    for label, face, edge in legend_items:
        ax.add_patch(Rectangle((x, legend_y), 0.018, 0.018, facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(x + 0.025, legend_y + 0.009, label, ha="left", va="center", fontsize=8, color=COLORS["muted"])
        x += 0.19

    fig.tight_layout(pad=0.2)
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = make_figure()
    pdf_path = os.path.join(OUT_DIR, "framework.pdf")
    png_path = os.path.join(OUT_DIR, "framework.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
