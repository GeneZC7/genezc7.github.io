"""
Generate all figures for the "Long Context From Scratch" blog post.
Style: clean & minimal, white background, muted colors.
Matches the style of the Ultra-Long Context Paradox and Sparse Attention figures.
Output: SVG files for Jekyll embedding.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Global style -----------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

# Color palette -- muted, professional (same as other blogs)
C_BLUE = "#4A90D9"
C_ORANGE = "#E8864A"
C_GREEN = "#5CB85C"
C_RED = "#D9534F"
C_PURPLE = "#9B59B6"
C_TEAL = "#2A9D8F"
C_GRAY = "#AAAAAA"
C_DARK = "#444444"
C_LIGHT_BLUE = "#D6E4F0"
C_LIGHT_ORANGE = "#FAE0CC"
C_LIGHT_GREEN = "#D4EDDA"
C_LIGHT_RED = "#F5CCCC"
C_LIGHT_PURPLE = "#E8D5F5"
C_LIGHT_TEAL = "#D0EFEB"


def save(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.svg")
    fig.savefig(path, format="svg", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ==============================================================
# Figure 1: Pipeline Placement -- where long context belongs
# ==============================================================
def fig1_pipeline_placement():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Three pipeline stages as proportional bars along a track
    stages = [
        ("Pre-training", "trillions of tokens\nshort, length drifting up", 5.6, C_BLUE, C_LIGHT_BLUE),
        ("Mid-training", "100s of B tokens\ncontext extension + short blend", 3.2, C_ORANGE, C_LIGHT_ORANGE),
        ("Post-training", "SFT + RL\non long & short tasks", 2.0, C_GREEN, C_LIGHT_GREEN),
    ]
    x = 0.6
    y = 3.2
    h = 1.3
    for name, desc, w, color, bg in stages:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                              facecolor=bg, edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.18, name, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h / 2 - 0.32, desc, ha="center", va="center",
                fontsize=7.5, color="#666666", fontstyle="italic")
        x += w + 0.3

    # Highlight: long-context extension lives in mid-training
    mid_x = 0.6 + 5.6 + 0.3
    mid_w = 3.2
    ax.annotate("", xy=(mid_x + mid_w / 2, 4.7), xytext=(mid_x + mid_w / 2, 5.6),
                arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=2))
    ax.text(mid_x + mid_w / 2, 5.85, "long-context extension lives here",
            ha="center", fontsize=10, fontweight="bold", color=C_ORANGE)

    # Context length annotations under the track
    ax.annotate("", xy=(11.5, 2.4), xytext=(0.6, 2.4),
                arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.5))
    ax.text(0.6, 2.0, "4K", fontsize=8, color="#999999", ha="center")
    ax.text(6.5, 2.0, "32K -> 128K", fontsize=8, color="#999999", ha="center")
    ax.text(10.2, 2.0, "256K+", fontsize=8, color="#999999", ha="center")
    ax.text(6.0, 1.5, "effective context length", fontsize=8.5,
            color="#888888", ha="center", fontstyle="italic")

    # Bottom rationale
    ax.text(6.0, 0.6, "Pre-training length is creeping up as data lengthens -- "
            "but the heavy extension still lives in mid-training",
            ha="center", fontsize=8.5, color=C_DARK, fontstyle="italic")

    ax.set_title("Where Long-Context Training Belongs in the Pipeline",
                 fontsize=14, fontweight="bold", pad=12)
    save(fig, "fig1_pipeline_placement")


# ==============================================================
# Figure 3: Data Composition & Balance
# ==============================================================
def fig3_data_composition():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # -- Left: the 40/60 long/short balance --
    axL.axis("off")
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 10)

    # Stacked bar: 40% long, 60% short
    bar_x, bar_w = 3.5, 3.0
    short_h = 4.8
    long_h = 3.2
    rect_s = FancyBboxPatch((bar_x, 1.2), bar_w, short_h, boxstyle="round,pad=0.02",
                            facecolor=C_LIGHT_BLUE, edgecolor=C_BLUE, linewidth=1.5)
    axL.add_patch(rect_s)
    axL.text(bar_x + bar_w / 2, 1.2 + short_h / 2, "60%\nshort", ha="center",
             va="center", fontsize=11, fontweight="bold", color=C_BLUE)

    rect_l = FancyBboxPatch((bar_x, 1.2 + short_h + 0.1), bar_w, long_h,
                            boxstyle="round,pad=0.02",
                            facecolor=C_LIGHT_ORANGE, edgecolor=C_ORANGE, linewidth=1.5)
    axL.add_patch(rect_l)
    axL.text(bar_x + bar_w / 2, 1.2 + short_h + 0.1 + long_h / 2, "40%\nlong",
             ha="center", va="center", fontsize=11, fontweight="bold", color=C_ORANGE)

    axL.text(5.0, 9.6, "Mixture Balance", ha="center", fontsize=11,
             fontweight="bold", color=C_DARK)
    axL.text(5.0, 0.5, "all-long -> forgetting + distribution shift",
             ha="center", fontsize=8, color=C_RED, fontstyle="italic")

    # -- Right: sources + train-longer-than-target --
    axR.axis("off")
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)

    axR.text(5.0, 9.6, "Long-Context Sources", ha="center", fontsize=11,
             fontweight="bold", color=C_DARK)
    sources = [
        ("Curated long-form", "books, papers, code (upsampled)", C_BLUE, C_LIGHT_BLUE),
        ("Synthetic", "chunk->QA, megadocs, paraphrase", C_PURPLE, C_LIGHT_PURPLE),
        ("Long-CoT QA", "reasoning traces, agent logs", C_GREEN, C_LIGHT_GREEN),
    ]
    for i, (name, desc, color, bg) in enumerate(sources):
        y = 7.6 - i * 1.7
        rect = FancyBboxPatch((0.4, y), 9.0, 1.15, boxstyle="round,pad=0.06",
                              facecolor=bg, edgecolor=color, linewidth=1.3)
        axR.add_patch(rect)
        axR.text(0.8, y + 0.58, name, ha="left", va="center", fontsize=9,
                 fontweight="bold", color=color)
        axR.text(9.0, y + 0.58, desc, ha="right", va="center", fontsize=7.5,
                 color="#666666", fontstyle="italic")

    axR.text(5.0, 0.9, "Real long-range structure beats synthetic", ha="center", fontsize=9,
             fontweight="bold", color=C_RED)
    axR.text(5.0, 0.35, "repo-level code & books carry verifiable long-range dependencies",
             ha="center", fontsize=7.5, color="#888888", fontstyle="italic")

    fig.suptitle("Data Composition & Balance for Long-Context Training",
                 fontsize=14, fontweight="bold", y=1.0)
    save(fig, "fig3_data_composition")


# ==============================================================
# Figure 2: Staged Extension Ladder
# ==============================================================
def fig2_staged_extension():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Step charts: context length vs stage index for three schedules
    schedules = [
        ("Two-stage (DeepSeek-V3, GLM, LongCat)", [8, 32, 128], C_BLUE, "o"),
        ("Llama-3 six-stage", [8, 16, 32, 64, 96, 128], C_ORANGE, "s"),
        ("Qwen2.5-Turbo progressive", [32, 64, 131, 262], C_GREEN, "^"),
    ]

    for label, lengths, color, marker in schedules:
        xs = np.arange(len(lengths))
        ax.step(xs, lengths, where="post", color=color, linewidth=2,
                marker=marker, markersize=7, label=label, alpha=0.9)

    ax.set_yscale("log", base=2)
    ax.set_yticks([8, 16, 32, 64, 128, 256])
    ax.set_yticklabels(["8K", "16K", "32K", "64K", "128K", "256K"])
    ax.set_xlabel("extension stage", fontsize=10)
    ax.set_ylabel("context length (log scale)", fontsize=10)
    ax.set_xticks(range(6))
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.grid(axis="y", linestyle=":", color="#DDDDDD", linewidth=0.7)

    ax.text(0.02, 0.97, "Progressive multi-stage extension is the dominant paradigm",
            transform=ax.transAxes, fontsize=9, color=C_DARK,
            fontstyle="italic", va="top")

    ax.set_title("Staged Context Extension Schedules", fontsize=14,
                 fontweight="bold", pad=12)
    save(fig, "fig2_staged_extension")


# ==============================================================
# Figure 4: Length Extrapolation -- RoPE base & method family
# ==============================================================
def fig4_extrapolation():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # -- Left: RoPE base frequency vs context length --
    ctx = np.array([4, 32, 128, 256, 1000])
    base = np.array([1e4, 1e6, 1e7, 1e7, 6.4e7])
    axL.plot(ctx, base, color=C_ORANGE, linewidth=2, marker="o", markersize=7)
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xticks([4, 32, 128, 256, 1000])
    axL.set_xticklabels(["4K", "32K", "128K", "256K", "1M"])
    axL.set_xlabel("training context length", fontsize=9.5)
    axL.set_ylabel("RoPE base frequency", fontsize=9.5)
    axL.set_title("ABF: base must match length", fontsize=11, fontweight="bold")
    axL.grid(True, linestyle=":", color="#DDDDDD", linewidth=0.7)
    for x, y in zip(ctx, base):
        axL.annotate(f"{y:.0e}".replace("e+0", "e"), (x, y),
                     textcoords="offset points", xytext=(0, 9),
                     fontsize=7, color="#888888", ha="center")

    # -- Right: method family tree --
    axR.axis("off")
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)
    axR.text(5.0, 9.6, "Extrapolation Method Family", ha="center",
             fontsize=11, fontweight="bold", color=C_DARK)

    methods = [
        ("PI", "linear interpolation", C_GRAY, C_LIGHT_BLUE),
        ("NTK-aware", "scale base frequency", C_BLUE, C_LIGHT_BLUE),
        ("NTK-by-parts", "per-frequency scaling", C_TEAL, C_LIGHT_TEAL),
        ("YaRN", "+ attention temperature", C_PURPLE, C_LIGHT_PURPLE),
    ]
    for i, (name, desc, color, bg) in enumerate(methods):
        y = 7.7 - i * 1.7
        rect = FancyBboxPatch((1.5, y), 7.0, 1.15, boxstyle="round,pad=0.06",
                              facecolor=bg, edgecolor=color, linewidth=1.3)
        axR.add_patch(rect)
        axR.text(2.0, y + 0.58, name, ha="left", va="center", fontsize=9.5,
                 fontweight="bold", color=color)
        axR.text(8.0, y + 0.58, desc, ha="right", va="center", fontsize=7.5,
                 color="#666666", fontstyle="italic")
        if i < len(methods) - 1:
            axR.annotate("", xy=(5.0, y - 0.5), xytext=(5.0, y - 0.05),
                         arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.2))

    axR.text(5.0, 0.6, "YaRN: SOTA with ~0.1% of pretrain tokens",
             ha="center", fontsize=8, color=C_PURPLE, fontstyle="italic")

    fig.suptitle("Length Extrapolation: Cheating the Length Tax",
                 fontsize=14, fontweight="bold", y=1.0)
    save(fig, "fig4_extrapolation")


# ==============================================================
# Figure 5: Context Parallelism for Training
# ==============================================================
def fig5_context_parallelism():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8))

    # -- Left: sequence sharded across CP ranks --
    axL.axis("off")
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 10)
    axL.text(5.0, 9.5, "Sequence Sharded Across CP Ranks", ha="center",
             fontsize=10.5, fontweight="bold", color=C_DARK)

    n = 4
    cw = 2.0
    for i in range(n):
        x = 0.5 + i * (cw + 0.1)
        rect = FancyBboxPatch((x, 5.5), cw, 1.6, boxstyle="round,pad=0.04",
                              facecolor=C_LIGHT_BLUE, edgecolor=C_BLUE, linewidth=1.3)
        axL.add_patch(rect)
        axL.text(x + cw / 2, 6.3, f"GPU {i}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color=C_BLUE)
        axL.text(x + cw / 2, 5.85, f"tokens [{i*256}K:{(i+1)*256}K]", ha="center",
                 va="center", fontsize=6.5, color="#666666", fontstyle="italic")

    axL.text(5.0, 4.6, "1M sequence  /  CP degree 4", ha="center", fontsize=8.5,
             color="#888888", fontstyle="italic")
    axL.text(5.0, 3.6, "KV exchanged across ranks", ha="center", fontsize=9,
             fontweight="bold", color=C_DARK)
    axL.text(5.0, 3.0, "CP degree grows 16x / 32x with length",
             ha="center", fontsize=8, color=C_RED, fontstyle="italic")
    axL.text(5.0, 1.8, "+ doc masking / varlen packing\n+ activation checkpointing",
             ha="center", fontsize=8, color="#666666")

    # -- Right: causal load imbalance & zigzag fix --
    axR.axis("off")
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)
    axR.text(5.0, 9.5, "Causal Load Balancing", ha="center",
             fontsize=10.5, fontweight="bold", color=C_DARK)

    # Naive: triangular workload (early ranks light, late ranks heavy)
    axR.text(2.5, 8.5, "naive split", ha="center", fontsize=8.5,
             fontweight="bold", color=C_RED)
    loads_naive = [1, 2, 3, 4]
    for i, L in enumerate(loads_naive):
        h = 0.45 * L
        rect = FancyBboxPatch((0.7 + i * 1.0, 5.5), 0.8, h, boxstyle="round,pad=0.02",
                              facecolor=C_LIGHT_RED, edgecolor=C_RED, linewidth=1)
        axR.add_patch(rect)
    axR.text(2.5, 4.9, "imbalanced", ha="center", fontsize=7.5,
             color=C_RED, fontstyle="italic")

    # Zigzag: balanced workload
    axR.text(7.5, 8.5, "zigzag / striped", ha="center", fontsize=8.5,
             fontweight="bold", color=C_GREEN)
    for i in range(4):
        h = 0.45 * 2.5
        rect = FancyBboxPatch((5.7 + i * 1.0, 5.5), 0.8, h, boxstyle="round,pad=0.02",
                              facecolor=C_LIGHT_GREEN, edgecolor=C_GREEN, linewidth=1)
        axR.add_patch(rect)
    axR.text(7.5, 4.9, "balanced", ha="center", fontsize=7.5,
             color=C_GREEN, fontstyle="italic")

    axR.text(5.0, 3.4, "Causal masking makes late tokens attend to more keys.",
             ha="center", fontsize=8, color="#666666", fontstyle="italic")
    axR.text(5.0, 2.6, "Zigzag assignment evens the per-rank work.",
             ha="center", fontsize=8, color="#666666", fontstyle="italic")
    axR.text(5.0, 1.5, "see: Striped Attention",
             ha="center", fontsize=8, color=C_PURPLE, fontweight="bold")

    fig.suptitle("Context Parallelism: The Systems Tax",
                 fontsize=14, fontweight="bold", y=1.0)
    save(fig, "fig5_context_parallelism")


# ==============================================================
# Figure 6: The Reference Recipe -- summary card
# ==============================================================
def fig6_reference_recipe():
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    steps = [
        ("1. STAGE", "Short-ish pre-training -> dedicated mid-training extension (short data kept in blend)",
         C_BLUE, C_LIGHT_BLUE),
        ("2. DATA", "~40% long / 60% short (tune per stage)  +  synthetic sparingly",
         C_ORANGE, C_LIGHT_ORANGE),
        ("3. EXTRAPOLATION", "ABF base matched to length  +  YaRN for cheap reach (not reliability)",
         C_PURPLE, C_LIGHT_PURPLE),
        ("4. PARALLELISM", "Hybrid CP  +  varlen packing  +  zigzag load balancing",
         C_GREEN, C_LIGHT_GREEN),
    ]
    for i, (name, desc, color, bg) in enumerate(steps):
        y = 7.6 - i * 1.75
        rect = FancyBboxPatch((0.5, y), 9.0, 1.3, boxstyle="round,pad=0.06",
                              facecolor=bg, edgecolor=color, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(0.9, y + 0.65, name, ha="left", va="center", fontsize=10,
                fontweight="bold", color=color)
        ax.text(3.1, y + 0.65, desc, ha="left", va="center", fontsize=8,
                color="#555555")
        if i < len(steps) - 1:
            ax.annotate("", xy=(5.0, y - 0.4), xytext=(5.0, y - 0.05),
                        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.3))

    ax.text(5.0, 0.5, "Four ingredients -- none alone suffices",
            ha="center", fontsize=10, fontweight="bold", color=C_DARK)

    ax.set_title("The Reference Long-Context Recipe", fontsize=14,
                 fontweight="bold", pad=12)
    save(fig, "fig6_reference_recipe")


# ==============================================================
# Run all
# ==============================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig1_pipeline_placement()
    fig2_staged_extension()
    fig3_data_composition()
    fig4_extrapolation()
    fig5_context_parallelism()
    fig6_reference_recipe()
    print("Done!")
