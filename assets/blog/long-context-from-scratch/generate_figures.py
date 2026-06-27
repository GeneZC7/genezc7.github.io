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

    # Line charts: context length vs stage index for three schedules
    schedules = [
        ("Coarse (DeepSeek-V3, GLM, LongCat)", [8, 32, 128], C_BLUE, "o"),
        ("Llama-3 six-stage", [8, 16, 32, 64, 96, 128], C_ORANGE, "s"),
        ("Qwen-2.5-Turbo progressive", [32, 64, 131, 262], C_GREEN, "^"),
    ]

    for label, lengths, color, marker in schedules:
        xs = np.arange(len(lengths))
        ax.plot(xs, lengths, "-", color=color, linewidth=2,
                marker=marker, markersize=8, label=label, alpha=0.9,
                markeredgecolor="white", markeredgewidth=0.8)

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
# Figure 4: RoPE Frequency Mechanics
# ==============================================================
def fig4_rope_mechanics():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # -- Left: high vs low frequency rotation across position --
    trained = 8.0          # trained length boundary (arbitrary units)
    pos = np.linspace(0, 12, 600)

    w_hi = 2.6             # high-frequency angular rate
    w_lo = 0.42           # low-frequency angular rate
    w_abf = 0.27          # low frequency after raising the base (ABF / YaRN)
    w_hi_pi = 1.5         # high freq if uniformly scaled by PI (period stretched)

    # high-frequency pair: many cycles within the trained window (local)
    hi = np.cos(pos * w_hi)
    # high-frequency under uniform PI scaling: period needlessly stretched (bad)
    hi_pi = np.cos(pos * w_hi_pi)
    # low-frequency pair: barely a fraction of a turn across the sequence (global)
    lo = np.cos(pos * w_lo)
    # low-frequency after raising the base (ABF): even slower
    lo_abf = np.cos(pos * w_abf)

    hi_off, lo_off = 3.4, 0.6

    # shade one period of each wave to make the period contrast explicit
    p_hi = 2 * np.pi / w_hi          # short high-freq period (~2.42)
    p_lo = 2 * np.pi / w_lo          # long low-freq period (~14.96, off-chart)
    seg_hi = (pos >= 0) & (pos <= p_hi)
    axL.fill_between(pos[seg_hi], hi_off - 1.0, hi[seg_hi] + hi_off,
                     color=C_BLUE, alpha=0.15, zorder=1)
    axL.annotate("", xy=(p_hi, hi_off - 1.25), xytext=(0, hi_off - 1.25),
                 arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=1.2))
    axL.text(p_hi / 2, hi_off - 1.75, "1 period (short)", ha="center",
             fontsize=7, color=C_BLUE)
    # low-freq: one period runs off the chart, so shade what's visible
    axL.fill_between(pos, lo_off - 1.0, lo + lo_off,
                     color=C_ORANGE, alpha=0.12, zorder=1)
    axL.annotate("", xy=(12, lo_off - 1.25), xytext=(0, lo_off - 1.25),
                 arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.2))
    axL.text(6.0, lo_off - 1.6,
             f"1 period ~ {p_lo:.0f} (longer than the sequence)", ha="center",
             fontsize=7, color=C_ORANGE)

    # high-freq band: original (kept by YaRN) vs PI's harmful uniform stretch
    axL.plot(pos, hi + hi_off, color=C_BLUE, linewidth=1.6,
             label="high freq (local)")
    axL.plot(pos, hi_pi + hi_off, color=C_BLUE, linewidth=1.2, linestyle=(0, (1, 1)),
             alpha=0.55, label="same, if PI-scaled (detail lost)")
    # low-freq band: original vs stretched (ABF / YaRN target)
    axL.plot(pos, lo + lo_off, color=C_ORANGE, linewidth=1.8,
             label="low freq (global)")
    axL.plot(pos, lo_abf + lo_off, color=C_GREEN, linewidth=1.8, linestyle="--",
             label="low freq stretched (ABF / YaRN)")

    # dedicated legend strip at bottom center, horizontal, white background
    leg = axL.legend(loc="upper center", fontsize=6.3, frameon=True,
                     framealpha=0.92, facecolor="white", edgecolor="#DDDDDD",
                     borderpad=0.5, labelspacing=0.3, columnspacing=1.1,
                     ncols=2, bbox_to_anchor=(0.5, -0.16))
    leg.set_zorder(20)

    # per-band YaRN action tags, parked in clear space beside each wave
    axL.text(2.55, hi_off + 1.45, "YaRN: keep the fast pair", ha="left", fontsize=7.5,
             color=C_BLUE, fontweight="bold")
    axL.text(3.4, lo_off + 0.95, "YaRN: stretch the slow pair", ha="left", fontsize=7.5,
             color=C_GREEN, fontweight="bold")

    # trained-length boundary + unseen region
    axL.axvspan(trained, 12, color="#F5F5F5", zorder=0)
    axL.axvline(trained, color=C_GRAY, linewidth=1.0, linestyle=":")
    axL.text(trained - 0.15, 4.9, "trained length", ha="right", fontsize=8,
             color="#888888", fontstyle="italic")
    axL.text((trained + 12) / 2, 4.9, "extrapolation\n(unseen angles)", ha="center",
             fontsize=8, color=C_RED, fontstyle="italic")

    axL.set_xlim(0, 12)
    axL.set_ylim(-2.3, 5.4)
    axL.set_yticks([])
    axL.set_xlabel("token position", fontsize=9.5)
    axL.spines["left"].set_visible(False)
    axL.set_title("YaRN stretches the slow period, keeps the fast one",
                  fontsize=10.5, fontweight="bold")

    # -- Right: ABF maps the same distance to a smaller angle --
    axR.axis("off")
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)
    axR.text(5.0, 9.4, "Why raising the base helps", ha="center",
             fontsize=10.5, fontweight="bold", color=C_DARK)

    from matplotlib.patches import Wedge

    def clock(cx, cy, r, angle_deg, seen_deg, color, label, sub, wraps):
        # shaded "seen during training" arc (0 .. seen_deg)
        axR.add_patch(Wedge((cx, cy), r, 0, seen_deg, facecolor=C_LIGHT_GREEN,
                            edgecolor="none", alpha=0.7, zorder=0))
        circ = plt.Circle((cx, cy), r, fill=False, edgecolor=C_GRAY, linewidth=1.2)
        axR.add_patch(circ)
        # boundary of the seen range
        sb = np.deg2rad(seen_deg)
        axR.plot([cx, cx + r * np.cos(sb)], [cy, cy + r * np.sin(sb)],
                 color=C_GREEN, linewidth=1.0, linestyle=":")
        # the far-token rotation vector
        a = np.deg2rad(angle_deg)
        axR.annotate("", xy=(cx + r * np.cos(a), cy + r * np.sin(a)),
                     xytext=(cx, cy),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
        tag = "lands OUTSIDE seen" if wraps else "lands inside seen"
        axR.text(cx, cy + r + 0.35, tag, ha="center", fontsize=6.8,
                 color=color, fontweight="bold")
        axR.text(cx, cy - r - 0.5, label, ha="center", fontsize=8.5,
                 fontweight="bold", color=color)
        axR.text(cx, cy - r - 1.0, sub, ha="center", fontsize=7,
                 color="#888888", fontstyle="italic")

    # green wedge = angles seen during training; same far token, two bases
    clock(2.7, 5.9, 1.25, 150, 95, C_RED, "small base",
          "far token -> large angle", wraps=True)
    clock(7.3, 5.9, 1.25, 40, 95, C_GREEN, "large base (ABF)",
          "same token -> small angle", wraps=False)
    axR.annotate("", xy=(5.75, 5.9), xytext=(4.25, 5.9),
                 arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.5))
    axR.text(5.0, 6.35, "raise base", ha="center", fontsize=7.5, color=C_DARK)

    axR.text(5.0, 2.7, "bigger base -> slower rotation ->\nlonger period -> more reach",
             ha="center", fontsize=8.5, color=C_DARK, fontstyle="italic")
    axR.text(5.0, 1.4, "base -> infinity  =>  angle -> 0  =>  NoPE",
             ha="center", fontsize=8, color=C_PURPLE, fontweight="bold")

    fig.suptitle("RoPE Frequency Mechanics", fontsize=14, fontweight="bold", y=1.0)
    save(fig, "fig4_rope_mechanics")


# ==============================================================
# Figure 5: Length Extrapolation -- RoPE base & method family
# ==============================================================
def fig5_extrapolation():
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
    save(fig, "fig5_extrapolation")


# ==============================================================
# Figure 5: Context Parallelism for Training
# ==============================================================
# ==============================================================
# Figure 6: Context Parallelism for Training
# ==============================================================
def fig6_context_parallelism():
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # -- Left: sequence sharded across CP ranks --
    axL.axis("off")
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 10)
    axL.text(5.0, 9.5, "Sequence Sharded Across CP Ranks", ha="center",
             fontsize=10, fontweight="bold", color=C_DARK)

    n = 4
    cw = 2.3
    for i in range(n):
        x = 0.3 + i * (cw + 0.05)
        rect = FancyBboxPatch((x, 5.8), cw, 1.6, boxstyle="round,pad=0.04",
                              facecolor=C_LIGHT_BLUE, edgecolor=C_BLUE, linewidth=1.3)
        axL.add_patch(rect)
        axL.text(x + cw / 2, 6.6, f"GPU {i}", ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color=C_BLUE)
        axL.text(x + cw / 2, 6.15, f"[{i*256}K:{(i+1)*256}K]", ha="center",
                 va="center", fontsize=6.5, color="#666666", fontstyle="italic")

    axL.text(5.0, 4.9, "1M sequence  /  CP degree 4", ha="center", fontsize=8.5,
             color="#888888", fontstyle="italic")
    axL.text(5.0, 3.9, "KV exchanged across ranks", ha="center", fontsize=9,
             fontweight="bold", color=C_DARK)
    axL.text(5.0, 3.3, "CP degree grows 16x / 32x with length",
             ha="center", fontsize=7.5, color=C_RED, fontstyle="italic")
    axL.text(5.0, 2.1, "+ doc masking / varlen packing\n+ activation checkpointing",
             ha="center", fontsize=8, color="#666666")

    # -- Middle: causal load imbalance & zigzag fix --
    axM.axis("off")
    axM.set_xlim(0, 10)
    axM.set_ylim(0, 10)
    axM.text(5.0, 9.5, "Causal Load Balancing", ha="center",
             fontsize=10, fontweight="bold", color=C_DARK)

    # Naive: triangular workload (early ranks light, late ranks heavy)
    axM.text(2.5, 8.5, "naive split", ha="center", fontsize=8.5,
             fontweight="bold", color=C_RED)
    loads_naive = [1, 2, 3, 4]
    for i, L in enumerate(loads_naive):
        h = 0.45 * L
        rect = FancyBboxPatch((0.7 + i * 1.0, 5.5), 0.8, h, boxstyle="round,pad=0.02",
                              facecolor=C_LIGHT_RED, edgecolor=C_RED, linewidth=1)
        axM.add_patch(rect)
    axM.text(2.5, 4.9, "imbalanced", ha="center", fontsize=7.5,
             color=C_RED, fontstyle="italic")

    # Zigzag: balanced workload
    axM.text(7.5, 8.5, "zigzag / striped", ha="center", fontsize=8.5,
             fontweight="bold", color=C_GREEN)
    for i in range(4):
        h = 0.45 * 2.5
        rect = FancyBboxPatch((5.7 + i * 1.0, 5.5), 0.8, h, boxstyle="round,pad=0.02",
                              facecolor=C_LIGHT_GREEN, edgecolor=C_GREEN, linewidth=1)
        axM.add_patch(rect)
    axM.text(7.5, 4.9, "balanced", ha="center", fontsize=7.5,
             color=C_GREEN, fontstyle="italic")

    axM.text(5.0, 3.6, "Causal masking makes late\ntokens attend to more keys.",
             ha="center", fontsize=8, color="#666666", fontstyle="italic")
    axM.text(5.0, 2.2, "see: Striped Attention,\nring-flash-attention (zigzag)",
             ha="center", fontsize=7.5, color=C_PURPLE, fontweight="bold")

    # -- Right: the sparse-attention twist --
    axR.axis("off")
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)
    axR.text(5.0, 9.5, "The Sparse-Attention Twist", ha="center",
             fontsize=10, fontweight="bold", color=C_DARK)

    # Data-dependent attended set: irregular per-rank loads
    axR.text(5.0, 8.5, "attended set is data-dependent", ha="center",
             fontsize=8, color=C_ORANGE, fontstyle="italic")
    loads_sparse = [2.0, 3.4, 1.4, 2.8]
    for i, L in enumerate(loads_sparse):
        h = 0.45 * L
        rect = FancyBboxPatch((1.2 + i * 1.0, 5.5), 0.8, h, boxstyle="round,pad=0.02",
                              facecolor=C_LIGHT_ORANGE, edgecolor=C_ORANGE, linewidth=1)
        axR.add_patch(rect)
    axR.text(5.0, 4.9, "fixed zigzag no longer balances", ha="center", fontsize=7.5,
             color=C_RED, fontstyle="italic")

    axR.text(5.0, 3.7, "CSA + HCA layers have different\ncache lifecycles.",
             ha="center", fontsize=8, color="#666666", fontstyle="italic")
    axR.text(5.0, 2.2, "DeepSeek-V4: two-stage CP\ntailored to compressed attention",
             ha="center", fontsize=7.5, color=C_PURPLE, fontweight="bold")

    fig.suptitle("Context Parallelism: The Systems Tax",
                 fontsize=14, fontweight="bold", y=1.0)
    save(fig, "fig6_context_parallelism")


# ==============================================================
# Figure 6: The Reference Recipe -- summary card
# ==============================================================
# ==============================================================
# Figure 7: The Reference Recipe -- summary card
# ==============================================================
def fig7_reference_recipe():
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
    save(fig, "fig7_reference_recipe")


# ==============================================================
# Run all
# ==============================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig1_pipeline_placement()
    fig2_staged_extension()
    fig3_data_composition()
    fig4_rope_mechanics()
    fig5_extrapolation()
    fig6_context_parallelism()
    fig7_reference_recipe()
    print("Done!")
