"""
Generate all figures for the Ultra-Long Context Paradox blog post.
Style: clean & minimal, white background, muted colors.
Output: SVG files for Hugo/Jekyll embedding.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Global style ──────────────────────────────────────────────
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

# Color palette — muted, professional
C_BLUE = "#4A90D9"
C_ORANGE = "#E8864A"
C_GREEN = "#5CB85C"
C_RED = "#D9534F"
C_PURPLE = "#9B59B6"
C_GRAY = "#AAAAAA"
C_LIGHT_BLUE = "#D6E4F0"
C_LIGHT_ORANGE = "#FAE0CC"
C_LIGHT_GREEN = "#D4EDDA"
C_LIGHT_RED = "#F5CCCC"
C_LIGHT_PURPLE = "#E8D5F5"


def save(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.svg")
    fig.savefig(path, format="svg", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════
# Figure 1: Context Rot — The Lost-in-the-Middle Effect
# ══════════════════════════════════════════════════════════════
def fig1_context_rot():
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # U-shaped curve: high at edges, low in middle
    x = np.linspace(0, 1, 200)
    y = 0.55 + 0.40 * (2 * (x - 0.5)) ** 4 + 0.03 * np.sin(8 * np.pi * x) * 0.1
    y = np.clip(y, 0, 1)

    ax.fill_between(x, y, alpha=0.15, color=C_BLUE)
    ax.plot(x, y, color=C_BLUE, linewidth=2.5)

    # Annotations — key term bold, descriptive italic
    ax.annotate("Strong recall\nat start", xy=(0.05, 0.93), fontsize=9,
                color=C_BLUE, ha="left", va="top", fontstyle="italic")
    ax.annotate("Lost in\nthe middle", xy=(0.5, 0.52), fontsize=9,
                color=C_RED, ha="center", va="top",
                fontweight="bold")
    ax.annotate("Strong recall\nat end", xy=(0.95, 0.93), fontsize=9,
                color=C_BLUE, ha="right", va="top", fontstyle="italic")

    # Arrow pointing to the dip
    ax.annotate("", xy=(0.5, 0.56), xytext=(0.5, 0.70),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5))

    ax.set_xlabel("Position in Context", fontsize=11, labelpad=8)
    ax.set_ylabel("Retrieval Accuracy", fontsize=11, labelpad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["Start", "Middle", "End"])
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["40%", "60%", "80%", "100%"])

    ax.set_title("Context Rot: The Lost-in-the-Middle Effect", fontsize=13,
                 fontweight="bold", pad=12)
    save(fig, "fig1_context_rot")


# ══════════════════════════════════════════════════════════════
# Figure 2: Context Rot vs. Context Anxiety
# ══════════════════════════════════════════════════════════════
def fig2_rot_vs_anxiety():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: Context Rot (positional)
    x = np.linspace(0, 1, 200)
    y = 0.55 + 0.40 * (2 * (x - 0.5)) ** 4
    ax1.fill_between(x, y, alpha=0.12, color=C_BLUE)
    ax1.plot(x, y, color=C_BLUE, linewidth=2.5)
    ax1.set_xlabel("Position in Context", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.3, 1.05)
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xticklabels(["Start", "Middle", "End"], fontsize=9)
    ax1.set_yticks([])
    ax1.set_title("Context Rot", fontsize=12, fontweight="bold", color=C_BLUE)
    ax1.text(0.5, 0.38, "Positional — accuracy dips\nin the middle", ha="center",
             fontsize=9, color="#666666", fontstyle="italic")

    # Right: Context Anxiety (behavioral)
    x2 = np.linspace(0, 1, 200)
    # Quality degrades as context fills up — gradual then steeper
    y2 = 0.95 - 0.15 * x2 - 0.35 * x2 ** 2.5
    ax2.fill_between(x2, y2, alpha=0.12, color=C_ORANGE)
    ax2.plot(x2, y2, color=C_ORANGE, linewidth=2.5)
    ax2.set_xlabel("Context Utilization", fontsize=10)
    ax2.set_ylabel("Output Quality", fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.3, 1.05)
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xticklabels(["Empty", "50%", "Full"], fontsize=9)
    ax2.set_yticks([])
    ax2.set_title("Context Anxiety", fontsize=12, fontweight="bold", color=C_ORANGE)
    ax2.text(0.5, 0.38, "Behavioral — overall quality\ndegrades as context fills", ha="center",
             fontsize=9, color="#666666", fontstyle="italic")

    # Mark the "premature wrap-up" zone
    ax2.axvspan(0.75, 1.0, alpha=0.08, color=C_RED)
    ax2.text(0.875, 0.75, "premature\nwrap-up\nzone", ha="center", fontsize=8,
             color=C_RED, fontstyle="italic")

    fig.suptitle("Two Distinct Failure Modes", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_rot_vs_anxiety")


# ══════════════════════════════════════════════════════════════
# Figure 3: Cumulative Compression Loss
# ══════════════════════════════════════════════════════════════
def fig3_compression_loss():
    fig, ax = plt.subplots(figsize=(8, 5.0))

    rounds = np.arange(0, 9)
    retention = 100 * (0.82 ** rounds)

    bars = ax.bar(rounds, retention, color=C_BLUE, alpha=0.7, width=0.6,
                  edgecolor=C_BLUE, linewidth=0.5)

    for i, bar in enumerate(bars):
        t = i / 8
        r = int(74 * (1 - t) + 217 * t)
        g = int(144 * (1 - t) + 83 * t)
        b = int(217 * (1 - t) + 79 * t)
        bar.set_color(f"#{r:02x}{g:02x}{b:02x}")
        bar.set_alpha(0.75)

    for i, (r, v) in enumerate(zip(rounds, retention)):
        ax.text(r, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9,
                fontweight="bold" if i == 0 else "normal",
                color="#444444")

    ax.axhline(y=100, color=C_GRAY, linewidth=0.8, linestyle=":", zorder=0)

    ax.annotate("", xy=(7.5, retention[7]), xytext=(7.5, 100),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5,
                                linestyle="--"))
    ax.text(8.2, 55, "signal\nloss", fontsize=9, color=C_RED, fontstyle="italic",
            ha="left", va="center")

    ax.set_xlabel("Number of Compression Rounds", fontsize=11, labelpad=8)
    ax.set_ylabel("Retained Signal Quality", fontsize=11, labelpad=8)
    ax.set_xlim(-0.5, 9.0)
    ax.set_ylim(-45, 115)
    ax.set_xticks(rounds)
    ax.set_xticklabels(["0", "1", "2", "3", "4", "5", "6", "7", "8"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.spines["bottom"].set_position(("data", 0))

    # Context window labels — below x-axis with enough clearance for xlabel
    ax.text(0, -32, "1M context", fontsize=8, ha="center", color=C_GREEN,
            fontweight="bold")
    ax.text(0, -37, "no comp.", fontsize=7.5, ha="center", color=C_GREEN,
            fontstyle="italic")

    ax.text(4, -32, "256K context", fontsize=8, ha="center", color=C_ORANGE,
            fontweight="bold")
    ax.text(4, -37, "4 rounds", fontsize=7.5, ha="center", color=C_ORANGE,
            fontstyle="italic")

    ax.text(8, -32, "128K context", fontsize=8, ha="center", color=C_RED,
            fontweight="bold")
    ax.text(8, -37, "8 rounds", fontsize=7.5, ha="center", color=C_RED,
            fontstyle="italic")

    ax.set_title("Cumulative Compression Loss Over a 4-Hour Agent Session",
                 fontsize=13, fontweight="bold", pad=12)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig3_compression_loss")


# ══════════════════════════════════════════════════════════════
# Figure 4: Evolution of Context Parallelism
# ══════════════════════════════════════════════════════════════
def fig4_cp_evolution():
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box_h = 1.2

    def draw_box(x, y, w, h, label, sublabel, color, bg):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=bg, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h / 2 - 0.25, sublabel, ha="center", va="center",
                fontsize=7.5, color="#666666", fontstyle="italic")

    y_top = 5.8
    y_bot = 1.0

    # Ring Attention
    draw_box(0.3, y_top, 2.0, box_h, "Ring Attention", "p2p ring exchange", C_BLUE, C_LIGHT_BLUE)
    ax.text(1.3, y_top - 0.45, "2023", fontsize=7.5, ha="center", color="#999999")

    # Ulysses
    draw_box(2.8, y_top, 2.0, box_h, "Ulysses", "all-to-all scatter/gather", C_ORANGE, C_LIGHT_ORANGE)
    ax.text(3.8, y_top - 0.45, "2023", fontsize=7.5, ha="center", color="#999999")

    # Llama 3
    draw_box(5.3, y_top, 2.0, box_h, "Llama 3 CP", "all-gather full KV", C_GREEN, C_LIGHT_GREEN)
    ax.text(6.3, y_top - 0.45, "2024", fontsize=7.5, ha="center", color="#999999")

    # Hybrid — wider box, centered
    draw_box(2.5, y_bot, 5.0, box_h, "Hybrid CP", "a2a intra-node + p2p inter-node", C_PURPLE, C_LIGHT_PURPLE)
    ax.text(5.0, y_bot - 0.45, "2025", fontsize=7.5, ha="center", color="#999999")

    # Arrows: top boxes converge into hybrid — start below year labels, end above bottom box
    for src_x in [1.3, 3.8, 6.3]:
        ax.annotate("", xy=(5.0, y_bot + box_h + 0.15), xytext=(src_x, y_top - 0.55),
                    arrowprops=dict(arrowstyle="-|>", color="#BBBBBB", lw=1.2,
                                    connectionstyle="arc3,rad=0"))

    # Arrows between top boxes (timeline)
    ax.annotate("", xy=(2.8, y_top + 0.6), xytext=(2.3, y_top + 0.6),
                arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.0))
    ax.annotate("", xy=(5.3, y_top + 0.6), xytext=(4.8, y_top + 0.6),
                arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.0))

    # Trade-off summary on the right side, vertically centered
    y_mid = (y_top + y_bot) / 2 + 0.6
    ax.text(8.3, y_mid + 0.7, "Trade-offs", fontsize=9, fontweight="bold", color="#444444")
    ax.text(8.3, y_mid + 0.2, "Ring: simple, high latency", fontsize=8, color=C_BLUE)
    ax.text(8.3, y_mid - 0.2, "A2A: fast, head-bounded", fontsize=8, color=C_ORANGE)
    ax.text(8.3, y_mid - 0.6, "AG: overlap-friendly, high mem", fontsize=8, color=C_GREEN)
    ax.text(8.3, y_mid - 1.1, "Hybrid: breaks head ceiling", fontsize=8, color=C_PURPLE, fontweight="bold")

    ax.set_title("Evolution of Context Parallelism", fontsize=14,
                 fontweight="bold", pad=12, loc="center")
    save(fig, "fig4_cp_evolution")


# ══════════════════════════════════════════════════════════════
# Figure 5: The Inference Memory Spectrum
# ══════════════════════════════════════════════════════════════
def fig5_inference_spectrum():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Spectrum arrow
    ax.annotate("", xy=(9, 6.2), xytext=(1, 6.2),
                arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=2))
    ax.text(5, 6.5, "Increasing compression", ha="center", fontsize=10,
            color="#888888", fontstyle="italic")

    # Method boxes along the spectrum — evenly spaced
    methods = [
        (1.3, "MLA", "O(n)", "KV to latent\nsmaller constant", C_BLUE, C_LIGHT_BLUE, "DeepSeek V3"),
        (3.6, "KSA", "O(n/k)", "summary tokens\nat ratio k", C_GREEN, C_LIGHT_GREEN, "Kwai"),
        (5.9, "DSA", "O(top-k)", "indexer + select\nKV offloading", C_ORANGE, C_LIGHT_ORANGE, "DS V3.2"),
        (8.2, "CSA+HCA", "O(top-k')", "compressed index\nSSD offloading", C_RED, C_LIGHT_RED, "DS V4"),
    ]

    for x, name, complexity, desc, color, bg, source in methods:
        bw = 1.7
        rect = FancyBboxPatch((x - bw / 2, 4.0), bw, 1.8, boxstyle="round,pad=0.15",
                              facecolor=bg, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 5.4, name, ha="center", va="center", fontsize=11,
                fontweight="bold", color=color)
        ax.text(x, 4.95, complexity, ha="center", va="center", fontsize=9,
                color="#444444", family="monospace")
        ax.text(x, 4.45, desc, ha="center", va="center", fontsize=7.5,
                color="#666666", fontstyle="italic", linespacing=1.3)
        ax.text(x, 3.55, source, ha="center", fontsize=7.5, color="#999999")

    # Hybrid architecture bars at bottom
    ax.text(0.3, 2.7, "Hybrid architectures interleave attention types:", fontsize=10,
            fontweight="bold", color="#444444")

    # DeepSeek V4 bar — correct pattern: Full, Full, then alternating CSA/HCA
    # Simplified to show the repeating unit: CSA, HCA, CSA, HCA, ...
    v4_layers = ["CSA", "HCA", "CSA", "HCA", "CSA", "HCA", "CSA", "HCA"]
    v4_colors_f = [C_LIGHT_ORANGE, C_LIGHT_RED] * 4
    v4_colors_e = [C_ORANGE, C_RED] * 4
    cell_w = 0.82
    x_start = 1.0
    for i, (layer, fc, ec) in enumerate(zip(v4_layers, v4_colors_f, v4_colors_e)):
        rect = FancyBboxPatch((x_start + i * cell_w, 1.7), cell_w - 0.08, 0.55,
                              boxstyle="round,pad=0.05",
                              facecolor=fc, edgecolor=ec, linewidth=1)
        ax.add_patch(rect)
        ax.text(x_start + i * cell_w + (cell_w - 0.08) / 2, 1.97, layer,
                ha="center", va="center", fontsize=7, color=ec, fontweight="bold")
    # Ellipsis after
    ax.text(x_start + 8 * cell_w + 0.15, 1.97, "...", fontsize=11,
            ha="center", va="center", color="#999999")
    ax.text(x_start + 8 * cell_w + 0.7, 1.97, "DeepSeek V4", fontsize=8.5,
            va="center", color="#666666")

    # MiMo bar — 6 SWA + 1 Full
    mimo_layers = ["SWA", "SWA", "SWA", "SWA", "SWA", "SWA", "Full"]
    for i, layer in enumerate(mimo_layers):
        c = C_LIGHT_GREEN if layer == "SWA" else C_LIGHT_BLUE
        ec = C_GREEN if layer == "SWA" else C_BLUE
        rect = FancyBboxPatch((x_start + i * cell_w, 0.8), cell_w - 0.08, 0.55,
                              boxstyle="round,pad=0.05",
                              facecolor=c, edgecolor=ec, linewidth=1)
        ax.add_patch(rect)
        ax.text(x_start + i * cell_w + (cell_w - 0.08) / 2, 1.07, layer,
                ha="center", va="center", fontsize=7, color=ec, fontweight="bold")
    # Ellipsis after
    ax.text(x_start + 7 * cell_w + 0.15, 1.07, "...", fontsize=11,
            ha="center", va="center", color="#999999")
    ax.text(x_start + 7 * cell_w + 0.7, 1.07, "MiMo-V2.5 (6:1)", fontsize=8.5,
            va="center", color="#666666")

    ax.set_title("Inference Memory Spectrum", fontsize=14,
                 fontweight="bold", pad=12)
    save(fig, "fig5_inference_spectrum")


# ══════════════════════════════════════════════════════════════
# Figure 6: The Impossible Triangle
# ══════════════════════════════════════════════════════════════
def fig6_impossible_triangle():
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.2, 1.7)
    ax.axis("off")
    ax.set_aspect("equal")

    # Triangle vertices — slightly larger
    top = (0, 1.3)
    left = (-1.2, -0.5)
    right = (1.2, -0.5)

    # Fill + edges
    triangle_fill = plt.Polygon([top, left, right], fill=True,
                                facecolor="#F8F8F8", edgecolor="none")
    ax.add_patch(triangle_fill)
    triangle_edge = plt.Polygon([top, left, right], fill=False,
                                edgecolor=C_GRAY, linewidth=2, linestyle="--")
    ax.add_patch(triangle_edge)

    # Vertex labels with circles
    labels_data = [(top, "Quality", C_BLUE, (0, 0.35)),
                   (left, "Diversity", C_ORANGE, (0, -0.35)),
                   (right, "Length", C_GREEN, (0, -0.35))]
    for (x, y), label, color, (ox, oy) in labels_data:
        circle = plt.Circle((x, y), 0.18, facecolor=color, edgecolor="white",
                            linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label[0], ha="center", va="center", fontsize=13,
                fontweight="bold", color="white", zorder=6)
        ax.text(x + ox, y + oy, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

    # Edge annotations — each edge labeled with data source that has those 2 properties
    ax.text(-0.85, 0.5, "Books", fontsize=8.5, ha="center", color="#888888",
            fontstyle="italic", rotation=38)
    ax.text(0.85, 0.5, "Web data", fontsize=8.5, ha="center", color="#888888",
            fontstyle="italic", rotation=-38)
    ax.text(0, -0.7, "Synthetic", fontsize=8.5, ha="center", color="#888888",
            fontstyle="italic")

    # Center label
    ax.text(0, 0.2, "Hard to get\nall three", ha="center", va="center",
            fontsize=9, color="#BBBBBB", fontstyle="italic", linespacing=1.3)

    # Bottom note
    ax.text(0, -1.05, "Trajectory data may break the triangle", ha="center",
            fontsize=9, color=C_PURPLE, fontweight="bold")

    ax.set_title("The Impossible Triangle of Long-Context Data", fontsize=13,
                 fontweight="bold", pad=12)
    save(fig, "fig6_impossible_triangle")


# ══════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig1_context_rot()
    fig2_rot_vs_anxiety()
    fig3_compression_loss()
    fig4_cp_evolution()
    fig5_inference_spectrum()
    fig6_impossible_triangle()
    print("Done!")
