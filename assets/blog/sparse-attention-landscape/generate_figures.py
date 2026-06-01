"""
Generate all figures for the Sparse Attention Landscape blog post.
Style: clean & minimal, white background, muted colors.
Matches the style of the Ultra-Long Context Paradox figures.
Output: SVG files for Jekyll embedding.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Color palette — muted, professional (same as first blog)
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


# ══════════════════════════════════════════════════════════════
# Figure 1: KV Cache Compression Taxonomy
# ══════════════════════════════════════════════════════════════
def fig1_kv_compression_taxonomy():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def draw_box(x, y, w, h, label, sublabel, color, bg):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=bg, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel, ha="center", va="center",
                fontsize=7, color="#666666", fontstyle="italic")

    # Section headers
    ax.text(3.0, 7.0, "Eviction-Based", fontsize=11, fontweight="bold",
            color=C_BLUE, ha="center")
    ax.text(8.0, 7.0, "Compaction-Based", fontsize=11, fontweight="bold",
            color=C_ORANGE, ha="center")

    # Dividing line
    ax.plot([5.8, 5.8], [0.5, 6.7], color=C_GRAY, linewidth=0.8,
            linestyle=":", zorder=0)

    # Eviction methods — timeline going down
    eviction = [
        ("StreamingLLM", "sinks + window", "2023"),
        ("H2O", "heavy-hitter eviction", "2023"),
        ("SnapKV", "observation voting", "2024"),
        ("PyramidKV", "pyramidal budgets", "2024"),
        ("ZigZagKV", "layer uncertainty", "2024"),
    ]
    bw, bh = 2.2, 0.85
    x0 = 1.9
    for i, (name, desc, year) in enumerate(eviction):
        y = 5.8 - i * 1.1
        draw_box(x0, y, bw, bh, name, desc, C_BLUE, C_LIGHT_BLUE)
        ax.text(x0 - 0.25, y + bh / 2, year, fontsize=7, ha="right",
                color="#999999", va="center")
        if i < len(eviction) - 1:
            ax.annotate("", xy=(x0 + bw / 2, y - 0.05),
                        xytext=(x0 + bw / 2, y + 0.0 - 0.2),
                        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1))

    # Compaction methods
    compaction = [
        ("Gist Tokens", "learnable prompt tokens", "2023"),
        ("Activation Beacon", "beacon condensation", "2024"),
    ]
    x1 = 6.9
    for i, (name, desc, year) in enumerate(compaction):
        y = 5.8 - i * 1.1
        draw_box(x1, y, bw, bh, name, desc, C_ORANGE, C_LIGHT_ORANGE)
        ax.text(x1 - 0.25, y + bh / 2, year, fontsize=7, ha="right",
                color="#999999", va="center")
        if i < len(compaction) - 1:
            ax.annotate("", xy=(x1 + bw / 2, y - 0.05),
                        xytext=(x1 + bw / 2, y + 0.0 - 0.2),
                        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1))

    # Shared trait at bottom
    ax.text(5.0, 0.15, "Both reduce KV entries  ->  less memory, partly less compute",
            ha="center", fontsize=9, color=C_DARK, fontstyle="italic")

    ax.set_title("KV Cache Compression Taxonomy", fontsize=14,
                 fontweight="bold", pad=12)
    save(fig, "fig1_kv_compression_taxonomy")


# ══════════════════════════════════════════════════════════════
# Figure 2: Sparse Attention Methods
# ══════════════════════════════════════════════════════════════
def fig2_sparse_attention_methods():
    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def draw_box(x, y, w, h, label, sublabel, color, bg):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=bg, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel, ha="center", va="center",
                fontsize=7, color="#666666", fontstyle="italic")

    # Central concept — note mostly full KV
    ax.text(5.0, 6.3, "Mostly full KV cache  --  sparse compute, not sparse storage",
            ha="center", fontsize=10, color=C_DARK, fontweight="bold")
    ax.text(5.0, 5.85, "(Duo Attention partially reduces KV for streaming heads)",
            ha="center", fontsize=7.5, color="#999999", fontstyle="italic")

    # Methods in a row
    methods = [
        ("InfLLM", "offload + retrieve", C_GREEN),
        ("Quest", "query-aware pages", C_BLUE),
        ("Double\nSparsity", "channel + token", C_ORANGE),
        ("Duo\nAttention", "retrieval vs\nstreaming heads", C_PURPLE),
        ("Hash\nAttention", "LSH lookup", C_RED),
    ]
    bw, bh = 1.55, 1.1
    x_start = 0.5
    gap = 0.3
    y_row = 4.0
    for i, (name, desc, color) in enumerate(methods):
        x = x_start + i * (bw + gap)
        bg = {"#5CB85C": C_LIGHT_GREEN, "#4A90D9": C_LIGHT_BLUE,
              "#E8864A": C_LIGHT_ORANGE, "#9B59B6": C_LIGHT_PURPLE,
              "#D9534F": C_LIGHT_RED}[color]
        draw_box(x, y_row, bw, bh, name, desc, color, bg)
        ax.text(x + bw / 2, y_row - 0.3, "2024", fontsize=7,
                ha="center", color="#999999")

    # SWA origin at bottom
    rect = FancyBboxPatch((2.5, 1.0), 5.0, 1.0, boxstyle="round,pad=0.15",
                          facecolor="#F5F5F5", edgecolor=C_GRAY, linewidth=1.2,
                          linestyle="--")
    ax.add_patch(rect)
    ax.text(5.0, 1.7, "Sliding Window Attention (SWA)", ha="center",
            fontsize=10, fontweight="bold", color=C_GRAY)
    ax.text(5.0, 1.25, "Longformer / BigBird (2020)  --  fixed local window, "
            "not query-dependent", ha="center", fontsize=7.5,
            color="#999999", fontstyle="italic")

    # Shared trait
    ax.text(5.0, 3.2, "All post-hoc  --  applied to dense-trained models",
            ha="center", fontsize=9, color=C_RED, fontstyle="italic")

    ax.set_title("Sparse Attention Methods", fontsize=14,
                 fontweight="bold", pad=12)
    save(fig, "fig2_sparse_attention_methods")


# ══════════════════════════════════════════════════════════════
# Figure 3: NSA Architecture
# ══════════════════════════════════════════════════════════════
def fig3_nsa_architecture():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def draw_box(x, y, w, h, label, sublabel, color, bg):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=bg, edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.2, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel, ha="center", va="center",
                fontsize=8, color="#666666", fontstyle="italic")

    # Three branches
    draw_box(0.5, 4.5, 2.6, 1.5, "Compressed", "coarse-grained\nblock summaries",
             C_ORANGE, C_LIGHT_ORANGE)
    draw_box(3.7, 4.5, 2.6, 1.5, "Selected", "fine-grained\ntop-k blocks",
             C_BLUE, C_LIGHT_BLUE)
    draw_box(6.9, 4.5, 2.6, 1.5, "Sliding Window", "local context\nrecent tokens",
             C_GREEN, C_LIGHT_GREEN)

    # Gating box at bottom
    draw_box(2.5, 1.5, 5.0, 1.2, "Learned Gating", "hardware-aligned combination",
             C_PURPLE, C_LIGHT_PURPLE)

    # Arrows from branches to gating
    for src_x in [1.8, 5.0, 8.2]:
        ax.annotate("", xy=(5.0, 2.8), xytext=(src_x, 4.4),
                    arrowprops=dict(arrowstyle="-|>", color="#BBBBBB", lw=1.5,
                                    connectionstyle="arc3,rad=0"))

    # Key insight at top
    ax.text(5.0, 7.0, "Native Sparse Attention (NSA)", ha="center",
            fontsize=13, fontweight="bold", color=C_DARK)
    ax.text(5.0, 6.5, "Sparsity baked into pre-training  —  not post-hoc",
            ha="center", fontsize=9.5, color=C_RED, fontstyle="italic")

    # Result at bottom
    ax.text(5.0, 0.7, "sparse ≈ dense quality when trained natively",
            ha="center", fontsize=10, color=C_DARK, fontweight="bold")
    ax.text(5.0, 0.25, "Best Paper, ACL 2025",
            ha="center", fontsize=9, color=C_PURPLE, fontstyle="italic")

    save(fig, "fig3_nsa_architecture")


# ══════════════════════════════════════════════════════════════
# Figure 4: Industry Adoption Timeline
# ══════════════════════════════════════════════════════════════
def fig4_industry_timeline():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Timeline arrow
    ax.annotate("", xy=(12.0, 4.3), xytext=(0.3, 4.3),
                arrowprops=dict(arrowstyle="-|>", color="#DDDDDD", lw=2.5))

    # NSA spark marker
    nsa_x = 5.2
    ax.plot([nsa_x, nsa_x], [3.5, 5.0], color=C_RED, linewidth=2, linestyle="-")
    ax.plot(nsa_x, 5.1, marker="*", markersize=14, color=C_RED, zorder=5)
    ax.text(nsa_x, 5.5, "NSA Best Paper", ha="center", fontsize=8,
            fontweight="bold", color=C_RED)

    # Before NSA — above timeline
    before = [
        (1.3, "Mistral", "SWA 4K window", "2023"),
        (2.9, "Gemma 2", "local/global alt.", "2024"),
        (4.5, "gpt-oss", "SWA hybrid", "2025"),
    ]
    bw_b, bh_b = 1.4, 0.9
    for x, name, desc, year in before:
        rect = FancyBboxPatch((x - bw_b / 2, 6.0), bw_b, bh_b,
                              boxstyle="round,pad=0.1",
                              facecolor=C_LIGHT_BLUE, edgecolor=C_BLUE,
                              linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, 6.65, name, ha="center", fontsize=8,
                fontweight="bold", color=C_BLUE)
        ax.text(x, 6.25, desc, ha="center", fontsize=6.5,
                color="#666666", fontstyle="italic")
        ax.plot([x, x], [4.4, 5.95], color="#CCCCCC",
                linewidth=0.8, linestyle=":")
        ax.text(x, 4.5, year, fontsize=7, ha="center", color="#999999")

    ax.text(3.0, 7.4, "Before NSA", fontsize=10, fontweight="bold",
            ha="center", color=C_BLUE)

    # After NSA — below timeline
    after = [
        (6.2, "MiMo / Step", "SWA + sinks", C_GREEN),
        (7.7, "LongCat", "zigzag sparse", C_ORANGE),
        (9.2, "MiniMax M2", "SWA abandoned", C_RED),
        (10.7, "GLM-5 / DS V4", "DSA, CSA+HCA", C_PURPLE),
    ]
    bw_a, bh_a = 1.5, 0.95
    for x, name, desc, color in after:
        bg = {C_GREEN: C_LIGHT_GREEN, C_ORANGE: C_LIGHT_ORANGE,
              C_RED: C_LIGHT_RED, C_PURPLE: C_LIGHT_PURPLE}[color]
        rect = FancyBboxPatch((x - bw_a / 2, 1.8), bw_a, bh_a,
                              boxstyle="round,pad=0.1",
                              facecolor=bg, edgecolor=color, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, 2.55, name, ha="center", fontsize=7.5,
                fontweight="bold", color=color)
        ax.text(x, 2.1, desc, ha="center", fontsize=6.5,
                color="#666666", fontstyle="italic")
        ax.plot([x, x], [2.8, 4.2], color="#CCCCCC",
                linewidth=0.8, linestyle=":")

    ax.text(8.7, 1.0, "After NSA  (2025-2026)", fontsize=10,
            fontweight="bold", ha="center", color=C_PURPLE)

    ax.set_title("Industry Sparse Attention Adoption", fontsize=14,
                 fontweight="bold", pad=12)
    save(fig, "fig4_industry_timeline")


# ══════════════════════════════════════════════════════════════
# Figure 5: DeepSeek V4 KV Cache Architecture
# ══════════════════════════════════════════════════════════════
def fig5_v4_cache_architecture():
    fig, ax = plt.subplots(figsize=(9, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def draw_box(x, y, w, h, label, sublabel, color, bg, fontsize_l=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=bg, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=fontsize_l, fontweight="bold", color=color)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.2, sublabel, ha="center",
                    va="center", fontsize=7, color="#666666", fontstyle="italic")

    # Architecture layer pattern — V4 has no standalone SWA layers
    # First 2 layers are pure SWA, then interleaved CSA/HCA with trailing tokens
    ax.text(1.5, 7.5, "Architecture: Interleaved CSA + HCA (+ trailing tokens)",
            fontsize=11, fontweight="bold", color=C_DARK)

    layers = ["CSA", "HCA", "CSA", "HCA", "CSA", "HCA", "CSA", "HCA",
              "CSA", "HCA"]
    cell_w = 0.78
    x_start = 0.5
    for i, layer in enumerate(layers):
        if layer == "CSA":
            c, ec = C_LIGHT_ORANGE, C_ORANGE
        else:
            c, ec = C_LIGHT_RED, C_RED
        rect = FancyBboxPatch((x_start + i * cell_w, 6.5), cell_w - 0.06, 0.55,
                              boxstyle="round,pad=0.04",
                              facecolor=c, edgecolor=ec, linewidth=1)
        ax.add_patch(rect)
        ax.text(x_start + i * cell_w + (cell_w - 0.06) / 2, 6.77, layer,
                ha="center", va="center", fontsize=7, color=ec, fontweight="bold")
    ax.text(x_start + 10 * cell_w + 0.1, 6.77, "...", fontsize=11,
            ha="center", va="center", color="#999999")
    ax.text(x_start + 10 * cell_w + 0.6, 6.77, "10% KV vs V3.2",
            fontsize=7.5, va="center", color=C_RED, fontweight="bold")

    # Storage tiers
    ax.text(5.0, 5.6, "Storage Hierarchy", fontsize=11, fontweight="bold",
            ha="center", color=C_DARK)

    tiers = [
        ("GPU HBM", "active requests", C_RED, C_LIGHT_RED),
        ("CPU DRAM", "warm entries", C_ORANGE, C_LIGHT_ORANGE),
        ("Disk (3FS)", "cold storage\n6.6 TiB/s aggregate", C_BLUE, C_LIGHT_BLUE),
    ]
    tw, th = 2.5, 0.9
    x_tier = 3.75
    for i, (name, desc, color, bg) in enumerate(tiers):
        y = 4.5 - i * 1.1
        draw_box(x_tier, y, tw, th, name, desc, color, bg, fontsize_l=9)
        if i < len(tiers) - 1:
            ax.annotate("", xy=(x_tier + tw / 2, y - 0.05),
                        xytext=(x_tier + tw / 2, y + 0.0 - 0.15),
                        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.2))

    # Trailing token caching strategies on the right
    ax.text(8.2, 5.2, "Trailing Token Strategies", fontsize=9, fontweight="bold",
            color=C_GREEN)
    strategies = [
        ("Full Cache", "store all, zero recompute"),
        ("Periodic Ckpt", "checkpoint every p tokens"),
        ("Zero Cache", "recompute nw x L tokens"),
    ]
    for i, (name, desc) in enumerate(strategies):
        y = 4.5 - i * 0.7
        ax.text(8.2, y, name, fontsize=8, color=C_GREEN,
                fontweight="bold")
        ax.text(8.2, y - 0.25, f"  {desc}", fontsize=7, color="#888888",
                fontstyle="italic")

    # CSA/HCA note on the left
    ax.text(1.5, 5.2, "CSA/HCA Cache", fontsize=9, fontweight="bold",
            color=C_ORANGE)
    ax.text(1.5, 4.7, "• Compressed entries", fontsize=8, color=C_ORANGE)
    ax.text(1.5, 4.35, "  stable, cache-friendly", fontsize=7,
            color="#888888", fontstyle="italic")
    ax.text(1.5, 4.0, "• Tail buffer", fontsize=8, color=C_ORANGE)
    ax.text(1.5, 3.65, "  uncompressed, position-\n  dependent", fontsize=7,
            color="#888888", fontstyle="italic")

    # Bottom note
    ax.text(5.0, 1.3, "Architecture × Systems co-design",
            ha="center", fontsize=10, fontweight="bold", color=C_PURPLE)
    ax.text(5.0, 0.8, "compression reduces what to store  +  "
            "selection reduces what to compute  +  3FS enables disk-tier KV reuse",
            ha="center", fontsize=8, color="#888888", fontstyle="italic")

    ax.set_title("DeepSeek V4: KV Cache Architecture", fontsize=14,
                 fontweight="bold", pad=12)
    save(fig, "fig5_v4_cache_architecture")


# ══════════════════════════════════════════════════════════════
# Figure 6: KV Cache Systems Ecosystem
# ══════════════════════════════════════════════════════════════
def fig6_systems_ecosystem():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def draw_box(x, y, w, h, label, sublabel, color, bg):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                              facecolor=bg, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.18, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h / 2 - 0.18, sublabel, ha="center", va="center",
                fontsize=7, color="#666666", fontstyle="italic")

    # Three systems in a row
    bw, bh = 2.7, 1.4

    draw_box(0.5, 4.8, bw, bh, "SGLang Radix Cache",
             "prefix sharing via radix tree", C_BLUE, C_LIGHT_BLUE)
    draw_box(3.65, 4.8, bw, bh, "HiCache (SGLang)",
             "GPU->CPU->SSD, tombstone", C_ORANGE, C_LIGHT_ORANGE)
    draw_box(6.8, 4.8, bw, bh, "Mooncake",
             "disaggregated KV via RDMA", C_GREEN, C_LIGHT_GREEN)

    # Arrow connecting to payoff
    ax.annotate("", xy=(5.0, 3.2), xytext=(5.0, 4.6),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=1.5))

    # Bottom: the payoff
    rect = FancyBboxPatch((1.2, 1.5), 7.6, 1.5, boxstyle="round,pad=0.15",
                          facecolor="#FFF9F0", edgecolor=C_ORANGE, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(5.0, 2.55, "Sparse attention amplifies all of these", ha="center",
            fontsize=11, fontweight="bold", color=C_DARK)
    ax.text(5.0, 2.0, "smaller per-token KV  --  more cached requests  "
            "--  higher hit rate  --  lower cost",
            ha="center", fontsize=8.5, color=C_ORANGE, fontstyle="italic")

    ax.set_title("KV Cache Systems Ecosystem", fontsize=14,
                 fontweight="bold", pad=12)
    save(fig, "fig6_systems_ecosystem")


# ══════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig1_kv_compression_taxonomy()
    fig2_sparse_attention_methods()
    fig3_nsa_architecture()
    fig4_industry_timeline()
    fig5_v4_cache_architecture()
    fig6_systems_ecosystem()
    print("Done!")
