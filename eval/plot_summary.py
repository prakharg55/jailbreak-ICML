"""Headline summary figure for the paper.

Produces a single 2x2 grid figure:
  Rows: (a) Safety (WildJailbreak ASR)  (b) Overrefusal (WildJailbreak adv-benign refusal)
  Cols: (left) Pure regime              (right) Mixed regime

Each subplot is internally compute-matched: Pure baselines (Base, Control,
Hard, Random) all see the same prompt budget on a given model (1,200 on 8B
/ 1,250 on 3B); Mixed baselines (Base, Control, Hard-Mixed, Random-Mixed)
all see twice that. The pure-Control and mixed-Control are different
checkpoints of the same control run, taken at the matched compute for
each regime.

Output: eval/plots_paper/headline_summary.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication styling consistent with eval/plot_paper.py
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.titleweight":  "semibold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    "#333333",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "grid.color":        "#d0d0d0",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
})

COLOR_8B = "#26c6da"   # cyan
COLOR_3B = "#fb8c00"   # orange
EDGE     = "#2c3e50"   # thin dark slate border
EDGE_W   = 0.7

# Hardcoded values from the 8B and 3B test-set tables in the paper.
# Pure-regime Control on 8B is at 1,200 prompts of vanilla training;
# Mixed-regime Control on 8B is at 2,350 prompts (closest matched-compute
# checkpoint to mixed's 2,400). 3B is analogous: pure Control at 1,250
# prompts, mixed Control at the closest pre-end save near 2,500.

# --- Pure regime (1,200 on 8B / 1,250 on 3B) ---
PURE_LABELS = ["Base", "Control", "Hard", "Random"]
PURE_ASR_8B = [11.5, 12.8, 2.1, 3.1]
PURE_ASR_3B = [20.1, 11.8, 1.1, 0.9]
PURE_REF_8B = [22.4, 32.4, 83.8, 78.6]
PURE_REF_3B = [14.3, 17.6, 90.0, 91.4]

# --- Mixed regime (2,400 on 8B / 2,500 on 3B) ---
MIXED_LABELS = ["Base", "Control", "Hard-Mixed", "Random-Mixed"]
MIXED_ASR_8B = [11.5, 13.6, 5.1, 7.9]
MIXED_ASR_3B = [20.1, 15.7, 3.4, 6.8]
MIXED_REF_8B = [22.4, 30.5, 49.0, 30.0]
MIXED_REF_3B = [14.3, 15.2, 66.2, 52.4]


def grouped_bars(ax, vals_8b, vals_3b, labels, ylabel, title, ymax,
                 show_legend=False):
    n = len(labels)
    x = list(range(n))
    width = 0.28

    bars_8b = ax.bar([xi - width / 2 for xi in x], vals_8b, width=width,
                     color=COLOR_8B, edgecolor=EDGE, linewidth=EDGE_W,
                     label="Llama-3-8B-Instruct")
    bars_3b = ax.bar([xi + width / 2 for xi in x], vals_3b, width=width,
                     color=COLOR_3B, edgecolor=EDGE, linewidth=EDGE_W,
                     label="Llama-3.2-3B-Instruct")

    for bars in (bars_8b, bars_3b):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + ymax * 0.012,
                    f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8.5, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    if show_legend:
        ax.legend(frameon=False, loc="upper right", fontsize=9)


def main():
    out_dir = "eval/plots_paper"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "headline_summary.png")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), sharey="row")
    (ax_pa, ax_ma), (ax_pr, ax_mr) = axes

    # Top row: ASR (shared y-axis 0..25)
    grouped_bars(ax_pa, PURE_ASR_8B,  PURE_ASR_3B,  PURE_LABELS,
                 ylabel="WildJailbreak ASR (%) $\\downarrow$",
                 title="(a) Safety, pure regime", ymax=25,
                 show_legend=True)
    grouped_bars(ax_ma, MIXED_ASR_8B, MIXED_ASR_3B, MIXED_LABELS,
                 ylabel=None,
                 title="(b) Safety, mixed regime", ymax=25)

    # Bottom row: Refusal (shared y-axis 0..100)
    grouped_bars(ax_pr, PURE_REF_8B,  PURE_REF_3B,  PURE_LABELS,
                 ylabel="WildJailbreak adv-benign\nrefusal (%) $\\downarrow$",
                 title="(c) Overrefusal, pure regime", ymax=100)
    grouped_bars(ax_mr, MIXED_REF_8B, MIXED_REF_3B, MIXED_LABELS,
                 ylabel=None,
                 title="(d) Overrefusal, mixed regime", ymax=100)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
