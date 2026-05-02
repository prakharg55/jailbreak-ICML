"""Paper-ready plots + LaTeX table from the 50%-checkpoint test eval.

Produces (under eval/plots_paper/<MODEL_SHORT>/):
  test_pure_safety.png        — 3 ASR subplots (wildjailbreak, wildguardmix, clearharm), shared y-axis
  test_pure_overrefusal.png   — 3 refusal subplots (WJ adv_benign, WGM adv_benign, WGM benign)
  test_mixed_safety.png       — same shape as pure_safety, mixed-regime baselines
  test_mixed_overrefusal.png  — same shape as pure_overrefusal, mixed-regime baselines
  results_table.tex           — LaTeX table ready to \\input{} into the paper

Safety subplots share a y-axis (0..max_observed, rounded up to nearest 5%) so the three
safety benchmarks are directly comparable. Overrefusal gets its own 0..100 range.

Run from project root: python3 eval/plot_paper.py
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication-polished styling
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "axes.titleweight":   "semibold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "axes.edgecolor":     "#333333",
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
    "grid.color":         "#d0d0d0",
    "grid.linestyle":     "-",
    "grid.linewidth":     0.6,
    "figure.dpi":         150,
})

BAR_WIDTH = 0.58
BAR_EDGECOLOR = "#ffffff"
BAR_LINEWIDTH = 0.0

SAFETY_DATASETS = ["wildjailbreak", "wildguardmix", "clearharm"]
OVERREFUSAL_DATASETS = [
    "wildjailbreak_adv_benign",
    "wildguardmix_adv_benign",
    "wildguardmix_benign",
]

PURE_BASELINES = [
    ("base",                      "Base",    "#6c757d"),  # slate gray
    ("control_checkpoint-120",    "Control", "#43a047"),  # muted green
    ("hard_checkpoint-120",       "Hard",    "#ef9a9a"),  # refined salmon (pure=lighter)
    ("random_checkpoint-120",     "Random",  "#90caf9"),  # refined light blue
]

MIXED_BASELINES = [
    ("base",                           "Base",          "#6c757d"),
    ("control_checkpoint-235",         "Control",       "#43a047"),
    ("hard_mixed_checkpoint-240",      "Hard+AdvBen",   "#c62828"),  # deeper red (mixed=darker)
    ("random_mixed_checkpoint-240",    "Random+AdvBen", "#1565c0"),  # deeper blue
]

DATASET_DISPLAY = {
    "wildjailbreak":             "WildJailbreak",
    "wildguardmix":              "WildGuardMix",
    "clearharm":                 "ClearHarm",
    "wildjailbreak_adv_benign":  "WJ adv_benign",
    "wildguardmix_adv_benign":   "WGM adv_benign",
    "wildguardmix_benign":       "WGM benign",
}


def load_summary(results_dir, dataset, fname):
    path = os.path.join(results_dir, dataset, f"{fname}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0] if data else None
    if data is None:
        return None
    for key in ("attack_success_rate", "mean_refusal_rate"):
        if key in data:
            return data[key]
    return None


def round_up_to(x, step):
    return int(math.ceil(x / step) * step)


def plot_subplots(results_dir, baselines, datasets, out_path, metric_label, ymax=None, figsize=None, layout=None):
    n = len(datasets)
    if layout is None:
        if n <= 3:
            nrows, ncols = 1, n
        elif n == 4:
            nrows, ncols = 2, 2
        else:
            ncols = 3
            nrows = math.ceil(n / ncols)
    else:
        nrows, ncols = layout

    if figsize is None:
        figsize = (5 * ncols, 4.5 * nrows) if n > 1 else (6, 5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    # Flatten axes into a single list regardless of shape
    flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    if ymax is None:
        all_values = []
        for ds in datasets:
            for fname, _, _ in baselines:
                v = load_summary(results_dir, ds, fname)
                if v is not None:
                    all_values.append(v * 100)
        ymax = round_up_to(max(all_values), 5) if all_values else 100

    for idx, ds in enumerate(datasets):
        ax = flat_axes[idx]
        labels, values, colors = [], [], []
        for fname, label, color in baselines:
            v = load_summary(results_dir, ds, fname)
            labels.append(label)
            values.append((v or 0.0) * 100)
            colors.append(color)

        bars = ax.bar(range(len(labels)), values, color=colors,
                       edgecolor=BAR_EDGECOLOR, linewidth=BAR_LINEWIDTH, width=BAR_WIDTH)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.012,
                    f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9.5, color="#222222")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(metric_label)
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.set_ylim(0, ymax)
        ax.grid(True, axis="y", alpha=0.55, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)  # no x-tick marks (labels alone)

    # Hide unused axes (e.g. when n isn't a perfect row×col fit)
    for j in range(len(datasets), len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def write_latex_table(results_dir, out_path):
    """One combined table: pure and mixed regimes as two row-groups, per-column bests bolded."""
    pure_rows = [
        ("Base",                  PURE_BASELINES[0][0]),
        ("Control",               PURE_BASELINES[1][0]),
        ("Hard",                  PURE_BASELINES[2][0]),
        ("Random",                PURE_BASELINES[3][0]),
    ]
    mixed_rows = [
        ("Base",                  MIXED_BASELINES[0][0]),
        ("Control (step 235)",    MIXED_BASELINES[1][0]),
        ("Hard+AdvBen",           MIXED_BASELINES[2][0]),
        ("Random+AdvBen",         MIXED_BASELINES[3][0]),
    ]
    columns = [
        ("WildJailbreak",   "wildjailbreak",             "ASR"),
        ("WildGuardMix",    "wildguardmix",              "ASR"),
        ("ClearHarm",       "clearharm",                 "ASR"),
        ("WJ adv\\_ben",    "wildjailbreak_adv_benign",  "Ref"),
        ("WGM adv\\_ben",   "wildguardmix_adv_benign",   "Ref"),
        ("WGM benign",      "wildguardmix_benign",       "Ref"),
    ]

    # Load all values
    def build_matrix(rows):
        matrix = []
        for label, fname in rows:
            vals = [load_summary(results_dir, ds, fname) for _, ds, _ in columns]
            matrix.append((label, vals))
        return matrix

    pure_matrix = build_matrix(pure_rows)
    mixed_matrix = build_matrix(mixed_rows)

    # Find best per column within each regime (lower = better for both ASR and Ref)
    def find_best(matrix, col_idx):
        vals = [row[1][col_idx] for row in matrix if row[1][col_idx] is not None]
        return min(vals) if vals else None

    pure_bests = [find_best(pure_matrix, i) for i in range(len(columns))]
    mixed_bests = [find_best(mixed_matrix, i) for i in range(len(columns))]

    def fmt_cell(v, best):
        if v is None:
            return "--"
        s = f"{v*100:.1f}\\%"
        return f"\\textbf{{{s}}}" if best is not None and abs(v - best) < 1e-9 else s

    lines = []
    lines.append("% Auto-generated by eval/plot_paper.py — do not hand-edit")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Test-set results at the 50\\%-of-hardness-pool checkpoint (Llama-3-8B). ")
    lines.append("ASR = Attack Success Rate ($\\downarrow$). Ref = Refusal Rate ($\\downarrow$). ")
    lines.append("Best per column within each regime in \\textbf{bold}.}")
    lines.append("\\label{tab:test-results}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * len(columns) + "}")
    lines.append("\\toprule")
    header = " & ".join(f"{name} ({metric}$\\downarrow$)" for name, _, metric in columns)
    lines.append(f"Baseline & {header} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(columns) + 1) + "}{l}{\\textit{Pure baselines (N=1200 prompts, $\\tau \\approx 0.094$)}} \\\\")
    for label, vals in pure_matrix:
        cells = " & ".join(fmt_cell(v, b) for v, b in zip(vals, pure_bests))
        lines.append(f"\\quad {label} & {cells} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(columns) + 1) + "}{l}{\\textit{Mixed baselines (N=2400 total = 1200 hard + 1200 adv\\_benign)}} \\\\")
    for label, vals in mixed_matrix:
        cells = " & ".join(fmt_cell(v, b) for v, b in zip(vals, mixed_bests))
        lines.append(f"\\quad {label} & {cells} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="eval/results/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", default="eval/plots_paper/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    # Compute shared safety y-axis max across BOTH regimes so the two safety figures are directly comparable
    all_safety_values = []
    for ds in SAFETY_DATASETS:
        for group in (PURE_BASELINES, MIXED_BASELINES):
            for fname, _, _ in group:
                v = load_summary(args.results_dir, ds, fname)
                if v is not None:
                    all_safety_values.append(v * 100)
    safety_ymax = round_up_to(max(all_safety_values), 5) if all_safety_values else 20

    # Pure
    plot_subplots(
        args.results_dir, PURE_BASELINES, SAFETY_DATASETS,
        os.path.join(args.output_dir, "test_pure_safety.png"),
        "Attack Success Rate (%)",
        ymax=safety_ymax,
    )
    plot_subplots(
        args.results_dir, PURE_BASELINES, OVERREFUSAL_DATASETS,
        os.path.join(args.output_dir, "test_pure_overrefusal.png"),
        "Refusal Rate (%)",
        ymax=100,
    )

    # Mixed
    plot_subplots(
        args.results_dir, MIXED_BASELINES, SAFETY_DATASETS,
        os.path.join(args.output_dir, "test_mixed_safety.png"),
        "Attack Success Rate (%)",
        ymax=safety_ymax,
    )
    plot_subplots(
        args.results_dir, MIXED_BASELINES, OVERREFUSAL_DATASETS,
        os.path.join(args.output_dir, "test_mixed_overrefusal.png"),
        "Refusal Rate (%)",
        ymax=100,
    )

    # LaTeX table
    write_latex_table(
        args.results_dir,
        os.path.join(args.output_dir, "results_table.tex"),
    )


if __name__ == "__main__":
    main()
