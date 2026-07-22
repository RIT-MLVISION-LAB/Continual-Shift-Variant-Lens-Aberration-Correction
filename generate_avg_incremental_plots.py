"""
Generates an average incremental PSNR comparison plot from two
lower-triangular PSNR matrices (ConDA adapter vs. sequential fine-tuning).

At each stage T_k the plotted value is:
    AIP(T_k) = (1/k) * sum_{j=1}^{k} PSNR(T_k, j)

i.e. the mean PSNR across all tasks seen so far, evaluated after training
on the k-th task.

Input format (JSON, per method):
    A lower-triangular matrix where entry [i][j] = PSNR on task j+1
    after sequential training through task i+1.

Usage:
    python generate_avg_incremental_plot.py \
        --adapter conda_psnr_matrix.json \
        --sequential sequential_psnr_matrix.json \
        --output avg_incremental_plot.pdf \
        --domain-labels "Optical Blur" "Denoising" "Dehazing" "Deraining" "Low-Light"
"""

import argparse
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


METHOD_STYLES = {
    "adapter": {
        "color": "#4477AA",
        "marker": "o",
        "linestyle": "-",
        "label": "ConDA (Adapter)",
    },
    "sequential": {
        "color": "#EE6677",
        "marker": "s",
        "linestyle": "--",
        "label": "Sequential FT",
    },
}

FILL_COLOR = "#4477AA"
FILL_ALPHA = 0.12


def setup_plot_style():
    plt.rcParams.update(
        {
            # Font
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "mathtext.fontset": "cm",
            # Font sizes — calibrated for single-column figures
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            # Lines
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            # Axes
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            # Grid
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            # Ticks
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
            # Legend
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "legend.fancybox": False,
            "legend.borderpad": 0.4,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_psnr_matrix(filepath):
    with open(filepath, "r") as f:
        matrix = json.load(f)

    for i, row in enumerate(matrix):
        assert len(row) == i + 1, f"Row {i} has {len(row)} entries, expected {i + 1}"
    return matrix


def compute_avg_incremental(psnr_matrix):
    return [np.mean(row) for row in psnr_matrix]


def plot_avg_incremental(adapter_matrix, sequential_matrix, output_path,
                         domain_labels=None, figsize=None, y_limits=None):
    setup_plot_style()

    n_stages_a = len(adapter_matrix)
    n_stages_s = len(sequential_matrix)
    assert n_stages_a == n_stages_s, f"Matrix size mismatch!"
    n_stages = n_stages_a

    avg_adapter = compute_avg_incremental(adapter_matrix)
    avg_sequential = compute_avg_incremental(sequential_matrix)

    if figsize is None:
        width = min(3.5 + 0.3 * n_stages, 7.0)
        figsize = (width, width * 0.65)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x_stages = list(range(1, n_stages + 1))

    ax.fill_between(x_stages, avg_sequential, avg_adapter, 
                    where=[a >= s for a, s in zip(avg_adapter, avg_sequential)],
                    color=FILL_COLOR, alpha=FILL_ALPHA, interpolate=True, zorder=1)

    for key in ("adapter", "sequential"):
        style = METHOD_STYLES[key]
        y = avg_adapter if key == "adapter" else avg_sequential
        ax.plot(x_stages, y, color=style["color"], linestyle=style["linestyle"], 
                marker=style["marker"], markeredgecolor="white", 
                markeredgewidth=0.8, label=style["label"], zorder=3)

    for k, (a, s) in enumerate(zip(avg_adapter, avg_sequential)):
        delta = a - s
        if abs(delta) > 0.05:  # skip negligible gaps
            y_mid = (a + s) / 2
            ax.annotate(f"Δ{delta:.2f}", xy=(x_stages[k], y_mid), 
                        fontsize=7.5, ha="left", va="center", 
                        xytext=(-8, 0), textcoords="offset points")

    ax.set_xlabel("Incremental Training Stage", labelpad=6)
    ax.set_ylabel("Avg. PSNR (dB)", labelpad=6)
    ax.set_xticks(x_stages)

    if domain_labels and len(domain_labels) == n_stages:
        xtick_labels = [domain_labels[0] if i == 0 else f"+{domain_labels[i]}" for i in range(n_stages)]
    else:
        xtick_labels = [f"D{i + 1}" if i == 0 else f"+D{i + 1}" for i in range(n_stages)]
    ax.set_xticklabels(xtick_labels, ha="right")

    if y_limits:
        ax.set_ylim(y_limits)
    else:
        all_vals = avg_adapter + avg_sequential
        y_min = min(all_vals) - 1.5
        y_max = max(all_vals) + 1.5
        ax.set_ylim(y_min, y_max)

    ax.set_xlim(0.6, n_stages + 0.4)

    legend = ax.legend(loc="best", borderaxespad=0.3)
    legend.get_frame().set_linewidth(0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"PDF saved: {output_path}")

    png_path = Path(output_path).with_suffix(".png")
    fig.savefig(str(png_path), dpi=300)
    print(f"PNG saved: {png_path}")

    plt.close(fig)
    return output_path, str(png_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compare average incremental PSNR: ConDA adapter vs. sequential fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--adapter", type=str, required=True,
                        help="JSON file: lower-triangular PSNR matrix for ConDA adapter")
    parser.add_argument("--sequential", type=str, required=True,
                        help="JSON file: lower-triangular PSNR matrix for sequential fine-tuning")
    parser.add_argument("--output", type=str, 
                        default="outputs/forgetting_visualizations/avg_incremental_plot.pdf",
                        help="Output PDF path")
    parser.add_argument("--domain-labels", type=str, nargs="+", default=None, 
                        help='x-tick labels: --domain-labels "Blur" "Denoise" "Dehaze" "Derain" "Low-Light"')
    parser.add_argument("--y-min", type=float, default=None,
                        help="Override y-axis lower bound")
    parser.add_argument("--y-max", type=float, default=None, 
                        help="Override y-axis upper bound")

    args = parser.parse_args()

    adapter_matrix = load_psnr_matrix(args.adapter)
    sequential_matrix = load_psnr_matrix(args.sequential)

    y_limits = None
    if args.y_min is not None and args.y_max is not None:
        y_limits = (args.y_min, args.y_max)

    plot_avg_incremental(adapter_matrix, sequential_matrix, output_path=args.output, 
                         domain_labels=args.domain_labels, y_limits=y_limits)


if __name__ == "__main__":
    main()