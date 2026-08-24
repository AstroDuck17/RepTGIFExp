#!/usr/bin/env python3
"""
plot_curves.py — Stage 4 (visualization) of the repetition probing pipeline.

Overlays layer-accuracy curves for multiple models (L, H, G) on a single
figure, using layer_fraction on the X-axis so models of different depths
are directly comparable.

Usage:
    # Single model
    python plot_curves.py --dirs artifacts/probes/16f_uniform_seed42/vjepa2_l/depth1

    # Multi-model overlay
    python plot_curves.py \\
        --dirs artifacts/probes/16f_uniform_seed42/vjepa2_l/depth1 \\
               artifacts/probes/16f_uniform_seed42/vjepa2_h/depth1 \\
               artifacts/probes/16f_uniform_seed42/vjepa2_g/depth1 \\
        --labels "V-JEPA2-L" "V-JEPA2-H" "V-JEPA2-G" \\
        --out artifacts/comparison_curve.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Palette for up to 6 curves
COLORS = ["#4A90D9", "#E05252", "#50C878", "#F5A623", "#9B59B6", "#1ABC9C"]


def plot_comparison(
    dirs: list[str],
    labels: list[str],
    out_path: str = "comparison_curve.png",
    metric: str = "test_balanced_acc",
    title: str = "Repetition Detection — Layer-wise Probing",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Chance (0.50)", zorder=1)

    for i, (d, label) in enumerate(zip(dirs, labels)):
        summary_path = Path(d) / "layer_summary.csv"
        if not summary_path.exists():
            print(f"WARNING: {summary_path} not found — skipping {label}")
            continue

        df = pd.read_csv(summary_path)
        x    = df["layer_fraction"].values
        mean_col = f"{metric}_mean"
        std_col  = f"{metric}_std"

        if mean_col not in df.columns:
            print(f"WARNING: column '{mean_col}' not in {summary_path} — skipping")
            continue

        y    = df[mean_col].values
        yerr = df[std_col].values if std_col in df.columns else np.zeros_like(y)

        color = COLORS[i % len(COLORS)]
        ax.plot(x, y, "-o", color=color, markersize=3.5, linewidth=2, label=label, zorder=3)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color, zorder=2)

        # Annotate peak
        peak_idx = np.argmax(y)
        ax.annotate(
            f"peak={y[peak_idx]:.3f}\n(f={x[peak_idx]:.2f})",
            xy=(x[peak_idx], y[peak_idx]),
            xytext=(x[peak_idx] + 0.03, y[peak_idx] + 0.02),
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

    ax.set_xlabel("Layer fraction (0 = first transformer layer, 1 = last)", fontsize=12)
    ax.set_ylabel(f"{'Balanced Accuracy' if 'balanced' in metric else metric} (mean ± std)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.40, 1.02)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Comparison curve saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot multi-model layer probing comparison curves.")
    parser.add_argument("--dirs",   nargs="+", required=True,  help="Probe result directories (one per model)")
    parser.add_argument("--labels", nargs="+", default=None,   help="Legend labels (one per dir)")
    parser.add_argument("--out",    default="comparison_curve.png", help="Output PNG path")
    parser.add_argument("--metric", default="test_balanced_acc",    help="Metric column prefix (without _mean/_std)")
    parser.add_argument("--title",  default="Repetition Detection — Layer-wise Probing", help="Plot title")
    args = parser.parse_args()

    labels = args.labels or [Path(d).parent.name for d in args.dirs]
    if len(labels) != len(args.dirs):
        raise ValueError("Number of --labels must match number of --dirs")

    plot_comparison(
        dirs=args.dirs,
        labels=labels,
        out_path=args.out,
        metric=args.metric,
        title=args.title,
    )


if __name__ == "__main__":
    main()
