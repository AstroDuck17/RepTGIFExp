#!/usr/bin/env python3
"""
plot_curves.py — Stage 4 (visualization) of the repetition probing pipeline.

Overlays layer-accuracy curves for multiple models (L, H, G) on a single
figure, using layer_fraction on the X-axis so models of different depths
are directly comparable.

Usage:
    # Single model — path must point to the directory containing layer_summary.csv
    python plot_curves.py \\
        --dirs /scratch/yashav/Reper/RepTGIFExp/artifacts/probes/16f_uniform_seed42/vjepa2_l/depth1

    # Multi-model overlay
    python plot_curves.py \\
        --dirs artifacts/probes/16f_uniform_seed42/vjepa2_l/depth1 \\
               artifacts/probes/16f_uniform_seed42/vjepa2_h/depth1 \\
               artifacts/probes/16f_uniform_seed42/vjepa2_g/depth1 \\
        --labels "V-JEPA2-L" "V-JEPA2-H" "V-JEPA2-G" \\
        --out artifacts/comparison_curve.png
"""

import argparse
import sys
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

    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(11, 6))

    n_plotted = 0
    for i, (d, label) in enumerate(zip(dirs, labels)):
        summary_path = Path(d) / "layer_summary.csv"

        if not summary_path.exists():
            print(f"[ERROR] layer_summary.csv not found at: {summary_path}")
            print(f"        Skipping '{label}'.")
            print(f"        Make sure run_probes.py finished and the path is correct.")
            continue

        df = pd.read_csv(summary_path)
        print(f"\n[INFO] Loaded {summary_path}")
        print(f"       Columns: {list(df.columns)}")
        print(f"       Rows: {len(df)}")

        if mean_col not in df.columns:
            print(f"[ERROR] Column '{mean_col}' not in {summary_path}")
            print(f"        Available columns: {list(df.columns)}")
            print(f"        Skipping '{label}'.")
            continue

        x    = df["layer_fraction"].values
        y    = df[mean_col].values
        yerr = df[std_col].values if std_col in df.columns else np.zeros_like(y)

        # Print a summary of values for debugging
        print(f"       layer_fraction range: {x.min():.3f} → {x.max():.3f}  ({len(x)} layers)")
        print(f"       {mean_col}: min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}")

        color = COLORS[i % len(COLORS)]
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color, zorder=2)
        ax.plot(x, y, "-o", color=color, markersize=3.5, linewidth=2, label=label, zorder=3)

        # Annotate peak
        if len(y) > 0:
            peak_idx = int(np.argmax(y))
            ax.annotate(
                f"peak={y[peak_idx]:.3f}\n(f={x[peak_idx]:.2f})",
                xy=(x[peak_idx], y[peak_idx]),
                xytext=(min(x[peak_idx] + 0.05, 0.9), y[peak_idx] + 0.02),
                fontsize=8,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            )
        n_plotted += 1

    # Draw chance line on top of everything
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Chance (0.50)", zorder=4)

    ax.set_xlabel("Layer fraction (0 = first transformer layer, 1 = last)", fontsize=12)
    ax.set_ylabel(
        f"{'Balanced Accuracy' if 'balanced' in metric else metric} (mean ± std across folds)",
        fontsize=12,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.40, 1.02)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    if n_plotted == 0:
        print(f"\n[WARNING] No curves were plotted — the output only shows the chance line.")
        print(f"          Check the paths above and make sure run_probes.py finished successfully.")
        sys.exit(1)

    print(f"\nComparison curve saved: {out_path}  ({n_plotted} model(s) plotted)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot multi-model layer probing comparison curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python plot_curves.py \\
      --dirs /path/to/artifacts/probes/16f_uniform_seed42/vjepa2_l/depth1

  # Multi-model
  python plot_curves.py \\
      --dirs /path/to/.../vjepa2_l/depth1 /path/to/.../vjepa2_h/depth1 \\
      --labels "V-JEPA2-L" "V-JEPA2-H" \\
      --out comparison.png
        """,
    )
    parser.add_argument("--dirs",   nargs="+", required=True,  help="Directories containing layer_summary.csv (one per model)")
    parser.add_argument("--labels", nargs="+", default=None,   help="Legend labels (one per --dirs entry)")
    parser.add_argument("--out",    default="comparison_curve.png", help="Output PNG path (default: comparison_curve.png)")
    parser.add_argument("--metric", default="test_balanced_acc",    help="Metric prefix in layer_summary.csv (default: test_balanced_acc)")
    parser.add_argument("--title",  default="Repetition Detection — Layer-wise Probing", help="Plot title")
    args = parser.parse_args()

    labels = args.labels or [Path(d).parent.parent.name + "/" + Path(d).name for d in args.dirs]
    if len(labels) != len(args.dirs):
        parser.error(f"Number of --labels ({len(labels)}) must match number of --dirs ({len(args.dirs)})")

    plot_comparison(
        dirs=args.dirs,
        labels=labels,
        out_path=args.out,
        metric=args.metric,
        title=args.title,
    )


if __name__ == "__main__":
    main()
