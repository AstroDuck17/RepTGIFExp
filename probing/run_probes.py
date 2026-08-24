#!/usr/bin/env python3
"""
run_probes.py — Stage 3 of the repetition probing pipeline.

For each (layer, fold) combination, trains an independent linear (or MLP) probe
on the cached features from extract_features.py, evaluates on held-out test folds,
and saves per-layer accuracy curves and raw results.

Output structure:
    artifacts/probes/{cache_name}/{model}/{depth_str}/
        fold_layer_results.csv    — per (fold, layer): all metrics + best hyperparams
        oof_predictions.parquet   — per (video, layer): logit, prob, prediction
        layer_summary.csv         — per layer: mean/std across folds
        layer_curve.png           — balanced_acc_mean ± std vs layer_fraction
        summary.json              — peak layer fraction + peak balanced accuracy

Usage:
    python run_probes.py --config config.yaml [--depth 1] [--cache-dir path/to/features/cache]
"""

import argparse
import copy
import itertools
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.probe_models import build_probe, build_optimizer, count_parameters
from utils.metrics import compute_all_metrics, sigmoid


# ---------------------------------------------------------------------------
# Feature loading & standardization
# ---------------------------------------------------------------------------

def load_features(cache_dir: Path) -> Tuple[np.ndarray, dict]:
    """Load pooled.npy and extraction metadata."""
    cache_path = cache_dir / "pooled.npy"
    meta_path  = cache_dir / "extraction_meta.json"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {cache_path}\n"
            "Run extract_features.py first."
        )
    features = np.load(cache_path, mmap_mode="r")  # [N, L, D]

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    print(f"Features loaded: shape={features.shape}  path={cache_path}")
    return features, meta


def standardize(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize using train statistics. Dead neurons get scale=1."""
    mean = x_train.mean(axis=0)
    std  = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (
        (x_train - mean) / std,
        (x_val   - mean) / std,
        (x_test  - mean) / std,
    )


# ---------------------------------------------------------------------------
# Train a single probe configuration
# ---------------------------------------------------------------------------

def train_probe_config(
    probe: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: str,
) -> Tuple[torch.nn.Module, Dict]:
    """
    Train a probe for one hyperparameter configuration.
    Returns the best probe (by val balanced_acc) and its val metrics.
    """
    probe = probe.to(device)
    optimizer = build_optimizer(probe, optimizer_name, lr, weight_decay)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val   = x_val.to(device)
    y_val   = y_val.to(device)

    best_val_bal_acc = -1.0
    best_val_bce     = float("inf")
    best_state = copy.deepcopy(probe.state_dict())
    no_improve = 0

    for epoch in range(max_epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(x_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(x_val)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val).item()
            val_metrics = compute_all_metrics(
                y_val.cpu().numpy(), val_logits.cpu().numpy()
            )

        val_bal_acc = val_metrics["balanced_acc"]

        # Checkpoint: higher balanced_acc, tie-break by lower BCE
        improved = (val_bal_acc > best_val_bal_acc) or (
            val_bal_acc == best_val_bal_acc and val_loss < best_val_bce
        )
        if improved:
            best_val_bal_acc = val_bal_acc
            best_val_bce     = val_loss
            best_state       = copy.deepcopy(probe.state_dict())
            no_improve       = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    probe.load_state_dict(best_state)
    probe.eval()
    return probe, {"val_bal_acc": best_val_bal_acc, "val_bce": best_val_bce}


# ---------------------------------------------------------------------------
# Main probing loop
# ---------------------------------------------------------------------------

def run_probes(
    config: dict,
    depth: int = 1,
    cache_dir: Optional[Path] = None,
):
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    pr_cfg        = config["probing"]
    mdl_cfg       = config["model"]

    # ── Locate feature cache ───────────────────────────────────────
    if cache_dir is None:
        ds_cfg = config["dataset"]
        ext_cfg = config["extraction"]
        cache_name = (
            f"{ext_cfg['n_frames']}f"
            f"_{ext_cfg['strategy']}"
            f"_seed{ds_cfg['seed']}"
        )
        cache_dir = artifacts_dir / "features" / cache_name / mdl_cfg["short_name"]

    features, feat_meta = load_features(cache_dir)
    N, N_layers, D = features.shape

    # ── Load manifest ──────────────────────────────────────────────
    manifest_path = artifacts_dir / "manifest.parquet"
    df = pd.read_parquet(manifest_path)
    df = df.sort_values("row_index").reset_index(drop=True)
    assert len(df) == N, f"Manifest has {len(df)} rows but features has {N}"

    y_all    = df["label"].values.astype(np.float32)
    folds    = df["fold"].values
    n_folds  = config["dataset"]["n_folds"]

    # ── Output directory ───────────────────────────────────────────
    depth_str = f"depth{depth}"
    out_dir = artifacts_dir / "probes" / cache_dir.parent.name / mdl_cfg["short_name"] / depth_str
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pr_cfg.get("device", "cpu")

    print(f"\n{'=' * 60}")
    print(f"  RUN PROBES — {mdl_cfg['short_name'].upper()} | depth={depth}")
    print(f"{'=' * 60}")
    print(f"  N videos   : {N:,}")
    print(f"  N layers   : {N_layers}")
    print(f"  Hidden dim : {D}")
    print(f"  N folds    : {n_folds}")
    print(f"  Device     : {device}")

    # Hyperparameter grid
    grid = list(itertools.product(
        pr_cfg["optimizers"],
        pr_cfg["learning_rates"],
        pr_cfg["weight_decays"],
    ))
    print(f"  Grid size  : {len(grid)} configs × {N_layers} layers × {n_folds} folds "
          f"= {len(grid) * N_layers * n_folds} total probe trains")

    # ── Storage ────────────────────────────────────────────────────
    fold_layer_rows = []  # one per (fold, layer)
    # oof: for each video, store logits per layer (average across folds where it's test)
    oof_logits = np.full((N, N_layers), np.nan, dtype=np.float32)

    # ── Outer loop: layer ──────────────────────────────────────────
    t_total_start = time.time()

    for layer_idx in range(N_layers):
        layer_fraction = layer_idx / max(N_layers - 1, 1)

        # Feature matrix for this layer: [N, D]
        X_layer = np.array(features[:, layer_idx, :], dtype=np.float32)

        layer_fold_metrics = []

        for test_fold in range(n_folds):
            val_fold = (test_fold + 1) % n_folds
            train_folds = [f for f in range(n_folds) if f not in (test_fold, val_fold)]

            train_mask = np.isin(folds, train_folds)
            val_mask   = (folds == val_fold)
            test_mask  = (folds == test_fold)

            x_train_raw = X_layer[train_mask]
            x_val_raw   = X_layer[val_mask]
            x_test_raw  = X_layer[test_mask]

            y_train = y_all[train_mask]
            y_val   = y_all[val_mask]
            y_test  = y_all[test_mask]

            # Standardize
            x_train, x_val, x_test = standardize(x_train_raw, x_val_raw, x_test_raw)

            # Convert to tensors
            Xtr = torch.from_numpy(x_train)
            Ytr = torch.from_numpy(y_train)
            Xvl = torch.from_numpy(x_val)
            Yvl = torch.from_numpy(y_val)
            Xts = torch.from_numpy(x_test)

            # ── Grid search ────────────────────────────────────────
            best_val_bal_acc = -1.0
            best_probe       = None
            best_hp          = {}

            base_seed = config["dataset"]["seed"]
            probe_seed = base_seed + 10000 * test_fold + 100 * layer_idx
            torch.manual_seed(probe_seed)

            for opt_name, lr, wd in grid:
                probe = build_probe(D, depth)

                trained_probe, val_info = train_probe_config(
                    probe=probe,
                    x_train=Xtr, y_train=Ytr,
                    x_val=Xvl,   y_val=Yvl,
                    optimizer_name=opt_name,
                    lr=lr,
                    weight_decay=wd,
                    max_epochs=pr_cfg["max_epochs"],
                    patience=pr_cfg["patience"],
                    device=device,
                )

                improved = (val_info["val_bal_acc"] > best_val_bal_acc) or (
                    val_info["val_bal_acc"] == best_val_bal_acc
                    and val_info["val_bce"] < best_probe.bce if best_probe else False
                )
                if val_info["val_bal_acc"] > best_val_bal_acc:
                    best_val_bal_acc = val_info["val_bal_acc"]
                    best_probe = trained_probe
                    best_probe.bce = val_info["val_bce"]  # tag for tie-break
                    best_hp = {"optimizer": opt_name, "lr": lr, "weight_decay": wd}

            # ── Test evaluation ────────────────────────────────────
            best_probe.eval()
            with torch.no_grad():
                test_logits = best_probe(Xts.to(device)).cpu().numpy()

            test_metrics = compute_all_metrics(y_test, test_logits)

            # Store OOF logits
            oof_logits[test_mask, layer_idx] = test_logits

            row = {
                "layer_idx":      layer_idx,
                "layer_fraction": round(layer_fraction, 4),
                "test_fold":      test_fold,
                "val_fold":       val_fold,
                **{f"test_{k}": v for k, v in test_metrics.items()},
                "val_bal_acc":    best_val_bal_acc,
                "n_train":        int(train_mask.sum()),
                "n_val":          int(val_mask.sum()),
                "n_test":         int(test_mask.sum()),
                **best_hp,
            }
            fold_layer_rows.append(row)
            layer_fold_metrics.append(test_metrics)

        # Print layer summary
        bal_accs = [m["balanced_acc"] for m in layer_fold_metrics]
        aucs     = [m["roc_auc"] for m in layer_fold_metrics]
        print(
            f"  Layer {layer_idx:3d} (f={layer_fraction:.2f}) | "
            f"bal_acc={np.mean(bal_accs):.4f}±{np.std(bal_accs):.4f}  "
            f"auc={np.mean(aucs):.4f}"
        )

    # ── Aggregate results ──────────────────────────────────────────
    results_df = pd.DataFrame(fold_layer_rows)
    results_df.to_csv(out_dir / "fold_layer_results.csv", index=False)

    # Layer summary
    metric_cols = ["test_accuracy", "test_balanced_acc", "test_roc_auc", "test_bce_loss"]
    layer_summary = (
        results_df.groupby(["layer_idx", "layer_fraction"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    layer_summary.columns = ["_".join(c).strip("_") for c in layer_summary.columns]
    layer_summary.to_csv(out_dir / "layer_summary.csv", index=False)

    # OOF predictions
    oof_df = df[["row_index", "gif_name", "label", "fold"]].copy()
    for layer_idx in range(N_layers):
        logits_col = oof_logits[:, layer_idx]
        probs_col  = sigmoid(logits_col)
        preds_col  = (probs_col >= 0.5).astype(int)
        oof_df[f"layer{layer_idx}_logit"] = logits_col
        oof_df[f"layer{layer_idx}_prob"]  = probs_col
        oof_df[f"layer{layer_idx}_pred"]  = preds_col
    oof_df.to_parquet(out_dir / "oof_predictions.parquet", index=False)

    # Summary JSON
    bal_acc_means = layer_summary["test_balanced_acc_mean"].values
    peak_layer_idx = int(np.argmax(bal_acc_means))
    summary = {
        "model":             mdl_cfg["short_name"],
        "depth":             depth,
        "n_layers":          N_layers,
        "peak_layer_idx":    peak_layer_idx,
        "peak_layer_fraction": float(peak_layer_idx / max(N_layers - 1, 1)),
        "peak_balanced_acc": float(bal_acc_means[peak_layer_idx]),
        "total_time_min":    round((time.time() - t_total_start) / 60, 2),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print(f"Peak balanced accuracy: {summary['peak_balanced_acc']:.4f} "
          f"at layer {peak_layer_idx} "
          f"(fraction={summary['peak_layer_fraction']:.2f})")

    # Plot
    _plot_layer_curve(layer_summary, out_dir, mdl_cfg["short_name"], depth)

    return results_df, layer_summary, summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_layer_curve(layer_summary: pd.DataFrame, out_dir: Path, model_name: str, depth: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = layer_summary["layer_fraction"].values
        y = layer_summary["test_balanced_acc_mean"].values
        yerr = layer_summary["test_balanced_acc_std"].values

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, "-o", color="#4A90D9", markersize=4, linewidth=2, label=f"{model_name} depth={depth}")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color="#4A90D9")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.50)")
        ax.set_xlabel("Layer fraction (0 = first transformer layer, 1 = last)", fontsize=12)
        ax.set_ylabel("Balanced Accuracy (mean ± std across folds)", fontsize=12)
        ax.set_title(f"Repetition Detection — Layer-wise Probing\n{model_name} | depth={depth}", fontsize=13)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "layer_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Layer curve saved: {out_dir / 'layer_curve.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot. Install with: pip install matplotlib")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train layer-wise probes on cached V-JEPA2 features.")
    parser.add_argument("--config",    "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--depth",     "-d", type=int, default=1,   help="Probe depth (1=linear, 2=MLP-2)")
    parser.add_argument("--cache-dir", "-f", default=None,          help="Override feature cache directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    run_probes(config, depth=args.depth, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
