#!/usr/bin/env python3
"""
build_manifest.py — Stage 1 of the repetition probing pipeline.

Reads the zero/nonzero count CSVs, optionally balances classes,
assigns 5-fold stratified cross-validation splits, and saves a
Parquet manifest for use by extract_features.py and run_probes.py.

Usage:
    python build_manifest.py --config config.yaml

Outputs:
    artifacts/manifest.parquet
    artifacts/manifest_stats.txt   (human-readable summary)
"""

import argparse
import os
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv_rows(csv_path: str, gif_dir: str, label: int) -> list[dict]:
    """Read a TGIF-QA count CSV and return rows as dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gif_name = row["gif_name"].strip()
            gif_path = os.path.join(gif_dir, gif_name + ".gif")
            rows.append({
                "gif_name": gif_name,
                "gif_path": gif_path,
                "question": row["question"].strip(),
                "answer":   int(float(row["answer"].strip())),
                "vid_id":   row["vid_id"].strip(),
                "key":      int(row["key"].strip()),
                "label":    label,
            })
    return rows


def assign_folds(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    stratify_col: str = "label",
) -> pd.DataFrame:
    """
    Add a 'fold' column via StratifiedKFold, stratified by stratify_col.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(df), dtype=int)

    for fold_idx, (_, val_idx) in enumerate(
        skf.split(np.zeros(len(df)), df[stratify_col].values)
    ):
        fold_assignments[val_idx] = fold_idx

    df = df.copy()
    df["fold"] = fold_assignments
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_manifest(config: dict) -> pd.DataFrame:
    paths      = config["paths"]
    ds_cfg     = config["dataset"]
    seed       = ds_cfg["seed"]
    n_folds    = ds_cfg["n_folds"]
    do_balance = ds_cfg["balance"]
    gif_dir    = paths["gif_dir"]

    print("=" * 60)
    print("  BUILD MANIFEST — Repetition Probing Pipeline")
    print("=" * 60)

    # ── Load rows ──────────────────────────────────────────────────
    print(f"\nLoading non-repetition (zero) CSV : {paths['zero_csv']}")
    zero_rows = load_csv_rows(paths["zero_csv"],    gif_dir, label=0)

    print(f"Loading repetition (nonzero) CSV  : {paths['nonzero_csv']}")
    nonzero_rows = load_csv_rows(paths["nonzero_csv"], gif_dir, label=1)

    print(f"  Zero rows loaded    : {len(zero_rows):,}")
    print(f"  Nonzero rows loaded : {len(nonzero_rows):,}")

    # ── Class balance ──────────────────────────────────────────────
    if do_balance:
        n_target = min(len(zero_rows), len(nonzero_rows))
        rng = np.random.default_rng(seed)

        if len(zero_rows) > n_target:
            idx = rng.choice(len(zero_rows), size=n_target, replace=False)
            zero_rows = [zero_rows[i] for i in idx]
        if len(nonzero_rows) > n_target:
            idx = rng.choice(len(nonzero_rows), size=n_target, replace=False)
            nonzero_rows = [nonzero_rows[i] for i in idx]

        print(f"\n  Balanced to {n_target:,} samples per class ({n_target * 2:,} total)")

    all_rows = zero_rows + nonzero_rows

    # Shuffle all rows before assigning row_index
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(all_rows)

    df = pd.DataFrame(all_rows)
    df.insert(0, "row_index", np.arange(len(df)))  # CRITICAL: sequential 0..N-1

    # ── Check GIF files exist ──────────────────────────────────────
    print(f"\nChecking GIF file existence in: {gif_dir}")
    missing = [row for row in df.itertuples() if not os.path.exists(row.gif_path)]
    if missing:
        print(f"  WARNING: {len(missing)} GIF files not found on disk.")
        print(f"  First 5 missing: {[m.gif_path for m in missing[:5]]}")
        print("  (This is expected if you're building the manifest locally")
        print("   before transferring GIFs to the remote server.)")
    else:
        print(f"  All {len(df):,} GIF files found.")

    # ── Assign folds ───────────────────────────────────────────────
    df = assign_folds(df, n_folds=n_folds, seed=seed)

    # Verify fold balance
    print(f"\nFold distribution (n_folds={n_folds}):")
    for fold in range(n_folds):
        fold_df = df[df["fold"] == fold]
        n0 = (fold_df["label"] == 0).sum()
        n1 = (fold_df["label"] == 1).sum()
        print(f"  Fold {fold}: {len(fold_df):5,} samples  [{n0} neg | {n1} pos]")

    # ── Save ───────────────────────────────────────────────────────
    artifacts_dir = Path(paths["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifacts_dir / "manifest.parquet"
    df.to_parquet(manifest_path, index=False)
    print(f"\nManifest saved to: {manifest_path}")

    # Human-readable summary
    stats = {
        "total_samples":    int(len(df)),
        "label_0_count":    int((df["label"] == 0).sum()),
        "label_1_count":    int((df["label"] == 1).sum()),
        "balanced":         do_balance,
        "n_folds":          n_folds,
        "seed":             seed,
        "missing_gifs":     len(missing),
        "manifest_path":    str(manifest_path),
    }
    stats_path = artifacts_dir / "manifest_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nManifest stats:")
    for k, v in stats.items():
        print(f"  {k:<25}: {v}")
    print("=" * 60)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build manifest parquet for repetition probing.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    build_manifest(config)


if __name__ == "__main__":
    main()
