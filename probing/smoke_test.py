#!/usr/bin/env python3
"""
smoke_test.py — Local validation of the probing pipeline (no model required).

Generates a synthetic feature cache and runs the full probe training loop
to verify that all metrics, fold logic, and output files work correctly
before running on the remote server.

Usage:
    python smoke_test.py
"""

import os
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


def make_synthetic_config(tmp_dir: str) -> dict:
    return {
        "paths": {
            "gif_dir":      tmp_dir,
            "zero_csv":     os.path.join(tmp_dir, "zero.csv"),
            "nonzero_csv":  os.path.join(tmp_dir, "nonzero.csv"),
            "artifacts_dir": os.path.join(tmp_dir, "artifacts"),
        },
        "dataset": {
            "balance": True,
            "seed":    42,
            "n_folds": 5,
        },
        "extraction": {
            "n_frames":        16,
            "strategy":        "uniform",
            "img_size":        224,
            "normalize_mean":  [0.485, 0.456, 0.406],
            "normalize_std":   [0.229, 0.224, 0.225],
            "inference_dtype": "float32",
            "batch_size":      4,
            "min_gif_frames":  8,
            "pad_short_gifs":  True,
        },
        "model": {
            "name":       "facebook/vjepa2-vitl-fpc64-256",
            "short_name": "vjepa2_l",
        },
        "probing": {
            "depths":             [1],
            "max_epochs":         200,   # reduced for smoke test
            "patience":           50,
            "optimizers":         ["adam"],
            "learning_rates":     [1e-3, 1e-2],
            "weight_decays":      [0.01, 0.1],
            "selection_criterion": "balanced_acc",
            "device":             "cpu",
        },
        "logging": {"use_wandb": False},
    }


def run_smoke_test():
    tmp_dir = tempfile.mkdtemp(prefix="rep_probe_smoke_")
    print(f"Smoke test directory: {tmp_dir}")

    try:
        config = make_synthetic_config(tmp_dir)
        artifacts_dir = Path(config["paths"]["artifacts_dir"])
        artifacts_dir.mkdir(parents=True)

        # ── 1. Build synthetic manifest ──────────────────────────────
        print("\n[1/4] Building synthetic manifest...")
        N_per_class = 200
        N = N_per_class * 2

        rows = []
        for i in range(N):
            label = 0 if i < N_per_class else 1
            rows.append({
                "row_index": i,
                "gif_name": f"synthetic_{i:04d}",
                "gif_path": f"/fake/path/synthetic_{i:04d}.gif",
                "label":    label,
                "question": "How many times does X happen?",
                "answer":   0 if label == 0 else (i % 9 + 1),
                "vid_id":   f"fake_{i}",
                "key":      i,
            })

        df = pd.DataFrame(rows)

        # Assign folds
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_col = np.zeros(N, dtype=int)
        for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(N), df["label"].values)):
            fold_col[val_idx] = fold_idx
        df["fold"] = fold_col

        df.to_parquet(artifacts_dir / "manifest.parquet", index=False)
        print(f"  Manifest: {N} samples, balanced={N_per_class}/class")

        # ── 2. Build synthetic features ──────────────────────────────
        print("\n[2/4] Generating synthetic feature cache...")
        N_LAYERS = 6   # small for speed
        HIDDEN_D = 64

        rng = np.random.default_rng(42)
        features = rng.standard_normal((N, N_LAYERS, HIDDEN_D)).astype(np.float32)

        # Inject signal: in later layers, add class-separating offset
        for layer in range(N_LAYERS):
            signal_strength = layer / (N_LAYERS - 1)  # 0..1
            labels_arr = df["label"].values
            features[:, layer, 0] += signal_strength * (labels_arr * 2 - 1) * 3.0

        cache_name = "16f_uniform_seed42"
        cache_dir = artifacts_dir / "features" / cache_name / "vjepa2_l"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "pooled.npy"
        np.save(cache_path, features)

        meta = {
            "model":      config["model"]["name"],
            "short_name": "vjepa2_l",
            "n_layers":   N_LAYERS,
            "hidden_dim": HIDDEN_D,
            "n_videos":   N,
            "n_success":  N,
            "n_fail":     0,
            "n_frames":   16,
            "strategy":   "uniform",
            "img_size":   224,
            "inference_dtype": "float32",
            "cache_path": str(cache_path),
        }
        with open(cache_dir / "extraction_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Features: shape={features.shape}  signal injected in dim 0")

        # ── 3. Run probes ────────────────────────────────────────────
        print("\n[3/4] Running probes...")
        from run_probes import run_probes
        results_df, layer_summary, summary = run_probes(
            config,
            depth=1,
            cache_dir=cache_dir,
        )

        # ── 4. Verify results ────────────────────────────────────────
        print("\n[4/4] Verifying results...")

        # The signal grows with layer index → last layer should have highest accuracy
        bal_accs = layer_summary["test_balanced_acc_mean"].values
        peak_layer = np.argmax(bal_accs)

        print(f"  Layer balanced accuracies: {np.round(bal_accs, 3).tolist()}")
        print(f"  Peak layer: {peak_layer} (expected: {N_LAYERS - 1})")
        print(f"  Peak balanced accuracy: {bal_accs[peak_layer]:.4f}")

        assert bal_accs[-1] > bal_accs[0], \
            f"Last layer ({bal_accs[-1]:.3f}) should be better than first ({bal_accs[0]:.3f})"
        assert bal_accs[-1] > 0.60, \
            f"Peak accuracy ({bal_accs[-1]:.3f}) should be well above chance (0.50)"

        # Check output files
        probe_dir = Path(config["paths"]["artifacts_dir"]) / "probes" / cache_name / "vjepa2_l" / "depth1"
        for fname in ["fold_layer_results.csv", "layer_summary.csv", "oof_predictions.parquet", "summary.json"]:
            fpath = probe_dir / fname
            assert fpath.exists(), f"Missing output file: {fpath}"
            print(f"  ✓ {fname}")

        # Check metrics utilities
        from utils.metrics import compute_all_metrics, sigmoid, roc_auc, balanced_accuracy
        y_true = np.array([0, 0, 0, 1, 1, 1])
        logits = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        metrics = compute_all_metrics(y_true, logits)
        assert metrics["balanced_acc"] == 1.0, "Perfect logits should give balanced_acc=1.0"
        assert metrics["roc_auc"] == 1.0,      "Perfect logits should give roc_auc=1.0"
        print("  ✓ metrics utility (perfect logits → acc=1.0, auc=1.0)")

        print("\n" + "=" * 50)
        print("  ALL SMOKE TESTS PASSED ✓")
        print("=" * 50)

    finally:
        shutil.rmtree(tmp_dir)
        print(f"\nCleaned up: {tmp_dir}")


if __name__ == "__main__":
    run_smoke_test()
