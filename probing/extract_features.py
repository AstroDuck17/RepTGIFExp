#!/usr/bin/env python3
"""
extract_features.py — Stage 2 of the repetition probing pipeline.

Loads a frozen V-JEPA2 encoder, processes each GIF in the manifest,
extracts all intermediate hidden states, mean-pools over spatiotemporal
tokens, and saves a single memory-mapped NumPy array:

    pooled.npy   shape: [N_videos, N_layers, hidden_dim]   dtype: float32

The row order in pooled.npy corresponds exactly to row_index in the manifest.

Usage:
    python extract_features.py --config config.yaml [--manifest path/to/manifest.parquet]

Notes:
    - The model is NEVER updated. All parameters are frozen.
    - Features are always saved as float32 (even if inference runs in float16).
    - Extraction is resumable: if pooled.npy already exists and has the correct
      shape, only missing rows are (re-)extracted. Use --force to re-extract all.
"""

import argparse
import os
import gc
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import torch

# Local utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.gif_decoder import load_gif, get_gif_frame_count


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_vjepa2_encoder(model_name: str, dtype: torch.dtype = torch.float32):
    """
    Load a V-JEPA2 encoder from HuggingFace Hub with all parameters frozen.

    Returns:
        model: The frozen encoder.
        n_layers: Number of transformer layers (excluding embedding).
        hidden_dim: Hidden dimension D.
    """
    print(f"Loading model: {model_name}")
    print("  (This may take a moment for large models...)")

    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{model_name}' from HuggingFace Hub.\n"
            f"  Ensure the model name is correct and weights are available.\n"
            f"  Original error: {e}"
        )

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(dtype=dtype)
    model.eval()

    # Determine architecture
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        n_layers = cfg.num_hidden_layers
    elif hasattr(cfg, "depth"):
        n_layers = cfg.depth
    else:
        raise AttributeError(
            "Cannot determine num_hidden_layers from model config. "
            "Check model.config attributes."
        )

    if hasattr(cfg, "hidden_size"):
        hidden_dim = cfg.hidden_size
    elif hasattr(cfg, "embed_dim"):
        hidden_dim = cfg.embed_dim
    else:
        raise AttributeError("Cannot determine hidden_size from model config.")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}, Params: {total_params:.0f}M")
    print("  All parameters frozen.")

    return model, n_layers, hidden_dim


# ---------------------------------------------------------------------------
# Single-video feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_one_video(
    model,
    gif_path: str,
    n_frames: int,
    strategy: str,
    img_size: int,
    mean: tuple,
    std: tuple,
    min_frames: int,
    pad_short: bool,
    device: torch.device,
    inference_dtype: torch.dtype,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """
    Extract hidden states for one video.

    Returns:
        np.ndarray of shape [N_layers, hidden_dim] in float32, or None on failure.
    """
    try:
        frames = load_gif(
            gif_path,
            n_frames=n_frames,
            strategy=strategy,
            img_size=img_size,
            mean=mean,
            std=std,
            min_frames=min_frames,
            pad_short=pad_short,
            seed=seed,
        )
        # frames: [T, C, H, W] float32
        # V-JEPA2 expects: [batch, T, C, H, W]
        x = frames.unsqueeze(0).to(device=device, dtype=inference_dtype)

        outputs = model(pixel_values=x, output_hidden_states=True)

        # hidden_states is a tuple of length (n_layers + 1):
        # index 0 = embedding output (discarded), indices 1..n_layers = transformer layers
        hidden_states = outputs.hidden_states  # tuple of [1, num_tokens, hidden_dim]

        layer_features = []
        for state in hidden_states[1:]:  # skip embedding layer (index 0)
            pooled = state.mean(dim=1)  # [1, hidden_dim] → mean over tokens
            layer_features.append(pooled.squeeze(0).float().cpu().numpy())

        return np.stack(layer_features, axis=0)  # [N_layers, hidden_dim]

    except Exception as e:
        print(f"  ERROR extracting {gif_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Batch extraction loop
# ---------------------------------------------------------------------------

def extract_features(config: dict, manifest_path: Optional[str] = None, force: bool = False):
    paths     = config["paths"]
    ext_cfg   = config["extraction"]
    mdl_cfg   = config["model"]

    artifacts_dir = Path(paths["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── Load manifest ──────────────────────────────────────────────
    if manifest_path is None:
        manifest_path = artifacts_dir / "manifest.parquet"
    df = pd.read_parquet(manifest_path)
    df = df.sort_values("row_index").reset_index(drop=True)

    assert list(df["row_index"]) == list(range(len(df))), \
        "row_index must be consecutive 0..N-1"

    N = len(df)
    print(f"\n{'=' * 60}")
    print(f"  EXTRACT FEATURES — {mdl_cfg['short_name'].upper()}")
    print(f"{'=' * 60}")
    print(f"  Videos     : {N:,}")
    print(f"  Strategy   : {ext_cfg['strategy']}, T={ext_cfg['n_frames']} frames")
    print(f"  Model      : {mdl_cfg['name']}")

    # ── Feature cache path ─────────────────────────────────────────
    cache_name = (
        f"{ext_cfg['n_frames']}f"
        f"_{ext_cfg['strategy']}"
        f"_seed{config['dataset']['seed']}"
    )
    cache_dir = artifacts_dir / "features" / cache_name / mdl_cfg["short_name"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "pooled.npy"

    # ── Load model ─────────────────────────────────────────────────
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    inference_dtype = dtype_map.get(ext_cfg["inference_dtype"], torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model, n_layers, hidden_dim = load_vjepa2_encoder(mdl_cfg["name"], dtype=inference_dtype)
    model = model.to(device)

    print(f"  N_layers   : {n_layers}  (probing layers 1..{n_layers})")
    print(f"  Hidden dim : {hidden_dim}")

    # ── Initialize or resume feature array ────────────────────────
    expected_shape = (N, n_layers, hidden_dim)
    done_mask = np.zeros(N, dtype=bool)

    if cache_path.exists() and not force:
        existing = np.load(cache_path, mmap_mode="r")
        if existing.shape == expected_shape:
            print(f"\n  Found existing cache: {cache_path}")
            # Mark rows with non-zero content as done
            # (all-zero rows = not yet extracted)
            sums = existing.reshape(N, -1).sum(axis=1)
            done_mask = (sums != 0)
            n_done = done_mask.sum()
            print(f"  Resuming: {n_done}/{N} already extracted, {N - n_done} remaining.")
            features = np.copy(existing)
            del existing
        else:
            print(f"  Shape mismatch ({existing.shape} vs {expected_shape}). Re-extracting.")
            features = np.zeros(expected_shape, dtype=np.float32)
    else:
        features = np.zeros(expected_shape, dtype=np.float32)

    # ── Extraction loop ────────────────────────────────────────────
    n_todo = int((~done_mask).sum())
    print(f"\nExtracting features for {n_todo} videos...")

    t_start = time.time()
    n_success = 0
    n_fail = 0

    for i, row in enumerate(df.itertuples()):
        if done_mask[row.row_index]:
            continue

        row_feat = extract_one_video(
            model=model,
            gif_path=row.gif_path,
            n_frames=ext_cfg["n_frames"],
            strategy=ext_cfg["strategy"],
            img_size=ext_cfg["img_size"],
            mean=tuple(ext_cfg["normalize_mean"]),
            std=tuple(ext_cfg["normalize_std"]),
            min_frames=ext_cfg["min_gif_frames"],
            pad_short=ext_cfg["pad_short_gifs"],
            device=device,
            inference_dtype=inference_dtype,
            seed=config["dataset"]["seed"],
        )

        if row_feat is not None:
            assert row_feat.shape == (n_layers, hidden_dim), \
                f"Unexpected shape {row_feat.shape} for video {row.gif_name}"
            features[row.row_index] = row_feat
            done_mask[row.row_index] = True
            n_success += 1
        else:
            n_fail += 1

        # Progress + periodic checkpoint
        if (i + 1) % 100 == 0 or (i + 1) == N:
            elapsed = time.time() - t_start
            rate = n_success / max(elapsed, 1)
            remaining = (n_todo - n_success - n_fail) / max(rate, 1e-9)
            print(
                f"  [{i + 1:>6}/{N}] "
                f"done={n_success+n_fail}  ok={n_success}  fail={n_fail}  "
                f"rate={rate:.1f} vid/s  eta={remaining/60:.1f} min"
            )

        # Checkpoint every 500 videos
        if (i + 1) % 500 == 0:
            np.save(cache_path, features)
            print(f"  Checkpoint saved → {cache_path}")

    # ── Final save ─────────────────────────────────────────────────
    np.save(cache_path, features)

    # Save extraction metadata
    meta = {
        "model":         mdl_cfg["name"],
        "short_name":    mdl_cfg["short_name"],
        "n_layers":      n_layers,
        "hidden_dim":    hidden_dim,
        "n_videos":      N,
        "n_success":     n_success,
        "n_fail":        n_fail,
        "n_frames":      ext_cfg["n_frames"],
        "strategy":      ext_cfg["strategy"],
        "img_size":      ext_cfg["img_size"],
        "inference_dtype": ext_cfg["inference_dtype"],
        "cache_path":    str(cache_path),
    }
    with open(cache_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed_total = time.time() - t_start
    print(f"\nExtraction complete in {elapsed_total / 60:.1f} min")
    print(f"Feature cache: {cache_path}")
    print(f"Shape: {features.shape}  ({features.nbytes / 1e9:.2f} GB on disk)")
    print(f"Failed videos: {n_fail}")

    return cache_path, meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA2 features from TGIF-QA GIFs.")
    parser.add_argument("--config",   "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--manifest", "-m", default=None,           help="Path to manifest.parquet (overrides config)")
    parser.add_argument("--force",    "-f", action="store_true",    help="Re-extract even if cache exists")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    extract_features(config, manifest_path=args.manifest, force=args.force)


if __name__ == "__main__":
    main()
