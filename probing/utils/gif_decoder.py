"""
utils/gif_decoder.py

GIF frame sampling and preprocessing utilities.
Supports: uniform, shuffle, center_repeat, reverse sampling strategies.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Literal

# Optional imports — prefer imageio for GIF support
try:
    import imageio.v3 as iio
    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False

try:
    import torch
    import torchvision.transforms.functional as TF
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


SamplingStrategy = Literal["uniform", "shuffle", "center_repeat", "reverse"]


def load_gif_frames_raw(gif_path: str | Path) -> np.ndarray:
    """
    Load all frames from a GIF as a uint8 numpy array of shape [T, H, W, C].
    Falls back gracefully if imageio is not available.
    """
    gif_path = str(gif_path)

    if _IMAGEIO_AVAILABLE:
        frames = iio.imread(gif_path, index=None)  # [T, H, W, C] or [T, H, W]
        if frames.ndim == 3:  # grayscale
            frames = np.stack([frames] * 3, axis=-1)
        # Some GIFs have RGBA — drop alpha channel
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        return frames.astype(np.uint8)

    # Fallback: torchvision read_video (slower for GIFs but widely available)
    try:
        import torchvision.io as tvio
        video, _, _ = tvio.read_video(gif_path, pts_unit="sec")  # [T, H, W, C]
        return video.numpy().astype(np.uint8)
    except Exception as e:
        raise RuntimeError(
            f"Cannot decode {gif_path}. Install imageio[ffmpeg] or torchvision. Error: {e}"
        )


def sample_frame_indices(
    total_frames: int,
    n_frames: int,
    strategy: SamplingStrategy,
    seed: int | None = None,
) -> np.ndarray:
    """
    Return an array of frame indices of length n_frames, given a sampling strategy.

    Args:
        total_frames: Number of frames in the GIF.
        n_frames: Number of frames to select.
        strategy: One of 'uniform', 'shuffle', 'center_repeat', 'reverse'.
        seed: Random seed (only used for 'shuffle').

    Returns:
        np.ndarray of shape [n_frames] with integer frame indices.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be > 0, got {total_frames}")

    if strategy == "uniform":
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    elif strategy == "reverse":
        indices = np.linspace(total_frames - 1, 0, n_frames, dtype=int)

    elif strategy == "shuffle":
        rng = np.random.default_rng(seed)
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        rng.shuffle(indices)

    elif strategy == "center_repeat":
        center = total_frames // 2
        indices = np.full(n_frames, center, dtype=int)

    else:
        raise ValueError(
            f"Unknown sampling strategy: '{strategy}'. "
            "Choose from: uniform, reverse, shuffle, center_repeat"
        )

    return indices


def pad_frames_by_looping(frames: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad a frame array by looping (repeating from the start) to reach target_length.

    Args:
        frames: [T, H, W, C] uint8 array.
        target_length: Desired number of frames.

    Returns:
        [target_length, H, W, C] uint8 array.
    """
    T = len(frames)
    if T >= target_length:
        return frames
    repeats = (target_length // T) + 1
    tiled = np.tile(frames, (repeats, 1, 1, 1))
    return tiled[:target_length]


def load_gif(
    gif_path: str | Path,
    n_frames: int = 16,
    strategy: SamplingStrategy = "uniform",
    img_size: int = 224,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    min_frames: int = 8,
    pad_short: bool = True,
    seed: int | None = None,
) -> "torch.Tensor":
    """
    Full GIF loading pipeline: decode → sample → resize → normalize.

    Args:
        gif_path: Path to the .gif file.
        n_frames: Number of frames to sample.
        strategy: Sampling strategy.
        img_size: Spatial resize resolution (square).
        mean: Normalization mean (per-channel).
        std: Normalization std (per-channel).
        min_frames: Skip/raise if GIF has fewer than this many frames.
        pad_short: If True, loop-pad short GIFs. If False, raise on short GIFs.
        seed: Random seed for shuffle strategy.

    Returns:
        Float32 tensor of shape [T, C, H, W], ready for model input.
        Values are in roughly [-2.3, 2.6] (normalized).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for load_gif(). Install torch.")

    frames_raw = load_gif_frames_raw(gif_path)  # [T, H, W, C], uint8

    T = len(frames_raw)
    if T < min_frames:
        if not pad_short:
            raise ValueError(
                f"GIF has only {T} frames (min={min_frames}): {gif_path}"
            )
        frames_raw = pad_frames_by_looping(frames_raw, min_frames)
        T = len(frames_raw)

    indices = sample_frame_indices(T, n_frames, strategy, seed)
    sampled = frames_raw[indices]  # [n_frames, H, W, C], uint8

    # Convert to float tensor [n_frames, C, H, W] in [0, 1]
    import torch
    tensor = torch.from_numpy(sampled).permute(0, 3, 1, 2).float() / 255.0

    # Resize each frame
    resized = TF.resize(tensor, [img_size, img_size], antialias=True)

    # Normalize
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t  = torch.tensor(std).view(1, 3, 1, 1)
    normalized = (resized - mean_t) / std_t

    return normalized  # [T, C, H, W]


def get_gif_frame_count(gif_path: str | Path) -> int:
    """
    Returns the number of frames in a GIF without loading full pixel data.
    Falls back to full decode if metadata is not available.
    """
    try:
        if _IMAGEIO_AVAILABLE:
            props = iio.improps(str(gif_path))
            if props.n_images is not None and props.n_images > 0:
                return props.n_images
    except Exception:
        pass

    # Fallback: full decode and count
    frames = load_gif_frames_raw(gif_path)
    return len(frames)
