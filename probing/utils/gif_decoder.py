"""
utils/gif_decoder.py

GIF frame sampling and preprocessing utilities.
Supports: uniform, shuffle, center_repeat, reverse sampling strategies.

Key design: frames are loaded ONE AT A TIME (not batch-stacked) to handle
GIFs where individual frames have different spatial dimensions — which is
valid per the GIF spec and common in TGIF-QA clips.
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


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize a single GIF frame to uint8 RGB [H, W, 3].
    Handles: grayscale [H,W], RGBA [H,W,4], standard RGB [H,W,3].
    """
    if frame.ndim == 2:                        # grayscale → RGB
        frame = np.stack([frame] * 3, axis=-1)
    if frame.ndim == 3 and frame.shape[-1] == 4:   # RGBA → RGB
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def _resize_frame_np(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize a single [H, W, C] uint8 numpy frame to (size, size) using PIL."""
    from PIL import Image
    img = Image.fromarray(frame)
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_gif_frames_raw(gif_path: str | Path) -> list[np.ndarray]:
    """
    Load all frames from a GIF as a LIST of [H_i, W_i, 3] uint8 arrays.

    Deliberately returns a list (not a stacked array) because GIF frames
    can have different spatial dimensions — stacking would raise:
        "all input arrays must have the same shape"

    Args:
        gif_path: Path to the .gif file.

    Returns:
        List of np.ndarray, each [H_i, W_i, 3] uint8.
        Note: H_i / W_i may vary across frames.
    """
    gif_path = str(gif_path)

    if _IMAGEIO_AVAILABLE:
        frames = []
        try:
            for frame in iio.imiter(gif_path, plugin="pillow"):
                frames.append(_normalize_frame(np.asarray(frame)))
        except Exception as e:
            raise RuntimeError(f"imageio failed to decode {gif_path}: {e}")
        if not frames:
            raise RuntimeError(f"No frames decoded from {gif_path}")
        return frames

    # Fallback: torchvision read_video
    try:
        import torchvision.io as tvio
        video, _, _ = tvio.read_video(gif_path, pts_unit="sec")  # [T, H, W, C]
        return [video[i].numpy().astype(np.uint8) for i in range(len(video))]
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

    Each selected frame is resized INDIVIDUALLY to img_size x img_size before
    stacking. This correctly handles variable-size GIF frames.

    Returns:
        Float32 tensor of shape [n_frames, C, H, W], normalized with ImageNet stats.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for load_gif(). Install torch.")

    import torch

    # Load as list — safe for variable-size frames
    frames_list = load_gif_frames_raw(gif_path)  # list of [H_i, W_i, 3]
    T = len(frames_list)

    # Handle short GIFs
    if T < min_frames:
        if not pad_short:
            raise ValueError(f"GIF has only {T} frames (min={min_frames}): {gif_path}")
        # Loop-pad by repeating from start
        while len(frames_list) < min_frames:
            frames_list = frames_list + frames_list
        frames_list = frames_list[:min_frames]
        T = len(frames_list)

    indices = sample_frame_indices(T, n_frames, strategy, seed)

    # Process each selected frame individually — handles varying H_i x W_i safely
    processed = []
    for idx in indices:
        frame = frames_list[int(idx)]              # [H_i, W_i, 3] uint8
        t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0  # [C, H, W]
        t = TF.resize(t.unsqueeze(0), [img_size, img_size], antialias=True).squeeze(0)
        processed.append(t)

    tensor = torch.stack(processed, dim=0)  # [n_frames, C, H, W]

    # ImageNet normalization
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t  = torch.tensor(std).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


def get_gif_frame_count(gif_path: str | Path) -> int:
                return props.n_images
    except Exception:
        pass

    # Fallback: full decode and count
    frames = load_gif_frames_raw(gif_path)
    return len(frames)
