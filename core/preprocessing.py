"""
Image Preprocessing Module
==========================
Deterministic preprocessing pipeline for image forgery detection.

Constraints:
    - Target resolution: 512 x 512
    - Resize: aspect-ratio preserved + letterbox padding (black)
    - Final dimensions guaranteed multiple of 8 (for 8x8 DCT blocks)
    - Output: np.ndarray, shape (512, 512), dtype float32, range [0, 1]
    - Minimum accepted input: 32 x 32 pixels
"""

import os
import cv2
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

TARGET_SIZE = 512          # Target width and height
BLOCK_SIZE = 8             # DCT block size — dimensions must be divisible by this
MIN_INPUT_SIZE = 32        # Reject images smaller than this in either dimension
PAD_VALUE = 0              # Black padding fill


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        Raw image as loaded by OpenCV (BGR or grayscale).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file cannot be decoded as a valid image.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Invalid or corrupted image: {path}")

    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert any input image (BGR, BGRA, or already grayscale) to single-channel grayscale.

    Parameters
    ----------
    image : np.ndarray
        Input image with 1, 3, or 4 channels.

    Returns
    -------
    np.ndarray
        Single-channel grayscale image, dtype uint8.
    """
    if image.ndim == 2:
        # Already grayscale
        return image

    channels = image.shape[2]

    if channels == 4:
        # BGRA → BGR (drop alpha), then to gray
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif channels == 3:
        # BGR → grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected number of channels: {channels}")


def _letterbox_resize(
    image: np.ndarray,
    target_size: int = TARGET_SIZE,
) -> tuple[np.ndarray, float]:
    """
    Resize image preserving aspect ratio and pad to target_size x target_size.

    The image is scaled so its longest side fits within `target_size`, then
    padded symmetrically with black (0) to fill the remaining space.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), dtype uint8.
    target_size : int
        Desired output dimension.

    Returns
    -------
    tuple[np.ndarray, float]
        - Letterboxed image (target_size x target_size), dtype uint8.
        - Scaling factor applied.
    """
    h, w = image.shape[:2]

    # Compute scale so the longest side fits target_size
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with area interpolation (best for downscaling)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create black canvas and center the resized image
    canvas = np.full((target_size, target_size), PAD_VALUE, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas, scale


def _normalize_intensity(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] uint8 to [0, 1] float32.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, dtype uint8.

    Returns
    -------
    np.ndarray
        Normalized image, dtype float32, range [0, 1].
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(path: str) -> tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline: load → validate → grayscale → letterbox → normalize.

    Parameters
    ----------
    path : str
        Path to the input image file.

    Returns
    -------
    tuple[np.ndarray, dict]
        - Preprocessed image: shape (512, 512), dtype float32, range [0, 1].
        - Metadata dict:
            - original_size: (width, height) of the raw input
            - padded_size: (width, height) after letterbox (always 512, 512)
            - scaling_factor: float ratio applied during resize

    Raises
    ------
    FileNotFoundError
        If image path does not exist.
    ValueError
        If image is corrupted or smaller than 32×32.
    """
    # 1. Load raw image
    raw = load_image(path)

    # 2. Validate minimum size
    h, w = raw.shape[:2]
    if h < MIN_INPUT_SIZE or w < MIN_INPUT_SIZE:
        raise ValueError(
            f"Image too small: {w}x{h}. Minimum accepted size is "
            f"{MIN_INPUT_SIZE}x{MIN_INPUT_SIZE}."
        )

    # 3. Convert to grayscale
    gray = _to_grayscale(raw)

    # 4. Letterbox resize with aspect ratio preservation
    letterboxed, scale = _letterbox_resize(gray, TARGET_SIZE)

    # 5. Normalize intensity to [0, 1] float32
    normalized = _normalize_intensity(letterboxed)

    # 6. Sanity checks (deterministic guarantees)
    assert normalized.shape == (TARGET_SIZE, TARGET_SIZE), \
        f"Output shape {normalized.shape} != expected ({TARGET_SIZE}, {TARGET_SIZE})"
    assert normalized.dtype == np.float32
    assert TARGET_SIZE % BLOCK_SIZE == 0, \
        f"Target size {TARGET_SIZE} is not divisible by block size {BLOCK_SIZE}"

    # 7. Build metadata
    metadata = {
        "original_size": (w, h),
        "padded_size": (TARGET_SIZE, TARGET_SIZE),
        "scaling_factor": round(float(scale), 6),
    }

    return normalized, metadata
