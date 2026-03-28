"""
Blockwise DCT Analysis Module
=============================
Efficient extraction of AC coefficients from 8x8 DCT blocks.
"""

import cv2 # type: ignore
import numpy as np # type: ignore


def extract_ac_coefficients(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Transform spatial image to frequency domain using block-wise DCT
    and extract only the AC coefficients.

    Parameters
    ----------
    image : np.ndarray
        2D image array of type float32. Dimensions must be divisible by block_size.
    block_size : int, default=8
        Size of the DCT blocks (e.g., 8x8).

    Returns
    -------
    np.ndarray
        Array of shape (num_blocks, block_size**2 - 1) containing only 
        the AC coefficients for each block.
        
    Raises
    ------
    ValueError
        If the image is not 2D or dimensions are not divisible by block_size.
    """
    if image.ndim != 2:
        raise ValueError(f"Input image must be 2D, got {image.ndim}D.")
        
    h, w = image.shape
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"Image dimensions ({w}x{h}) must be strictly divisible by {block_size}."
        )

    # Efficiently partition image into blocks without nested spatial loops
    # Original: (H, W) -> Partitions: (H//8, 8, W//8, 8)
    # Swapped: (H//8, W//8, 8, 8) -> Reshaped: (num_blocks, 8, 8)
    blocks = (
        image.reshape(h // block_size, block_size, w // block_size, block_size)
        .swapaxes(1, 2)
        .reshape(-1, block_size, block_size)
    )

    # Compute 2D DCT per block. A list comprehension on the reshaped contiguous 
    # array is significantly faster than nested spatial for-loops.
    dct_blocks = np.array([cv2.dct(block) for block in blocks])

    # Flatten the (8, 8) blocks to (64,), then slice [:, 1:] to skip 
    # the DC component (index 0) across all blocks simultaneously.
    # Output shape will be (num_blocks, 63)
    ac_coeffs = dct_blocks.reshape(dct_blocks.shape[0], -1)[:, 1:]

    return ac_coeffs
