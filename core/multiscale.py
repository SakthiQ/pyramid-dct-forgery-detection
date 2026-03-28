"""
Multi-Scale Pyramid Module
==========================
Generates multi-scale representations of preprocessed images
using Gaussian pyramids according to project constraints.
"""

import cv2
import numpy as np


def generate_gaussian_pyramid(image: np.ndarray, levels: int = 4) -> list[np.ndarray]:
    """
    Generate a Gaussian pyramid with exactly `levels` resolutions.
    
    The first level (index 0) is the original image. Each subsequent level
    is downsampled exactly by a factor of 2 using cv2.pyrDown().
    
    Parameters
    ----------
    image : np.ndarray
        Preprocessed image (e.g. 512x512, float32, range [0, 1]).
    levels : int, default=4
        Total number of levels in the pyramid (including original).
        
    Returns
    -------
    list[np.ndarray]
        List of length `levels`, containing the pyramid starting from
        the highest resolution to the lowest.
        
    Raises
    ------
    ValueError
        If the image is not 2-dimensional or dimensions are not sufficient
        for the requested number of levels.
    """
    if image.ndim != 2:
        raise ValueError(f"Input image must be 2D grayscale, got {image.ndim}D")
        
    h, w = image.shape
    # To create N levels, the dimensions must be divisible by 2^(N-1)
    if h % (2 ** (levels - 1)) != 0 or w % (2 ** (levels - 1)) != 0:
        raise ValueError(
            f"Image dimensions ({w}x{h}) are not divisible by 2^{levels-1} "
            f"required for {levels} pyramid levels."
        )

    pyramid = [image.copy()]
    
    current_img = image
    for _ in range(1, levels):
        # cv2.pyrDown performs Gaussian blurring and then downsampling by a factor of 2
        current_img = cv2.pyrDown(current_img)
        pyramid.append(current_img)
        
    return pyramid
