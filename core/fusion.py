import numpy as np

def compute_score_map(
    normalized_features: np.ndarray, 
    grid_shape: tuple, 
    weights: tuple = (0.5, 0.3, 0.2)
) -> np.ndarray:
    """
    Combine normalized features into a single scalar score map per block.

    Parameters
    ----------
    normalized_features : np.ndarray
        Array of shape (N, 3) containing [kurtosis, entropy, variance].
    grid_shape : tuple
        The expected output grid shape (height_blocks, width_blocks).
        Note: N must equal height_blocks * width_blocks.
    weights : tuple, default=(0.5, 0.3, 0.2)
        Weights applied to kurtosis, entropy, and variance respectively.

    Returns
    -------
    np.ndarray
        A 2D array of shape `grid_shape` containing the fused score mapping.
    """
    h_blocks, w_blocks = grid_shape
    num_blocks = h_blocks * w_blocks
    
    if normalized_features.size == 0 or num_blocks == 0:
        return np.empty(grid_shape, dtype=np.float32)
        
    if normalized_features.shape[0] != num_blocks:
        raise ValueError(
            f"Feature count ({normalized_features.shape[0]}) does not match "
            f"grid size ({num_blocks} = {h_blocks}x{w_blocks})."
        )

    weights_array = np.array(weights, dtype=np.float32)
    
    # Vectorized dot product computation of the weighted combination
    # score_values = (0.5 * kurtosis) + (0.3 * entropy) + (0.2 * variance)
    score_values = np.dot(normalized_features, weights_array)
    
    # Reshape scores to maintain spatial 2D coherence with original block layout
    score_map = score_values.reshape(grid_shape)
    
    return score_map.astype(np.float32)
