import numpy as np
import scipy.stats

def extract_features(dct_blocks: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    """
    Extract statistical features (Kurtosis, Entropy, Variance) from AC coefficients.

    Parameters
    ----------
    dct_blocks : np.ndarray
        Array of shape (N, num_ac_coeffs) containing AC coefficients per block.
    clip_percentile : float
        Percentile for clipping extreme outliers before normalization.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N, 3) where columns are [kurtosis, entropy, variance].
    """
    if dct_blocks.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    num_blocks = dct_blocks.shape[0]

    # 1. Variance
    variance = np.var(dct_blocks, axis=1)

    # 2. Kurtosis
    valid_mask = variance > 1e-10
    kurtosis = np.zeros(num_blocks, dtype=np.float64)

    if np.any(valid_mask):
        kurt_vals = scipy.stats.kurtosis(
            dct_blocks[valid_mask], axis=1, fisher=True, bias=False
        )
        kurtosis[valid_mask] = np.nan_to_num(kurt_vals, nan=0.0)

    # 3. Entropy
    entropy = np.zeros(num_blocks, dtype=np.float64)

    for i in range(num_blocks):
        if not valid_mask[i]:
            continue

        counts, _ = np.histogram(dct_blocks[i], bins="auto")
        probs = counts / counts.sum()
        entropy[i] = scipy.stats.entropy(probs, base=2)
        if np.isnan(entropy[i]):
            entropy[i] = 0.0

    # Combine into N x 3 matrix
    feature_matrix = np.column_stack((kurtosis, entropy, variance))

    # Optional (Advanced): Clip extreme outliers
    if clip_percentile < 100.0:
        lower_bounds = np.percentile(feature_matrix, 100.0 - clip_percentile, axis=0)
        upper_bounds = np.percentile(feature_matrix, clip_percentile, axis=0)
        feature_matrix = np.clip(feature_matrix, lower_bounds, upper_bounds)

    return feature_matrix.astype(np.float32)
