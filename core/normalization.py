import numpy as np

def normalize_features(feature_vectors: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Normalize feature vectors using Z-score normalization feature-wise.

    Parameters
    ----------
    feature_vectors : np.ndarray
        Array of shape (N, 3) containing [kurtosis, entropy, variance].

    Returns
    -------
    tuple[np.ndarray, dict]
        - normalized_features: Array of shape (N, 3) with normalized values.
        - stats_dict: Dictionary containing the mean and std for each feature column.
    """
    if feature_vectors.size == 0:
        return np.empty((0, 3), dtype=np.float32), {}

    # Compute mean and standard deviation feature-wise (column-wise)
    mean_vals = np.mean(feature_vectors, axis=0)
    std_vals = np.std(feature_vectors, axis=0)

    # Z-score normalization with epsilon (1e-6) to prevent division by zero
    epsilon = 1e-6
    normalized_features = (feature_vectors - mean_vals) / (std_vals + epsilon)

    stats_dict = {
        "mean_kurtosis": float(mean_vals[0]),
        "mean_entropy": float(mean_vals[1]),
        "mean_variance": float(mean_vals[2]),
        "std_kurtosis": float(std_vals[0]),
        "std_entropy": float(std_vals[1]),
        "std_variance": float(std_vals[2]),
    }

    return normalized_features.astype(np.float32), stats_dict
