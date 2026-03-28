import numpy as np

def classify_image(p_value: float, score_map: np.ndarray) -> dict:
    """
    Classify the image as 'FAKE' or 'AUTHENTIC' based on the p-value.
    
    Parameters
    ----------
    p_value : float
        P-value resulting from the chi-square test across pyramid levels.
    score_map : np.ndarray
        The 2D spatial score map (typically from the primary pyramid level).
        Passed for context or potential future spatial heuristic logic.
        
    Returns
    -------
    dict
        Dictionary containing 'classification' (string) and 'confidence' (float).
    """
    classification = "FAKE" if p_value < 0.01 else "AUTHENTIC"
    confidence = 1.0 - p_value
    
    return {
        "classification": classification,
        "confidence": float(confidence)
    }
