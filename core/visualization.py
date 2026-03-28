import cv2
import numpy as np

def generate_visuals(score_map: np.ndarray, original_image: np.ndarray, metadata: dict) -> tuple:
    """
    Generate normalized heatmap overlay and binary threshold anomaly mask.

    Parameters
    ----------
    score_map : np.ndarray
        Raw score map representing spatial anomaly evaluations.
    original_image : np.ndarray
        The original input BGR image array.
    metadata : dict
        Metadata from preprocessing containing 'original_size', parsed as (width, height).

    Returns
    -------
    tuple
        (heatmap_overlay, binary_mask) as numpy structured arrays correctly sized.
    """
    # 1. Normalize score_map to [0, 255]
    score_min = score_map.min()
    score_max = score_map.max()
    
    if score_max - score_min < 1e-8:
        normalized_score = np.zeros_like(score_map, dtype=np.uint8)
    else:
        normalized_score = 255.0 * (score_map - score_min) / (score_max - score_min)
        
    # 2. Convert to uint8
    normalized_score = np.clip(normalized_score, 0, 255).astype(np.uint8)

    # 3. Apply color map (cv2.COLORMAP_JET)
    heatmap_colored = cv2.applyColorMap(normalized_score, cv2.COLORMAP_JET)
    
    # 4. Resize to original image size
    target_size = metadata.get("original_size")
    heatmap_resized = cv2.resize(heatmap_colored, target_size, interpolation=cv2.INTER_CUBIC)
    
    # 5. Overlay heatmap on original image
    # Safely assert dims (ensure original_image matches BGR size)
    if original_image.shape[:2] != (target_size[1], target_size[0]):
        original_image = cv2.resize(original_image, target_size)
        
    overlay = cv2.addWeighted(original_image, 0.6, heatmap_resized, 0.4, 0)
    
    # 6. Apply Gaussian blur to the blended layout
    heatmap_blurred = cv2.GaussianBlur(overlay, (15, 15), 0)
    
    # 7. Generate binary mask based on 85th percentile
    score_map_resized = cv2.resize(score_map, target_size, interpolation=cv2.INTER_CUBIC)
    threshold = np.percentile(score_map_resized, 85)
    
    binary_mask = (score_map_resized > threshold).astype(np.uint8) * 255
    
    return heatmap_blurred, binary_mask
