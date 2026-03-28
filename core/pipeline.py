"""
Main Pipeline Integration Module
================================
Ties together preprocessing, multi-scale pyramid generation, 
DCT processing, and statistical feature extraction.
"""

from modules.preprocessing import preprocess_image # type: ignore
from modules.multiscale import generate_gaussian_pyramid # type: ignore
from modules.dct import extract_ac_coefficients # type: ignore
from modules.features import extract_features # type: ignore
from modules.normalization import normalize_features # type: ignore
from modules.fusion import compute_score_map # type: ignore
from modules.stats import chi_square_test # type: ignore
from modules.decision import classify_image # type: ignore
from modules.visualization import generate_visuals # type: ignore
import cv2 # type: ignore

def analyze_image(image_path: str) -> dict:
    """
    Run the full image forgery detection pipeline on an image file.
    
    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image to analyze.
        
    Returns
    -------
    dict
        A structured dictionary containing the multi-scale feature 
        analysis and original metadata.
    """
    # 1. Preprocess
    preprocessed_img, metadata = preprocess_image(image_path)
    
    # 2. Multi-scale Pyramids (4 levels exactly)
    pyramids = generate_gaussian_pyramid(preprocessed_img, levels=4)
    
    # 3. For each scale: DCT & Statistical Features
    multi_scale_features = []
    
    for level_idx, level_img in enumerate(pyramids):
        # Generate AC coefficients (8x8 blocks)
        ac_coeffs = extract_ac_coefficients(level_img, block_size=8)
        
        # Extract raw features (N x 3 matrix: kurtosis, entropy, variance)
        raw_features = extract_features(ac_coeffs)
        
        # Sequentially apply Normalization
        norm_features, stats_dict = normalize_features(raw_features)
        
        # Compute spatial grid context
        h, w = level_img.shape
        grid_shape = (h // 8, w // 8)
        
        # Apply fusion
        score_map = compute_score_map(norm_features, grid_shape)
        
        multi_scale_features.append({
            "level": level_idx,
            "resolution": level_img.shape,
            "grid_shape": grid_shape,
            "features": norm_features,
            "normalization_stats": stats_dict,
            "score_map": score_map
        })
        
    # Statistical Testing Stage
    score_maps = [level["score_map"] for level in multi_scale_features]
    chi_square_value, p_value = chi_square_test(score_maps)
    
    # Final Decision Stage
    # Pass the finest resolution score_map (level 0) for context
    decision_dict = classify_image(p_value, score_maps[0])
    
    # Visualization Stage
    original_image = cv2.imread(image_path)
    heatmap, mask = generate_visuals(score_maps[0], original_image, metadata)
        
    return {
        "metadata": metadata,
        "multi_scale_analysis": multi_scale_features,
        "statistical_test": {
            "chi_square_value": chi_square_value,
            "p_value": p_value
        },
        "decision": decision_dict,
        "visuals": {
            "heatmap": heatmap,
            "mask": mask
        }
    }
