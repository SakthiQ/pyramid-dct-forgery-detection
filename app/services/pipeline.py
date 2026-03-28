import os
import cv2 # type: ignore
import json
import uuid
import numpy as np # type: ignore
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import existing modules from the actual functional core
from modules.preprocessing import preprocess_image # type: ignore
from modules.multiscale import generate_gaussian_pyramid # type: ignore
from modules.dct import extract_ac_coefficients # type: ignore
from modules.features import extract_features # type: ignore
from modules.normalization import normalize_features # type: ignore
from modules.fusion import compute_score_map # type: ignore
from modules.stats import chi_square_test # type: ignore
from modules.decision import classify_image # type: ignore
from modules.visualization import generate_visuals # type: ignore
from app.services.report_service import generate_report # type: ignore



def run_pipeline(image) -> dict:
    """
    Run the complete 11-step image forgery detection pipeline in strict order.
    
    Parameters
    ----------
    image : str, bytes, or np.ndarray
        Input image data for analysis.
        
    Returns
    -------
    dict
        Dictionary containing classification outcomes and paths to saved outputs.
    """
    # 0. Output Structure Generation
    os.makedirs("outputs", exist_ok=True)
    # Avoid string slicing to bypass Pyre's __getitem__ slice parsing bug
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4()).split("-")[0]
    
    # Safely convert parameter 'image' into a numpy format and store a local copy if required
    if isinstance(image, bytes):
        nparr = np.frombuffer(image, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_img is None:
            raise ValueError("Could not decode image bytes.")
        img_input = f"outputs/temp_{run_id}.png"
        cv2.imwrite(img_input, original_img)
    elif isinstance(image, np.ndarray):
        original_img = image.copy()
        img_input = f"outputs/temp_{run_id}.png"
        cv2.imwrite(img_input, original_img)
    else:
        # Assume it is a valid str path
        original_img = cv2.imread(image)
        if original_img is None:
            raise ValueError(f"Could not load image from path: {image}")
        img_input = image

    # 1. Preprocess image
    preprocessed_img, metadata = preprocess_image(img_input)
    
    # Clean up temp file immediately since preprocessing has safely read it
    if isinstance(image, (bytes, np.ndarray)) and os.path.exists(img_input):
        os.remove(img_input)

    # 2. Generate pyramid levels
    pyramid_levels = generate_gaussian_pyramid(preprocessed_img, levels=4)

    # 3. For each pyramid level: Apply DCT, Extract features
    all_raw_features = []
    level_grid_shapes = []
    
    for level_img in pyramid_levels:
        ac_coeffs = extract_ac_coefficients(level_img)
        raw_features = extract_features(ac_coeffs) # N x 3 feature matrices
        all_raw_features.append(raw_features)
        
        h, w = level_img.shape
        level_grid_shapes.append((h // 8, w // 8))

    # 4. Combine all features
    combined_raw_features = np.vstack(all_raw_features)

    # 5. Normalize features globally across all collected pyramid dimensions
    combined_norm_features, _ = normalize_features(combined_raw_features)

    # 6. Compute score map (feature fusion)
    score_maps = []
    start_idx = 0
    for grid_shape in level_grid_shapes:
        num_blocks = grid_shape[0] * grid_shape[1]
        level_norm_features = combined_norm_features[start_idx : start_idx + num_blocks]
        start_idx += num_blocks
        
        score_map = compute_score_map(level_norm_features, grid_shape)
        score_maps.append(score_map)

    # 7. Perform chi-square test
    _, p_value = chi_square_test(score_maps)

    # 8. Make classification decision (passing standard pyramid size 0)
    decision = classify_image(p_value, score_maps[0])
    classification = decision.get("classification", "UNKNOWN")
    confidence = decision.get("confidence", 0.0)

    # 9. Generate heatmap + mask using the highest fidelity map
    heatmap, mask = generate_visuals(score_maps[0], original_img, metadata)

    # 10. Generate report (Call report service orchestrator)
    report_outputs = generate_report(classification, confidence, p_value, output_dir="outputs")
    report_path = report_outputs["report_path"]

    # 11. Save visual outputs independently
    heatmap_path = f"outputs/heatmap_{run_id}.png"
    mask_path = f"outputs/mask_{run_id}.png"
    
    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(mask_path, mask)

    # Enforce exact return dictionary fields mapping requirements
    return {
        "classification": classification,
        "confidence": confidence,
        "p_value": p_value,
        "heatmap_path": heatmap_path,
        "mask_path": mask_path,
        "report_path": report_path
    }
