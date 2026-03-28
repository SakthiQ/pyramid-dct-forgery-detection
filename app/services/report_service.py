import os
import json
import uuid
from datetime import datetime

def generate_report(classification: str, confidence: float, p_value: float, output_dir: str = "outputs") -> dict:
    """
    Generate and save formatted analysis reports and statistical JSON descriptors.

    Parameters
    ----------
    classification : str
        The final diagnostic boundary ('FAKE' or 'AUTHENTIC').
    confidence : float
        Scalar threshold capturing confidence probability.
    p_value : float
        Statistical Chi-Square probability metrics mapping deviation severity.
    output_dir : str
        Target root output directory for saved artifacts.
        
    Returns
    -------
    dict
        Paths to the generated report and stats files:
        {"report_path": ..., "stats_path": ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique ID for this report instance avoiding string slice syntax
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{str(uuid.uuid4()).split('-')[0]}"
    
    report_path = os.path.join(output_dir, f"report_{run_id}.txt")
    stats_path = os.path.join(output_dir, f"stats_{run_id}.json")
    
    # 1. Text Report Generation
    explanation = (
        "Analysis confirms spatial structural anomalies consistent with block manipulations."
        if classification == "FAKE" else
        "No significant deviations detected. Global scaling metrics are consistent with authentic captures."
    )
    
    report_text = (
        f"--- Image Forgery Detection Report ---\n"
        f"Run ID: {run_id}\n"
        f"Timestamp: {datetime.now().isoformat()}\n"
        f"Classification: {classification}\n"
        f"Confidence: {confidence:.4f}\n"
        f"P-Value: {p_value:.6e}\n\n"
        f"Conclusion: {explanation}\n"
        f"--------------------------------------\n"
    )
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    # 2. JSON Stats Generation
    stats_data = {
        "run_id": run_id,
        "classification": classification,
        "confidence": float(confidence),
        "chi_square_p_value": float(p_value)
    }
    
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=4)
        
    return {
        "report_path": report_path,
        "stats_path": stats_path
    }
