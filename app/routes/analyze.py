from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.services.pipeline import run_pipeline
import sys
from pathlib import Path

# Add project root to sys.path to strictly bypass Pyre sub-folder missing-local-module errors 
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

router = APIRouter(prefix="/api/v1")

@router.get("/health")
async def health_check():
    """
    Basic application viability check.
    """
    return {"status": "ok", "message": "Forgery Detection API is completely active."}

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Primary endpoint parsing uploaded images and routing straight into the 11-step analysis Orchestrator.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid media. File must be a standard image type.")
    
    try:
        # Load absolute byte stream natively from the HTTP structure
        image_bytes = await file.read()
        
        # Route execution bounds synchronously back downward into our standalone local runner
        result = run_pipeline(image_bytes)
        
        # Safely wrap raw dictionary constraints out via JSON HTTP Response
        return JSONResponse(content=result)
        
    except Exception as e:
        # Intercept core analysis failures surfacing diagnostic issues smoothly
        raise HTTPException(status_code=500, detail=f"Pipeline Execution Failed: {str(e)}")
