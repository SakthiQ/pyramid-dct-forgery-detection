import sys
from pathlib import Path

# Add project root to path for absolute module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
from fastapi import FastAPI # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from app.routes.analyze import router as analyze_router # type: ignore
import uvicorn # type: ignore

app = FastAPI(
    title="Image Forgery Detection API",
    description="API for detecting image manipulation using classical DIP methods.",
    version="1.0.0"
)

# 1. Enable secure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Include the routing logic (this integrates both POST /analyze and GET /health natively)
app.include_router(analyze_router)

# 3. Mount Static React SPA Array
os.makedirs("outputs", exist_ok=True)
os.makedirs("frontend/dist", exist_ok=True)

# Ensure our newly generated masks/heatmaps are available to the DOM natively
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Instruct FastAPI to serve our new frontend strictly on the root / path mapping natively across Vite artifacts
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
