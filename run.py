from fastapi import FastAPI
from app.routes.analyze import router as analyze_router
import uvicorn

app = FastAPI(
    title="Antigravity Image Forgery System",
    description="Deterministic statistical diagnostic API securely extracting mapped structural authenticity metrics.",
    version="1.0.0"
)

# Bind the extracted /api/v1 execution paths
app.include_router(analyze_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
