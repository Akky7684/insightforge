"""FastAPI application entrypoint for InsightForge."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router as api_router

app = FastAPI(
    title="InsightForge API",
    description="Autonomous Multi-Agent Data Analyst Backend",
    version="1.0.0",
)

# CORS middleware for local and web UI communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint for container and load-balancer probes."""
    return {"status": "healthy", "service": "insightforge-backend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
