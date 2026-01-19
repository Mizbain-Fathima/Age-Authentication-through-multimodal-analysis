"""
FastAPI Application for Age Authentication System
Main entry point for the REST API
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import API_HOST, API_PORT, MODELS_DIR
from src.api.routes import router
from src.api.services import AgeAuthenticationService


# Global service instance
auth_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global auth_service
    
    # Startup
    logger.info("Starting Age Authentication API...")
    
    # Initialize authentication service
    auth_service = AgeAuthenticationService()
    app.state.auth_service = auth_service
    
    logger.info("Age Authentication API ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Age Authentication API...")


# Create FastAPI app
app = FastAPI(
    title="Age Authentication System",
    description="""
    Multimodal Age Authentication API
    
    This API provides age verification using:
    - **Face Analysis**: CNN-based age prediction from face images
    - **Voice Analysis**: Age prediction from audio features (MFCC, Mel Spectrograms)
    - **Liveness Detection**: 
        - Face liveness (blink detection, motion analysis, texture analysis)
        - Voice liveness (captcha verification via speech-to-text)
        - Lip-sync verification
    - **Multimodal Fusion**: Combined face + voice age prediction
    
    ## Workflow
    1. Request a captcha sentence (`/api/captcha`)
    2. Record video + audio of user reading the captcha
    3. Submit for verification (`/api/verify`)
    4. Get age estimation and verification results
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve static files (frontend)
frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
async def root():
    """Serve the frontend application"""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Age Authentication API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": auth_service is not None
    }


def run_server():
    """Run the API server"""
    import uvicorn
    
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()

