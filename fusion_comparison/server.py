"""
Run the API server with fusion comparison enabled.

Usage (from project root):
  python fusion_comparison/server.py

Then open:
  - Normal app:     http://localhost:8000
  - Compare page:   http://localhost:8000/compare

No changes to existing code; this only adds the fusion route and /compare page.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app
from src.config import API_HOST, API_PORT

# Add fusion comparison route and /compare page
from fusion_comparison.api_route import router as fusion_router
app.include_router(fusion_router, prefix="")


if __name__ == "__main__":
    import uvicorn
    from loguru import logger
    logger.info(f"Starting server with fusion comparison at http://{API_HOST}:{API_PORT}")
    logger.info("Normal app: http://localhost:8000  |  Compare: http://localhost:8000/compare")
    uvicorn.run(
        "fusion_comparison.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
