import logging
import logging.config
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routes import router


# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Interview Trainer Agent",
    description=(
        "An AI-powered interview training system "
        "using IBM watsonx foundation models."
    ),
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────
# Allow the UI (served from the same origin) and common local dev ports.
# In production, replace with the actual deployed domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── API routes ─────────────────────────────────────────────────────────
app.include_router(router)

# ── Static files (frontend) ────────────────────────────────────────────
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
    _logger.info("Frontend static files mounted from %s", _static_dir)
else:
    _logger.warning("static/ directory not found — UI will not be served.")


# ── Routes ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def ui():
    """Serve the single-page interview UI."""
    index = _static_dir / "index.html"
    if index.is_file():
        return FileResponse(str(index), media_type="text/html")
    return {
        "name": "AI Interview Trainer Agent",
        "status": "running",
        "version": "1.0.0",
        "ui": "static/index.html not found — run from project root.",
    }


@app.get("/api", include_in_schema=False)
def api_root():
    """JSON status endpoint for health checks / API clients."""
    return {
        "name": "AI Interview Trainer Agent",
        "status": "running",
        "version": "1.0.0",
    }
