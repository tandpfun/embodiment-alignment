"""Local web UI for the deployed OpenVLA Modal worker.

Run:

    uv sync                       # one-time
    modal deploy inference_app.py # one-time, brings up the GPU worker
    uv run python local_site.py   # serves http://127.0.0.1:8000

The site is a thin FastAPI app that lives entirely on your laptop. When you
click "Predict", it base64-decodes the uploaded image, calls the OpenVLA
Modal class via the Modal SDK, and renders the returned action vector.

No public URL is exposed — the only thing that talks to Modal is your local
process, authenticated by your local ~/.modal.toml.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Must match `APP_NAME` and the class name in inference_app.py.
APP_NAME = "vla-inference"
OPENVLA_CLASS = "OpenVLAWorker"

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


# -----------------------------------------------------------------------------
# Modal class handle
#
# We resolve and cache it lazily so the server can boot before the user has
# deployed the Modal app (they'll just get a clear error when they click
# Predict). Instantiation is cheap; the GPU container is only spun up on the
# first .remote() call.
# -----------------------------------------------------------------------------

_worker: Any = None


def get_worker() -> Any:
    global _worker
    if _worker is None:
        Cls = modal.Cls.from_name(APP_NAME, OPENVLA_CLASS)
        _worker = Cls()
    return _worker


# -----------------------------------------------------------------------------
# Request / response schemas
# -----------------------------------------------------------------------------


class PredictRequest(BaseModel):
    image_b64: str = Field(
        ...,
        description="Base64-encoded image bytes (any common format: JPEG, PNG, ...).",
    )
    task: str = Field(..., description="Natural-language instruction.")
    unnorm_key: str = Field(
        default="bridge_orig",
        description="OpenVLA un-normalisation key (per-OXE-subset stats).",
    )


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

api = FastAPI(title="OpenVLA Playground", docs_url="/docs", redoc_url=None)

api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@api.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(INDEX_HTML))


@api.get("/api/info")
async def info() -> dict[str, Any]:
    """Tells the UI which Modal app it talks to. Handy for sanity checks."""
    return {
        "app_name": APP_NAME,
        "openvla_class": OPENVLA_CLASS,
    }


@api.post("/api/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        image_bytes = base64.b64decode(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"Bad image_b64: {e}") from e

    worker = get_worker()
    t0 = time.perf_counter()
    try:
        out = await worker.predict.remote.aio(
            image_bytes=image_bytes,
            task=req.task,
            unnorm_key=req.unnorm_key,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {"ok": True, "elapsed_ms": elapsed_ms, **out}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "ok": False,
            "elapsed_ms": elapsed_ms,
            "error": f"{type(e).__name__}: {e}",
        }


# -----------------------------------------------------------------------------
# Entrypoint: `python local_site.py` (no reload) or `uvicorn local_site:api`.
# -----------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    uvicorn.run("local_site:api", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
