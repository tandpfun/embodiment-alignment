"""Local web UI for the deployed Modal VLA workers.

Run:

    uv sync                       # one-time
    modal deploy inference_app.py # one-time, brings up the GPU workers
    uv run python local_site.py   # serves http://127.0.0.1:8000

The site is a thin FastAPI app that lives entirely on your laptop. When you
click "Predict", it base64-decodes the uploaded image, calls one or both
Modal classes via the Modal SDK, and renders the returned action vectors.

No public URL is exposed — the only thing that talks to Modal is your local
process, authenticated by your local ~/.modal.toml.
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Must match `APP_NAME` and the class names in inference_app.py.
APP_NAME = "vla-inference"
SMOLVLA_CLASS = "SmolVLAWorker"
OPENVLA_CLASS = "OpenVLAWorker"

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


# -----------------------------------------------------------------------------
# Modal class handles
#
# We resolve and cache them lazily so the server can boot before the user has
# deployed the Modal app (you'll just get a clear error when you click Predict).
# Modal.Cls.from_name itself is cheap; instantiation is also cheap (it doesn't
# spin up a container — that happens on the first .remote() call).
# -----------------------------------------------------------------------------


def _instance(class_name: str):
    Cls = modal.Cls.from_name(APP_NAME, class_name)
    return Cls()


_workers: dict[str, Any] = {}


def get_worker(name: str):
    name = name.lower()
    if name in _workers:
        return _workers[name]
    if name == "smolvla":
        _workers[name] = _instance(SMOLVLA_CLASS)
    elif name == "openvla":
        _workers[name] = _instance(OPENVLA_CLASS)
    else:
        raise HTTPException(400, f"Unknown model: {name!r}")
    return _workers[name]


# -----------------------------------------------------------------------------
# Request / response schemas
# -----------------------------------------------------------------------------


class PredictRequest(BaseModel):
    image_b64: str = Field(
        ...,
        description="Base64-encoded image bytes (any common format: JPEG, PNG, ...).",
    )
    task: str = Field(..., description="Natural-language instruction.")
    models: list[str] = Field(
        default_factory=lambda: ["smolvla", "openvla"],
        description='Subset of {"smolvla", "openvla"}.',
    )
    state: list[float] | None = Field(
        default=None,
        description="Optional 6-dim proprio state for SmolVLA. Zeros if omitted.",
    )
    unnorm_key: str = Field(
        default="bridge_orig",
        description="OpenVLA un-normalisation key (per-OXE-subset stats).",
    )


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

api = FastAPI(title="VLA Playground", docs_url="/docs", redoc_url=None)

api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@api.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(INDEX_HTML))


@api.get("/api/info")
async def info() -> dict[str, Any]:
    """Tells the UI which Modal app it talks to. Handy for sanity checks."""
    return {
        "app_name": APP_NAME,
        "smolvla_class": SMOLVLA_CLASS,
        "openvla_class": OPENVLA_CLASS,
    }


@api.post("/api/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        image_bytes = base64.b64decode(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"Bad image_b64: {e}") from e

    models = [m.lower().strip() for m in req.models if m.strip()]
    unknown = set(models) - {"smolvla", "openvla"}
    if unknown:
        raise HTTPException(400, f"Unknown models: {sorted(unknown)}")
    if not models:
        raise HTTPException(400, "Pick at least one model.")

    # Fire both workers in parallel via Modal's async client.
    coros = {}
    if "smolvla" in models:
        worker = get_worker("smolvla")
        coros["smolvla"] = worker.predict.remote.aio(
            image_bytes=image_bytes,
            task=req.task,
            state=req.state,
        )
    if "openvla" in models:
        worker = get_worker("openvla")
        coros["openvla"] = worker.predict.remote.aio(
            image_bytes=image_bytes,
            task=req.task,
            unnorm_key=req.unnorm_key,
        )

    async def timed(key: str, coro):
        t0 = time.perf_counter()
        try:
            out = await coro
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return key, {"ok": True, "elapsed_ms": elapsed_ms, **out}
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return key, {
                "ok": False,
                "elapsed_ms": elapsed_ms,
                "error": f"{type(e).__name__}: {e}",
            }

    pairs = await asyncio.gather(*(timed(k, c) for k, c in coros.items()))
    return {k: v for k, v in pairs}


# -----------------------------------------------------------------------------
# Entrypoint: `python local_site.py` (no reload) or `uvicorn local_site:api`.
# -----------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    uvicorn.run("local_site:api", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
