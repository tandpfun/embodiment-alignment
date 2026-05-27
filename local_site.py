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

import asyncio
import base64
import math
import time
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import h_and_r as hr

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


# -----------------------------------------------------------------------------
# H&R dataset browser (dannyXSC/HumanAndRobot)
#
# - /api/hr/tasks                       -> ["grab_cube", ...]
# - /api/hr/episodes?task=...           -> {"task": ..., "episodes": [...]}
# - /api/hr/info?task=...&episode=...   -> {"num_frames": ..., "instruction": ...}
# - /api/hr/frame?task=...&episode=...&frame=...&view=human|robot -> JPEG image
#
# The browser uses these endpoints to let the user pick a (task, episode, frame)
# pair and load *either* the human or robot view into the OpenVLA input.
# Episode HDF5 files are lazily downloaded into the HuggingFace cache on first
# access (~70-220 MB each).
# -----------------------------------------------------------------------------


@api.get("/api/hr/tasks")
async def hr_tasks() -> dict[str, Any]:
    tasks = hr.list_tasks()
    return {
        "tasks": [{"name": t, "instruction": hr.task_instruction(t)} for t in tasks],
    }


@api.get("/api/hr/episodes")
async def hr_episodes(task: str = Query(...)) -> dict[str, Any]:
    try:
        eps = hr.list_episodes(task)
    except Exception as e:
        raise HTTPException(400, f"Bad task {task!r}: {e}") from e
    return {"task": task, "episodes": eps}


@api.get("/api/hr/info")
async def hr_info(task: str = Query(...), episode: int = Query(...)) -> dict[str, Any]:
    try:
        return hr.episode_info(task, episode)
    except Exception as e:
        raise HTTPException(400, f"Failed to read {task}/episode_{episode}: {e}") from e


@api.get("/api/hr/frame")
async def hr_frame(
    task: str = Query(...),
    episode: int = Query(...),
    frame: int = Query(...),
    view: str = Query("human", pattern="^(human|robot)$"),
) -> Response:
    try:
        jpeg = hr.get_frame_jpeg(task, episode, frame, view)  # type: ignore[arg-type]
    except IndexError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Frame read failed: {e}") from e
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@api.post("/api/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        image_bytes = base64.b64decode(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"Bad image_b64: {e}") from e

    return await _run_one(image_bytes, req.task, req.unnorm_key)


# -----------------------------------------------------------------------------
# H&R compare-pair endpoint: same frame, two views, two parallel Modal calls.
# -----------------------------------------------------------------------------


class HRPairRequest(BaseModel):
    task: str = Field(..., description="H&R task folder (e.g. 'push_box').")
    episode: int = Field(..., description="Episode index within the task.")
    frame: int = Field(..., description="Frame index inside the episode.")
    instruction: str | None = Field(
        default=None,
        description=(
            "Natural-language instruction sent to OpenVLA. Defaults to the "
            "instruction associated with the task in h_and_r.py."
        ),
    )
    unnorm_key: str = Field(
        default="bridge_orig",
        description="OpenVLA un-normalisation key (per-OXE-subset stats).",
    )


async def _run_one(image_bytes: bytes, task: str, unnorm_key: str) -> dict[str, Any]:
    worker = get_worker()
    t0 = time.perf_counter()
    try:
        out = await worker.predict.remote.aio(
            image_bytes=image_bytes,
            task=task,
            unnorm_key=unnorm_key,
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


def _action_diff(a: list[float], b: list[float]) -> dict[str, Any]:
    n = min(len(a), len(b))
    per_dim = [float(a[i] - b[i]) for i in range(n)]
    l2 = math.sqrt(sum(d * d for d in per_dim))
    linf = max((abs(d) for d in per_dim), default=0.0)
    return {"per_dim": per_dim, "l2": l2, "linf": linf, "dim": n}


@api.post("/api/hr/predict_pair")
async def hr_predict_pair(req: HRPairRequest) -> dict[str, Any]:
    """Run OpenVLA on the human and robot views of the same H&R frame in parallel."""
    try:
        human_jpeg = hr.get_frame_jpeg(req.task, req.episode, req.frame, "human")
        robot_jpeg = hr.get_frame_jpeg(req.task, req.episode, req.frame, "robot")
    except IndexError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Frame read failed: {e}") from e

    instruction = req.instruction or hr.task_instruction(req.task)

    t0 = time.perf_counter()
    human_res, robot_res = await asyncio.gather(
        _run_one(human_jpeg, instruction, req.unnorm_key),
        _run_one(robot_jpeg, instruction, req.unnorm_key),
    )
    wall_ms = (time.perf_counter() - t0) * 1000

    diff: dict[str, Any] | None = None
    if (
        human_res.get("ok")
        and robot_res.get("ok")
        and isinstance(human_res.get("action"), list)
        and isinstance(robot_res.get("action"), list)
    ):
        diff = _action_diff(human_res["action"], robot_res["action"])

    return {
        "task": req.task,
        "episode": req.episode,
        "frame": req.frame,
        "instruction": instruction,
        "unnorm_key": req.unnorm_key,
        "wall_ms": wall_ms,
        "human": human_res,
        "robot": robot_res,
        "diff": diff,
    }


# -----------------------------------------------------------------------------
# Entrypoint: `python local_site.py` (no reload) or `uvicorn local_site:api`.
# -----------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    uvicorn.run("local_site:api", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
