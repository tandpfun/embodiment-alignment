"""Browse the H&R (dannyXSC/HumanAndRobot) dataset locally.

The dataset has the layout:

    data/v0/<task>/episode_<N>.hdf5
    data/v1/<task>/episode_<N>.hdf5

Each HDF5 file has two aligned tensors of shape (T, H, W, 3) uint8:

    /cam_data/human_camera
    /cam_data/robot_camera

This module exposes a small read-only browsing API:

    list_tasks() -> [task, ...]
    list_episodes(task) -> [0, 1, 2, ...]
    episode_info(task, ep) -> {"num_frames": T, "height": H, "width": W}
    get_frame_jpeg(task, ep, frame_idx, view) -> bytes

The first request for an episode triggers a one-time ~70-220 MB download
into the HuggingFace cache; subsequent reads are zero-copy from disk.
"""

from __future__ import annotations

import io
import re
import threading
from functools import lru_cache
from typing import Literal

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO_ID = "dannyXSC/HumanAndRobot"
REPO_TYPE = "dataset"

# Pick one of these. v1 also has a precomputed /action; v0 is smaller / older.
VERSION = "v0"

# Human-readable task instructions used when a frame is loaded into the
# OpenVLA input. Tweak freely.
TASK_INSTRUCTIONS: dict[str, str] = {
    "grab_cube": "grab the cube",
    "grab_cup": "grab the cup",
    "grab_to_plate1": "grab the object and place it on the plate",
    "grab_to_plate2": "grab the object and place it on the plate",
    "grab_two_cubes1": "grab the two cubes",
    "grab_two_cubes2": "grab the two cubes",
    "pull_plate": "pull the plate",
    "push_box": "push the box",
    "push_plate": "push the plate",
}


_api = HfApi()
_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Catalog (cached in-memory).
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def list_tasks() -> list[str]:
    """List task subfolders under data/<VERSION>/."""
    entries = _api.list_repo_tree(
        REPO_ID, repo_type=REPO_TYPE, path_in_repo=f"data/{VERSION}"
    )
    # RepoFolder objects expose `tree_id`; RepoFile objects expose `size`.
    return sorted(
        e.path.split("/")[-1] for e in entries if getattr(e, "tree_id", None) is not None
    )


@lru_cache(maxsize=64)
def list_episodes(task: str) -> list[int]:
    """List episode indices for a given task, sorted numerically."""
    entries = _api.list_repo_tree(
        REPO_ID, repo_type=REPO_TYPE, path_in_repo=f"data/{VERSION}/{task}"
    )
    eps: list[int] = []
    pat = re.compile(r"episode_(\d+)\.hdf5$")
    for e in entries:
        m = pat.search(e.path)
        if m:
            eps.append(int(m.group(1)))
    eps.sort()
    return eps


def task_instruction(task: str) -> str:
    """Return the human-readable task instruction for a task name."""
    return TASK_INSTRUCTIONS.get(task, task.replace("_", " "))


# -----------------------------------------------------------------------------
# Episode files (lazy download).
# -----------------------------------------------------------------------------


def _ensure_episode(task: str, ep: int) -> str:
    """Download the HDF5 for (task, ep) if not cached, return local path."""
    rel = f"data/{VERSION}/{task}/episode_{ep}.hdf5"
    return hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=rel)


def episode_info(task: str, ep: int) -> dict:
    path = _ensure_episode(task, ep)
    with _lock, h5py.File(path, "r") as f:
        ds = f["/cam_data/human_camera"]
        t, h, w, _c = ds.shape
        return {
            "task": task,
            "episode": ep,
            "num_frames": int(t),
            "height": int(h),
            "width": int(w),
            "instruction": task_instruction(task),
        }


def get_frame_jpeg(
    task: str,
    ep: int,
    frame_idx: int,
    view: Literal["human", "robot"],
    quality: int = 88,
) -> bytes:
    """Decode one frame from the episode and re-encode as JPEG bytes."""
    if view not in ("human", "robot"):
        raise ValueError(f"view must be 'human' or 'robot', got {view!r}")

    path = _ensure_episode(task, ep)
    key = "/cam_data/human_camera" if view == "human" else "/cam_data/robot_camera"
    with _lock, h5py.File(path, "r") as f:
        ds = f[key]
        t = ds.shape[0]
        if frame_idx < 0:
            frame_idx += t
        if not 0 <= frame_idx < t:
            raise IndexError(f"frame {frame_idx} out of range (0..{t - 1})")
        arr: np.ndarray = ds[frame_idx]

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)  # HxWxRGB
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
