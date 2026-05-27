"""Scan a few frames across H&R episodes and print a comparison table.

Calls the locally-running site (`local_site.py`) which in turn fires both
Modal calls in parallel via `/api/hr/predict_pair`.

Usage:
    uv run python scripts/scan_episode.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"

DEFAULT_TASK = "grab_cube"
DEFAULT_EPISODE = 0
DEFAULT_NUM_FRAMES = 30


def get_json(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def post_json(path: str, body: dict, timeout: float = 240.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fmt(x: float | None, w: int = 8) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—".rjust(w)
    return f"{x:+.4f}".rjust(w)


LABELS = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]


def evenly_spaced_frames(num_frames: int, total: int) -> list[int]:
    """Return `num_frames` indices spread across [0, total-1], always including endpoints."""
    if total <= 0:
        return []
    if num_frames <= 1:
        return [0]
    if num_frames >= total:
        return list(range(total))
    out: list[int] = []
    for i in range(num_frames):
        idx = round(i * (total - 1) / (num_frames - 1))
        if not out or idx != out[-1]:
            out.append(idx)
    return out


def scan_task(
    task: str, episode: int, num_frames: int, out_path: str | None = None
) -> None:
    info = get_json(f"/api/hr/info?task={task}&episode={episode}")
    n = info["num_frames"]
    instruction = info["instruction"]
    frames = evenly_spaced_frames(num_frames, n)
    print(f"\n=== {task}/episode_{episode}  ({n} frames, instruction: {instruction!r}) ===")

    header = (
        f"{'frame':>5} {'view':>6}  "
        + "  ".join(L.rjust(8) for L in LABELS)
        + f"  {'||a||₂':>8}  {'lat ms':>7}"
    )
    print(header)
    print("-" * len(header))

    summaries: list[dict] = []
    out_file = None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path, "w")

    for i, fr in enumerate(frames):
        print(f"  [{i + 1}/{len(frames)}] frame {fr} …", flush=True)
        body = {
            "task": task,
            "episode": episode,
            "frame": fr,
            "instruction": instruction,
            "unnorm_key": "bridge_orig",
        }
        try:
            data = post_json("/api/hr/predict_pair", body)
        except Exception as e:
            print(f"  frame {fr}: request failed: {e}")
            continue

        for label, key in (("human", "human"), ("robot", "robot")):
            r = data.get(key) or {}
            if not r.get("ok"):
                print(f"  {fr:>5} {label:>6}  ERROR  {r.get('error')!r}")
                continue
            action = r.get("action") or []
            mag = math.sqrt(sum(a * a for a in action))
            print(
                f"  {fr:>5} {label:>6}  "
                + "  ".join(fmt(a) for a in action)
                + f"  {fmt(mag)}  {r.get('elapsed_ms', 0):>7.0f}"
            )

        diff = data.get("diff") or {}
        if diff:
            print(
                "        diff  "
                + "  ".join(fmt(d) for d in diff.get("per_dim", []))
                + f"  L2={fmt(diff.get('l2'))}  max|Δ|={fmt(diff.get('linf'))}"
            )
            row = {
                "frame": fr,
                "l2": diff.get("l2"),
                "linf": diff.get("linf"),
                "human_gripper": (data["human"].get("action") or [None] * 7)[-1],
                "robot_gripper": (data["robot"].get("action") or [None] * 7)[-1],
                "human_mag_pose": math.sqrt(
                    sum(a * a for a in (data["human"].get("action") or [0] * 7)[:6])
                ),
                "robot_mag_pose": math.sqrt(
                    sum(a * a for a in (data["robot"].get("action") or [0] * 7)[:6])
                ),
                "human_action": data["human"].get("action"),
                "robot_action": data["robot"].get("action"),
            }
            summaries.append(row)
            if out_file:
                out_file.write(json.dumps(row) + "\n")
                out_file.flush()

    if out_file:
        out_file.close()
        print(f"  wrote {out_path}")

    print()
    print(f"  summary for {task}/ep{episode}:")
    print(f"  {'frame':>5} {'h_grip':>8} {'r_grip':>8} {'h‖pose‖':>9} {'r‖pose‖':>9} {'L2':>8} {'max|Δ|':>8}")
    for s in summaries:
        print(
            f"  {s['frame']:>5} "
            f"{fmt(s['human_gripper'])} {fmt(s['robot_gripper'])} "
            f"{fmt(s['human_mag_pose'])} {fmt(s['robot_mag_pose'])} "
            f"{fmt(s['l2'])} {fmt(s['linf'])}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--episode", type=int, default=DEFAULT_EPISODE)
    p.add_argument("-n", "--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    p.add_argument(
        "--out",
        default="outputs/scan_results.jsonl",
        help="Write one JSON object per frame (for plotting later).",
    )
    args = p.parse_args()

    try:
        get_json("/api/info")
    except urllib.error.URLError as e:
        print(f"Site not reachable at {BASE}: {e}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    scan_task(args.task, args.episode, args.num_frames, args.out)
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
