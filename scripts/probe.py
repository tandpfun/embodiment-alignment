"""Linear-probe analysis of OpenVLA hidden states on H&R human/robot frame pairs.

For each of N evenly-spaced frames in a single H&R episode we:

    1. Run BOTH views (human, robot) through OpenVLA on Modal once, capturing
       the mean-pooled hidden state at every Llama decoder layer (and the
       embedding output).
    2. Compute, per layer:
         - mean cosine similarity between matched (human, robot) pairs
         - leave-one-pair-out cross-validated accuracy of an L2-regularized
           logistic regression that tries to classify human vs robot from
           the layer's representation.

High cosine + ~50% probe accuracy at a layer means the model has "abstracted
away" embodiment at that depth. Low cosine + ~100% probe accuracy means the
embodiment is still linearly decodable there.

Usage:
    uv run python scripts/probe.py                          # defaults
    uv run python scripts/probe.py --task push_box -n 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the project root importable when running `python scripts/probe.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modal
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

import h_and_r as hr

APP_NAME = "vla-inference"


def evenly_spaced(num: int, total: int) -> list[int]:
    if total <= 1 or num <= 1:
        return [0]
    if num >= total:
        return list(range(total))
    return sorted(
        {round(i * (total - 1) / (num - 1)) for i in range(num)}
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="grab_cube")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("-n", "--num-frames", type=int, default=30)
    p.add_argument("--out", default="outputs/probe.npz")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve frame range from the H&R dataset directly (no need to hit the
    # local site).
    info = hr.episode_info(args.task, args.episode)
    n_total = info["num_frames"]
    instruction = info["instruction"]
    frame_idx = evenly_spaced(args.num_frames, n_total)
    print(
        f"Task '{args.task}' episode_{args.episode}: "
        f"{len(frame_idx)} frames of {n_total} (instruction: {instruction!r})"
    )

    # Pre-fetch frames as JPEG bytes (cached in HF disk cache after first run).
    print("Loading frames…")
    pairs: list[tuple[int, bytes, bytes]] = []
    for fi in frame_idx:
        h_jpeg = hr.get_frame_jpeg(args.task, args.episode, fi, "human")
        r_jpeg = hr.get_frame_jpeg(args.task, args.episode, fi, "robot")
        pairs.append((fi, h_jpeg, r_jpeg))

    # Connect to the deployed OpenVLA worker.
    OpenVLA = modal.Cls.from_name(APP_NAME, "OpenVLAWorker")
    worker = OpenVLA()

    # Run extraction. We send each call sequentially; Modal keeps the
    # container warm so each call after the first is ~600ms-1s.
    print(f"Calling OpenVLA on {2 * len(pairs)} images…")
    t0 = time.perf_counter()
    feats_human: list[np.ndarray] = []
    feats_robot: list[np.ndarray] = []
    n_layers = None
    hidden_dim = None
    for i, (fi, h_jpeg, r_jpeg) in enumerate(pairs, start=1):
        # Two calls per frame — keep it simple, not strictly parallel.
        out_h = worker.extract_features.remote(image_bytes=h_jpeg, task=instruction)
        out_r = worker.extract_features.remote(image_bytes=r_jpeg, task=instruction)
        for tag, out, sink in (
            ("human", out_h, feats_human),
            ("robot", out_r, feats_robot),
        ):
            if n_layers is None:
                n_layers = out["n_layers"]
                hidden_dim = out["hidden_dim"]
            arr = np.frombuffer(out["features_f16"], dtype=np.float16).reshape(
                out["n_layers"], out["hidden_dim"]
            )
            sink.append(arr.astype(np.float32))
        print(
            f"  [{i:>2}/{len(pairs)}] frame {fi}: "
            f"shape={n_layers}x{hidden_dim}  "
            f"({time.perf_counter() - t0:.1f}s elapsed)"
        )

    H = np.stack(feats_human, axis=0)  # (N, L, D)
    R = np.stack(feats_robot, axis=0)
    print(f"Got features: H{tuple(H.shape)}  R{tuple(R.shape)}")

    # ------------------------------------------------------------------ analysis

    L = H.shape[1]
    cosines = np.zeros(L)
    l2s = np.zeros(L)
    probe_acc = np.zeros(L)

    logo = LeaveOneGroupOut()
    pair_groups = np.repeat(np.arange(len(pairs)), 2)  # group id per row in X

    for layer in range(L):
        h = H[:, layer, :]
        r = R[:, layer, :]
        # Cosine similarity between matched pairs.
        num = (h * r).sum(axis=1)
        den = np.linalg.norm(h, axis=1) * np.linalg.norm(r, axis=1) + 1e-8
        cosines[layer] = (num / den).mean()
        l2s[layer] = np.linalg.norm(h - r, axis=1).mean()

        # Linear probe: human vs robot.
        X = np.concatenate([h, r], axis=0)  # (2N, D)
        y = np.concatenate(
            [np.zeros(len(pairs)), np.ones(len(pairs))]
        )  # 0=human, 1=robot
        # Center & l2-normalize so logistic regression behaves on different layer
        # scales.
        X = X - X.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X = X / norms
        scores = []
        for train, test in logo.split(X, y, groups=pair_groups):
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            clf.fit(X[train], y[train])
            scores.append(clf.score(X[test], y[test]))
        probe_acc[layer] = float(np.mean(scores))

    # ------------------------------------------------------------------ report

    print()
    print(f"{'layer':>5}  {'cos_sim':>8}  {'l2_dist':>9}  {'probe_acc':>9}  bar")
    print("-" * 70)
    for layer in range(L):
        # ASCII bar for probe accuracy (1.0 = full bar of 30 chars).
        bar_len = int(round(probe_acc[layer] * 30))
        bar = "█" * bar_len + " " * (30 - bar_len)
        print(
            f"{layer:>5}  {cosines[layer]:+8.4f}  {l2s[layer]:9.3f}  "
            f"{probe_acc[layer]:9.3f}  |{bar}|"
        )

    print()
    chance = 0.5
    abstracted = [layer for layer in range(L) if probe_acc[layer] < 0.7]
    separable = [layer for layer in range(L) if probe_acc[layer] > 0.95]
    print(f"Layers where probe < 70% (≈embodiment-abstracted): {abstracted}")
    print(f"Layers where probe > 95% (embodiment fully separable): {separable}")

    np.savez(
        out_path,
        cosines=cosines,
        l2s=l2s,
        probe_acc=probe_acc,
        features_human=H,
        features_robot=R,
        frames=np.array([fi for fi, _, _ in pairs]),
        instruction=instruction,
        task=args.task,
        episode=args.episode,
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
