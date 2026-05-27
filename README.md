# VLA inference on Modal

SmolVLA (`lerobot/smolvla_base`, T4) and OpenVLA (`openvla/openvla-7b`, L4) run as Modal GPU workers. A local web UI calls them from your browser.

## One-time setup

```bash
uv sync
uv run modal token new   # skip if ~/.modal.toml already exists
uv run modal deploy inference_app.py
```

First deploy downloads ~15 GB of model weights into the Modal images (several minutes).

## Smoke test (terminal)

```bash
uv run modal run inference_app.py
```

## Local web UI

```bash
uv run python local_site.py
```

Open http://127.0.0.1:8000 — upload an image, enter a task, click **Predict**. First request after idle may take 30–60 s while GPU containers warm up.

## Call from Python

```python
import modal

SmolVLA = modal.Cls.from_name("vla-inference", "SmolVLAWorker")
OpenVLA = modal.Cls.from_name("vla-inference", "OpenVLAWorker")

out = SmolVLA().predict.remote(image_bytes=jpeg_bytes, task="pick up the cup")
```

OpenVLA accepts `unnorm_key` (default `bridge_orig` for Bridge V2).
