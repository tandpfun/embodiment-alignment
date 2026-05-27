# OpenVLA inference on Modal

`openvla/openvla-7b` runs as a Modal GPU worker (L4, bf16). A local web UI calls it from your browser.

## One-time setup

```bash
uv sync
uv run modal token new   # skip if ~/.modal.toml already exists
uv run modal deploy inference_app.py
```

First deploy downloads ~14 GB of model weights into the Modal image (several minutes).

## Smoke test (terminal)

```bash
uv run modal run inference_app.py
```

## Local web UI

```bash
uv run python local_site.py
```

Open http://127.0.0.1:8000 — upload an image, enter a task, click **Predict**. First request after idle may take 30–60 s while the GPU container warms up.

## Call from Python

```python
import modal

OpenVLA = modal.Cls.from_name("vla-inference", "OpenVLAWorker")
out = OpenVLA().predict.remote(
    image_bytes=jpeg_bytes,
    task="pick up the cup",
    unnorm_key="bridge_orig",  # OXE subset stats; default is BridgeV2
)
```
