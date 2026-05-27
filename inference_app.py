"""Modal app exposing OpenVLA-7B as a GPU inference worker + web playground.

Deploy:

    modal deploy inference_app.py

The web playground is served at the Modal-assigned URL. The GPU worker can
also be called from anywhere with the Modal SDK:

    import modal
    OpenVLA = modal.Cls.from_name("vla-inference", "OpenVLAWorker")
    out = OpenVLA().predict.remote(image_bytes=img_jpeg, task="pick up the cup")

Inputs are JPEG/PNG bytes + a natural-language task string. Outputs are
JSON-serialisable dicts containing the predicted 7-DoF action vector.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "vla-inference"

OPENVLA_MODEL_ID = "openvla/openvla-7b"

OPENVLA_DEFAULT_UNNORM_KEY = "bridge_orig"

STATIC_DIR = Path(__file__).parent / "static"


def _download_openvla() -> None:
    """Cache OpenVLA weights + processor inside the image."""
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

    AutoProcessor.from_pretrained(OPENVLA_MODEL_ID, trust_remote_code=True)
    AutoModelForVision2Seq.from_pretrained(
        OPENVLA_MODEL_ID,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )


openvla_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.40,<4.50",
        "timm>=0.9.10,<1.0.0",
        "tokenizers>=0.19",
        "accelerate>=0.30",
        "huggingface_hub>=0.24",
        "pillow>=10",
        "numpy>=1.26",
    )
    .run_function(_download_openvla)
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi>=0.110")
    .add_local_dir(STATIC_DIR, remote_path="/assets/static")
)

app = modal.App(APP_NAME)


@app.cls(
    image=openvla_image,
    gpu="L4",
    scaledown_window=120,
    timeout=900,
)
class OpenVLAWorker:
    """OpenVLA-7B inference worker.

    OpenVLA outputs a 7-DoF end-effector delta action
    ``(dx, dy, dz, droll, dpitch, dyaw, gripper)``. The action is
    un-normalised using per-dataset statistics keyed by ``unnorm_key``;
    common choices are ``"bridge_orig"``, ``"fractal20220817_data"``,
    ``"jaco_play"``, ``"berkeley_autolab_ur5"``, etc.
    """

    @modal.enter()
    def setup(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.device = "cuda:0"
        self.dtype = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(
            OPENVLA_MODEL_ID, trust_remote_code=True
        )
        self.vla = (
            AutoModelForVision2Seq.from_pretrained(
                OPENVLA_MODEL_ID,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

        # Cache the list of available un-normalisation keys so we can fall back
        # cleanly if the caller passes an unknown one.
        self.available_unnorm_keys = sorted(
            getattr(self.vla, "norm_stats", {}).keys()
        )

    @modal.method()
    def predict(
        self,
        image_bytes: bytes,
        task: str,
        unnorm_key: str = OPENVLA_DEFAULT_UNNORM_KEY,
        return_activations: bool = False,
        comparison_image_bytes: bytes | None = None,
    ) -> dict:
        import io

        from PIL import Image

        torch = self.torch
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        comparison_image = None
        if comparison_image_bytes is not None:
            comparison_image = Image.open(io.BytesIO(comparison_image_bytes)).convert("RGB")

        prompt = f"In: What action should the robot take to {task}?\nOut:"

        chosen_key = unnorm_key
        if chosen_key not in self.available_unnorm_keys:
            chosen_key = OPENVLA_DEFAULT_UNNORM_KEY

        inputs = self.processor(prompt, image).to(self.device, dtype=self.dtype)

        activations_data = {}
        cka_data = {}
        if return_activations:
            with torch.inference_mode():
                fwd_out = self.vla(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=True,
                )
            hidden = fwd_out.hidden_states
            attentions = fwd_out.attentions

            n_layers = len(hidden) - 1
            layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
            layer_norms = []
            for li in layer_indices:
                h = hidden[li + 1][0]  # (seq_len, hidden_dim)
                norms = h.float().norm(dim=-1).cpu().numpy().tolist()
                layer_norms.append({"layer": int(li), "norms": norms})

            attn_indices = [0, len(attentions) // 2, len(attentions) - 1]
            attn_maps = []
            for ai in attn_indices:
                a = attentions[ai][0]  # (n_heads, seq, seq)
                avg_attn = a.float().mean(dim=0)  # (seq, seq)
                seq_len = avg_attn.shape[0]
                stride = max(1, seq_len // 64)
                downsampled = avg_attn[::stride, ::stride].cpu().numpy().tolist()
                attn_maps.append({
                    "layer": int(ai),
                    "map": downsampled,
                    "original_seq_len": int(seq_len),
                })

            activations_data = {
                "hidden_norms": layer_norms,
                "attention_maps": attn_maps,
                "n_layers": int(n_layers),
                "seq_len": int(hidden[0].shape[1]),
            }

            if comparison_image is not None:
                comparison_inputs = self.processor(prompt, comparison_image).to(
                    self.device, dtype=self.dtype
                )
                with torch.inference_mode():
                    comparison_out = self.vla(
                        **comparison_inputs,
                        output_hidden_states=True,
                    )

                comparison_hidden = comparison_out.hidden_states
                cka_layers = list(range(n_layers))
                seq_len = min(hidden[1].shape[1], comparison_hidden[1].shape[1])

                def centered_gram(layer_hidden):
                    x = layer_hidden[0, :seq_len].float()
                    x = x - x.mean(dim=0, keepdim=True)
                    gram = x @ x.T
                    return (
                        gram
                        - gram.mean(dim=0, keepdim=True)
                        - gram.mean(dim=1, keepdim=True)
                        + gram.mean()
                    )

                robot_grams = [centered_gram(hidden[i + 1]) for i in cka_layers]
                human_grams = [
                    centered_gram(comparison_hidden[i + 1]) for i in cka_layers
                ]
                robot_norms = [g.norm() for g in robot_grams]
                human_norms = [g.norm() for g in human_grams]

                cka_matrix = []
                for rg, rn in zip(robot_grams, robot_norms):
                    row = []
                    for hg, hn in zip(human_grams, human_norms):
                        denom = rn * hn
                        cka = (rg * hg).sum() / denom if denom > 0 else torch.tensor(0.0)
                        row.append(float(cka.clamp(0, 1).cpu()))
                    cka_matrix.append(row)

                diagonal = [
                    cka_matrix[i][i] for i in range(min(len(cka_matrix), len(cka_matrix[0])))
                ]
                cka_data = {
                    "kind": "linear_cka_centered_gram",
                    "layers": cka_layers,
                    "matrix": cka_matrix,
                    "diagonal": diagonal,
                    "mean_diagonal": float(sum(diagonal) / len(diagonal)),
                    "seq_len_used": int(seq_len),
                    "robot_view": "robot_camera",
                    "human_view": "human_camera",
                }

        with torch.inference_mode():
            action = self.vla.predict_action(
                **inputs, unnorm_key=chosen_key, do_sample=False
            )

        action_list = action.tolist() if hasattr(action, "tolist") else list(action)
        shape = list(action.shape) if hasattr(action, "shape") else [len(action_list)]

        result = {
            "model": "openvla",
            "model_id": OPENVLA_MODEL_ID,
            "task": task,
            "action": action_list,
            "shape": shape,
            "action_dim": int(shape[-1]) if shape else len(action_list),
            "unnorm_key_used": chosen_key,
            "unnorm_key_requested": unnorm_key,
        }
        if return_activations:
            result["activations"] = activations_data
        if cka_data:
            result["cka"] = cka_data
        return result

    @modal.method()
    def list_unnorm_keys(self) -> list[str]:
        """Return all OXE-subset stats keys baked into this OpenVLA checkpoint."""
        return list(self.available_unnorm_keys)


@app.function(
    image=web_image,
    scaledown_window=300,
)
@modal.asgi_app()
def web():
    import base64
    import time
    from typing import Any

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    api = FastAPI(title="OpenVLA Playground", docs_url="/docs", redoc_url=None)
    api.mount("/static", StaticFiles(directory="/assets/static"), name="static")

    _worker = None

    def get_worker():
        nonlocal _worker
        if _worker is None:
            _worker = OpenVLAWorker()
        return _worker

    class PredictRequest(BaseModel):
        image_b64: str = Field(..., description="Base64-encoded image bytes.")
        comparison_image_b64: str | None = Field(
            default=None,
            description="Optional paired image for representation comparison.",
        )
        task: str = Field(..., description="Natural-language instruction.")
        unnorm_key: str = Field(default="bridge_orig")
        return_activations: bool = Field(default=False)

    @api.get("/", include_in_schema=False)
    async def index():
        return FileResponse("/assets/static/index.html")

    @api.get("/api/info")
    async def info() -> dict[str, Any]:
        return {"app_name": APP_NAME, "openvla_class": "OpenVLAWorker"}

    @api.post("/api/predict")
    async def predict(req: PredictRequest) -> dict[str, Any]:
        try:
            image_bytes = base64.b64decode(req.image_b64)
        except Exception as e:
            raise HTTPException(400, f"Bad image_b64: {e}") from e
        comparison_image_bytes = None
        if req.comparison_image_b64:
            try:
                comparison_image_bytes = base64.b64decode(req.comparison_image_b64)
            except Exception as e:
                raise HTTPException(400, f"Bad comparison_image_b64: {e}") from e

        worker = get_worker()
        t0 = time.perf_counter()
        try:
            out = await worker.predict.remote.aio(
                image_bytes=image_bytes,
                task=req.task,
                unnorm_key=req.unnorm_key,
                return_activations=req.return_activations,
                comparison_image_bytes=comparison_image_bytes,
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

    return api


@app.local_entrypoint()
def smoke_test() -> None:
    """Quick check: ``modal run inference_app.py``."""
    import io

    from PIL import Image

    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    task = "pick up the cup on the table"

    out = OpenVLAWorker().predict.remote(image_bytes=image_bytes, task=task)
    print("OpenVLA ->", out)
