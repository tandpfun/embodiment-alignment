"""Modal app exposing OpenVLA-7B as a GPU inference worker.

Deploy:

    modal deploy inference_app.py

Once deployed, the worker can be called from anywhere with the Modal SDK:

    import modal
    OpenVLA = modal.Cls.from_name("vla-inference", "OpenVLAWorker")
    out = OpenVLA().predict.remote(image_bytes=img_jpeg, task="pick up the cup")

Inputs are JPEG/PNG bytes + a natural-language task string. Outputs are
JSON-serialisable dicts containing the predicted 7-DoF action vector.
"""

from __future__ import annotations

import modal

APP_NAME = "vla-inference"

OPENVLA_MODEL_ID = "openvla/openvla-7b"

# Default unnorm key for OpenVLA. OpenVLA stores per-OXE-subset normalisation
# statistics in ``vla.norm_stats``; ``"bridge_orig"`` (BridgeV2 / WidowX) is a
# sensible generic default. Override per dataset (e.g. ``"jaco_play"``,
# ``"fractal20220817_data"``).
OPENVLA_DEFAULT_UNNORM_KEY = "bridge_orig"


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
        "timm>=0.9.10,<1.0.0",  # OpenVLA requires timm < 1.0
        "tokenizers>=0.19",
        "accelerate>=0.30",
        "huggingface_hub>=0.24",
        "pillow>=10",
        "numpy>=1.26",
    )
    .run_function(_download_openvla)
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
    ) -> dict:
        import io

        from PIL import Image

        torch = self.torch
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # OpenVLA's expected prompt template (see model card).
        prompt = f"In: What action should the robot take to {task}?\nOut:"

        # Fall back if caller passed an unsupported unnorm_key.
        chosen_key = unnorm_key
        if chosen_key not in self.available_unnorm_keys:
            chosen_key = OPENVLA_DEFAULT_UNNORM_KEY

        inputs = self.processor(prompt, image).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            action = self.vla.predict_action(
                **inputs, unnorm_key=chosen_key, do_sample=False
            )

        # ``predict_action`` returns a numpy array of shape (7,).
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)
        shape = list(action.shape) if hasattr(action, "shape") else [len(action_list)]

        return {
            "model": "openvla",
            "model_id": OPENVLA_MODEL_ID,
            "task": task,
            "action": action_list,
            "shape": shape,
            "action_dim": int(shape[-1]) if shape else len(action_list),
            "unnorm_key_used": chosen_key,
            "unnorm_key_requested": unnorm_key,
        }

    @modal.method()
    def list_unnorm_keys(self) -> list[str]:
        """Return all OXE-subset stats keys baked into this OpenVLA checkpoint."""
        return list(self.available_unnorm_keys)

    # -------------------------------------------------------------------------
    # Per-layer feature extraction for probing experiments.
    #
    # We register forward hooks on every Llama decoder layer and capture the
    # block output (shape: [1, seq_len, hidden_dim]). We return the activations
    # mean-pooled over the token dimension as float16 bytes — much smaller than
    # JSON. The probe script reshapes back to (n_layers + 1, hidden_dim).
    #
    # Including the embedding layer (index 0) gives n_layers+1 representations:
    # layer 0 == post-embedding, layer N == output of decoder block N-1.
    # -------------------------------------------------------------------------

    @modal.method()
    def extract_features(self, image_bytes: bytes, task: str) -> dict:
        import io

        import numpy as np
        from PIL import Image

        torch = self.torch
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt = f"In: What action should the robot take to {task}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=self.dtype)

        # Find the Llama decoder layers anywhere under the model.
        decoder_layers: list[tuple[str, "torch.nn.Module"]] = []
        for name, module in self.vla.named_modules():
            if module.__class__.__name__ == "LlamaDecoderLayer":
                decoder_layers.append((name, module))
        if not decoder_layers:
            raise RuntimeError("No LlamaDecoderLayer modules found on OpenVLA model.")

        n_decoder = len(decoder_layers)
        # Capture only the *first* forward pass per layer. predict_action
        # autoregresses several action tokens, each of which re-runs the
        # entire decoder; we want the prompt-encoding pass (image+text +
        # "Out:" -> the position right before the action is sampled).
        captured: dict[int, "torch.Tensor"] = {}

        def make_hook(idx: int):
            def hook(_mod, _inp, out):
                if idx in captured:
                    return
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h.detach()
            return hook

        handles = [
            mod.register_forward_hook(make_hook(i))
            for i, (_, mod) in enumerate(decoder_layers)
        ]

        try:
            with torch.inference_mode():
                _ = self.vla.predict_action(
                    **inputs, unnorm_key="bridge_orig", do_sample=False
                )
        finally:
            for h in handles:
                h.remove()

        if len(captured) != n_decoder:
            raise RuntimeError(
                f"Hook capture mismatch: got {len(captured)} layers, expected {n_decoder}."
            )

        # Mean-pool over the token dimension to a single H-dim vector per layer.
        # captured[i] has shape (B=1, T, H); T is the full prompt length.
        pooled = []
        seq_lens = []
        for i in range(n_decoder):
            h = captured[i]
            seq_lens.append(int(h.shape[1]))
            pooled.append(h[0].mean(dim=0).to(torch.float16).cpu().numpy())

        arr = np.stack(pooled, axis=0)  # (n_layers, hidden_dim) float16
        return {
            "features_f16": arr.tobytes(),
            "n_layers": int(arr.shape[0]),
            "hidden_dim": int(arr.shape[1]),
            "seq_lens": seq_lens,
            "task": task,
        }


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
