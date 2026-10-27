"""Replay-safe real image Mappers for the compute-flow experiment.

Local weights only. No artificial delays or synthetic GPU predictions. These
operators are benchmark adapters, not new default DJ operators.
"""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from data_juicer.ops.base_op import OPERATORS, Mapper


@OPERATORS.register_module("ej_flow_decode")
class Decode(Mapper):
    _batched_op = True
    elastic_juicer_dag_safe = True

    def process_batched(self, batch):
        output = {k: [] for k in ("id", "pixels", "image_hash", "width", "height")}
        for identity, path, expected in zip(batch["id"], batch["image_path"], batch["image_sha256"]):
            raw = Path(path).read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected:
                raise ValueError(f"source image hash changed: {identity}")
            import io

            with Image.open(io.BytesIO(raw)) as source:
                pixels = np.array(source.convert("RGB"))
            for key, value in dict(
                id=identity, pixels=pixels, image_hash=actual, width=pixels.shape[1], height=pixels.shape[0]
            ).items():
                output[key].append(value)
        return output


@OPERATORS.register_module("ej_flow_model")
class ImageModel(Mapper):
    _batched_op = True
    _accelerator = "cuda"
    elastic_juicer_dag_safe = True

    def __init__(self, stage, model_path, **kwargs):
        super().__init__(**kwargs)
        import torch

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.stage = stage
        if stage == "quality":
            path = Path(model_path)
            spec = importlib.util.spec_from_file_location("ej_flow_qualiclip", path / "clip/model.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            checkpoint = torch.load(path / "qualiclip.pth", map_location="cpu", weights_only=True)
            state = {k[len("clip_model.") :]: v for k, v in checkpoint.items() if k.startswith("clip_model.")}
            self.model = module.build_model(dict(state)).float()
            self.model.load_state_dict(state, strict=True)
            self.model = self.model.eval().cuda()
            self.prompts = checkpoint["prompts_features"].float().cuda()
            self.dtype = torch.float32
            from torchvision.transforms import (
                CenterCrop,
                Compose,
                InterpolationMode,
                Normalize,
                Resize,
                ToTensor,
            )

            self.transform = Compose(
                [
                    Resize(224, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
        elif stage == "embedding":
            from transformers import Dinov2Model

            self.dtype = torch.float16
            self.model = (
                Dinov2Model.from_pretrained(
                    model_path, local_files_only=True, torch_dtype=self.dtype, attn_implementation="eager"
                )
                .eval()
                .cuda()
            )
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        else:
            raise ValueError("unknown model stage")
        torch.cuda.synchronize()  # Both models load in every arm's pool startup.

    def process_batched(self, batch):
        import torch

        # Group by resolution; preserve the original row positions after inference.
        sides = [224] * len(batch["id"]) if self.stage == "quality" else batch["input_side"]
        results = [None] * len(sides)
        for side in sorted(set(sides)):
            positions = [i for i, value in enumerate(sides) if value == side]
            tensors = []
            for i in positions:
                image = Image.fromarray(batch["pixels"][i])
                if self.stage == "quality":
                    tensor = self.transform(image)
                else:
                    image = image.resize((side, side), Image.Resampling.BICUBIC)
                    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)
                    tensor = ((tensor - self.mean) / self.std).contiguous()
                tensors.append(tensor)
            values = torch.stack(tensors).to(device="cuda", dtype=self.dtype)
            with torch.inference_mode():
                if self.stage == "quality":
                    features = torch.nn.functional.normalize(self.model.encode_image(values), dim=-1)
                    logits = (100 * features @ self.prompts.T).reshape(len(values), 7, 2)
                    prediction = logits.softmax(dim=-1)[..., 0].mean(dim=1)
                else:
                    prediction = self.model(pixel_values=values).pooler_output
            prediction = prediction.float().cpu().numpy()
            if not np.isfinite(prediction).all():
                raise ValueError("non-finite model output")
            for i, value in zip(positions, prediction.tolist()):
                results[i] = value
        return {"id": list(batch["id"]), self.stage: results}


@OPERATORS.register_module("ej_flow_pack")
class Pack(Mapper):
    _batched_op = True
    elastic_juicer_dag_safe = True

    def process_batched(self, batch):
        norms = []
        for score, embedding in zip(batch["quality"], batch["embedding"]):
            values = np.asarray(embedding, dtype=np.float32)
            if not np.isfinite(score) or values.shape != (1024,) or not np.isfinite(values).all():
                raise ValueError("invalid real model output")
            norms.append(float(np.linalg.norm(values)))
        return {"id": list(batch["id"]), "embedding_norm": norms}
