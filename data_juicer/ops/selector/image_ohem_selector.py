"""Select hard image examples using a user supplied per-sample loss function."""

import heapq
import importlib.util
import os
from typing import Optional

import numpy as np

from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import load_image

from ..base_op import OPERATORS, Selector


@OPERATORS.register_module("image_ohem_selector")
class ImageOHEMSelector(Selector):
    """Keep the highest-loss image samples (Online Hard Example Mining).

    ``score_fn`` is supplied by the user and must have the signature
    ``score_fn(model, samples, images, device, **score_kwargs)``. It must return one
    numeric loss per sample in the batch. ``model_factory`` is optional and, when
    supplied, is called once as ``model_factory(**model_kwargs)``.
    """

    _accelerator = "cuda"

    def __init__(
        self,
        score_fn=None,
        model_factory=None,
        score_file: str = "",
        score_function: str = "score_fn",
        model_function: str = "model_factory",
        top_ratio: Optional[float] = None,
        topk: Optional[int] = None,
        batch_size: int = 8,
        image_key: str = "images",
        image_bytes_key: str = "",
        loss_field: str = "image_ohem_loss",
        device: str = "auto",
        model_kwargs: Optional[dict] = None,
        score_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        super().__init__(image_key=image_key, image_bytes_key=image_bytes_key, *args, **kwargs)
        if top_ratio is None and topk is None:
            raise ValueError("At least one of top_ratio or topk must be specified")
        if top_ratio is not None and not 0 <= top_ratio <= 1:
            raise ValueError("top_ratio must be in [0, 1]")
        if topk is not None and topk < 0:
            raise ValueError("topk must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.score_fn = score_fn
        self.model_factory = model_factory
        self.score_file = score_file
        self.score_function = score_function
        self.model_function = model_function
        self.top_ratio, self.topk = top_ratio, topk
        self.batch_size, self.loss_field = batch_size, loss_field
        self.device_name = device
        self.model_kwargs = model_kwargs or {}
        self.score_kwargs = score_kwargs or {}
        self._model = None
        if score_file:
            self._load_user_functions()
        if not callable(self.score_fn):
            raise ValueError("score_fn or a score_file containing a callable score_fn is required")

    def _load_user_functions(self):
        if not os.path.isfile(self.score_file) or not self.score_file.endswith(".py"):
            raise ValueError(f"score_file must be an existing Python file: {self.score_file}")
        spec = importlib.util.spec_from_file_location("data_juicer_image_ohem_user", self.score_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.score_fn = getattr(module, self.score_function, self.score_fn)
        self.model_factory = getattr(module, self.model_function, self.model_factory)

    def _device(self):
        if self.device_name != "auto":
            return self.device_name
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_model(self):
        if self._model is None and self.model_factory is not None:
            self._model = self.model_factory(**self.model_kwargs)
            if hasattr(self._model, "to"):
                self._model.to(self._device())
            if hasattr(self._model, "eval"):
                self._model.eval()
        return self._model

    def _images(self, sample):
        paths = sample.get(self.image_key, []) or []
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        byte_values = sample.get(self.image_bytes_key, []) if self.image_bytes_key else []
        if byte_values and len(byte_values) == len(paths):
            return [load_image(value) for value in byte_values]
        return [load_image(path) for path in paths]

    def process(self, dataset):
        if not len(dataset):
            return dataset
        model, device = self._ensure_model(), self._device()
        losses = []
        for start in range(0, len(dataset), self.batch_size):
            samples = dataset.select(range(start, min(start + self.batch_size, len(dataset)))).to_list()
            images = [self._images(sample) for sample in samples]
            batch_losses = self.score_fn(model, samples, images, device, **self.score_kwargs)
            if hasattr(batch_losses, "detach"):
                batch_losses = batch_losses.detach().cpu().reshape(-1).tolist()
            batch_losses = np.asarray(batch_losses, dtype=float).reshape(-1).tolist()
            if len(batch_losses) != len(samples):
                raise ValueError("score_fn must return exactly one loss per sample")
            losses.extend(batch_losses)

        def attach_loss(sample, index):
            stats = dict(sample.get(Fields.stats, {}))
            stats[self.loss_field] = float(losses[index])
            sample[Fields.stats] = stats
            return sample

        dataset = dataset.map(attach_loss, with_indices=True)
        select_num = len(dataset)
        if self.top_ratio is not None:
            select_num = int(self.top_ratio * len(dataset))
        if self.topk is not None:
            select_num = min(select_num, self.topk)
        if select_num <= 0:
            return dataset.select([])
        values = [float(sample[Fields.stats][self.loss_field]) for sample in dataset]
        if not np.isfinite(values).all():
            raise ValueError("score_fn returned NaN or infinite loss")
        indices = heapq.nlargest(select_num, range(len(dataset)), values.__getitem__)
        return dataset.select(indices)
