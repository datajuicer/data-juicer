from typing import Optional

from data_juicer.ops.base_op import Pipeline
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.ray_utils import is_ray_mode

ray = LazyLoader("ray")


class RayVLLMEnginePipeline(Pipeline):
    """Pipeline for Ray vLLM engine."""

    _accelerator = "cuda"

    def __init__(
        self,
        accelerator_type: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.
        :param accelerator_type: The type of accelerator to use (e.g., "V100", "A100").
            Default to None, meaning that only the CPU will be used.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.accelerator_type = accelerator_type

        assert is_ray_mode(), "Ray vLLM engine only works in Ray mode."

        from ray.llm._internal.serve.core.configs.llm_config import GPUType

        if self.accelerator_type:
            all_accelerator_types = [t.value for t in GPUType]
            assert self.accelerator_type in all_accelerator_types, (
                f"Unsupported accelerator type: {self.accelerator_type}. "
                f"Supported types are: {all_accelerator_types}"
            )

    def run(self, dataset: "ray.data.Dataset", *, exporter=None, tracer=None, reduce=True) -> "ray.data.Dataset":
        raise NotImplementedError
