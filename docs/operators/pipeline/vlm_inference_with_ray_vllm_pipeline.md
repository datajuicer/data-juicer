# vlm_ray_vllm_engine_pipeline

Use `vlm_ray_vllm_engine_pipeline` as the operator name in YAML recipes. This registered name differs from the source filename.

YAML 菜谱中的算子名为 `vlm_ray_vllm_engine_pipeline`，与源码文件名不同。

Pipeline to generate response using vLLM engine on Ray. This pipeline leverages the vLLM engine for efficient large vision language model inference. More details about ray vLLM engine can be found at: https://docs.ray.io/en/latest/data/working-with-llms.html

使用 Ray 上的 vLLM 引擎生成响应的流水线。该流水线利用 vLLM 引擎实现高效的大视觉语言模型推理。有关 Ray vLLM 引擎的更多详情，请参见：https://docs.ray.io/en/latest/data/working-with-llms.html

Type 算子类型: **pipeline**

Tags 标签: gpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'Qwen/Qwen2.5-7B-Instruct'` | API or huggingface model name. |
| `is_hf_model` | <class 'bool'> | `True` |  |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for guiding the optimization task. |
| `accelerator_type` | typing.Optional[str] | `None` | The type of accelerator to use (e.g., "V100", "A100"). Default to None, meaning that only the CPU will be used. |
| `sampling_params` | typing.Optional[typing.Dict] | `None` | Sampling parameters for text generation (e.g., {'temperature': 0.9, 'top_p': 0.95}). |
| `engine_kwargs` | typing.Optional[typing.Dict] | `None` | The kwargs to pass to the vLLM engine. See documentation for details: https://docs.vllm.ai/en/latest/api/vllm/engine/arg_utils/#vllm.engine.arg_utils.AsyncEngineArgs. |
| `kwargs` |  | `''` | Extra keyword arguments. |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/pipeline/vlm_inference_with_ray_vllm_pipeline.py)
- [unit test 单元测试](../../../tests/ops/pipeline/test_vlm_inference_with_ray_vllm_pipeline.py)
- [Return operator list 返回算子列表](../../Operators.md)