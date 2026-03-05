# optimize_prompt_mapper

Optimize prompts based on existing ones in the same batch.

This operator uses the existing prompts and newly optimized prompts as examples to generate better prompts. It supports using a Hugging Face model or an API for text generation. The operator can be configured to keep the original samples or replace them with the generated ones. The optimization process involves multiple retries if the generated prompt is empty. The operator operates in batch mode and can leverage vLLM for inference acceleration on CUDA devices.

- Uses existing and newly generated prompts to optimize future prompts.
- Supports both Hugging Face models and API-based text generation.
- Can keep or replace original samples with generated ones.
- Retries up to a specified number of times if the generated prompt is empty.
- Operates in batch mode and can use vLLM for acceleration on CUDA.
- References: https://doc.agentscope.io/v0/en/build_tutorial/prompt_optimization.html

根据同一批次中现有的提示进行优化。

该算子使用现有的提示和新优化的提示作为示例来生成更好的提示。它支持使用Hugging Face模型或API进行文本生成。可以配置该算子保留原始样本或将它们替换为生成的样本。如果生成的提示为空，则优化过程会涉及多次重试。该算子以批处理模式运行，并且可以在CUDA设备上利用vLLM进行推理加速。

- 使用现有的和新生成的提示来优化未来的提示。
- 支持Hugging Face模型和基于API的文本生成。
- 可以保留原始样本或将它们替换为生成的样本。
- 如果生成的提示为空，则最多重试指定次数。
- 以批处理模式运行，并且可以在CUDA上使用vLLM进行加速。
- 参考：https://doc.agentscope.io/v0/en/build_tutorial/prompt_optimization.html

Type 算子类型: **mapper**

Tags 标签: gpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'Qwen/Qwen2.5-7B-Instruct'` | API or huggingface model name. |
| `gen_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of new prompts to generate. |
| `max_example_num` | typing.Annotated[int, Gt(gt=0)] | `3` | Maximum number of example prompts to include as context when generating new optimized prompts. |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only generated texts in the final datasets and the original texts will be removed. It's True in default. |
| `retry_num` | <class 'int'> | `3` | how many times to retry to generate the prompt if the parsed generated prompt is empty. It's 3 in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for guiding the generation task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the input prompt. It must include one placeholder '{}', which will be replaced by `example_num` formatted examples defined by `example_template`. |
| `example_template` | typing.Optional[str] | `None` | Template for formatting one prompt example. It must include one placeholder '{}', which will be replaced by one formatted prompt. |
| `prompt_template` | typing.Optional[str] | `None` | Template for formatting a single prompt within each example. Must include two placeholders '{}' for the question and answer. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression pattern to extract questions and answers from model response. |
| `enable_vllm` | <class 'bool'> | `False` | Whether to use vllm for inference acceleration. |
| `is_hf_model` | <class 'bool'> | `False` | If true, use Transformers for loading hugging face or local llm. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the model. |
| `sampling_params` | typing.Optional[typing.Dict] | `None` | Sampling parameters for text generation. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_api_model
```python
OptimizePromptMapper(api_or_hf_model='qwen2.5-72b-instruct', gen_num=3, max_example_num=3, enable_vllm=False, is_hf_model=False, sampling_params={'max_new_tokens': 200})
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>prompt</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>Act as a math solution evaluator. Score the given solution on a scale of 0 to 1, where 1 means perfect logical reasoning and final answer correctness, and 0 means completely invalid. Focus on:\n1. Mathematical validity of steps\n2. Proper application of mathematical principles\n3. Final answer accuracy\n4. Logical flow and coherence\n5. Handling of edge cases and constraints\nand so on.\n\nReturn ONLY a numerical score between 0 and 1 (e.g., 0.85) without any explanation.</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>prompt</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>Act as a math solution evaluator. Score the given solution on a scale of 0 to 1, where 1 means perfect logical reasoning and final answer correctness, and 0 means completely invalid. Focus on:\n1. Mathematical validity of steps\n2. Proper application of mathematical principles\n3. Final answer accuracy\n4. Logical flow and coherence\n5. Handling of edge cases and constraints\nand so on.\n\nReturn ONLY a numerical score between 0 and 1 (e.g., 0.85) without any explanation.</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>prompt</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>Act as a math solution assessor. Evaluate the provided solution on a scale from 0 to 1, with 1 indicating flawless logic and correct final answers, and 0 indicating total invalidity. Consider the following aspects:\n1. Validity of mathematical steps\n2. Correct use of mathematical concepts\n3. Precision of the final answer\n4. Coherence and logical progression\n5. Management of special cases and constraints\nand other relevant factors.\n\nProvide ONLY a numeric score between 0 and 1 (e.g., 0.92)... [truncated, total 535 chars]</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>prompt</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>Act as a math solution reviewer. Assess the given solution on a scale from 0 to 1, where 1 represents flawless logical reasoning and accurate final answers, and 0 indicates complete invalidity. Focus on the following criteria:\n1. Validity of mathematical steps\n2. Proper application of mathematical principles\n3. Accuracy of the final answer\n4. Logical flow and coherence\n5. Handling of edge cases and constraints\nand other pertinent factors.\n\nReturn ONLY a numerical score between 0 and 1 (e... [truncated, total 535 chars]</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>prompt</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>Act as a math solution examiner. Evaluate the provided solution on a scale from 0 to 1, with 1 indicating flawless logical reasoning and correct final answers, and 0 indicating total invalidity. Consider the following aspects:\n1. Validity of mathematical steps\n2. Correct application of mathematical principles\n3. Accuracy of the final answer\n4. Logical flow and coherence\n5. Handling of special cases and constraints\nand other relevant factors.\n\nProvide ONLY a numeric score between 0 and 1 ... [truncated, total 547 chars]</td></tr></table></div></div>



## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/optimize_prompt_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_optimize_prompt_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)