# image_tagging_vlm_mapper

Mapper to generates image tags. This operator generates tags based on the content of given images. The tags are generated using a vlm model and stored in the specified field name. If the tags are already present in the sample, the operator skips processing.

用于生成图像标签的 Mapper。该算子根据给定图像的内容生成标签，使用 vlm 模型生成标签并存储到指定字段名中。如果样本中已存在标签，则跳过处理。

Type 算子类型: **mapper**

Tags 标签: gpu, api, vllm, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'Qwen/Qwen2.5-VL-7B-Instruct'` | API model name or HF model name. |
| `is_api_model` | <class 'bool'> | `False` | Whether the model is an API model. If true, use openai api to generate tags, otherwise use vllm. |
| `tag_field_name` | <class 'str'> | `'image_tags'` | the field name to store the tags. It's "image_tags" in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_api_model
```python
ImageTaggingVLMMapper(api_or_hf_model='qwen2.5-vl-3b-instruct', is_api_model=True, sampling_params={'max_new_tokens': 512})
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>images</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>__dj__meta__</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>{}</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>__dj__meta__</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>{}</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>images</th></tr><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>image_tags</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[[]]</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>__dj__meta__</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>image_tags</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>[[&#x27;public-transit&#x27;, &#x27;sw-manila&#x27;, &#x27;city-street&#x27;, &#x27;passenger-bus&#x27;, &#x27;bus-advertising&#x27;, &#x27;red-wine-drink&#x27;, &#x27;advertisements&#x27;, &#x27;urban-landscape&#x27;]]</td></tr></table></div></div>



## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_tagging_vlm_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_tagging_vlm_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)