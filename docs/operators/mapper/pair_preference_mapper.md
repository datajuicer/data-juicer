# pair_preference_mapper

Mapper to construct paired preference samples by generating a rejected response and its reason.

This operator uses an API model to generate a new response that is opposite in style, factuality, or stance to the original response. The generated response and the reason for its generation are stored in the sample. The default system prompt and input template are provided, but can be customized. The output is parsed using a regular expression to extract the new response and the reason. If parsing fails, the operator retries up to a specified number of times. The generated response and reason are stored in the sample under the keys 'rejected_response' and 'reason', respectively.

构造配对偏好样本的映射器，通过生成一个被拒绝的回答及其原因。

该算子使用API模型生成与原始回答在风格、事实性或立场上相反的新回答。生成的回答及其生成原因会被存储在样本中。提供了默认的系统提示和输入模板，但可以自定义。输出使用正则表达式解析以提取新回答和原因。如果解析失败，算子将重试指定次数。生成的回答和原因分别存储在样本的'rejected_response'和'reason'键下。

Type 算子类型: **mapper**

Tags 标签: cpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for guiding the generation task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. It must contain placeholders '{query}' and '{response}', and can optionally include '{reference}'. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing model output. |
| `rejected_key` | <class 'str'> | `'rejected_response'` | The field name in the sample to store the generated rejected response. Defaults to 'rejected_response'. |
| `reason_key` | <class 'str'> | `'reason'` | The field name in the sample to store the reason for generating the response. Defaults to 'reason'. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retries for the API call in case of response parsing failure. Defaults to 3. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/pair_preference_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_pair_preference_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)