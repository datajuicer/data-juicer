# calibrate_response_mapper

Calibrate response in question-answer pairs based on reference text.

This mapper calibrates the 'response' part of a question-answer pair by using a reference text. It aims to make the response more detailed and accurate while ensuring it still answers the original question. The calibration process uses a default system prompt, which can be customized. The output is stripped of any leading or trailing whitespace.

基于参考文本校准问答对中的响应。

该映射器通过使用参考文本来校准问答对中的“响应”部分。它的目标是使响应更加详细和准确，同时确保它仍然能够回答原始问题。校准过程使用默认的系统提示，该提示可以自定义。输出去除了任何前导或尾随空白。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` | API model name. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the calibration task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `reference_template` | typing.Optional[str] | `None` | Template for formatting the reference text. |
| `qa_pair_template` | typing.Optional[str] | `None` | Template for formatting question-answer pairs. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression for parsing model output. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/calibrate_response_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_calibrate_response_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)