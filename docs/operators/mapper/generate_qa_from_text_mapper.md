# generate_qa_from_text_mapper

Generates question and answer pairs from text using a specified model.

This operator supports local Hugging Face and vLLM models and API models. Each generated QA pair becomes a row with the original input fields and the generated `query` and `response`. The default parser understands the `Human:` and `Assistant:` format used by PAI Doc2QA models.

使用指定模型从文本生成问题和答案对。

该算子支持本地 Hugging Face、vLLM 模型和 API 模型。每对生成的问答对应一条记录，包含原始输入字段和生成的 `query`、`response`。默认解析器识别 PAI Doc2QA 模型使用的 `Human:` 和 `Assistant:` 格式。

Type 算子类型: **mapper**

Tags 标签: gpu, api, vllm, hf, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'alibaba-pai/pai-qwen1_5-7b-doc2qa'` | Huggingface model ID. |
| `max_num` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | The max num of returned QA sample for each text. Not limit if it is None. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression pattern to extract questions and answers from model response. |
| `enable_vllm` | <class 'bool'> | `False` | Whether to use vllm for inference acceleration. |
| `api_model` | typing.Optional[str] | `None` | API model name. Selects API inference when set. |
| `api_endpoint` | typing.Optional[str] | `None` | API URL or path. Defaults to `/chat/completions`. |
| `response_path` | typing.Optional[str] | `None` | Dot-separated path to text in the API response. |
| `system_prompt` | typing.Optional[str] | `None` | Instructions for the model. API mode supplies a QA format prompt by default. Pass an empty string to omit it. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the model. |
| `sampling_params` | typing.Optional[typing.Dict] | `None` | Sampling parameters for text generation, e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. The default data format parsed by this interface is as follows: Model Input:     蒙古国的首都是乌兰巴托（Ulaanbaatar）     冰岛的首都是雷克雅未克（Reykjavik） Model Output:     蒙古国的首都是乌兰巴托（Ulaanbaatar）     冰岛的首都是雷克雅未克（Reykjavik）     Human: 请问蒙古国的首都是哪里？     Assistant: 你好，根据提供的信息，蒙古国的首都是乌兰巴托（Ulaanbaatar）。     Human: 冰岛的首都是哪里呢？     Assistant: 冰岛的首都是雷克雅未克（Reykjavik）。     ... |


## Examples 配置示例

For a local instruction model, set a system prompt that matches the parser's expected format. You can also set `enable_vllm: true` to run the local model with vLLM.

使用本地指令模型时，设置与解析格式一致的系统提示词。也可以通过 `enable_vllm: true` 使用 vLLM 推理。

```yaml
process:
  - generate_qa_from_text_mapper:
      hf_model: Qwen/Qwen2.5-7B-Instruct
      system_prompt: >-
        Generate question-answer pairs grounded in the source text.
        Write each question after Human: and its answer after Assistant:.
      max_num: 2
```

For an OpenAI-compatible chat API, set `api_model` to a model served by your endpoint. Configure the base URL with `OPENAI_BASE_URL` or `model_params.base_url`, and supply credentials through the provider environment. API mode adds QA format instructions by default.

使用兼容 OpenAI 的聊天接口时，将 `api_model` 设为接口提供的模型名。可通过 `OPENAI_BASE_URL` 或 `model_params.base_url` 配置基础地址，并通过服务商环境变量提供凭据。API 模式默认会附加问答格式说明。

```yaml
process:
  - generate_qa_from_text_mapper:
      api_model: my-chat-model
      model_params:
        base_url: http://localhost:8000/v1
      sampling_params:
        max_tokens: 512
      max_num: 2
```

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/generate_qa_from_text_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_generate_qa_from_text_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)