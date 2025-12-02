# 后微调工具

对于 后微调 数据格式，我们主要考虑 4 种格式来覆盖支持 [ModelScope-Swift](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md) 和 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) :

- Swift的 Messages 格式（与LLaMA-Factory的 ShareGPT 格式几乎一致，采用了略微不同的key字段命名）：

```python
{
  "messages": [
    {
      "role": "system",
      "content": "<system>"
    },
    {
      "role": "user",
      "content": "<query1>"
    },
    {
      "role": "assistant",
      "content": "<response1>"
    },
    {
      "role": "user",
      "content": "<query2>"
    },
    {
      "role": "assistant",
      "content": "<response2>"
    }
  ]
}
```

- Swift的 ShareGPT 格式：

```python
{
  "system": "<system>",
  "conversation": [
    {
      "human": "<query1>",
      "assistant": "<response1>"
    },
    {
      "human": "<query2>",
      "assistant": "<response2>"
    }
  ]
}
```

- Alpaca 格式 （在Swift和LLaMA-Factory中定义一致）：

```python
{
  "system": "<system>",
  "instruction": "<query-inst>",
  "input": "<query-input>",
  "output": "<response>"
}
```

- Swift的Query-Response 格式：

```python
{
  "system": "<system>",
  "query": "<query2>",
  "response": "<response2>",
  "history": [
    [
      "<query1>",
      "<response1>"
    ]
  ]
}
```

在 Data-Juicer 中，我们预设了一些字段来对齐最后两种格式（Alpaca和Query-Response），并将如下格式作为 后微调对话 数据集的统一中间表示。
相应地，我们提供了若干内置工具将其他格式的数据集转换为 DJ 格式以及反向转换。


- DJ的多轮对话缺省格式（DJ post-tuning算子实现时假设基于该格式进行字段解析和处理）:

```python
{
  "system": "<system>",
  "instruction": "<query-inst>",
  "query": "<query2>",
  "response": "<response2>",
  "history": [
    [
      "<query1>",
      "<response1>"
    ]
  ]
}
```

## 用法

对于所有工具，您可以运行以下命令来查看它们的用法：

```shell
# e.g. messages_to_dj.py
python tools/fmt_conversion/post_tuning_dialog/source_format_to_data_juicer_format/messages_to_dj.py --help
```

对于从源格式转换到Data-Juicer格式，您可以使用`source_format_to_data_juicer_format`文件夹中的工具。
对于从Data-Juicer格式转换回目标格式，您可以使用`data_juicer_format_to_target_format`文件夹中的工具。
