# 格式转换工具

在这里，Data-Juicer 为各式各样的数据集提供了十数种格式转换工具，包括多模态数据集，后微调数据集等等。
这些工具能够将原始格式的数据集转换为Data-Juicer所需的统一中间格式（我们称之为"DJ格式"）。DJ算子的默认实现基于该格式进行设计，例如会直接从'text'字段读取数据载荷（payload）执行相应处理操作。对于特殊格式需求，用户既可以通过调整算子参数配置来适配，也可通过自定义算子实现进行扩展支持。

DJ 格式的一个示例如下所示：

```python
{
  // >>> 核心内容：文本，对话，......
  "text": "xxx",
  "query": "xxx",
  "response": "xxx",
  ......
  // <<< 核心内容

  // >>> 额外数据内容：多模态数据路径，......
  "images": [
    "path/to/the/image/of/antarctica_snowfield",
    "path/to/the/image/of/antarctica_map",
    "path/to/the/image/of/europe_map"
  ],
  "audios": [
    "path/to/the/audio/of/sound_of_waves_in_Antarctic_Ocean"
  ],
  "videos": [
    "path/to/the/video/of/remote_sensing_view_of_antarctica"
  ],
  // <<< 额外数据内容

  // >>> meta 信息和 stats，它们可能是数据集原生的，也可以由 Data-Juicer 产出
  "meta": {
    "src": "customized",
    "version": "0.1",
    "author": "xxx"
  },
  "stats": {
    "lang": "en",
    "image_widths": [224, 336, 512],
    ...
  },
  // <<< meta 信息和 stats
}
```

在 DJ 格式中大概包括三个部分：
1. 核心内容：例如 LLM 的预训练数据集中的文本内容，后微调数据集中的对话内容等。它们与数据集的下游使用的训练或者微调过程直接相关。
2. 额外数据内容：例如多模态数据集中的多模态数据路径。它们被组织为路径列表。
3. Meta 信息和 Stats：例如从原始数据集中继承而来的数据集版本或来源信息，或者由 Data-Juicer 的算子产出的类别 tags 和 stats 信息。

其中，第 2 和第 3 部分对于不同的数据集来说是通用的，而且都会被组织为几乎相同的结构。
作为对比，第 1 部分，也就是核心内容部分，对于各种数据集来说可能非常不同。
这里列举了针对不同种类数据集介绍这个部分更多细节的对应的文档：
- [多模态数据集](multimodal/README_ZH.md)
- [后微调数据集](post_tuning_dialog/README_ZH.md)