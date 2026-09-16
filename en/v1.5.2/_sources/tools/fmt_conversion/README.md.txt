# Format Conversion Tools

Here Data-Juicer provides tens of format conversion tools for diverse datasets, including multimodal datasets, post tuning datasets, and so on.
These tools can convert raw datasets into a unified intermediate format required by Data-Juicer (referred to as "DJ Format"). The default implementation of DJ operators is designed based on this format - for example, directly reading data payloads from the 'text' field for processing operations. For special format requirements, users can either adjust operator parameter configurations for adaptation or implement custom operators to enable extended support.


An overview of DJ format is shown below:

```python
{
  // >>> core contents: texts, dialogs, ...
  "text": "xxx",
  "query": "xxx",
  "response": "xxx",
  ......
  // <<< core contents

  // >>> extra data contents: multimodal data paths, ...
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
  // <<< extra data contents

  // >>> meta infos and stats, which could be primitive or produced by Data-Juicer
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
  // <<< meta infos and stats
}
```

There are about three parts in DJ format:
1. Core contents: such as texts in the pretraining dataset of LLMs, dialogs in the post tuning dataset, and so on. They are directly related to the training or fine-tuning procedures in the downstream usage of the dataset.
2. Extra data contents: such as the paths to the multimodal data in the multimodal datasets. They are organized as path lists.
3. Meta infos & Stats: such as version or source information of the dataset that are inherent from the original dataset, or category tags and stats produced by OPs of Data-Juicer.

The 2nd and 3rd parts of them are common used and organized in nearly the same structures for diverse datasets.
As a contrast, the 1st part, which is the core contents, might be quite different for different kinds of datasets.
Here are the corresponding documents for different datasets that introduce more details about this part:
- [Multimodal datasets](multimodal/README.md)
- [Post Tuning](post_tuning_dialog/README.md)