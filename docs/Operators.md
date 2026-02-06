
# Operator Schemas 算子提要

Operators are a collection of basic processes that assist in data modification,
cleaning, filtering, deduplication, etc. We support a wide range of data
sources and file formats, and allow for flexible extension to custom datasets.

算子 (Operator) 是协助数据修改、清理、过滤、去重等基本流程的集合。我们支持广泛的数据来源和文件格式，并支持对自定义数据集的灵活扩展。

This page offers a basic description of the operators (OPs) in Data-Juicer.
Users can consult the
[API documentation](https://datajuicer.github.io/data-juicer/en/main/api.html)
for the operator API reference. To learn more about each operator, click its
adjacent 'info' link to access the operator's details page, which includes its
detailed parameters, effect demonstrations, and links to relevant unit tests
and source code.

Additionally, the 'Reference' column in the table is intended to cite research,
libraries, or resource links that the operator's design or implementation is
based on. We welcome contributions of known or relevant reference sources to
enrich this section.

Users can also refer to and run the unit tests (`tests/ops/...`) for
[examples of operator-wise usage](../tests/ops) as well as the effects of each
operator when applied to built-in test data samples. Besides, you can try to
use agent to automatically route suitable OPs and call them. E.g., refer to
[Agentic Filters of DJ](../demos/api_service/react_data_filter_process.ipynb), [Agentic Mappers of DJ](../demos/api_service/react_data_mapper_process.ipynb)

这个页面提供了Data-Juicer中算子的基本描述。算子的API参考，用户可以直接查阅[API文档](https://datajuicer.github.io/data-juicer/en/main/api.html)。
要详细了解每个算子，请点击其旁的info链接进入算子详情页，其中包含了算子参数、效果演示，以及相关单元测试和源码的链接。

此外，表格中的『参考』（Reference）列则用于注明算子设计或实现所依据的研究、库或资料链接，欢迎您提供已知或相关的参考来源，共同完善此部分内容。

用户还可以查看、运行单元测试 (`tests/ops/...`)，来体验[各OP的用法示例](../tests/ops)以及每个OP作用于内置测试数据样本时的效果。例如，参考[Agentic Filters of DJ](../demos/api_service/react_data_filter_process.ipynb), [Agentic Mappers of DJ](../demos/api_service/react_data_mapper_process.ipynb)


## Overview  概览

The operators in Data-Juicer are categorized into 8 types.
Data-Juicer 中的算子分为以下 8 种类型。

| Type 类型 | Number 数量 | Description 描述 |
|------|:------:|-------------|
| [aggregator](#aggregator) | 4 | Aggregate for batched samples, such as summary or conclusion. 对批量样本进行汇总，如得出总结或结论。 |
| [deduplicator](#deduplicator) | 10 | Detects and removes duplicate samples. 识别、删除重复样本。 |
| [filter](#filter) | 56 | Filters out low-quality samples. 过滤低质量样本。 |
| [formatter](#formatter) | 8 | Discovers, loads, and canonicalizes source data. 发现、加载、规范化原始数据。 |
| [grouper](#grouper) | 3 | Group samples to batched samples. 将样本分组，每一组组成一个批量样本。 |
| [mapper](#mapper) | 102 | Edits and transforms samples. 对数据样本进行编辑和转换。 |
| [pipeline](#pipeline) | 3 | Applies dataset-level processing; both input and output are datasets. 执行数据集级别的操作，输入和输出均为完整数据集。 |
| [selector](#selector) | 5 | Selects top samples based on ranking. 基于排序选取高质量样本。 |

All the specific operators are listed below, each featured with several capability tags. 
下面列出所有具体算子，每种算子都通过多个标签来注明其主要功能。
* Modality Tags
  - 🔤Text: process text data specifically. 专用于处理文本。
  - 🏞Image: process image data specifically. 专用于处理图像。
  - 📣Audio: process audio data specifically. 专用于处理音频。
  - 🎬Video: process video data specifically. 专用于处理视频。
  - 🔮Multimodal: process multimodal data. 用于处理多模态数据。
* Resource Tags
  - 💻CPU: only requires CPU resource. 只需要 CPU 资源。
  - 🚀GPU: requires GPU/CUDA resource as well. 额外需要 GPU/CUDA 资源。
* Usability Tags
  - 🔴Alpha: alpha version OP. Only the basic OP implementations are finished. 表示 alpha 版本算子。只完成了基础的算子实现。
  - 🟡Beta: beta version OP. Based on the alpha version, unittests for this OP are added as well. 表示 beta 版本算子。基于 alpha 版本，添加了算子的单元测试。
  - 🟢Stable: stable version OP. Based on the beta version, OP optimizations related to DJ (e.g. model management, batched processing, OP fusion, ...) are added to this OP. 表示 stable 版本算子。基于 beta 版本，完善了DJ相关的算子优化项（如模型管理，批处理，算子融合等）。
* Model Tags
  - 🔗API: equipped with API-based models. (e.g. ChatGPT, GPT-4o). 支持基于 API 调用模型（如 ChatGPT，GPT-4o）。
  - 🌊vLLM: equipped with models supported by vLLM. 支持基于 vLLM 进行模型推理。
  - 🧩HF: equipped with models from HuggingFace Hub. 支持来自于 HuggingFace Hub 的模型。

## aggregator <a name="aggregator"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| entity_attribute_aggregator | 💻CPU 🔗API 🟢Stable | Summarizes a given attribute of an entity from a set of documents. 汇总一组文档中实体的给定属性。 | [info](operators/aggregator/entity_attribute_aggregator.md) | - |
| meta_tags_aggregator | 💻CPU 🔗API 🟢Stable | Merge similar meta tags into a single, unified tag. 将类似的元标记合并到一个统一的标记中。 | [info](operators/aggregator/meta_tags_aggregator.md) | - |
| most_relevant_entities_aggregator | 💻CPU 🔗API 🟢Stable | Extracts and ranks entities closely related to a given entity from provided texts. 从提供的文本中提取与给定实体密切相关的实体并对其进行排名。 | [info](operators/aggregator/most_relevant_entities_aggregator.md) | - |
| nested_aggregator | 🔤Text 💻CPU 🔗API 🟢Stable | Aggregates nested content from multiple samples into a single summary. 将多个示例中的嵌套内容聚合到单个摘要中。 | [info](operators/aggregator/nested_aggregator.md) | - |

## deduplicator <a name="deduplicator"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| document_deduplicator | 🔤Text 💻CPU 🟢Stable | Deduplicates samples at the document level using exact matching. 使用完全匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/document_deduplicator.md) | - |
| document_minhash_deduplicator | 🔤Text 💻CPU 🟢Stable | Deduplicates samples at the document level using MinHash LSH. 使用MinHash LSH在文档级别删除重复样本。 | [info](operators/deduplicator/document_minhash_deduplicator.md) | - |
| document_simhash_deduplicator | 🔤Text 💻CPU 🟢Stable | Deduplicates samples at the document level using SimHash. 使用SimHash在文档级别删除重复的样本。 | [info](operators/deduplicator/document_simhash_deduplicator.md) | - |
| image_deduplicator | 🏞Image 💻CPU 🟢Stable | Deduplicates samples at the document level by exact matching of images. 通过图像的精确匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/image_deduplicator.md) | - |
| ray_basic_deduplicator | 💻CPU 🔴Alpha | Backend for deduplicator. deduplicator的后端。 | - | - |
| ray_bts_minhash_deduplicator | 🔤Text 💻CPU 🟡Beta | A distributed implementation of Union-Find with load balancing. 具有负载平衡的Union-Find的分布式实现。 | [info](operators/deduplicator/ray_bts_minhash_deduplicator.md) | - |
| ray_document_deduplicator | 🔤Text 💻CPU 🟡Beta | Deduplicates samples at the document level using exact matching in Ray distributed mode. 在Ray分布式模式下使用精确匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/ray_document_deduplicator.md) | - |
| ray_image_deduplicator | 🏞Image 💻CPU 🟡Beta | Deduplicates samples at the document level using exact matching of images in Ray distributed mode. 在光线分布模式下使用图像的精确匹配在文档级别删除重复样本。 | [info](operators/deduplicator/ray_image_deduplicator.md) | - |
| ray_video_deduplicator | 🎬Video 💻CPU 🟡Beta | Deduplicates samples at document-level using exact matching of videos in Ray distributed mode. 在Ray分布式模式下使用视频的精确匹配在文档级删除重复样本。 | [info](operators/deduplicator/ray_video_deduplicator.md) | - |
| video_deduplicator | 🎬Video 💻CPU 🟢Stable | Deduplicates samples at the document level using exact matching of videos. 使用视频的精确匹配在文档级别删除重复的样本。 | [info](operators/deduplicator/video_deduplicator.md) | - |

## filter <a name="filter"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| alphanumeric_filter | 🔤Text 💻CPU 🧩HF 🟢Stable | Filter to keep samples with an alphabet/numeric ratio within a specific range. 过滤器，以保持具有特定范围内的字母/数字比率的样本。 | [info](operators/filter/alphanumeric_filter.md) | - |
| audio_duration_filter | 📣Audio 💻CPU 🟢Stable | Keep data samples whose audio durations are within a specified range. 保留音频持续时间在指定范围内的数据样本。 | [info](operators/filter/audio_duration_filter.md) | - |
| audio_nmf_snr_filter | 📣Audio 💻CPU 🟢Stable | Keep data samples whose audio Signal-to-Noise Ratios (SNRs) are within a specified range. 保留音频信噪比 (snr) 在指定范围内的数据样本。 | [info](operators/filter/audio_nmf_snr_filter.md) | - |
| audio_size_filter | 📣Audio 💻CPU 🟢Stable | Keep data samples based on the size of their audio files. 根据音频文件的大小保留数据样本。 | [info](operators/filter/audio_size_filter.md) | - |
| average_line_length_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with average line length within a specific range. 过滤器，以保持平均线长度在特定范围内的样本。 | [info](operators/filter/average_line_length_filter.md) | - |
| character_repetition_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with character-level n-gram repetition ratio within a specific range. 过滤器将具有字符级n-gram重复比的样本保持在特定范围内。 | [info](operators/filter/character_repetition_filter.md) | - |
| flagged_words_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with flagged-word ratio in a specified range. 过滤器将标记词比率的样本保留在指定范围内。 | [info](operators/filter/flagged_words_filter.md) | - |
| general_field_filter | 💻CPU 🟡Beta | Filter to keep samples based on a general field filter condition. 根据常规字段筛选条件保留样本。 | [info](operators/filter/general_field_filter.md) | - |
| image_aesthetics_filter | 🏞Image 🚀GPU 🧩HF 🟢Stable | Filter to keep samples with aesthetics scores within a specific range. 过滤以保持美学分数在特定范围内的样品。 | [info](operators/filter/image_aesthetics_filter.md) | - |
| image_aspect_ratio_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with image aspect ratio within a specific range. 过滤器，以保持样本的图像纵横比在特定范围内。 | [info](operators/filter/image_aspect_ratio_filter.md) | - |
| image_face_count_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with the number of faces within a specific range. 过滤以保持样本的面数在特定范围内。 | [info](operators/filter/image_face_count_filter.md) | - |
| image_face_ratio_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with face area ratios within a specific range. 过滤以保持面面积比在特定范围内的样本。 | [info](operators/filter/image_face_ratio_filter.md) | - |
| image_nsfw_filter | 🏞Image 🚀GPU 🧩HF 🟢Stable | Filter to keep samples whose images have nsfw scores in a specified range. 过滤器保留其图像的nsfw分数在指定范围内的样本。 | [info](operators/filter/image_nsfw_filter.md) | - |
| image_pair_similarity_filter | 🏞Image 🚀GPU 🧩HF 🟢Stable | Filter to keep image pairs with similarities between images within a specific range. 过滤器将图像之间具有相似性的图像对保持在特定范围内。 | [info](operators/filter/image_pair_similarity_filter.md) | - |
| image_shape_filter | 🏞Image 💻CPU 🟢Stable | Filter to keep samples with image shape (width, height) within specific ranges. 过滤器，以保持样本的图像形状 (宽度，高度) 在特定的范围内。 | [info](operators/filter/image_shape_filter.md) | - |
| image_size_filter | 🏞Image 💻CPU 🟢Stable | Keep data samples whose image size (in Bytes/KB/MB/...) is within a specific range. 保留图像大小 (以字节/KB/MB/... 为单位) 在特定范围内的数据样本。 | [info](operators/filter/image_size_filter.md) | - |
| image_subplot_filter | 🏞Image 💻CPU 🟡Beta | Filter to detect and remove samples with images containing subplots. 过滤器来检测和删除包含子图的图像样本。 | [info](operators/filter/image_subplot_filter.md) | - |
| image_text_matching_filter | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Filter to keep samples with image-text matching scores within a specific range. 过滤器将图像文本匹配分数的样本保持在特定范围内。 | [info](operators/filter/image_text_matching_filter.md) | - |
| image_text_similarity_filter | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Filter to keep samples with image-text similarity within a specified range. 过滤器将具有图像-文本相似性的样本保持在指定范围内。 | [info](operators/filter/image_text_similarity_filter.md) | - |
| image_watermark_filter | 🏞Image 🚀GPU 🧩HF 🟢Stable | Filter to keep samples whose images have no watermark with high probability. 过滤器以保持其图像没有水印的样本具有高概率。 | [info](operators/filter/image_watermark_filter.md) | - |
| in_context_influence_filter | 🚀GPU 🟢Stable | Filter to keep texts based on their in-context influence on a validation set. 过滤以根据文本在上下文中对验证集的影响来保留文本。 | [info](operators/filter/in_context_influence_filter.md) | - |
| instruction_following_difficulty_filter | 🚀GPU 🟡Beta | Filter to keep texts based on their instruction following difficulty (IFD, https://arxiv.org/abs/2308.12032) score. 过滤以保持文本基于他们的指令跟随难度 (IFD， https://arxiv.org/abs/ 2308.12032) 分数。 | [info](operators/filter/instruction_following_difficulty_filter.md) | - |
| language_id_score_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples in a specific language with a confidence score above a threshold. 过滤器以保留置信度高于阈值的特定语言的样本。 | [info](operators/filter/language_id_score_filter.md) | - |
| llm_analysis_filter | 🚀GPU 🌊vLLM 🧩HF 🔗API 🟡Beta | Base filter class for leveraging LLMs to analyze and filter data samples. 用于利用LLMs分析和过滤数据样本的基本筛选器类。 | [info](operators/filter/llm_analysis_filter.md) | - |
| llm_difficulty_score_filter | 💻CPU 🟡Beta | Filter to keep samples with high difficulty scores estimated by an LLM. 过滤器以保留由LLM估计的高难度分数的样本。 | [info](operators/filter/llm_difficulty_score_filter.md) | - |
| llm_perplexity_filter | 🚀GPU 🧩HF 🟡Beta | Filter to keep samples with perplexity scores within a specified range, computed using a specified LLM. 过滤器将困惑分数的样本保留在指定范围内，使用指定的LLM计算。 | [info](operators/filter/llm_perplexity_filter.md) | - |
| llm_quality_score_filter | 💻CPU 🟡Beta | Filter to keep samples with a high quality score estimated by a language model. 过滤器，以保留具有语言模型估计的高质量分数的样本。 | [info](operators/filter/llm_quality_score_filter.md) | - |
| llm_task_relevance_filter | 💻CPU 🟡Beta | Filter to keep samples with high relevance scores to validation tasks estimated by an LLM. 过滤器以保留与LLM估计的验证任务具有高相关性分数的样本。 | [info](operators/filter/llm_task_relevance_filter.md) | - |
| maximum_line_length_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with a maximum line length within a specified range. 筛选器将最大行长度的样本保持在指定范围内。 | [info](operators/filter/maximum_line_length_filter.md) | - |
| perplexity_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with perplexity score in a specified range. 过滤以保持困惑分数在指定范围内的样本。 | [info](operators/filter/perplexity_filter.md) | - |
| phrase_grounding_recall_filter | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Filter to keep samples based on the phrase grounding recall of phrases extracted from text in images. 根据从图像中的文本中提取的短语接地召回来过滤以保留样本。 | [info](operators/filter/phrase_grounding_recall_filter.md) | - |
| special_characters_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with special-character ratio within a specific range. 过滤器，以将具有特殊字符比率的样本保持在特定范围内。 | [info](operators/filter/special_characters_filter.md) | - |
| specified_field_filter | 💻CPU 🟢Stable | Filter samples based on the specified field information. 根据指定的字段信息筛选样本。 | [info](operators/filter/specified_field_filter.md) | - |
| specified_numeric_field_filter | 💻CPU 🟢Stable | Filter samples based on a specified numeric field value. 根据指定的数值字段值筛选样本。 | [info](operators/filter/specified_numeric_field_filter.md) | - |
| stopwords_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with stopword ratio within a specified range. 过滤器将停止词比率的样本保持在指定范围内。 | [info](operators/filter/stopwords_filter.md) | - |
| suffix_filter | 💻CPU 🟢Stable | Filter to keep samples with specified suffix. 过滤器以保留具有指定后缀的样本。 | [info](operators/filter/suffix_filter.md) | - |
| text_action_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep texts that contain a minimum number of actions. 过滤以保留包含最少数量操作的文本。 | [info](operators/filter/text_action_filter.md) | - |
| text_embd_similarity_filter | 🔤Text 🚀GPU 🔗API 🟡Beta | Filter to keep texts whose average embedding similarity to a set of given validation texts falls within a specific range. 过滤器，以保留与一组给定验证文本的平均嵌入相似度在特定范围内的文本。 | [info](operators/filter/text_embd_similarity_filter.md) | - |
| text_entity_dependency_filter | 🔤Text 💻CPU 🟢Stable | Identify and filter text samples based on entity dependencies. 根据实体依赖关系识别和过滤文本样本。 | [info](operators/filter/text_entity_dependency_filter.md) | - |
| text_length_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with total text length within a specific range. 过滤以保持文本总长度在特定范围内的样本。 | [info](operators/filter/text_length_filter.md) | - |
| text_pair_similarity_filter | 🔤Text 🚀GPU 🧩HF 🟢Stable | Filter to keep text pairs with similarities within a specific range. 过滤以将具有相似性的文本对保持在特定范围内。 | [info](operators/filter/text_pair_similarity_filter.md) | - |
| token_num_filter | 🔤Text 💻CPU 🧩HF 🟢Stable | Filter to keep samples with a total token number within a specified range. 筛选器将总令牌数的样本保留在指定范围内。 | [info](operators/filter/token_num_filter.md) | - |
| video_aesthetics_filter | 🎬Video 🚀GPU 🧩HF 🟢Stable | Filter to keep data samples with aesthetics scores for specified frames in the videos within a specific range. 过滤器将视频中指定帧的美学得分数据样本保留在特定范围内。 | [info](operators/filter/video_aesthetics_filter.md) | - |
| video_aspect_ratio_filter | 🎬Video 💻CPU 🟢Stable | Filter to keep samples with video aspect ratio within a specific range. 过滤器将视频纵横比的样本保持在特定范围内。 | [info](operators/filter/video_aspect_ratio_filter.md) | - |
| video_duration_filter | 🎬Video 💻CPU 🟢Stable | Keep data samples whose videos' durations are within a specified range. 保留视频持续时间在指定范围内的数据样本。 | [info](operators/filter/video_duration_filter.md) | - |
| video_frames_text_similarity_filter | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Filter to keep samples based on the similarity between video frame images and text within a specific range. 根据视频帧图像和文本之间的相似性进行过滤，以保持样本在特定范围内。 | [info](operators/filter/video_frames_text_similarity_filter.md) | - |
| video_motion_score_filter | 🎬Video 💻CPU 🟢Stable | Filter to keep samples with video motion scores from OpenCV within a specific range. 过滤器将来自OpenCV的视频运动分数的样本保持在特定范围内。 | [info](operators/filter/video_motion_score_filter.md) | - |
| video_motion_score_ptlflow_filter | 🎬Video 🚀GPU 🟡Beta | Filter to keep samples with video motion scores from ptlflow within a specified range. 过滤器以将来自ptlflow的具有视频运动分数的样本保持在指定范围内。 | [info](operators/filter/video_motion_score_ptlflow_filter.md) | - |
| video_motion_score_raft_filter | 🎬Video 🚀GPU 🟢Stable | Filter to keep samples with video motion scores within a specified range. 过滤器将视频运动分数的样本保持在指定范围内。 | [info](operators/filter/video_motion_score_raft_filter.md) | [RAFT](https://arxiv.org/abs/2003.12039) |
| video_nsfw_filter | 🎬Video 🚀GPU 🧩HF 🟢Stable | Filter to keep samples whose videos have nsfw scores in a specified range. 过滤器以保留其视频的nsfw分数在指定范围内的样本。 | [info](operators/filter/video_nsfw_filter.md) | - |
| video_ocr_area_ratio_filter | 🎬Video 🚀GPU 🟢Stable | Keep data samples whose detected text area ratios for specified frames in the video are within a specified range. 保留检测到的视频中指定帧的文本面积比率在指定范围内的数据样本。 | [info](operators/filter/video_ocr_area_ratio_filter.md) | - |
| video_resolution_filter | 🎬Video 💻CPU 🟢Stable | Keep data samples whose videos' resolutions are within a specified range. 保留视频分辨率在指定范围内的数据样本。 | [info](operators/filter/video_resolution_filter.md) | - |
| video_tagging_from_frames_filter | 🎬Video 🚀GPU 🟢Stable | Filter to keep samples whose videos contain specified tags. 过滤器以保留其视频包含指定标签的样本。 | [info](operators/filter/video_tagging_from_frames_filter.md) | - |
| video_watermark_filter | 🎬Video 🚀GPU 🧩HF 🟢Stable | Filter to keep samples whose videos have no watermark with high probability. 过滤器以保持其视频具有高概率没有水印的样本。 | [info](operators/filter/video_watermark_filter.md) | - |
| word_repetition_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with word-level n-gram repetition ratio within a specific range. 过滤器将单词级n-gram重复比率的样本保持在特定范围内。 | [info](operators/filter/word_repetition_filter.md) | - |
| words_num_filter | 🔤Text 💻CPU 🟢Stable | Filter to keep samples with a total word count within a specified range. 过滤器将样本的总字数保持在指定范围内。 | [info](operators/filter/words_num_filter.md) | - |

## formatter <a name="formatter"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| csv_formatter | 🟢Stable | The class is used to load and format csv-type files. 类用于加载和格式化csv类型的文件。 | [info](operators/formatter/csv_formatter.md) | - |
| empty_formatter | 🟢Stable | The class is used to create empty data. 类用于创建空数据。 | [info](operators/formatter/empty_formatter.md) | - |
| json_formatter | 🟡Beta | The class is used to load and format json-type files. 类用于加载和格式化json类型的文件。 | [info](operators/formatter/json_formatter.md) | - |
| local_formatter | 🟢Stable | The class is used to load a dataset from local files or local directory. 类用于从本地文件或本地目录加载数据集。 | - | - |
| parquet_formatter | 🟢Stable | The class is used to load and format parquet-type files. 该类用于加载和格式化镶木地板类型的文件。 | [info](operators/formatter/parquet_formatter.md) | - |
| remote_formatter | 🟢Stable | The class is used to load a dataset from repository of huggingface hub. 该类用于从huggingface hub的存储库加载数据集。 | - | - |
| text_formatter | 🟡Beta | The class is used to load and format text-type files. 类用于加载和格式化文本类型文件。 | [info](operators/formatter/text_formatter.md) | - |
| tsv_formatter | 🟢Stable | The class is used to load and format tsv-type files. 该类用于加载和格式化tsv类型的文件。 | [info](operators/formatter/tsv_formatter.md) | - |

## grouper <a name="grouper"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| key_value_grouper | 🔤Text 💻CPU 🟢Stable | Groups samples into batches based on values in specified keys. 根据指定键中的值将样本分组为批处理。 | [info](operators/grouper/key_value_grouper.md) | - |
| naive_grouper | 💻CPU 🟢Stable | Group all samples in a dataset into a single batched sample. 将数据集中的所有样本分组为单个批处理样本。 | [info](operators/grouper/naive_grouper.md) | - |
| naive_reverse_grouper | 💻CPU 🟢Stable | Split batched samples into individual samples. 将批处理的样品分成单个样品。 | [info](operators/grouper/naive_reverse_grouper.md) | - |

## mapper <a name="mapper"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| audio_add_gaussian_noise_mapper | 📣Audio 💻CPU 🟡Beta | Mapper to add Gaussian noise to audio samples. 映射器将高斯噪声添加到音频样本。 | [info](operators/mapper/audio_add_gaussian_noise_mapper.md) | - |
| audio_ffmpeg_wrapped_mapper | 📣Audio 💻CPU 🟢Stable | Wraps FFmpeg audio filters for processing audio files in a dataset. 包装FFmpeg音频过滤器，用于处理数据集中的音频文件。 | [info](operators/mapper/audio_ffmpeg_wrapped_mapper.md) | - |
| calibrate_qa_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Calibrates question-answer pairs based on reference text using an API model. 使用API模型根据参考文本校准问答对。 | [info](operators/mapper/calibrate_qa_mapper.md) | - |
| calibrate_query_mapper | 💻CPU 🟢Stable | Calibrate query in question-answer pairs based on reference text. 基于参考文本校准问答对中的查询。 | [info](operators/mapper/calibrate_query_mapper.md) | - |
| calibrate_response_mapper | 💻CPU 🟢Stable | Calibrate response in question-answer pairs based on reference text. 根据参考文本校准问答对中的回答。 | [info](operators/mapper/calibrate_response_mapper.md) | - |
| chinese_convert_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to convert Chinese text between Traditional, Simplified, and Japanese Kanji. 映射器在繁体、简体和日文汉字之间转换中文文本。 | [info](operators/mapper/chinese_convert_mapper.md) | - |
| clean_copyright_mapper | 🔤Text 💻CPU 🟢Stable | Cleans copyright comments at the beginning of text samples. 清除文本示例开头的版权注释。 | [info](operators/mapper/clean_copyright_mapper.md) | - |
| clean_email_mapper | 🔤Text 💻CPU 🟢Stable | Cleans email addresses from text samples using a regular expression. 使用正则表达式从文本示例中清除电子邮件地址。 | [info](operators/mapper/clean_email_mapper.md) | - |
| clean_html_mapper | 🔤Text 💻CPU 🟢Stable | Cleans HTML code from text samples, converting HTML to plain text. 从文本示例中清除HTML代码，将HTML转换为纯文本。 | [info](operators/mapper/clean_html_mapper.md) | - |
| clean_ip_mapper | 🔤Text 💻CPU 🟢Stable | Cleans IPv4 and IPv6 addresses from text samples. 从文本示例中清除IPv4和IPv6地址。 | [info](operators/mapper/clean_ip_mapper.md) | - |
| clean_links_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to clean links like http/https/ftp in text samples. 映射器来清理链接，如文本示例中的http/https/ftp。 | [info](operators/mapper/clean_links_mapper.md) | - |
| detect_character_attributes_mapper | 🚀GPU 🟡Beta | Takes an image, a caption, and main character names as input to extract the characters' attributes. 根据给定的图像、图像描述信息和（多个）角色名称，提取图像中主要角色的属性。 | [info](operators/mapper/detect_character_attributes_mapper.md) | [DetailMaster](https://arxiv.org/abs/2505.16915) |
| detect_character_locations_mapper | 🚀GPU 🟡Beta | Given an image and a list of main character names, extract the bounding boxes for each present character. 给定一张图像和主要角色的名称列表，提取每个在场角色的边界框。(YOLOE + MLLM) | [info](operators/mapper/detect_character_locations_mapper.md) | [DetailMaster](https://arxiv.org/abs/2505.16915) |
| detect_main_character_mapper | 🚀GPU 🟡Beta | Extract all main character names based on the given image and its caption. 根据给定的图像及其图像描述，提取所有主要角色的名字。 | [info](operators/mapper/detect_main_character_mapper.md) | [DetailMaster](https://arxiv.org/abs/2505.16915) |
| dialog_intent_detection_mapper | 💻CPU 🔗API 🟢Stable | Generates user's intent labels in a dialog by analyzing the history, query, and response. 通过分析历史记录、查询和响应，在对话框中生成用户的意图标签。 | [info](operators/mapper/dialog_intent_detection_mapper.md) | - |
| dialog_sentiment_detection_mapper | 💻CPU 🔗API 🟢Stable | Generates sentiment labels and analysis for user queries in a dialog. 在对话框中为用户查询生成情绪标签和分析。 | [info](operators/mapper/dialog_sentiment_detection_mapper.md) | - |
| dialog_sentiment_intensity_mapper | 💻CPU 🔗API 🟢Stable | Mapper to predict user's sentiment intensity in a dialog, ranging from -5 to 5. Mapper预测用户在对话框中的情绪强度，范围从-5到5。 | [info](operators/mapper/dialog_sentiment_intensity_mapper.md) | - |
| dialog_topic_detection_mapper | 💻CPU 🔗API 🟢Stable | Generates user's topic labels and analysis in a dialog. 在对话框中生成用户的主题标签和分析。 | [info](operators/mapper/dialog_topic_detection_mapper.md) | - |
| download_file_mapper | 💻CPU 🟡Beta | Mapper to download URL files to local files or load them into memory. 映射器将URL文件下载到本地文件或将其加载到内存中。 | [info](operators/mapper/download_file_mapper.md) | - |
| expand_macro_mapper | 🔤Text 💻CPU 🟢Stable | Expands macro definitions in the document body of LaTeX samples. 展开LaTeX示例文档主体中的宏定义。 | [info](operators/mapper/expand_macro_mapper.md) | - |
| extract_entity_attribute_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extracts attributes for given entities from the text and stores them in the sample's metadata. 从文本中提取给定实体的属性，并将其存储在示例的元数据中。 | [info](operators/mapper/extract_entity_attribute_mapper.md) | - |
| extract_entity_relation_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extracts entities and relations from text to build a knowledge graph. 从文本中提取实体和关系以构建知识图谱。 | [info](operators/mapper/extract_entity_relation_mapper.md) | - |
| extract_event_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extracts events and relevant characters from the text. 从文本中提取事件和相关字符。 | [info](operators/mapper/extract_event_mapper.md) | - |
| extract_keyword_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Generate keywords for the text. 为文本生成关键字。 | [info](operators/mapper/extract_keyword_mapper.md) | - |
| extract_nickname_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extracts nickname relationships in the text using a language model. 使用语言模型提取文本中的昵称关系。 | [info](operators/mapper/extract_nickname_mapper.md) | - |
| extract_support_text_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Extracts a supporting sub-text from the original text based on a given summary. 根据给定的摘要从原始文本中提取支持子文本。 | [info](operators/mapper/extract_support_text_mapper.md) | - |
| extract_tables_from_html_mapper | 🔤Text 💻CPU 🟡Beta | Extracts tables from HTML content and stores them in a specified field. 从HTML内容中提取表并将其存储在指定字段中。 | [info](operators/mapper/extract_tables_from_html_mapper.md) | - |
| fix_unicode_mapper | 🔤Text 💻CPU 🟢Stable | Fixes unicode errors in text samples. 修复文本示例中的unicode错误。 | [info](operators/mapper/fix_unicode_mapper.md) | - |
| generate_qa_from_examples_mapper | 🚀GPU 🌊vLLM 🧩HF 🟢Stable | Generates question and answer pairs from examples using a Hugging Face model. 使用拥抱面部模型从示例生成问题和答案对。 | [info](operators/mapper/generate_qa_from_examples_mapper.md) | - |
| generate_qa_from_text_mapper | 🔤Text 🚀GPU 🌊vLLM 🧩HF 🟢Stable | Generates question and answer pairs from text using a specified model. 使用指定的模型从文本生成问题和答案对。 | [info](operators/mapper/generate_qa_from_text_mapper.md) | - |
| image_blur_mapper | 🏞Image 💻CPU 🟢Stable | Blurs images in the dataset with a specified probability and blur type. 使用指定的概率和模糊类型对数据集中的图像进行模糊处理。 | [info](operators/mapper/image_blur_mapper.md) | - |
| image_captioning_from_gpt4v_mapper | 🔮Multimodal 💻CPU 🟡Beta | Generates text captions for images using the GPT-4 Vision model. 使用GPT-4视觉模型为图像生成文本标题。 | [info](operators/mapper/image_captioning_from_gpt4v_mapper.md) | - |
| image_captioning_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Generates image captions using a Hugging Face model and appends them to samples. 使用拥抱面部模型生成图像标题，并将其附加到样本中。 | [info](operators/mapper/image_captioning_mapper.md) | - |
| image_detection_yolo_mapper | 🏞Image 🚀GPU 🟡Beta | Perform object detection using YOLO on images and return bounding boxes and class labels. 使用YOLO对图像执行对象检测，并返回边界框和类标签。 | [info](operators/mapper/image_detection_yolo_mapper.md) | - |
| image_diffusion_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Generate images using a diffusion model based on provided captions. 使用基于提供的字幕的扩散模型生成图像。 | [info](operators/mapper/image_diffusion_mapper.md) | - |
| image_face_blur_mapper | 🏞Image 💻CPU 🟢Stable | Mapper to blur faces detected in images. 映射器模糊图像中检测到的人脸。 | [info](operators/mapper/image_face_blur_mapper.md) | - |
| image_mmpose_mapper | 🏞Image 🚀GPU 🟡Beta | Mapper to perform human keypoint detection inference using MMPose models. Mapper使用MMPose模型执行人体关键点检测推断。 | [info](operators/mapper/image_mmpose_mapper.md) | - |
| image_remove_background_mapper | 🏞Image 💻CPU 🟢Stable | Mapper to remove the background of images. 映射器删除图像的背景。 | [info](operators/mapper/image_remove_background_mapper.md) | - |
| image_sam_3d_body_mapper | 🏞Image 🚀GPU 🟡Beta | SAM 3D Body (3DB) is a promptable model for single-image full-body 3D human mesh recovery (HMR). SAM 3D Body (3DB) 是用于单图像全身3D人体网格恢复 (HMR) 的可提示模型。 | [info](operators/mapper/image_sam_3d_body_mapper.md) | - |
| image_segment_mapper | 🏞Image 🚀GPU 🟢Stable | Perform segment-anything on images and return the bounding boxes. 对图像执行segment-任何操作并返回边界框。 | [info](operators/mapper/image_segment_mapper.md) | - |
| image_tagging_mapper | 🏞Image 🚀GPU 🟢Stable | Generates image tags for each image in the sample. 为样本中的每个图像生成图像标记。 | [info](operators/mapper/image_tagging_mapper.md) | - |
| image_tagging_vlm_mapper | 🔮Multimodal 🚀GPU 🔗API 🌊vLLM 🟡Beta | Mapper to generates image tags. 映射器生成图像标签。 | [info](operators/mapper/image_tagging_vlm_mapper.md) | - |
| imgdiff_difference_area_generator_mapper | 🚀GPU 🟡Beta | Generates and filters bounding boxes for image pairs based on similarity, segmentation, and text matching. 根据相似性、分割和文本匹配生成和过滤图像对的边界框。 | [info](operators/mapper/imgdiff_difference_area_generator_mapper.md) | [ImgDiff](https://arxiv.org/abs/2408.04594) |
| imgdiff_difference_caption_generator_mapper | 🚀GPU 🟡Beta | Generates difference captions for bounding box regions in two images. 为两个图像中的边界框区域生成差异字幕。 | [info](operators/mapper/imgdiff_difference_caption_generator_mapper.md) | [ImgDiff](https://arxiv.org/abs/2408.04594) |
| mllm_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to use MLLMs for visual question answering tasks. Mapper使用MLLMs进行视觉问答任务。 | [info](operators/mapper/mllm_mapper.md) | - |
| nlpaug_en_mapper | 🔤Text 💻CPU 🟢Stable | Augments English text samples using various methods from the nlpaug library. 使用nlpaug库中的各种方法增强英语文本样本。 | [info](operators/mapper/nlpaug_en_mapper.md) | - |
| nlpcda_zh_mapper | 🔤Text 💻CPU 🟢Stable | Augments Chinese text samples using the nlpcda library. 使用nlpcda库扩充中文文本样本。 | [info](operators/mapper/nlpcda_zh_mapper.md) | - |
| optimize_prompt_mapper | 🚀GPU 🌊vLLM 🧩HF 🔗API 🟡Beta | Optimize prompts based on existing ones in the same batch. 根据同一批次中的现有提示优化提示。 | [info](operators/mapper/optimize_prompt_mapper.md) | - |
| optimize_qa_mapper | 🚀GPU 🌊vLLM 🧩HF 🔗API 🟢Stable | Mapper to optimize question-answer pairs. 映射器来优化问题-答案对。 | [info](operators/mapper/optimize_qa_mapper.md) | - |
| optimize_query_mapper | 🚀GPU 🟢Stable | Optimize queries in question-answer pairs to make them more specific and detailed. 优化问答对中的查询，使其更加具体和详细。 | [info](operators/mapper/optimize_query_mapper.md) | - |
| optimize_response_mapper | 🚀GPU 🟢Stable | Optimize response in question-answer pairs to be more detailed and specific. 优化问答对中的响应，使其更加详细和具体。 | [info](operators/mapper/optimize_response_mapper.md) | - |
| pair_preference_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Mapper to construct paired preference samples by generating a rejected response and its reason. Mapper通过生成拒绝响应及其原因来构造成对的偏好样本。 | [info](operators/mapper/pair_preference_mapper.md) | - |
| punctuation_normalization_mapper | 🔤Text 💻CPU 🟢Stable | Normalizes unicode punctuations to their English equivalents in text samples. 将unicode标点规范化为文本示例中的英语等效项。 | [info](operators/mapper/punctuation_normalization_mapper.md) | - |
| python_file_mapper | 💻CPU 🟢Stable | Executes a Python function defined in a file on input data. 对输入数据执行文件中定义的Python函数。 | [info](operators/mapper/python_file_mapper.md) | - |
| python_lambda_mapper | 💻CPU 🟢Stable | Mapper for applying a Python lambda function to data samples. Mapper，用于将Python lambda函数应用于数据样本。 | [info](operators/mapper/python_lambda_mapper.md) | - |
| query_intent_detection_mapper | 🚀GPU 🧩HF 🧩HF 🟢Stable | Predicts the user's intent label and corresponding score for a given query. 为给定查询预测用户的意图标签和相应的分数。 | [info](operators/mapper/query_intent_detection_mapper.md) | - |
| query_sentiment_detection_mapper | 🚀GPU 🧩HF 🧩HF 🟢Stable | Predicts user's sentiment label ('negative', 'neutral', 'positive') in a query. 在查询中预测用户的情绪标签 (“负面” 、 “中性” 、 “正面”)。 | [info](operators/mapper/query_sentiment_detection_mapper.md) | - |
| query_topic_detection_mapper | 🚀GPU 🧩HF 🧩HF 🟢Stable | Predicts the topic label and its corresponding score for a given query. 预测给定查询的主题标签及其相应的分数。 | [info](operators/mapper/query_topic_detection_mapper.md) | - |
| relation_identity_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Identify the relation between two entities in a given text. 确定给定文本中两个实体之间的关系。 | [info](operators/mapper/relation_identity_mapper.md) | - |
| remove_bibliography_mapper | 🔤Text 💻CPU 🟢Stable | Removes bibliography sections at the end of LaTeX documents. 删除LaTeX文档末尾的参考书目部分。 | [info](operators/mapper/remove_bibliography_mapper.md) | - |
| remove_comments_mapper | 🔤Text 💻CPU 🟢Stable | Removes comments from documents, currently supporting only 'tex' format. 从文档中删除注释，当前仅支持 “文本” 格式。 | [info](operators/mapper/remove_comments_mapper.md) | - |
| remove_header_mapper | 🔤Text 💻CPU 🟢Stable | Removes headers at the beginning of documents in LaTeX samples. 删除LaTeX示例中文档开头的标题。 | [info](operators/mapper/remove_header_mapper.md) | - |
| remove_long_words_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove long words within a specific range. 映射器删除特定范围内的长词。 | [info](operators/mapper/remove_long_words_mapper.md) | - |
| remove_non_chinese_character_mapper | 🔤Text 💻CPU 🟢Stable | Removes non-Chinese characters from text samples. 从文本样本中删除非中文字符。 | [info](operators/mapper/remove_non_chinese_character_mapper.md) | - |
| remove_repeat_sentences_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove repeat sentences in text samples. 映射器删除文本样本中的重复句子。 | [info](operators/mapper/remove_repeat_sentences_mapper.md) | - |
| remove_specific_chars_mapper | 🔤Text 💻CPU 🟢Stable | Removes specific characters from text samples. 从文本示例中删除特定字符。 | [info](operators/mapper/remove_specific_chars_mapper.md) | - |
| remove_table_text_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove table texts from text samples. 映射器从文本样本中删除表文本。 | [info](operators/mapper/remove_table_text_mapper.md) | - |
| remove_words_with_incorrect_substrings_mapper | 🔤Text 💻CPU 🟢Stable | Mapper to remove words containing specified incorrect substrings. 映射程序删除包含指定的不正确子字符串的单词。 | [info](operators/mapper/remove_words_with_incorrect_substrings_mapper.md) | - |
| replace_content_mapper | 🔤Text 💻CPU 🟢Stable | Replaces content in the text that matches a specific regular expression pattern with a designated replacement string. 用指定的替换字符串替换与特定正则表达式模式匹配的文本中的内容。 | [info](operators/mapper/replace_content_mapper.md) | - |
| s3_download_file_mapper | 💻CPU 🟡Beta | Mapper to download files from S3 to local files or load them into memory. Mapper将文件从S3下载到本地文件或将其加载到内存中。 | [info](operators/mapper/s3_download_file_mapper.md) | - |
| s3_upload_file_mapper | 💻CPU 🟡Beta | Mapper to upload local files to S3 and update paths to S3 URLs. Mapper将本地文件上传到S3并更新S3 url的路径。 | [info](operators/mapper/s3_upload_file_mapper.md) | - |
| sdxl_prompt2prompt_mapper | 🔤Text 🚀GPU 🟢Stable | Generates pairs of similar images using the SDXL model. 使用SDXL模型生成成对的相似图像。 | [info](operators/mapper/sdxl_prompt2prompt_mapper.md) | - |
| sentence_augmentation_mapper | 🔤Text 🚀GPU 🧩HF 🟢Stable | Augments sentences by generating enhanced versions using a Hugging Face model. 通过使用拥抱面部模型生成增强版本来增强句子。 | [info](operators/mapper/sentence_augmentation_mapper.md) | - |
| sentence_split_mapper | 🔤Text 💻CPU 🟢Stable | Splits text samples into individual sentences based on the specified language. 根据指定的语言将文本样本拆分为单个句子。 | [info](operators/mapper/sentence_split_mapper.md) | - |
| text_chunk_mapper | 🔤Text 💻CPU 🔗API 🟢Stable | Split input text into chunks based on specified criteria. 根据指定的条件将输入文本拆分为块。 | [info](operators/mapper/text_chunk_mapper.md) | - |
| text_tagging_by_prompt_mapper | 🔤Text 🚀GPU 🌊vLLM 🧩HF 🟡Beta | Mapper to generate text tags using prompt with LLM. Mapper使用带有LLM的prompt生成文本标记。 | [info](operators/mapper/text_tagging_by_prompt_mapper.md) | - |
| vggt_mapper | 🎬Video 🚀GPU 🟡Beta | Input a video of a single scene, and use VGGT to extract information including Camera Pose, Depth Maps, Point Maps, and 3D Point Tracks. 输入单个场景的视频，并使用VGGT提取包括相机姿态、深度图、点图和3D点轨迹的信息。 | [info](operators/mapper/vggt_mapper.md) | - |
| video_camera_calibration_static_deepcalib_mapper | 🎬Video 🚀GPU 🟡Beta | Compute the camera intrinsics and field of view (FOV) for a static camera using DeepCalib. 使用DeepCalib计算静态摄像机的摄像机内部和视场 (FOV)。 | [info](operators/mapper/video_camera_calibration_static_deepcalib_mapper.md) | - |
| video_camera_calibration_static_moge_mapper | 🎬Video 🚀GPU 🟡Beta | Compute the camera intrinsics and field of view (FOV) for a static camera using Moge-2 (more accurate than DeepCalib). 使用Moge-2 (比DeepCalib更准确) 计算静态摄像机的摄像机内部函数和视场 (FOV)。 | [info](operators/mapper/video_camera_calibration_static_moge_mapper.md) | - |
| video_captioning_from_audio_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to caption a video according to its audio streams based on Qwen-Audio model. 映射器根据基于qwen-audio模型的音频流为视频添加字幕。 | [info](operators/mapper/video_captioning_from_audio_mapper.md) | - |
| video_captioning_from_frames_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Generates video captions from sampled frames using an image-to-text model. 使用图像到文本模型从采样帧生成视频字幕。 | [info](operators/mapper/video_captioning_from_frames_mapper.md) | - |
| video_captioning_from_summarizer_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Mapper to generate video captions by summarizing several kinds of generated texts (captions from video/audio/frames, tags from audio/frames, ...). 映射器通过总结几种生成的文本 (来自视频/音频/帧的字幕，来自音频/帧的标签，...) 来生成视频字幕。 | [info](operators/mapper/video_captioning_from_summarizer_mapper.md) | - |
| video_captioning_from_video_mapper | 🔮Multimodal 🚀GPU 🧩HF 🟢Stable | Generates video captions using a Hugging Face video-to-text model and sampled video frames. 使用拥抱面部视频到文本模型和采样视频帧生成视频字幕。 | [info](operators/mapper/video_captioning_from_video_mapper.md) | - |
| video_captioning_from_vlm_mapper | 🔮Multimodal 🚀GPU 🌊vLLM 🧩HF 🟡Beta | Generates video captions using a VLM that accepts videos as inputs. 使用接受视频作为输入的VLM生成视频字幕。 | [info](operators/mapper/video_captioning_from_vlm_mapper.md) | - |
| video_depth_estimation_mapper | 🎬Video 🚀GPU 🟡Beta | Perform depth estimation on the video. 对视频进行深度估计。 | [info](operators/mapper/video_depth_estimation_mapper.md) | - |
| video_extract_frames_mapper | 🔮Multimodal 💻CPU 🟢Stable | Mapper to extract frames from video files according to specified methods. 映射器根据指定的方法从视频文件中提取帧。 | [info](operators/mapper/video_extract_frames_mapper.md) | - |
| video_face_blur_mapper | 🎬Video 💻CPU 🟢Stable | Mapper to blur faces detected in videos. 映射器模糊在视频中检测到的人脸。 | [info](operators/mapper/video_face_blur_mapper.md) | - |
| video_ffmpeg_wrapped_mapper | 🎬Video 💻CPU 🟢Stable | Wraps FFmpeg video filters for processing video files in a dataset. 包装FFmpeg视频过滤器，用于处理数据集中的视频文件。 | [info](operators/mapper/video_ffmpeg_wrapped_mapper.md) | - |
| video_hand_reconstruction_hawor_mapper | 🎬Video 🚀GPU 🟡Beta | Use HaWoR and MoGe-2 for hand reconstruction. 使用HaWoR和MoGe-2进行手部重建。 | - | - |
| video_hand_reconstruction_mapper | 🎬Video 🚀GPU 🟡Beta | Use the WiLoR model for hand localization and reconstruction. 使用WiLoR模型进行手部定位和重建。 | [info](operators/mapper/video_hand_reconstruction_mapper.md) | - |
| video_object_segmenting_mapper | 🎬Video 🚀GPU 🧩HF 🟡Beta | Text-guided semantic segmentation of valid objects throughout the video (YOLOE + SAM2). 在整个视频中对有效对象进行文本引导的语义分割 (YOLOE SAM2)。 | [info](operators/mapper/video_object_segmenting_mapper.md) | - |
| video_remove_watermark_mapper | 🎬Video 💻CPU 🟢Stable | Remove watermarks from videos based on specified regions. 根据指定区域从视频中删除水印。 | [info](operators/mapper/video_remove_watermark_mapper.md) | - |
| video_resize_aspect_ratio_mapper | 🎬Video 💻CPU 🟢Stable | Resizes videos to fit within a specified aspect ratio range. 调整视频大小以适应指定的宽高比范围。 | [info](operators/mapper/video_resize_aspect_ratio_mapper.md) | - |
| video_resize_resolution_mapper | 🎬Video 💻CPU 🟢Stable | Resizes video resolution based on specified width and height constraints. 根据指定的宽度和高度限制调整视频分辨率。 | [info](operators/mapper/video_resize_resolution_mapper.md) | - |
| video_split_by_duration_mapper | 🔮Multimodal 💻CPU 🟢Stable | Splits videos into segments based on a specified duration. 根据指定的持续时间将视频拆分为多个片段。 | [info](operators/mapper/video_split_by_duration_mapper.md) | - |
| video_split_by_key_frame_mapper | 🔮Multimodal 💻CPU 🟢Stable | Splits a video into segments based on key frames. 根据关键帧将视频分割为多个片段。 | [info](operators/mapper/video_split_by_key_frame_mapper.md) | - |
| video_split_by_scene_mapper | 🔮Multimodal 💻CPU 🟢Stable | Splits videos into scene clips based on detected scene changes. 根据检测到的场景变化将视频拆分为场景剪辑。 | [info](operators/mapper/video_split_by_scene_mapper.md) | - |
| video_tagging_from_audio_mapper | 🎬Video 🚀GPU 🧩HF 🟢Stable | Generates video tags from audio streams using the Audio Spectrogram Transformer. 使用音频频谱图转换器从音频流生成视频标签。 | [info](operators/mapper/video_tagging_from_audio_mapper.md) | - |
| video_tagging_from_frames_mapper | 🎬Video 🚀GPU 🟢Stable | Generates video tags from frames extracted from videos. 从视频中提取的帧生成视频标签。 | [info](operators/mapper/video_tagging_from_frames_mapper.md) | - |
| video_undistort_mapper | 🎬Video 💻CPU 🟡Beta | Undistort raw videos with corresponding camera intrinsics and distortion coefficients. 使用相应的相机固有特性和失真系数对原始视频进行失真。 | [info](operators/mapper/video_undistort_mapper.md) | - |
| video_whole_body_pose_estimation_mapper | 🎬Video 🚀GPU 🟡Beta | Input a video containing people, and use the DWPose model to extract the body, hand, feet, and face keypoints of the human subjects in the video, i.e., 2D Whole-body Pose Estimation. 输入包含人的视频，并使用DWPose模型来提取视频中人类主体的身体、手、脚和面部关键点，即2D全身姿态估计。 | [info](operators/mapper/video_whole_body_pose_estimation_mapper.md) | - |
| whitespace_normalization_mapper | 🔤Text 💻CPU 🟢Stable | Normalizes various types of whitespace characters to standard spaces in text samples. 将文本样本中各种类型的空白字符规范化为标准空格。 | [info](operators/mapper/whitespace_normalization_mapper.md) | - |

## pipeline <a name="pipeline"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| llm_inference_with_ray_vllm_pipeline | 🚀GPU 🟡Beta | Pipeline to generate response using vLLM engine on Ray. 使用Ray上的vLLM引擎生成响应的管道。 | - | - |
| ray_vllm_pipeline | 🚀GPU 🟡Beta | Pipeline for Ray vLLM engine. Ray vLLM引擎的管道。 | - | - |
| vlm_inference_with_ray_vllm_pipeline | 🏞Image 🚀GPU 🟡Beta | Pipeline to generate response using vLLM engine on Ray. 使用Ray上的vLLM引擎生成响应的管道。 | - | - |

## selector <a name="selector"/>

| Operator 算子 | Tags 标签 | Description 描述 | Details 详情 | Reference 参考 |
|----------|------|-------------|-------------|-------------|
| frequency_specified_field_selector | 💻CPU 🟢Stable | Selector to filter samples based on the frequency of a specified field. 选择器根据指定字段的频率过滤样本。 | [info](operators/selector/frequency_specified_field_selector.md) | - |
| random_selector | 💻CPU 🟢Stable | Randomly selects a subset of samples from the dataset. 从数据集中随机选择样本子集。 | [info](operators/selector/random_selector.md) | - |
| range_specified_field_selector | 💻CPU 🟢Stable | Selects a range of samples based on the sorted values of a specified field. 根据指定字段的排序值选择采样范围。 | [info](operators/selector/range_specified_field_selector.md) | - |
| tags_specified_field_selector | 💻CPU 🟢Stable | Selector to filter samples based on the tags of a specified field. 选择器根据指定字段的标签过滤样本。 | [info](operators/selector/tags_specified_field_selector.md) | - |
| topk_specified_field_selector | 💻CPU 🟢Stable | Selects top samples based on the sorted values of a specified field. 根据指定字段的排序值选择顶部样本。 | [info](operators/selector/topk_specified_field_selector.md) | - |


## Contributing  贡献

We welcome contributions of adding new operators. Please refer to [How-to Guide
for Developers](DeveloperGuide.md).

我们欢迎社区贡献新的算子，具体请参考[开发者指南](DeveloperGuide_ZH.md)。
