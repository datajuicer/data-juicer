# DJ-Cookbook
## 资源合集
- [KDD'24 相关教程](https://datajuicer.github.io/data-juicer/en/main/_static/tutorial_kdd24.html)
- [Awesome LLM-Data](../awesome_llm_data.md)
- ["坏"数据展览](../BadDataExhibition_ZH.md)

## 编写Data-Juicer (DJ) 代码
- 基础
  - [DJ概览](../../README_ZH.md)
  - [快速上手](QuickStart_ZH.md)
  - [配置](https://datajuicer.github.io/data-juicer-hub/zh_CN/main/docs/RecipeGallery_ZH.html)
  - [数据格式转换](../../tools/fmt_conversion/README_ZH.md)
- 信息速查
  - [算子库](../Operators.md)
  - [API参考](https://datajuicer.github.io/data-juicer/zh_CN/main/api)
- 进阶
  - [开发者指南](../DeveloperGuide_ZH.md)
  - [预处理工具](../../tools/preprocess/README_ZH.md)
  - [后处理工具](../../tools/postprocess/README_ZH.md)
  - [沙盒](https://datajuicer.github.io/data-juicer-sandbox/zh_CN/main/index_ZH.html)
  - [API服务化](../DJ_service_ZH.md)
  - [给数据打分](../../data_juicer/tools/quality_classifier/README_ZH.md)
  - [自动评估](../../tools/evaluator/README_ZH.md)
  - [第三方集成](../../thirdparty/LLM_ecosystems/README_ZH.md)

## 用例与数据菜谱
* [数据菜谱Gallery](https://datajuicer.github.io/data-juicer-hub/zh_CN/main/docs/RecipeGallery_ZH.html)
  - Data-Juicer 最小示例配方
  - 复现开源文本数据集
  - 改进开源文本预训练数据集
  - 改进开源文本后处理数据集
  - 合成对比学习图像文本数据集
  - 改进开源图像文本数据集
  - 视频数据的基本示例菜谱
  - 合成以人为中心的视频评测集
  - 改进现有的开源视频数据集
* Data-Juicer相关竞赛
  - [Better Synth](https://tianchi.aliyun.com/competition/entrance/532251)，在DJ-沙盒实验室和多模态大模型上，探索大模型合成数据对图像理解能力的影响
  - [Modelscope-Sora挑战赛](https://tianchi.aliyun.com/competition/entrance/532219)，基于Data-Juicer和[EasyAnimate](https://github.com/aigc-apps/EasyAnimate)框架，调优文本-视频数据集，在类SORA小模型上训练以生成更好的视频
  - [Better Mixture](https://tianchi.aliyun.com/competition/entrance/532174)，针对指定多个候选数据集，仅调整数据混合和采样策略
  - FT-Data Ranker ([1B Track](https://tianchi.aliyun.com/competition/entrance/532157)、 [7B Track](https://tianchi.aliyun.com/competition/entrance/532158))，针对指定候选数据集，仅调整数据过滤和增强策略
  - [可图Kolors-LoRA风格故事挑战赛](https://tianchi.aliyun.com/competition/entrance/532254)，基于Data-Juicer和[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)框架，探索Diffusion模型微调
* [DJ-SORA](../DJ_SORA_ZH.md)
* 基于Data-Juicer和[AgentScope](https://github.com/agentscope-ai/agentscope)框架，通过[智能体调用DJ Filters](../../demos/api_service/react_data_filter_process.ipynb)和[调用DJ Mappers](../../demos/api_service/react_data_mapper_process.ipynb)
  


## 交互类示例
* Data-Juicer 介绍 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/overview_scan/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/overview_scan)]
* 数据可视化:
  * 基础指标统计 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_statistics)]
  * 词汇多样性 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_diversity)]
  * 算子洞察（单OP） [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visualization_op_insight/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_insight)]
  * 算子效果（多OP） [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_effect)]
* 数据处理:
  * 科学文献 (例如 [arXiv](https://info.arxiv.org/help/bulk_data_s3.html)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sci_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_sci_data)]
  * 编程代码 (例如 [TheStack](https://huggingface.co/datasets/bigcode/the-stack)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_code_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_code_data)]
  * 中文指令数据 (例如 [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sft_zh_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_cft_zh_data)]
* 工具池:
  * 按语言分割数据集 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_dataset_splitting_by_language/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_dataset_splitting_by_language)]
  * CommonCrawl 质量分类器 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_quality_classifier/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_quality_classifier)]
  * 基于 [HELM](https://github.com/stanford-crfm/helm) 的自动评测 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/auto_evaluation_helm/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/auto_evaluation_helm)]
  * 数据采样及混合 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_mixture/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_mixture)]
* 数据处理回路 [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_loop/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_loop)]

