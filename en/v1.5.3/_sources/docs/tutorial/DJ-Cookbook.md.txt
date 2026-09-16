# DJ-Cookbook

## Curated Resources
- [KDD-Tutorial](https://datajuicer.github.io/data-juicer/en/main/_static/tutorial_kdd24.html)
- [Awesome LLM-Data](../awesome_llm_data.md)
- ["Bad" Data Exhibition](../BadDataExhibition.md)


## Coding with Data-Juicer (DJ)
- Basics
  - [Overview of DJ](../../README.md)
  - [Quick Start](QuickStart.md)
  - [Configuration](https://datajuicer.github.io/data-juicer-hub/en/main/docs/RecipeGallery.html)
  - [Data Format Conversion](../../tools/fmt_conversion/README.md)
- Lookup Materials
  - [DJ OperatorZoo](../Operators.md)
  - [API references](https://datajuicer.github.io/data-juicer/en/main/api)
- Advanced
  - [Developer Guide](../DeveloperGuide.md)
  - [Preprocess Tools](../../tools/preprocess/README.md)
  - [Postprocess Tools](../../tools/postprocess/README.md)
  - [Sandbox](https://datajuicer.github.io/data-juicer-sandbox/en/main/index.html)
  - [API Service](../DJ_service.md)
  - [Data Scoring](../../data_juicer/tools/quality_classifier/README.md)
  - [Auto Evaluation](../../tools/evaluator/README.md)
  - [Third-parties Integration](../../thirdparty/LLM_ecosystems/README.md)


## Use Cases & Data Recipes
- [Data Recipe Gallery](https://datajuicer.github.io/data-juicer-hub/en/main/docs/RecipeGallery.html)
  - Data-Juicer Minimal Example Recipe
  - Reproducing Open Source Text Datasets
  - Improving Open Source Pre-training Text Datasets
  - Improving Open Source Post-tuning Text Datasets
  - Synthetic Contrastive Learning Image-text Datasets
  - Improving Open Source Image-text Datasets
  - Basic Example Recipes for Video Data
  - Synthesizing Human-centric Video Benchmarks
  - Improving Existing Open Source Video Datasets
- Data-Juicer related Competitions
  - [Better Synth](https://tianchi.aliyun.com/competition/entrance/532251), explore the impact of large model synthetic data on image understanding ability with DJ-Sandbox Lab and multimodal large models
  - [Modelscope-Sora Challenge](https://tianchi.aliyun.com/competition/entrance/532219), based on Data-Juicer and [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) framework,  optimize data and train SORA-like small models to generate better videos
  - [Better Mixture](https://tianchi.aliyun.com/competition/entrance/532174), only adjust data mixing and sampling strategies for given multiple candidate datasets
  - FT-Data Ranker ([1B Track](https://tianchi.aliyun.com/competition/entrance/532157), [7B Track](https://tianchi.aliyun.com/competition/entrance/532158)), For a specified candidate dataset, only adjust the data filtering and enhancement strategies
  - [Kolors-LoRA Stylized Story Challenge](https://tianchi.aliyun.com/competition/entrance/532254), based on Data-Juicer and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) framework, explore Diffusion model fine-tuning
- [DJ-SORA](../DJ_SORA.md)
- Based on Data-Juicer and [AgentScope](https://github.com/agentscope-ai/agentscope) framework, leverage [agents to call DJ Filters](../../demos/api_service/react_data_filter_process.ipynb) and [call DJ Mappers](../../demos/api_service/react_data_mapper_process.ipynb)


## Interactive Examples
- Introduction to Data-Juicer [[ModelScope](https://modelscope.cn/studios/Data-Juicer/overview_scan/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/overview_scan)]
- Data Visualization:
  - Basic Statistics [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_statistics/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_statistics)]
  - Lexical Diversity [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_diversity/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_diversity)]
  - Operator Insight (Single OP) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visualization_op_insight/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_insight)]
  - Operator Effect (Multiple OPs) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_visulization_op_effect/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_visualization_op_effect)]
- Data Processing:
  - Scientific Literature (e.g. [arXiv](https://info.arxiv.org/help/bulk_data_s3.html)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sci_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_sci_data)]
  - Programming Code (e.g. [TheStack](https://huggingface.co/datasets/bigcode/the-stack)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_code_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_code_data)]
  - Chinese Instruction Data (e.g. [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/process_sft_zh_data/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/process_cft_zh_data)]
- Tool Pool:
  - Dataset Splitting by Language [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_dataset_splitting_by_language/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_dataset_splitting_by_language)]
  - Quality Classifier for CommonCrawl [[ModelScope](https://modelscope.cn/studios/Data-Juicer/tool_quality_classifier/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/tool_quality_classifier)]
  - Auto Evaluation on [HELM](https://github.com/stanford-crfm/helm) [[ModelScope](https://modelscope.cn/studios/Data-Juicer/auto_evaluation_helm/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/auto_evaluation_helm)]
  - Data Sampling and Mixture [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_mixture/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_mixture)]
- Data Processing Loop [[ModelScope](https://modelscope.cn/studios/Data-Juicer/data_process_loop/summary)] [[HuggingFace](https://huggingface.co/spaces/datajuicer/data_process_loop)]