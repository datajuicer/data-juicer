[[中文主页]](README_ZH.md) | [[DJ-Cookbook]](docs/tutorial/DJ-Cookbook.md) | [[OperatorZoo]](docs/Operators.md) | [[API]](https://datajuicer.github.io/data-juicer/en/main/api) | [[Awesome LLM Data]](docs/awesome_llm_data.md)


# Data Processing for and with Foundation Models

 <img src="https://img.alicdn.com/imgextra/i1/O1CN01fUfM5A1vPclzPQ6VI_!!6000000006165-0-tps-1792-1024.jpg" width = "533" height = "300" alt="Data-Juicer"/>
 
![](https://img.shields.io/badge/language-Python-214870.svg)
![](https://img.shields.io/badge/license-Apache--2.0-000000.svg)
[![pypi version](https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad)](https://pypi.org/project/py-data-juicer)
[![Docker version](https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf)](https://hub.docker.com/r/datajuicer/data-juicer)
[![Docker on OSS](https://img.shields.io/badge/OSS%20latest-none?logo=docker&label=Docker&color=498bdf)](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/docker_images/data-juicer-latest.tar.gz)
![](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FHYLcool%2Ff856b14416f08f73d05d32fd992a9c29%2Fraw%2Ftotal_cov.json)

[![DataModality](https://img.shields.io/badge/DataModality-Text,Image,Audio,Video-brightgreen.svg)](https://datajuicer.github.io/data-juicer/en/main/docs/tutorial/DJ-Cookbook.html)
[![Usage](https://img.shields.io/badge/Usage-Cleaning,Synthesis,Analysis-FFD21E.svg)](https://datajuicer.github.io/data-juicer/en/main/docs/hub/RecipeGallery.html)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/py-data-juicer?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/py-data-juicer)



[![OpZoo](https://img.shields.io/badge/Doc-OperatorZoo-blue?logo=Markdown)](https://datajuicer.github.io/data-juicer/en/main/docs/Operators.html)
[![Paper](http://img.shields.io/badge/cs.LG-1.0Paper(SIGMOD'24)-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2309.02033)
[![Paper](http://img.shields.io/badge/cs.AI-2.0Paper(NeurIPS'25)-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2501.14755)





Data-Juicer is a one-stop system to process text and multimodal data for and with foundation models (typically LLMs).
We provide a [playground](http://8.138.149.181/) with a managed JupyterLab. [Try Data-Juicer](http://8.138.149.181/) straight away in your browser! If you find Data-Juicer useful for your research or development, please kindly support us by starting it (then be instantly notified of our new releases) and citing our [works](#references).

[Platform for AI of Alibaba Cloud (PAI)](https://www.aliyun.com/product/bigdata/learn) has deeply integrated Data-Juicer into its data processing products. PAI is an AI Native large model and AIGC engineering platform that provides dataset management, computing power management, model tool chain, model development, model training, model deployment, and AI asset management. For documentation on data processing, please refer to: [Quickly submit a DataJuicer job](https://www.alibabacloud.com/help/en/pai/user-guide/quickly-submit-a-datajuicer-task).

Data-Juicer is being actively updated and maintained. We will periodically enhance and add more features, data recipes and datasets.  We welcome you to [join us](#contribution-and-acknowledgements), in promoting data-model co-development along with research and applications of foundation models!

## News
- 🚀[2026-01-15] [Data-Juicer Q&A Copilot](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot/README.md) is available on the official [Doc Site](https://datajuicer.github.io/data-juicer/en/main/index.html) | [DingTalk](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?) | [Discord](https://discord.gg/ngQbB9hEVK) ! Feel free to ask anything related to Data-Juicer ecosystem! 
  -  Check 🤖 [Data-Juicer Agents](https://github.com/datajuicer/data-juicer-agents/blob/main) | 📃 [Deploy-ready codes](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot) | 🎬[ More demos](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot/DEMO.md).
<div align="center">
<img src="https://github.com/user-attachments/assets/d10a95a8-fb7a-494f-b858-f21e5996790b" width=80%>
</div>

- 🎉 [2025-09-19] Our work of [Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](https://arxiv.org/abs/2501.14755) has been accepted as a **NeurIPS'25 Spotlight** (top 3.1% of all submissions)!
- 🎉 [2025-09-19] Our two works regarding data mixture/selection/synthesis: [Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data](https://arxiv.org/abs/2502.04380) and [MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?](https://arxiv.org/abs/2503.09499) have been accepted by **NeurIPS'25**!
- 🛠️ [2025-06-04] How to process feedback data in the "era of experience"? We propose [Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of LLMs](https://arxiv.org/abs/2505.17826), which leverages Data-Juicer for its data pipelines tailored for RFT scenarios.
- 🎉 [2025-06-04] Our [Data-Model Co-development Survey](https://ieeexplore.ieee.org/document/11027559) has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (**TPAMI**)! Welcome to explore and contribute the [awesome-list](https://datajuicer.github.io/data-juicer/en/main/docs/awesome_llm_data.html).
- 🔎 [2025-06-04] We introduce [DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?](https://www.arxiv.org/abs/2505.16915) A synthetic benchmark revealing notable performance drops despite large models' proficiency with short descriptions.
- 🎉 [2025-05-06] Our work of [Data-Juicer Sandbox](https://arxiv.org/abs/2407.11784) has been accepted as a **ICML'25 Spotlight** (top 2.6% of all submissions)!
- 💡 [2025-03-13] We propose [MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?](https://arxiv.org/abs/2503.09499) A new data synthesis method that enables large models to self-synthesize high-quality, low-variance data for efficient fine-tuning, (e.g., *16%* gain on [MathVision](https://mathllm.github.io/mathvision/#leaderboard) using only *400 samples*). 
- 🤝 [2025-02-28] DJ has been integrated in [Ray's official Ecosystem](https://docs.ray.io/en/latest/ray-overview/ray-libraries.html) and [Example Gallery](https://docs.ray.io/en/latest/ray-more-libs/data_juicer_distributed_data_processing.html). Besides, our patch in DJ2.0 for the streaming JSON reader has been officially integrated by [Apache Arrow](https://github.com/apache/arrow/pull/45084). 
- 🎉 [2025-02-27] Our work on contrastive data synthesis, [ImgDiff](https://arxiv.org/pdf/2408.04594), has been accepted by **CVPR'25**!
- 💡 [2025-02-05] We propose a new data selection method, [Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data](https://www.arxiv.org/abs/2502.04380). It is theoretically informed, via treating diversity as a reward, achieves better overall performance across 7 benchmarks when post-training SOTA LLMs. 
- 🎉 [2025-01-11] We release our 2.0 paper, [Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](https://arxiv.org/abs/2501.14755). It now can process 70B data samples within 2.1h, using 6400 CPU cores on 50 Ray nodes from Alibaba Cloud cluster, and deduplicate 5TB data within 2.8h using 1280 CPU cores on 8 Ray nodes.

<details>
<summary> History News:
</summary>>

- [2025-01-03] We support post-tuning scenarios better, via 20+ related new [OPs](https://github.com/datajuicer/data-juicer/releases/tag/v1.0.2), and via unified [dataset format](https://github.com/datajuicer/data-juicer/releases/tag/v1.0.3) compatible to LLaMA-Factory and ModelScope-Swift.
- [2024-12-17] We propose *HumanVBench*, which comprises 16 human-centric tasks with synthetic data, benchmarking 22 video-MLLMs' capabilities from views of inner emotion and outer manifestations. See more details in our [paper](https://arxiv.org/abs/2412.17574), and try to [evaluate](https://github.com/datajuicer/data-juicer/tree/HumanVBench) your models with it.
- [2024-11-22] We release DJ [v1.0.0](https://github.com/datajuicer/data-juicer/releases/tag/v1.0.0), in which we refactored Data-Juicer's *Operator*, *Dataset*, *Sandbox* and many other modules for better usability, such as supporting fault-tolerant, FastAPI and adaptive resource management.
- [2024-08-25] We give a [tutorial](https://datajuicer.github.io/data-juicer/_static/tutorial_kdd24.html) about data processing for multimodal LLMs in KDD'2024.
- [2024-08-09] We propose Img-Diff, which enhances the performance of multimodal large language models through *contrastive data synthesis*, achieving a score that is 12 points higher than GPT-4V on the [MMVP benchmark](https://tsb0601.github.io/mmvp_blog/). See more details in our [paper](https://arxiv.org/abs/2408.04594), and download the dataset from [huggingface](https://huggingface.co/datasets/datajuicer/Img-Diff) and [modelscope](https://modelscope.cn/datasets/Data-Juicer/Img-Diff).
- [2024-07-24] "Tianchi Better Synth Data Synthesis Competition for Multimodal Large Models" — Our 4th data-centric LLM competition has kicked off! Please visit the competition's [official website](https://tianchi.aliyun.com/competition/entrance/532251) for more information.
- [2024-07-17] We utilized the Data-Juicer [Sandbox Laboratory Suite](https://datajuicer.github.io/data-juicer-sandbox/en/main/index.html) to systematically optimize data and models through a co-development workflow between data and models, achieving a new top spot on the [VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) text-to-video leaderboard. The related achievements have been compiled and published in a [paper](http://arxiv.org/abs/2407.11784), and the model has been released on the [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) and [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V) platforms.
- [2024-07-12] Our *awesome list of MLLM-Data* has evolved into a systemic [survey](https://arxiv.org/abs/2407.08583) from model-data co-development perspective. Welcome to [explore](docs/awesome_llm_data.md) and contribute!
- [2024-06-01] ModelScope-Sora "Data Directors" creative sprint—Our third data-centric LLM competition has kicked off! Please visit the competition's [official website](https://tianchi.aliyun.com/competition/entrance/532219) for more information.
- [2024-03-07] We release **Data-Juicer [v0.2.0](https://github.com/datajuicer/data-juicer/releases/tag/v0.2.0)** now! 
In this new version, we support more features for **multimodal data (including video now)**, and introduce **[DJ-SORA](docs/DJ_SORA.md)** to provide open large-scale, high-quality datasets for SORA-like models.
- [2024-02-20] We have actively maintained an *awesome list of LLM-Data*, welcome to [visit](docs/awesome_llm_data.md) and contribute!
- [2024-02-05] Our paper has been accepted by SIGMOD'24 industrial track!
- [2024-01-10] Discover new horizons in "Data Mixture"—Our second data-centric LLM competition has kicked off! Please visit the competition's [official website](https://tianchi.aliyun.com/competition/entrance/532174) for more information.
- [2024-01-05] We release **Data-Juicer v0.1.3** now!
In this new version, we support **more Python versions** (3.8-3.10), and support **multimodal** dataset [converting](tools/fmt_conversion/multimodal/README.md)/[processing](docs/Operators.md) (Including texts, images, and audios. More modalities will be supported in the future).
Besides, our paper is also updated to [v3](https://arxiv.org/abs/2309.02033).
- [2023-10-13] Our first data-centric LLM competition begins! Please
  visit the competition's official websites, FT-Data Ranker ([1B Track](https://tianchi.aliyun.com/competition/entrance/532157), [7B Track](https://tianchi.aliyun.com/competition/entrance/532158)), for more information.
</details>

## Demos

DataJuicer-Agent: Quick start your data processing journey!

https://github.com/user-attachments/assets/6eb726b7-6054-4b0c-905e-506b2b9c7927

DataJuicer-Sandbox: Better data-model co-dev at a lower cost!

https://github.com/user-attachments/assets/a45f0eee-0f0e-4ffe-9a42-d9a55370089d

## Why Data-Juicer?

<img src="https://img.alicdn.com/imgextra/i4/O1CN015URK6i21KU3XdkUpK_!!6000000006966-2-tps-3994-3956.png" align="center" width="500" />

- **Systematic & Reusable**:
  Empowering users with a systematic library of 100+ core [OPs](docs/Operators.md), and 50+ reusable config recipes and 
  dedicated toolkits, designed to
  function independently of specific multimodal LLM datasets and processing pipelines. Supporting data analysis, cleaning, and synthesis in pre-training, post-tuning, en, zh, and more scenarios.

- **User-Friendly & Extensible**: 
  Designed for simplicity and flexibility, with easy-start [guides](docs/tutorial/QuickStart.md), and [DJ-Cookbook](docs/tutorial/DJ-Cookbook.md) containing fruitful demo usages. Feel free to [implement your own OPs](docs/DeveloperGuide.md#build-your-own-ops) for customizable data processing.

    Data-Juicer now uses AI to automatically rewrite and optimize operator docstrings, generating detailed operator documentation to help users quickly understand the functionality and usage of each operator.  
    For details about the implementation of this documentation enhancement workflow, please visit the [op_doc_enhance_workflow](https://github.com/datajuicer/data-juicer/tree/main/docs/op_doc_enhance_workflow).

- **Efficient & Robust**: Providing performance-optimized [parallel data processing](docs/Distributed.md) (Aliyun-PAI\Ray\CUDA\OP Fusion),
  faster with less resource usage, verified in large-scale production environments.


- **Effect-Proven & Sandbox**: Supporting data-model co-development, enabling rapid iteration
  through the [sandbox laboratory](https://datajuicer.github.io/data-juicer-sandbox/en/main/index.html), and providing features such as feedback loops and visualization, so that you can better understand and improve your data and models. Many effect-proven datasets and models have been derived from DJ, in scenarios such as pre-training, text-to-video and image-to-text generation.
  ![Data-in-the-loop](https://img.alicdn.com/imgextra/i2/O1CN017U7Zz31Y7XtCJ5GOz_!!6000000003012-0-tps-3640-1567.jpg) 

## Documentation

For detailed documentation, please see [here](https://datajuicer.github.io/data-juicer/en/main/docs_index.html).

## License
Data-Juicer is released under Apache License 2.0.

## Contribution and Acknowledgements

Data-Juicer has benefited greatly from and continues to welcome contributions at all levels: new operators (from simple functions to advanced algorithms based on existing papers), data-recipes & processing scenarios, feature requests, efficiency enhancements, bug fixes, better documentation and usage feedback. Please refer to our [Developer Guide](docs/DeveloperGuide.md) to get started. Spreading the word in the community and giving the repository a star ⭐ are also invaluable forms of support!

Our sincere gratitude goes to all our [code contributors](https://github.com/datajuicer/data-juicer/graphs/contributors) who are the cornerstone of this project. We strive to keep the list below updated and look forward to including more names (alphabetical order); please reach out if we have missed any acknowledgements.
- **Initiated by:** Alibaba Tongyi Lab
- **Co-developed and Optimized with:** Alibaba Cloud PAI, Anyscale (Ray Team), Sun Yat-sen University ([Knowledge Engineering Lab](https://github.com/YingShen-SYSU/AIGC)), NVIDIA (NeMo Team), ...
- **Used by & Valuable Feedback from:** [AgentScope](https://github.com/agentscope-ai/agentscope), Alibaba Group, Ant Group, BYD Auto, Bytedance, CAS, [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), DTSTACK, [EasyAnimate](https://github.com/aigc-apps/EasyAnimate), [Eval-Scope](https://github.com/modelscope/evalscope), JD.com, [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), Nanjing University, OPPO, Peking University, [RM-Gallery](https://github.com/modelscope/RM-Gallery), RUC, Tsinghua University, [Trinity-RFT](https://github.com/modelscope/Trinity-RFT), UCAS, Xiaohongshu, Xiaomi, Ximalaya, Zhejiang University, ...
- **Inspired by:** Data-Juicer also thanks pioneering open-source projects such as [Apache Arrow](https://github.com/apache/arrow), [BLOOM](https://huggingface.co/bigscience/bloom), [RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1), [Ray](https://github.com/ray-project/ray), [Hugging Face Datasets](https://github.com/huggingface/datasets), ...

We look forward to your feedback and collaboration, including partnership inquiries or proposals for new sub-projects related to Data-Juicer. Feel free to contact via issues, PRs, [Slack](https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg?spm=a2c22.12281976.0.0.7a8253f30mgpjw), [DingTalk](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?), [Discord](https://discord.gg/ngQbB9hEVK), and [e-mails](mailto:datajuicer@outlook.com).

Welcome to join our community on

| [Discord](https://discord.gg/ngQbB9hEVK)                                                                                         | [DingTalk](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?)                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| <img src="https://gw.alicdn.com/imgextra/i1/O1CN011Oj8CB1f8Bw5JpgJA_!!6000000003961-0-tps-762-769.jpg" width="100" height="100"> | <img src="https://gw.alicdn.com/imgextra/i3/O1CN01bBPoaX1EwZsiYudtd_!!6000000000416-2-tps-656-660.png" width="100" height="100"> |



## References
If you find Data-Juicer useful for your research or development, please kindly cite the following works, [1.0paper](https://arxiv.org/abs/2309.02033), [2.0paper](https://arxiv.org/abs/2501.14755).
```
@inproceedings{djv1,
  title={Data-Juicer: A One-Stop Data Processing System for Large Language Models},
  author={Daoyuan Chen and Yilun Huang and Zhijian Ma and Hesen Chen and Xuchen Pan and Ce Ge and Dawei Gao and Yuexiang Xie and Zhaoyang Liu and Jinyang Gao and Yaliang Li and Bolin Ding and Jingren Zhou},
  booktitle={International Conference on Management of Data},
  year={2024}
}

@article{djv2,
  title={Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models},
  author={Chen, Daoyuan and Huang, Yilun and Pan, Xuchen and Jiang, Nana and Wang, Haibin and Zhang, Yilei and Ge, Ce and Chen, Yushuo and Zhang, Wenhao and Ma, Zhijian and Huang, Jun and Lin, Wei and Li, Yaliang and Ding, Bolin and Zhou, Jingren},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

<details>
<summary> More data-related papers from the Data-Juicer Team:
</summary>>

- (ICML'25 Spotlight) [Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development](https://arxiv.org/abs/2407.11784)

- (CVPR'25) [ImgDiff: Contrastive Data Synthesis for Vision Large Language Models](https://arxiv.org/abs/2408.04594)
 
- (TPAMI'25) [The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective](https://arxiv.org/abs/2407.08583)

- (NeurIPS'25) [Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data](https://arxiv.org/abs/2502.04380)

- (NeurIPS'25) [MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?](https://arxiv.org/abs/2503.09499)

- (Benchmark Data) [HumanVBench: Exploring Human-Centric Video Understanding Capabilities of MLLMs with Synthetic Benchmark Data](https://arxiv.org/abs/2412.17574)
 
- (Benchmark Data) [DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?](https://www.arxiv.org/abs/2505.16915)

- (Data Scaling) [BiMix: A Bivariate Data Mixing Law for Language Model Pretraining](https://arxiv.org/abs/2405.14908)

</details>


