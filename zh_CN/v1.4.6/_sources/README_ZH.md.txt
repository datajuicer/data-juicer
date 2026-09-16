[[英文主页]](README.md) | [[DJ-Cookbook]](docs/tutorial/DJ-Cookbook_ZH.md) | [[算子池]](docs/Operators.md) | [[API]](https://datajuicer.github.io/data-juicer/zh_CN/main/api) | [[Awesome LLM Data]](docs/awesome_llm_data.md)

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



[![算子池](https://img.shields.io/badge/文档-算子池-blue?logo=Markdown)](https://datajuicer.github.io/data-juicer/en/main/docs/Operators.html)
[![Paper](http://img.shields.io/badge/cs.LG-1.0Paper(SIGMOD'24)-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2309.02033)
[![Paper](http://img.shields.io/badge/cs.AI-2.0Paper(NeurIPS'25)-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2501.14755)



Data-Juicer 是一个一站式系统，面向大模型的文本及多模态数据处理。我们提供了一个基于 JupyterLab 的 [Playground](http://8.138.149.181/)，您可以从浏览器中在线试用 Data-Juicer。 如果Data-Juicer对您的研发有帮助，请支持加星（自动订阅我们的新发布）、以及引用我们的[工作](#参考文献) 。

[阿里云人工智能平台 PAI](https://www.aliyun.com/product/bigdata/learn) 已深度集成Data-Juicer到其数据处理产品中。PAI提供包含数据集管理、算力管理、模型工具链、模型开发、模型训练、模型部署、AI资产管理在内的功能模块，为用户提供高性能、高稳定、企业级的大模型工程化能力。数据处理的使用文档请参考：[快速提交DataJuicer任务](https://www.alibabacloud.com/help/zh/pai/user-guide/quickly-submit-a-datajuicer-task)。

Data-Juicer正在积极更新和维护中，我们将定期强化和新增更多的功能和数据菜谱。热烈欢迎您[加入我们](#贡献与致谢)，一起推进大模型的数据-模型协同开发和研究应用！

## 新消息
- 🚀[2026-01-15] [Data-Juicer Q&A Copilot](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot/README_ZH.md) 已上线 [文档页](https://datajuicer.github.io/data-juicer/zh_CN/main/index_ZH.html) | [钉钉](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?) | [Discord](https://discord.gg/ngQbB9hEVK) ! 欢迎询问任何与Data-Juicer生态系统有关的问题!

  - 关注 🤖 [Data-Juicer Agents](https://github.com/datajuicer/data-juicer-agents/blob/main) | 📃 [Deploy-ready codes](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot) | 🎬[ More demos](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot/DEMO.md).
<div align="center">
<img src="https://github.com/user-attachments/assets/a0099ce2-4ed3-4fab-8cfa-b0bbd3beeac9" width=80%>
</div>

- 🎉 [2025-09-19] 我们的 [Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](https://arxiv.org/abs/2501.14755) 已被接收为 **NeurIPS'25 Spotlight**（处于所有投稿中的前 3.1%）！
- 🎉 [2025-09-19] 我们关于数据配比/选择/合成的两个工作：[Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data](https://arxiv.org/abs/2502.04380) 和 [MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?](https://arxiv.org/abs/2503.09499)，已被 **NeurIPS'25** 接收！
- 🛠️ [2025-06-04] 如何在“经验时代”处理反馈数据？我们提出了 [Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of LLMs](https://arxiv.org/abs/2505.17826)，该框架利用 Data-Juicer 为 RFT 场景量身定制数据处理管道。
- 🎉 [2025-06-04] 我们的 [Data-Model Co-development 综述](https://ieeexplore.ieee.org/document/11027559) 已被 IEEE Transactions on Pattern Analysis and Machine Intelligence（**TPAMI**）接收！欢迎探索并贡献[awesome-list](https://datajuicer.github.io/data-juicer/en/main/docs/awesome_llm_data.html)。
- 🔎 [2025-06-04] 我们推出了 [DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?](https://www.arxiv.org/abs/2505.16915) 一项合成基准测试，揭示了大模型虽擅长处理短描述，但在长提示下性能显著下降的问题。
- 🎉 [2025-05-06] 我们的 [Data-Juicer Sandbox](https://arxiv.org/abs/2407.11784) 已被接收为 **ICML'25 Spotlight**（处于所有投稿中的前 2.6%）！
- 💡 [2025-03-13] 我们提出[MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?](https://arxiv.org/abs/2503.09499)。一种新的数据合成方法鼓励大模型自我合成高质量、低方差数据，实现高效SFT（如仅使用 *400 个样本* 即可在 [MathVision](https://mathllm.github.io/mathvision/#leaderboard) 上获得 *16%* 的增益）。
- 🤝 [2025-02-28] DJ 已被集成到 [Ray官方 Ecosystem](https://docs.ray.io/en/latest/ray-overview/ray-libraries.html) 和 [Example Gallery](https://docs.ray.io/en/latest/ray-more-libs/data_juicer_distributed_data_processing.html)。此外，我们在 DJ2.0 中的流式 JSON 加载补丁已被 [Apache Arrow 官方集成](https://github.com/apache/arrow/pull/45084)。
- 🎉 [2025-02-27] 我们的对比数据合成工作， [ImgDiff](https://arxiv.org/pdf/2408.04594)， 已被 **CVPR'25** 接收！
- 💡 [2025-02-05] 我们提出了一种新的数据选择方法 [Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data](https://www.arxiv.org/abs/2502.04380)，该方法基于理论指导，将数据多样性建模为奖励信号，在 7 个基准测试中，微调 SOTA LLMs 取得了更好的整体表现。
- 🎉 [2025-01-11] 我们发布了 2.0 版论文 [Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](https://arxiv.org/abs/2501.14755)。DJ现在可以使用阿里云集群中 50 个 Ray 节点上的 6400 个 CPU 核心在 2.1 小时内处理 70B 数据样本，并使用 8 个 Ray 节点上的 1280 个 CPU 核心在 2.8 小时内对 5TB 数据进行重复数据删除。

<details>
<summary> History News:
</summary>>

- [2025-01-03] 我们通过 20 多个相关的新 [OP](https://github.com/datajuicer/data-juicer/releases/tag/v1.0.2) 以及与 LLaMA-Factory 和 ModelScope-Swift 兼容的统一 [数据集格式](https://github.com/datajuicer/data-juicer/releases/tag/v1.0.3) 更好地支持Post-Tuning场景。
- [2024-12-17] 我们提出了 *HumanVBench*，它包含 16 个以人为中心的任务，使用合成数据，从内在情感和外在表现的角度对22个视频 MLLM 的能力进行基准测试。请参阅我们的 [论文](https://arxiv.org/abs/2412.17574) 中的更多详细信息，并尝试使用它 [评估](https://github.com/datajuicer/data-juicer/tree/HumanVBench) 您的模型。
- [2024-11-22] 我们发布 DJ [v1.0.0](https://github.com/datajuicer/data-juicer/releases/tag/v1.0.0)，其中我们重构了 Data-Juicer 的 *Operator*、*Dataset*、*Sandbox* 和许多其他模块以提高可用性，例如支持容错、FastAPI 和自适应资源管理。
- [2024-08-25] 我们在 KDD'2024 中提供了有关多模态 LLM 数据处理的[教程](https://datajuicer.github.io/data-juicer/_static/tutorial_kdd24.html)。
- [2024-08-09] 我们提出了Img-Diff，它通过*对比数据合成*来增强多模态大型语言模型的性能，在[MMVP benchmark](https://tsb0601.github.io/mmvp_blog/)中比GPT-4V高出12个点。 更多细节请参阅我们的 [论文](https://arxiv.org/abs/2408.04594), 以及从 [huggingface](https://huggingface.co/datasets/datajuicer/Img-Diff) 和 [modelscope](https://modelscope.cn/datasets/Data-Juicer/Img-Diff)下载这份数据集。
- [2024-07-24] "天池 Better Synth 多模态大模型数据合成赛"——第四届Data-Juicer大模型数据挑战赛已经正式启动！立即访问[竞赛官网](https://tianchi.aliyun.com/competition/entrance/532251)，了解赛事详情。
- [2024-07-17] 我们利用Data-Juicer[沙盒实验室套件](https://datajuicer.github.io/data-juicer-sandbox/zh_CN/main/index_ZH.html)，通过数据与模型间的系统性研发工作流，调优数据和模型，在[VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)文生视频排行榜取得了新的榜首。相关成果已经整理发表在[论文](http://arxiv.org/abs/2407.11784)中，并且模型已在[ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V)和[HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V)平台发布。
- [2024-07-12] 我们的MLLM-Data精选列表已经演化为一个模型-数据协同开发的角度系统性[综述](https://arxiv.org/abs/2407.08583)。欢迎[浏览](docs/awesome_llm_data.md)或参与贡献!
- [2024-06-01] ModelScope-Sora"数据导演"创意竞速——第三届Data-Juicer大模型数据挑战赛已经正式启动！立即访问[竞赛官网](https://tianchi.aliyun.com/competition/entrance/532219)，了解赛事详情。
- [2024-03-07] 我们现在发布了 **Data-Juicer [v0.2.0](https://github.com/datajuicer/data-juicer/releases/tag/v0.2.0)**! 在这个新版本中，我们支持了更多的 **多模态数据(包括视频)** 相关特性。我们还启动了 **[DJ-SORA](docs/DJ_SORA_ZH.md)** ，为SORA-like大模型构建开放的大规模高质量数据集！
- [2024-02-20] 我们在积极维护一份关于LLM-Data的*精选列表*，欢迎[访问](docs/awesome_llm_data.md)并参与贡献！
- [2024-02-05] 我们的论文被SIGMOD'24 industrial track接收！
- [2024-01-10] 开启"数据混合"新视界——第二届Data-Juicer大模型数据挑战赛已经正式启动！立即访问[竞赛官网](https://tianchi.aliyun.com/competition/entrance/532174)，了解赛事详情。
- [2024-01-05] **Data-Juicer v0.1.3** 版本发布了。 
在这个新版本中，我们支持了**更多Python版本**（3.8-3.10），同时支持了**多模态**数据集的[转换](tools/fmt_conversion/multimodal/README_ZH.md)和[处理](docs/Operators.md)（包括文本、图像和音频。更多模态也将会在之后支持）！
此外，我们的论文也更新到了[第三版](https://arxiv.org/abs/2309.02033) 。
- [2023-10-13] 我们的第一届以数据为中心的 LLM 竞赛开始了！
  请访问大赛官网，FT-Data Ranker（[1B赛道](https://tianchi.aliyun.com/competition/entrance/532157) 、[7B赛道](https://tianchi.aliyun.com/competition/entrance/532158) ) ，了解更多信息。
</details>

## Demos

DataJuicer-Agent:数据处理，即刻启程！

https://github.com/user-attachments/assets/6eb726b7-6054-4b0c-905e-506b2b9c7927

DataJuicer-Sandbox: 降本增效，优化数据-模型协同开发！

https://github.com/user-attachments/assets/a45f0eee-0f0e-4ffe-9a42-d9a55370089d

## 为什么选择 Data-Juicer？

<img src="https://img.alicdn.com/imgextra/i4/O1CN015URK6i21KU3XdkUpK_!!6000000006966-2-tps-3994-3956.png" align="center" width="500" />

- **系统化和可重用**：
系统化地为用户提供 100 多个核心 [算子](docs/Operators.md) 和 50 多个可重用的数据菜谱和
专用工具套件，旨在解耦于特定的多模态 LLM 数据集和处理管道运行。支持预训练、后训练、英语、中文等场景中的数据分析、清洗和合成。

- **易用、可扩展**：
简洁灵活，提供快速[入门指南](docs/tutorial/QuickStart_ZH.md)和包含丰富使用示例的[DJ-Cookbook](docs/tutorial/DJ-Cookbook_ZH.md)。您可以灵活实现自己的OP，[自定义](docs/DeveloperGuide_ZH.md)数据处理工作流。

  Data-Juicer 现采用 AI 自动重写和优化算子的 docstring，并生成详细的算子文档，帮助更快理解每个算子的功能及用法。  
  如需了解该文档增强流程的具体实现，欢迎访问 [op_doc_enhance_workflow](https://github.com/datajuicer/data-juicer/tree/main/docs/op_doc_enhance_workflow)。

- **高效、稳定**：提供性能优化的[并行数据处理能力](docs/Distributed_ZH.md)（Aliyun-PAI\Ray\CUDA\OP Fusion），
更快、更少资源消耗，基于大规模生产环境打磨。

- **效果验证、沙盒**：支持数据模型协同开发，通过[沙盒实验室](https://datajuicer.github.io/data-juicer-sandbox/zh_CN/main/index_ZH.html)实现快速迭代，提供反馈循环、可视化等功能，让您更好地理解和改进数据和模型。已经有许多基于 DJ 衍生的数据菜谱和模型经过了效用验证，譬如在预训练、文生视频、图文生成等场景。
![Data-in-the-loop](https://img.alicdn.com/imgextra/i2/O1CN017U7Zz31Y7XtCJ5GOz_!!6000000003012-0-tps-3640-1567.jpg)

## 文档

详细文档请看[此处](https://datajuicer.github.io/data-juicer/zh_CN/main/docs_index_ZH.html)。

## 开源协议

Data-Juicer 在 Apache License 2.0 协议下发布。

## 贡献与致谢

Data-Juicer 的发展离不开社区的参与和反馈，非常欢迎各方面的贡献：开发新的算子（无论是简单函数还是现有论文的先进算法）、分享新的数据菜谱和使用场景、提出新功能需求、提升代码效率、修复程序错误、完善项目文档、反馈使用体验等。您可参考[开发者指南](docs/DeveloperGuide_ZH.md)开启贡献；在社区中宣传本项目，或为我们的代码仓库点亮星标 ⭐，同样是对该项目非常宝贵的支持！

我们由衷感谢所有为本项目做出贡献的[代码贡献者](https://github.com/datajuicer/data-juicer/graphs/contributors)，他们是本项目的基石。我们尽力确保以下名单的完整和及时，并期待更多名字的加入（英文字母序排列）。若有疏漏，请随时联系我们。

- **发起方：** 阿里巴巴通义实验室
- **联合研发优化：** 阿里云PAI、Anyscale (Ray Team)、中山大学 ([知识工程实验室](https://github.com/YingShen-SYSU/AIGC))、NVIDIA (NeMo Team) 等
- **用户/提供无价反馈：** [AgentScope](https://github.com/agentscope-ai/agentscope)、阿里巴巴集团、蚂蚁集团、比亚迪、字节跳动、[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)、袋鼠云、[EasyAnimate](https://github.com/aigc-apps/EasyAnimate)、[Eval-Scope](https://github.com/modelscope/evalscope)、京东、[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、南京大学、OPPO、北京大学、[RM-Gallery](https://github.com/modelscope/RM-Gallery)、中国人民大学、清华大学、[Trinity-RFT](https://github.com/modelscope/Trinity-RFT)、中国科学院、中国科学院大学、小红书、小米、喜马拉雅、浙江大学等
- **其它项目：** Data-Juicer 也感谢许多先驱开源项目，例如 [Apache Arrow](https://github.com/apache/arrow)、[BLOOM](https://huggingface.co/bigscience/bloom)、[Hugging Face Datasets](https://github.com/huggingface/datasets)、[RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1)、[Ray](https://github.com/ray-project/ray)、[vLLM](https://github.com/vllm-project/vllm) 等

我们期待您的反馈与合作。如您有合作意向或关于新子项目的提案，欢迎通过 GitHub Issues、Pull Requests、[Slack](https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg?spm=a2c22.12281976.0.0.7a8253f30mgpjw)、[钉钉](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?)、 [Discord](https://discord.gg/ngQbB9hEVK)或[邮件](mailto:datajuicer@outlook.com)与我们联系。

欢迎加入我们的社群：

|[钉钉](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?)                                                                                          | [Discord](https://discord.gg/ngQbB9hEVK)                                                                  |
|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| <img src="https://gw.alicdn.com/imgextra/i3/O1CN01bBPoaX1EwZsiYudtd_!!6000000000416-2-tps-656-660.png" width="100" height="100"> | <img src="https://gw.alicdn.com/imgextra/i1/O1CN011Oj8CB1f8Bw5JpgJA_!!6000000003961-0-tps-762-769.jpg" width="100" height="100"> |

## 参考文献
如果您发现Data-Juicer对您的研发有帮助，请引用以下工作，[1.0paper](https://arxiv.org/abs/2309.02033)，[2.0paper](https://arxiv.org/abs/2501.14755)。

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
<summary>更多Data-Juicer团队关于数据的论文:
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

