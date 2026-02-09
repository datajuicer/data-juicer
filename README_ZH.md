# Data-Juicer: 基础模型时代的*数据操作*系统

<p align="center">
  <a href="https://pypi.org/project/py-data-juicer"><img src="https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad" alt="PyPI"></a>
  <a href="https://pepy.tech/projects/py-data-juicer"><img src="https://static.pepy.tech/personalized-badge/py-data-juicer?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=green&left_text=downloads" alt="Downloads"></a>
   <a href="https://hub.docker.com/r/datajuicer/data-juicer"><img src="https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf" alt="Docker"></a>
  <br>
  <a href="https://datajuicer.github.io/data-juicer/zh_CN/main/index_ZH.html"><img src="https://img.shields.io/badge/📖_文档-网站-026cad" alt="Docs"></a>
  <a href="https://datajuicer.github.io/data-juicer/en/main/docs/Operators.html"><img src="https://img.shields.io/badge/🧩_算子-200+-blue" alt="Operators"></a>
  <a href="https://github.com/datajuicer/data-juicer-hub"><img src="https://img.shields.io/badge/🍳_配方-50+-brightgreen" alt="Recipes"></a>
  <br>
  <a href="https://datajuicer.github.io/data-juicer/en/main/index.html"><img src="https://img.shields.io/badge/🇬🇧_English-主页-red" alt="English"></a>
  <a href="https://arxiv.org/abs/2501.14755"><img src="https://img.shields.io/badge/NeurIPS'25_Spotlight-2.0-B31B1B?logo=arxiv" alt="Paper"></a>
  <a href="https://github.com/datajuicer/data-juicer">
    <img src="https://img.shields.io/endpoint?style=flat&url=https%3A%2F%2Fgist.githubusercontent.com%2FHYLcool%2Ff856b14416f08f73d05d32fd992a9c29%2Fraw%2Ftotal_cov.json&label=coverage&logo=codecov&color=4c1" alt="Coverage">
  </a>
</p>

<p align="center">
  <b>多模态 | 云原生 | AI就绪 | 大规模 </b>
</p>

Data-Juicer (DJ) 将原始数据转化为 AI 就绪的智能。它将数据处理视为*可组合的基础设施*——提供模块化构建块，在整个 AI 生命周期中清洗、合成和分析数据，释放每份数据的潜在价值。

无论您是在去重网络规模的预训练语料库、整理智能体交互轨迹，还是准备特定领域的 RAG 索引，DJ 都可以从您的笔记本电脑无缝扩展到数千节点的集群——无需编写胶水代码。

> **阿里云 PAI** 已深度集成 Data-Juicer 到其数据处理产品中。请参阅 **[快速提交 DataJuicer 任务](https://www.alibabacloud.com/help/zh/pai/user-guide/quickly-submit-a-datajuicer-task)**。

---

## 🚀 快速开始

**零安装探索**：
- [带教程的 JupyterLab 在线环境](http://8.138.149.181/) 
- [询问 DJ Copilot](https://datajuicer.github.io/data-juicer/zh_CN/main/docs_index_ZH.html)

**安装并运行**：
```bash
uv pip install py-data-juicer
dj-process --config demos/process_simple/process.yaml
```

**或在 Python 中组合**：
```python
from data_juicer.core.data import NestedDataset
from data_juicer.ops.filter import TextLengthFilter
from data_juicer.ops.mapper import WhitespaceNormalizationMapper

ds = NestedDataset.from_dict({
    "text": ["Short", "This passes the filter.", "Text   with   spaces"]
})
res_ds = ds.process([
    TextLengthFilter(min_len=10),
    WhitespaceNormalizationMapper()
])

for s in res_ds:
    print(s)
```

---

## ✨ 为什么选择 Data-Juicer？

### 1. 模块化与可扩展架构
- **200+ 算子** 涵盖文本、图像、音频、视频和多模态数据
- **配方优先**：可复现的 YAML 管道，您可以像代码一样进行版本管理、共享和分叉
- **可组合**：可插入单个算子、链接复杂工作流或编排完整管道
- **热重载**：无需重启管道即可迭代算子

### 2. 全栈数据智能
- **基础模型**：预训练、微调、强化学习和评估级数据整理
- **智能体系统**：清洗工具轨迹、结构化上下文、去标识化和质量把关
- **RAG与分析**：提取、规范化、语义分块、去重和数据画像分析

### 3. 生产就绪的性能
- **规模**：在 50 个 Ray 节点（6400 核心）上 2 小时处理 700 亿样本
- **效率**：使用 1280 核心在 2.8 小时内对 5TB 进行去重
- **优化**：自动 OP 融合（2-10 倍加速）、自适应并行、CUDA 加速、鲁棒性
- **可观测性**：内置追踪功能，用于调试、审计和迭代改进

> *⭐ 如果 Data-Juicer 为您节省了时间或改进了您的数据工作，请考虑为仓库加星。* 它帮助更多人发现项目，并让您及时了解新发布和功能。

---

## 📰 动态

<details open>
<summary>[2026-02-02] Release v1.4.6: <b>Copilot、视频字节 I/O 与 Ray 追踪</b></summary>

- 🤖 *Q&A Copilot* — 现已上线我们的[文档站点](https://datajuicer.github.io/data-juicer/zh_CN/main/index_ZH.html) | [钉钉](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?) | [Discord](https://discord.gg/ngQbB9hEVK)。欢迎询问任何与 Data-Juicer 生态系统相关的问题！  
    - 查看 🤖 [Data-Juicer Agents](https://github.com/datajuicer/data-juicer-agents/blob/main) | 📃 [部署就绪代码](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot) | 🎬[更多演示](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot/DEMO.md) 了解更多详情。
- 🎬 *视频字节 I/O* — 视频管道的直接字节处理  
- 🫆 *Ray 模式追踪器* — 在分布式处理中追踪变更的样本  
- 🐳 *增强与修复* — 刷新 Docker 镜像、小幅性能提升、GitHub Insights 流量工作流、Ray 兼容性更新以及 Bug/文档修复。
</details>

<details open>
<summary>[2026-01-15] Release v1.4.5: <b>20+ 新 OP、Ray vLLM 管道与 Sphinx 文档升级</b> </summary>

- *具身 AI OP*：添加/增强了用于视频标题生成（VLM）、视频对象分割（YOLOE+SAM2）、视频深度估计（可视化 + 点云）、人体姿态（MMPose）、图像标签（VLM）、单图像 3D 人体网格恢复（SAM 3D Body）的映射器，以及 *S3 上传/下载*。
- *新管道 OP*：将多个 OP 组合成一个管道；引入了用于 LLM/VLM 推理的 *Ray + vLLM* 管道。
- *文档升级*：迁移到统一的基于 *Sphinx* 的文档构建/部署工作流，具有隔离的主题/架构仓库。
- *增强与修复*：依赖更新、改进的 Ray 去重和 S3 加载、OpenAI Responses API 支持、追踪器一致性、Docker 基础更新为 CUDA 12.6.3 + Ubuntu 24.04 + Py3.11，以及多个 Bug 修复。 

</details>

<details>
<summary>[2025-12-01] Release v1.4.4: <b>NeurIPS'25 Spotlight、6 个新视频/多模态 OP 与 S3 I/O</b> </summary>

- Data-Juicer 2.0 获得 NeurIPS'25 **Spotlight**
- *仓库拆分*：sandbox/recipes/agents 移至独立仓库
- *S3 I/O* 添加到加载器/导出器
- *6 个新的视频和多模态 OP*（角色检测、VGGT、全身姿态、手部重建）+ 文档/Ray/视频 I/O 改进和 Bug 修复
</details>

查看 [所有发布](https://github.com/datajuicer/data-juicer/releases) 和 [动态归档](docs/news_zh.md)

---

## 🔌 用户与生态系统
> 以下列表重点关注*面向开发者的集成和使用*，按*字母顺序*排列。  
> 缺少您的项目/名称？欢迎[提交 PR](https://github.com/datajuicer/data-juicer/pulls) 或[联系我们](#贡献与社区)。

Data-Juicer 可无缝集成到您现有的技术栈，并随着社区贡献而不断发展：

### 扩展
- **[data-juicer-agents](https://github.com/datajuicer/data-juicer-agents)** — DJ Copilot 和智能体工作流  
- **[data-juicer-hub](https://github.com/datajuicer/data-juicer-hub)** — 社区配方和最佳实践  
- **[data-juicer-sandbox](https://github.com/datajuicer/data-juicer-sandbox)** — 带反馈循环的数据-模型协同开发  

### 框架与平台
[阿里云 PAI](https://www.alibabacloud.com/zh/product/machine-learning?_p_lc=1) · [Delta Lake](https://delta.io/)[AgentScope](https://github.com/agentscope-ai/agentscope) · [Apache Arrow](https://github.com/apache/arrow) · [Apache HDFS](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsUserGuide.html) · [Apache Hudi](https://hudi.apache.org/) · [Apache Iceberg](https://iceberg.apache.org/) · [Apache Paimon](https://paimon.apache.org/) · [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) · [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) · [Eval-Scope](https://github.com/modelscope/evalscope) · [华为昇腾](https://www.huawei.com/en/products/cloud-computing-dc/atlas/ascend) · [Hugging Face](https://huggingface.co/) · [LanceDB](https://lancedb.github.io/lance/) · [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) · [ModelScope](https://modelscope.cn/) · [ModelScope Swift](https://github.com/modelscope/ms-swift) · [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) · [Ray](https://docs.ray.io/) · [RM-Gallery](https://github.com/modelscope/RM-Gallery) · [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) · [火山引擎](https://www.volcengine.com/)

### 企业
阿里巴巴集团、蚂蚁集团、比亚迪、字节跳动、袋鼠云、京东、NVIDIA、OPPO、小红书、小米、喜马拉雅等。

### 学术机构
中科院、南京大学、北京大学、中国人民大学、清华大学、中科院大学、浙江大学等。

### 贡献与社区
我们相信*共同建设*。无论您是修复拼写错误、开发新算子还是分享数据处理配方，每一次贡献都塑造着数据处理的未来。

我们欢迎各个层面的贡献：
- **[Good First Issues](https://github.com/datajuicer/data-juicer/labels/good%20first%20issue)** — 添加算子、改进文档、报告问题或修复 Bug
- **[开发者指南](https://datajuicer.github.io/data-juicer/en/main/docs/DeveloperGuide.html)** — 优化引擎、添加功能或增强核心基础设施
- **[DJ-Hub](https://github.com/datajuicer/data-juicer-hub)** — 分享知识：配方、论文和最佳实践
- **联系**：[Slack](https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg) · [钉钉](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?) · [Discord](https://discord.gg/ngQbB9hEVK)

| Discord | 钉钉 |
|:---:|:---:|
| <img src="https://gw.alicdn.com/imgextra/i1/O1CN011Oj8CB1f8Bw5JpgJA_!!6000000003961-0-tps-762-769.jpg" width="100"> | <img src="https://gw.alicdn.com/imgextra/i3/O1CN01bBPoaX1EwZsiYudtd_!!6000000000416-2-tps-656-660.png" width="100"> |

Data-Juicer 由用户和社区共同打造：
- **发起方**：阿里巴巴通义实验室  
- **联合开发**：阿里云 PAI、Anyscale（Ray 团队）、中山大学、NVIDIA（NeMo 团队）以及[全球贡献者](https://github.com/datajuicer/data-juicer/graphs/contributors)
- **启发来源**：Apache Arrow、Ray、Hugging Face Datasets、BLOOM、RedPajama-Data、...

---

## 文档

详细文档请查看[此处](https://datajuicer.github.io/data-juicer/zh_CN/main/docs_index_ZH.html)。

**快速链接：**
- **[算子池](https://datajuicer.github.io/data-juicer/en/main/docs/Operators.html)** — 浏览 200+ 带示例的算子
- **[data-juicer-hub](https://github.com/datajuicer/data-juicer-hub)** — 社区驱动的配方和最佳实践
- **[开发者指南](https://datajuicer.github.io/data-juicer/en/main/docs/DeveloperGuide.html)** — 构建您自己的代码并为 DJ 贡献
- **[data-juicer-cookbook](https://datajuicer.github.io/data-juicer/en/main/docs/tutorial/DJ-Cookbook.html)** — 资源归档
- **[awesome_llm_data](https://datajuicer.github.io/data-juicer/en/main/docs/awesome_llm_data)** — 数据-模型协同开发的"Awesome List"

---

## 📄 许可证与致谢

Data-Juicer 在 [Apache License 2.0](LICENSE) 下发布。  
如果您项目中要致谢DataJuicer：请使用我们的[Badge](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/assets/DJ-Org-Logo.jpeg)，或文本譬如 "本项目使用Data-Juicer: https://github.com/datajuicer"。

---

## 📖 引用

如果您发现 Data-Juicer 帮助了您的项目，请考虑如下引用：

```bibtex
@inproceedings{djv1,
  title={Data-Juicer: A One-Stop Data Processing System for Large Language Models},
  author={Chen, Daoyuan and Huang, Yilun and Ma, Zhijian and Chen, Hesen and Pan, Xuchen and Ge, Ce and Gao, Dawei and Xie, Yuexiang and Liu, Zhaoyang and Gao, Jinyang and Li, Yaliang and Ding, Bolin and Zhou, Jingren},
  booktitle={SIGMOD},
  year={2024}
}

@article{djv2,
  title={Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models},
  author={Chen, Daoyuan and Huang, Yilun and Pan, Xuchen and Jiang, Nana and Wang, Haibin and Zhang, Yilei and Ge, Ce and Chen, Yushuo and Zhang, Wenhao and Ma, Zhijian and Huang, Jun and Lin, Wei and Li, Yaliang and Ding, Bolin and Zhou, Jingren},
  journal={NeurIPS},
  year={2025}
}
```

<details>
<summary><b>更多出版物</b>（点击展开）</summary>

- (ICML'25 Spotlight) [Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development](https://arxiv.org/abs/2407.11784)

- (CVPR'25) [ImgDiff: Contrastive Data Synthesis for Vision Large Language Models](https://arxiv.org/abs/2408.04594)
 
- (TPAMI'25) [The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective](https://arxiv.org/abs/2407.08583)

- (NeurIPS'25) [Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data](https://arxiv.org/abs/2502.04380)

- (NeurIPS'25) [MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?](https://arxiv.org/abs/2503.09499)

- (Benchmark Data) [HumanVBench: Exploring Human-Centric Video Understanding Capabilities of MLLMs with Synthetic Benchmark Data](https://arxiv.org/abs/2412.17574)
 
- (Benchmark Data) [DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?](https://www.arxiv.org/abs/2505.16915)

- (Data Scaling) [BiMix: A Bivariate Data Mixing Law for Language Model Pretraining](https://arxiv.org/abs/2405.14908)

</details>
