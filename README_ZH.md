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

- 🎉 [2026-08-25] 我们发布了 [Juicer](docs/Juicer_ZH.md)，一个支持本地部署的数据精炼模型，可通过自然语言指令完成文本清洗、过滤与语义标注。欢迎下载[模型](https://huggingface.co/datajuicer/Juicer-35B-A3B)，并在 [Juicer Playground](https://github.com/datajuicer/data-juicer-hub/tree/main/juicer_playground) 中体验数据处理配方。

<details open>
<summary>[2026-08-07] Release v1.5.5: <b>外部算子插件；HDFS I/O 与 Ray Data 优化；多节点弹性分片</b></summary>

* 🧩 外部算子插件 — 第三方算子现在可作为独立 pip 包分发，并通过 Python entry points 自动注册。
* 🗄️ HDFS I/O — 新增 `hdfs://` 支持，可用于数据集加载与导出，与现有 S3 路径对齐。
* ⚡ Ray Data 优化 — 移除会强制提前执行的急切操作，迁移至公开的 `TaskPoolStrategy`/`ActorPoolStrategy` API，并行化分区执行，保留弹性 actor pool 并发度，并使分区检查点在 block 布局变化后仍然有效。
* 🌐 多节点弹性分片 — `dj-process-sharded` 通过节点本地 Ray 将大型 JSONL 数据集跨节点弹性处理，支持自动安全降级、重试与有序合并；调度示例见 `demos/elastic_sharding/`。
* 📊 分布式 RayAnalyzer — 新增 `RayAnalyzer`，以 Ray 原生算子计算并汇总数据集统计量，避免 pandas 物化。
* 🧮 内存优化 — 重复度过滤器的流式 n-gram 计数与分块式 MinHash 置换降低了内存峰值（RSS 768 → 272 MiB），且结果不变。
* 🔧 健壮性与依赖修复 — 修复 `text_chunk_mapper` 分隔符泄漏、`calibrate_response_mapper` 的 `output_pattern` 处理以及 `image_diffusion_mapper` 的空 caption 问题；为 `general_field_filter` 增加成员运算符；放宽 `numpy`/`fsspec`/`pandas` 版本约束以兼容 Python 3.13+ 与 pyarrow>=17。
</details>

<details open>
<summary>[2026-07-17] Release v1.5.4: <b>HumanVBench 视频算子；批内阶段算子融合；健壮性修复</b></summary>

* 🧑‍🤝‍🧑 新算子 — 新增 9 个以人为中心的视频理解算子（人物轨迹提取、活跃说话人检测、音频 ASR、语音情绪与年龄/性别检测、人脸人口统计与属性/情绪描述、人脸占比过滤），用于构建 HumanVBench (CVPR'26) 风格的处理流水线。
* ⚡ 批内阶段算子融合 — 新增 `FusedSequentialBatchOp`，在一个 batch 内融合连续算子，减少算子间开销，加速顺序处理。
* 🔧 健壮性与安装修复 — 修复 Ray 去重算子的共享状态处理；修复 Ray checkpoint writer 的相对路径问题；使 `clean_html_mapper` 对 null 文本值更健壮；`ColumnWiseAnalysis` 支持处理布尔型统计列；并通过为 `decord`/`torchcodec` 添加精确的平台标记解锁 ARM64 (aarch64) 平台的安装。
* 🧪 测试覆盖与清理 — 扩展工具函数与模型处理相关的测试，并进行若干代码清理。
</details>

<details>
<summary>[2026-06-26] Release v1.5.3: <b>VLA 算子升级；Ray 重分区管道；可扩展性与健壮性</b></summary>

* 🤖 VLA 算子升级 — 新增/重命名 10+ 个 VLA 算子（DeepCalib/DroidCalib/MoGe 相机标定、原子动作分割、手部动作计算与运动平滑、片段重组、轨迹叠加、LeRobot 导出），并提供完整的 VLA pipeline 示例。
* 🔄 Ray 重分区管道 — 新增 'ray_repartition_pipeline'，支持 Ray 模式下的数据集级块重分区。
* ⚡ 可扩展的 Ray Data 读取 — 将 `override_num_blocks` 贯通整个调用链，支持通过 CLI 控制 PB 级数据集的块并行度。
* 🧪 测试覆盖扩展 — 新增 409 个测试用例，覆盖 18 个测试文件。
* 🐳 稳定性与健壮性修复 — JSONStreamDatasource 跨批次 schema 统一、OP 环境版本解析、PartitionedRayExecutor FUSE 安全 rmtree、废弃模型名更新、num_proc 处理修复等。
</details>

<details>
<summary>[2026-05-29] Release v1.5.2: <b>语义化 LLM 算子；跨文档行级去重；更轻量的依赖</b></summary>

- 🧹 新增去重算子：DocumentLineDeduplicator 支持跨文档的行级去重，依据全局文档频率识别并移除模板文本、版权声明、导航栏等样板行。
- 🤖 Agent 数据质量工具集：上线交互质量算子与配方（recipe），新增 bad-case HTML 报告，并增强 JSONL / HuggingFace meta 加载的健壮性。
- 📦 更轻量、更快的安装：精简默认依赖集（Ray、音频、spaCy、av 等移至按需安装的可选依赖），显著加快安装速度。
- 🐳 稳定性与健壮性修复：库友好的错误处理（以 raise 替代 exit(1)）、Ray 初始化与临时目录修复、清理非法 API 参数（移除无效的 max_new_tokens）、PyArrow 20+ 批量读取 JSON 修复、支持本地路径的 aesthetics 模型，以及更多性能与 Bug 修复。
- 🧠 新增语义化 LLM 算子：引入 llm_extract_mapper、llm_condition_filter 和 llm_structured_ops，统一 llm_* 命名规范，并支持可配置的推理策略（join/agg/top-k 等能力规划中）。
</details>

<details>
<summary>[2026-03-17] Release v1.5.1: <b>LaTeX 算子上线；压缩格式支持；算子健壮性修复</b></summary>

- 📄 新增两个面向 LaTeX 的 Mapper 算子，将 data-juicer 的文档处理能力延伸至 .tex 压缩包和图片上下文的提取与处理。
- 🗜️ 支持压缩数据集格式：现在可以直接加载 json[l].gz 文件，Ray 数据集也同步支持读取压缩 JSON 文件。
- 📚 新增文档，覆盖缓存、导出和执行追踪等工作流，帮助用户更好地理解和调试数据处理流水线。
- 🤖 对 data-juicer-agents 的重大重构与升级已经完成：项目架构及 CLI/会话能力经过全面重新设计，以提升可维护性与可扩展性。详情请参阅 [date-juicer-agents](https://github.com/datajuicer/data-juicer-agents).
</details>

<details>
<summary>[2026-02-12] Release v1.5.0: <b>分区Ray执行器，OP级环境隔离，以及更多具身算子</b></summary>

- 🚀 *分布式执行框架升级* — 新增分区Ray执行器与OP级隔离环境，强化容错性、可扩展性及依赖冲突管理。
- 🤖 *具身AI视频处理能力扩展* — 集成相机校准、视频去畸变、手部重建、位姿估计等专用操作符，提升多视角视频处理能力。
- 💪🏻 *系统性能与开发体验优化* — 支持批处理推理、内存/日志精简、关键逻辑重构，并更新文档与问题模板。
- 🐳 *关键问题修复与稳定性提升* — 修复重复项追踪、参数冲突、首页渲染等缺陷，增强系统可靠性。
</details>

<details>
<summary>[2026-02-02] Release v1.4.6: <b>Copilot、视频字节 I/O 与 Ray 追踪</b></summary>

- 🤖 *Q&A Copilot* — 现已上线我们的[文档站点](https://datajuicer.github.io/data-juicer/zh_CN/main/index_ZH.html) | [钉钉](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?) | [Discord](https://discord.gg/ngQbB9hEVK)。欢迎询问任何与 Data-Juicer 生态系统相关的问题！
    - 查看 🤖 [Data-Juicer Agents](https://github.com/datajuicer/data-juicer-agents/blob/main) | 📃 [部署就绪代码](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot) | 🎬[更多演示](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot/DEMO.md) 了解更多详情。
- 🎬 *视频字节 I/O* — 视频管道的直接字节处理
- 🫆 *Ray 模式追踪器* — 在分布式处理中追踪变更的样本
- 🐳 *增强与修复* — 刷新 Docker 镜像、小幅性能提升、GitHub Insights 流量工作流、Ray 兼容性更新以及 Bug/文档修复。
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

- (CVPR'26) [HumanVBench: Exploring Human-Centric Video Understanding Capabilities of MLLMs with Synthetic Benchmark Data](https://arxiv.org/abs/2412.17574)

- (ICML'26) [DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?](https://www.arxiv.org/abs/2505.16915)

- (Data Scaling) [BiMix: A Bivariate Data Mixing Law for Language Model Pretraining](https://arxiv.org/abs/2405.14908)

</details>
