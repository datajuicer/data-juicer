

# Data-Juicer: Your Data Operating System for and with Foundation Models

![](https://img.shields.io/badge/language-Python-214870.svg)
![](https://img.shields.io/badge/license-Apache--2.0-000000.svg)
[![pypi version](https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad)](https://pypi.org/project/py-data-juicer)
[![Docker version](https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf)](https://hub.docker.com/r/datajuicer/data-juicer)
[![Docker on OSS](https://img.shields.io/badge/OSS%20latest-none?logo=docker&label=Docker&color=498bdf)](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/docker_images/data-juicer-latest.tar.gz)
![](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FHYLcool%2Ff856b14416f08f73d05d32fd992a9c29%2Fraw%2Ftotal_cov.json)

[![DataModality](https://img.shields.io/badge/DataModality-Text,Image,Audio,Video-brightgreen.svg)](https://datajuicer.github.io/data-juicer/en/main/docs/tutorial/DJ-Cookbook.html)
[![Usage](https://img.shields.io/badge/Usage-Cleaning,Synthesis,Analysis-FFD21E.svg)](https://datajuicer.github.io/data-juicer/en/main/docs/hub/RecipeGallery.html)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/py-data-juicer?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/py-data-juicer)

[![中文主页](https://img.shields.io/badge/中文主页-README_ZH.md-red?logo=github)](README_ZH.md)
[![Document](https://img.shields.io/badge/Document-Website-026cad?logo=readthedocs)](https://datajuicer.github.io/data-juicer/)
[![OperatorZoo](https://img.shields.io/badge/OperatorZoo-200%2B%20Ops-blue?logo=markdown)](https://datajuicer.github.io/data-juicer/en/main/docs/Operators.html)
[![Examples](https://img.shields.io/badge/Examples-Cookbook-brightgreen?logo=github)](https://github.com/datajuicer/data-juicer-hub)

[![Agentic Usage](https://img.shields.io/badge/Agentic%20Usage-Copilot-purple?logo=robot)](https://github.com/datajuicer/data-juicer-agents)
[![Paper](http://img.shields.io/badge/cs.LG-1.0Paper(SIGMOD'24)-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2309.02033)
[![Paper](http://img.shields.io/badge/cs.AI-2.0Paper(NeurIPS'25)-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2501.14755)

**Data-Juicer** is a production-ready, open-source data operating system that powers data workflows for and with foundation models. Beyond training data preparation, Data-Juicer enables the entire data lifecycle—from collection and cleaning to synthesis, analysis, and model co-development. Whether you're building LLMs, VLMs, agents, BI systems, or any intelligent application, Data-Juicer provides the infrastructure to make your data work smarter.

[Platform for AI of Alibaba Cloud (PAI)](https://www.aliyun.com/product/bigdata/learn) has deeply integrated Data-Juicer into its data processing products. PAI is an AI Native large model and AIGC engineering platform that provides dataset management, computing power management, model tool chain, model development, model training, model deployment, and AI asset management. For documentation on data processing, please refer to: [Quickly submit a DataJuicer job](https://www.alibabacloud.com/help/en/pai/user-guide/quickly-submit-a-datajuicer-task).

## 🚀 Quick Start

**Installation:**

```bash
uv pip install py-data-juicer
```

**Quick example:**

```python
from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor

# Initialize config from YAML
cfg = init_configs(['--config', './configs/basic.yaml'])

# Create and run executor
executor = DefaultExecutor(cfg)
dataset = executor.run()
```

👉 **[Try it online](http://8.138.149.181/)** in our managed JupyterLab playground  
👉 **[📓 Notebook Tutorials](https://github.com/datajuicer/data-juicer-hub/tree/notebook)** — Step-by-step guides from basics to advanced topics  
👉 **[Ask DJ Copilot](https://datajuicer.github.io/data-juicer/en/main/docs_index.html)** — "Ask me with DJ everything!"  
👉 **[Browse 200+ Operators](docs/Operators.md)** to see what's possible

## ✨ Why Data-Juicer?

Data-Juicer provides a comprehensive, production-ready solution for data workflows across the entire lifecycle of foundation models. Here's what makes it powerful:

### 🧩 Modular & Extensible
- **200+ reusable operators** covering text, image, audio, video, and multimodal processing
- **50+ pre-built recipes** for common scenarios (pre-training, fine-tuning, multilingual, etc.)
- **Composable pipelines** — use only what you need, extend with your own operators
- **Zero-code processing** — define pipelines in YAML, no Python required for basic workflows
- **Extensible architecture** — add custom operators in minutes, not hours

### ⚡ Production-Ready Performance
- **Cloud-scale processing** — Process 70B samples in 2.1h on 50 Ray nodes (6400 CPU cores)
- **Distributed deduplication** — Deduplicate 5TB of data in 2.8h using 1280 CPU cores
- **Optimized I/O** — Support for S3, local storage, and streaming pipelines
- **Deep optimization** — OP fusion (2-10x speedup), adaptive parallelism, CUDA acceleration
- **Streaming processing** — Handle datasets larger than memory with streaming I/O
- **Battle-tested** — Used in production by Alibaba Cloud PAI, BYD Auto, ByteDance, and more

### 🤝 Seamless Ecosystem Integration
Works seamlessly with your existing stack:
- **Ray** — Native integration for distributed processing with auto-scaling
- **Apache Arrow** — Streaming JSON reader contributions
- **Hugging Face** — Compatible dataset formats
- **ModelScope / LLaMA-Factory** — Unified dataset formats
- **Docker** — Pre-built images for embodied AI and general use

### 🎯 Core Capabilities

**Data Processing & Transformation:**
- **Clean & filter** — Remove duplicates, filter low-quality samples, detect and fix data issues
- **Enhance & synthesize** — Generate captions, extract features, create synthetic data, augment datasets
- **Transform & format** — Convert between formats, normalize structures, prepare for training
- **Analyze & visualize** — Quality metrics, diversity analysis, statistical insights, data profiling

**Production Workflows:**
- **Pre-training** — Clean and deduplicate large-scale web data for foundation model training
- **Fine-tuning** — 20+ specialized operators for instruction following, RLHF, and domain adaptation
- **Multimodal** — Process text, images, videos, and audio in unified pipelines
- **Data-model co-development** — Iterative improvement with feedback loops via [Sandbox](https://datajuicer.github.io/data-juicer-sandbox/en/main/index.html)
- **Agent Development** — Process and structure data for AI agents, RAG systems, and autonomous applications
- **BI & Analytics** — Transform and analyze data for business intelligence, document intelligence, and knowledge extraction

👉 **Explore more use cases and recipes**: [Recipe Gallery](https://github.com/datajuicer/data-juicer-hub) | [DJ-Cookbook](docs/tutorial/DJ-Cookbook.md)

**AI-Powered Features:**
- **DJ Copilot** — Ask questions, get instant help with data processing tasks
- **Auto documentation** — AI-generated operator docs with examples and usage guides
- **Smart recipes** — Pre-built configurations for common scenarios
- **Intelligent agents** — Automated data exploration and pipeline suggestions

**Developer Experience:**
- **Hot reload** — Modify operators and recipes without restarting pipelines
- **Tracer system** — Track exactly what changed in your data, perfect for debugging and auditing
- **Cloud integration** — Direct read/write from cloud modules, no local download needed
- **Active community** — Support via Discord, DingTalk, and GitHub

## 📰 History & News

### Latest Release: v1.4.6
- 🤖 **Q&A Copilot** — Get instant help from our AI assistant in docs, DingTalk, and Discord
- 🎬 **Video Bytes I/O** — Direct bytes reading/storing for video data
- 🫆 **Ray Mode Tracer** — Track changed samples in distributed Ray processing
- 🐳 **Embodied AI Docker** — New Dockerfile optimized for embodied AI workloads

<details>
<summary><b>Recent Releases</b> (Click to expand)</summary>

### v1.4.5: Embodied-AI OPs & Doc System Upgrade
- New embodied-AI operators: video captioning, object segmentation, depth estimation, pose estimation
- Upgraded documentation system with unified Sphinx framework
- Enhanced S3 I/O support and Ray integration improvements

### v1.4.4: NeurIPS Spotlight & Repo Reorganization
- 🎉 Data-Juicer 2.0 paper accepted as NeurIPS'25 Spotlight (top 3.1%)
- Split sandbox, recipes, and agents into independent repos for faster iteration
- New video & multimodal operators for character detection, pose estimation, and more
- S3 I/O support for seamless cloud storage integration

[View all releases →](https://github.com/datajuicer/data-juicer/releases) 
</details>

<details>
<summary><b>Research Highlights</b> (Click to expand)</summary>

- 🎉 **NeurIPS'25 Spotlight**: [Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing](https://arxiv.org/abs/2501.14755)
- 🎉 **NeurIPS'25**: [Diversity as a Reward](https://arxiv.org/abs/2502.04380) and [MindGYM](https://arxiv.org/abs/2503.09499)
- 🎉 **ICML'25 Spotlight**: [Data-Juicer Sandbox](https://arxiv.org/abs/2407.11784)
- 🎉 **CVPR'25**: [ImgDiff: Contrastive Data Synthesis](https://arxiv.org/abs/2408.04594)
- 🎉 **TPAMI'25**: [Data-Model Co-development Survey](https://arxiv.org/abs/2407.08583)

[View all publications →](#references)
</details>

<details>
<summary><b>Community Activity</b> (Click to expand)</summary>

- 🚀 **Growing Community**: Join 1000+ developers using Data-Juicer
- 📈 **Active Development**: Regular releases with new features and improvements
- 🤝 **Open Collaboration**: Contributions from Alibaba, NVIDIA, Ray Team, and more
- 💬 **Active Discussions**: Get help on Discord, DingTalk, and GitHub Discussions
- 📚 **Rich Resources**: 200+ operators, 50+ recipes, comprehensive documentation

[View community stats →](https://github.com/datajuicer/data-juicer/graphs/contributors)
</details>

## 🔗 Integration & Users

Data-Juicer is **used by**, has received **valuable feedback from**, and is **integrated with** a growing open-source community. We strive to keep the list below updated and look forward to including more names (alphabetical order); please reach out if we have missed any acknowledgements.

**Cloud Platforms & Infrastructure:**
- [Alibaba Cloud PAI](https://www.aliyun.com/product/bigdata/learn) — Deeply integrated into enterprise AI platform
- [Apache Arrow](https://github.com/apache/arrow) — Streaming JSON reader contributions and format compatibility
- [Huawei Ascend](https://www.huawei.com/en/products/cloud-computing-dc/atlas/ascend) — Open community integration for AI acceleration
- [Ray](https://docs.ray.io/en/latest/ray-overview/ray-libraries.html) — Official ecosystem integration for distributed processing
- [Volcano Engine](https://www.volcengine.com/) — Open community integration for cloud-native AI workloads

**AI Frameworks & Model Training Tools:**
- [AgentScope](https://github.com/agentscope-ai/agentscope) — Agent development framework
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) — Multimodal synthesis
- [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) — Video generation
- [Eval-Scope](https://github.com/modelscope/evalscope) — Evaluation framework
- [Hugging Face](https://huggingface.co/) — Compatible dataset formats and model integration
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — Model fine-tuning toolkit
- [ModelScope](https://modelscope.cn/) — Unified dataset formats and model ecosystem
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — Open community integration for large language model training
- [RM-Gallery](https://github.com/modelscope/RM-Gallery) — Reward model gallery
- [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) — Reinforcement fine-tuning framework

**Enterprises:**
Alibaba Group, Ant Group, BYD Auto, ByteDance, DTSTACK, JD.com, OPPO, Xiaohongshu, Xiaomi, Ximalaya, and more.

**Academic Institutions:**
CAS, Nanjing University, Peking University, RUC, Tsinghua University, UCAS, Zhejiang University, and more.

## 🔮 Future Works & Roadmap

We're actively expanding Data-Juicer's capabilities and ecosystem. Here's what's coming:

### 🌐 Ecosystem Expansion
- **ModelScope (魔搭)** — Enhanced integration with Alibaba's model ecosystem
- **Alibaba PAI & 百炼** — Deeper integration with enterprise AI platforms
- **Anyscale** — Advanced Ray-based distributed processing
- **Ant Group** — Financial and fintech data processing capabilities
- **NVIDIA NeMo** — Seamless integration for large language model training pipelines

### ⚙️ Engine Enhancements
- **Multi-level fault tolerance** — Robust error handling and recovery mechanisms
- **Ultra-large scale processing** — Support for petabyte-scale datasets
- **Fully managed deployment** — One-click deployment on cloud platforms
- **Adaptive parallel scheduling** — Intelligent resource allocation and optimization
- **Auto-scaling** — Dynamic cluster scaling based on workload
- **Lakehouse pushdown** — Integration with data lake architectures
- **GPU-CPU fusion computing** — Unified compute resource utilization

### 🔧 Operators & Recipes
- **Multimodal processing** — Enhanced support for text, image, video, audio, and 3D data
- **Embodied intelligence** — Specialized operators for robotics and embodied AI
- **Feedback-driven optimization** — Data quality improvement through model feedback loops
- **Intelligent agents** — Data processing pipelines optimized for agent workflows
- **Data attribution** — Traceability and provenance tracking for training data

### 🤝 Community & Collaboration
We're committed to lowering the barrier for contribution and welcoming more community members to build together. Check out our [Developer Guide](docs/DeveloperGuide.md) to get started with your first contribution!

## 📚 Documentation

**🤖 Have questions? Ask [DJ Copilot](https://datajuicer.github.io/data-juicer/en/main/docs_index.html) — "Ask me with DJ everything!"**

👉 **[Full Documentation](https://datajuicer.github.io/data-juicer/en/main/docs_index.html)** — Complete documentation, tutorials, and API reference

**Quick Links:**
- **[DJ-Cookbook](docs/tutorial/DJ-Cookbook.md)** — Real-world examples and recipes
- **[Operator Zoo](docs/Operators.md)** — Browse 200+ operators with examples
- **[Quick Start](docs/tutorial/QuickStart.md)** — Get started in 5 minutes
- **[Developer Guide](docs/DeveloperGuide.md)** — Build your own operators

**More Modular Repos:**
- **[data-juicer](https://github.com/datajuicer/data-juicer)** (this repo) — Core processing framework with 200+ operators
- **[data-juicer-hub](https://github.com/datajuicer/data-juicer-hub)** — Community-driven recipes and best practices
- **[data-juicer-sandbox](https://github.com/datajuicer/data-juicer-sandbox)** — Data-model co-development suite with feedback loops
- **[data-juicer-agents](https://github.com/datajuicer/data-juicer-agents)** — AI copilot for data exploration and processing

## 📄 License
Data-Juicer is released under [Apache License 2.0](LICENSE).

## 🤝 Contributing & Acknowledgements

We welcome contributions at all levels! Whether you're:
- Adding new operators or recipes
- Improving documentation
- Reporting bugs or suggesting features
- Sharing use cases and feedback
- Creating tutorials or examples
- Translating documentation

👉 **[Read our Developer Guide](docs/DeveloperGuide.md)** — We've made it easier than ever to contribute!  
👉 **[Good First Issues](https://github.com/datajuicer/data-juicer/labels/good%20first%20issue)** — Perfect for new contributors  
👉 **[Join our community](https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg)** on Slack, [DingTalk](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?), or [Discord](https://discord.gg/ngQbB9hEVK)

**⭐ Star this repo** to show your support and stay updated!

| [Discord](https://discord.gg/ngQbB9hEVK)                                                                                         | [DingTalk](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?)                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| <img src="https://gw.alicdn.com/imgextra/i1/O1CN011Oj8CB1f8Bw5JpgJA_!!6000000003961-0-tps-762-769.jpg" width="100" height="100"> | <img src="https://gw.alicdn.com/imgextra/i3/O1CN01bBPoaX1EwZsiYudtd_!!6000000000416-2-tps-656-660.png" width="100" height="100"> |

**Acknowledgements:**

Data-Juicer is made possible by our amazing community:

- **Initiated by:** Alibaba Tongyi Lab
- **Co-developed with:** Alibaba Cloud PAI, Anyscale (Ray Team), Sun Yat-sen University, NVIDIA (NeMo Team)
- **Inspired by:** Apache Arrow, BLOOM, RedPajama-Data, Ray, Hugging Face Datasets

See our [contributors](https://github.com/datajuicer/data-juicer/graphs/contributors) for the complete list.

## 📖 References
If you find Data-Juicer useful, please cite our work:
```bibtex
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
<summary><b>More Publications</b> (Click to expand)</summary>

- (ICML'25 Spotlight) [Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development](https://arxiv.org/abs/2407.11784)

- (CVPR'25) [ImgDiff: Contrastive Data Synthesis for Vision Large Language Models](https://arxiv.org/abs/2408.04594)
 
- (TPAMI'25) [The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective](https://arxiv.org/abs/2407.08583)

- (NeurIPS'25) [Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data](https://arxiv.org/abs/2502.04380)

- (NeurIPS'25) [MindGYM: What Matters in Question Synthesis for Thinking-Centric Fine-Tuning?](https://arxiv.org/abs/2503.09499)

- (Benchmark Data) [HumanVBench: Exploring Human-Centric Video Understanding Capabilities of MLLMs with Synthetic Benchmark Data](https://arxiv.org/abs/2412.17574)
 
- (Benchmark Data) [DetailMaster: Can Your Text-to-Image Model Handle Long Prompts?](https://www.arxiv.org/abs/2505.16915)

- (Data Scaling) [BiMix: A Bivariate Data Mixing Law for Language Model Pretraining](https://arxiv.org/abs/2405.14908)

</details>