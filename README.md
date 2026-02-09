#  Data-Juicer: The Data Operating System for the Foundation Model Era
<p align="center">
  <a href="https://pypi.org/project/py-data-juicer"><img src="https://img.shields.io/pypi/v/py-data-juicer?logo=pypi&color=026cad" alt="PyPI"></a>
  <a href="https://pepy.tech/projects/py-data-juicer"><img src="https://static.pepy.tech/personalized-badge/py-data-juicer?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=green&left_text=downloads" alt="Downloads"></a>
   <a href="https://hub.docker.com/r/datajuicer/data-juicer"><img src="https://img.shields.io/docker/v/datajuicer/data-juicer?logo=docker&label=Docker&color=498bdf" alt="Docker"></a>
  <br>
  <a href="https://datajuicer.github.io/data-juicer/"><img src="https://img.shields.io/badge/📖_Docs-Website-026cad" alt="Docs"></a>
  <a href="https://datajuicer.github.io/data-juicer/en/main/docs/Operators.html"><img src="https://img.shields.io/badge/🧩_Operators-200+-blue" alt="Operators"></a>
  <a href="https://github.com/datajuicer/data-juicer-hub"><img src="https://img.shields.io/badge/🍳_Recipes-50+-brightgreen" alt="Recipes"></a>
  <br>
  <a href="https://datajuicer.github.io/data-juicer/zh_CN/main/index_ZH.html"><img src="https://img.shields.io/badge/🇨🇳_文档-主页-red" alt="Chinese"></a>
  <a href="https://arxiv.org/abs/2501.14755"><img src="https://img.shields.io/badge/NeurIPS'25_Spotlight-2.0-B31B1B?logo=arxiv" alt="Paper"></a>
  <a href="https://github.com/datajuicer/data-juicer">
    <img src="https://img.shields.io/endpoint?style=flat&url=https%3A%2F%2Fgist.githubusercontent.com%2FHYLcool%2Ff856b14416f08f73d05d32fd992a9c29%2Fraw%2Ftotal_cov.json&label=coverage&logo=codecov&color=4c1" alt="Coverage">
  </a>
</p>

<p align="center">
  <b>Multimodal | Cloud-Native | AI-Ready | Large-Scale </b>
</p>

Data-Juicer (DJ) transforms raw data chaos into AI-ready intelligence. It treats data processing as *composable infrastructure*—providing modular building blocks to clean, synthesize, and analyze data across the entire AI lifecycle, unlocking latent value in every byte.

Whether you're deduplicating web-scale pre-training corpora, curating agent interaction traces, or preparing domain-specific RAG indices, DJ scales seamlessly from your laptop to thousand-node clusters—no glue code required.

> **Alibaba Cloud PAI** has deeply integrated Data-Juicer into its data processing products.  See **[Quickly submit a DataJuicer job](https://www.alibabacloud.com/help/en/pai/user-guide/quickly-submit-a-datajuicer-task)**.

---

## 🚀 Quick Start

**Zero-install exploration**: 
- [JupyterLab Playground with Tutorials](http://8.138.149.181/) 
- [Ask DJ Copilot](https://datajuicer.github.io/data-juicer/en/main/docs_index.html)

**Install & run**:
```bash
uv pip install py-data-juicer
dj-process --config demos/process_simple/process.yaml
```

**Or compose in Python**:
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

## ✨ Why Data-Juicer?

### 1. Modular & Extensible Architecture
- **200+ operators** spanning text, image, audio, video, and multimodal data
- **Recipe-first**: Reproducible YAML pipelines you can version, share, and fork like code
- **Composable**: Drop in a single operator, chain complex workflows, or orchestrate full pipelines
- **Hot-reload**: Iterate on operators without pipeline restarts

### 2. Full-Spectrum Data Intelligence
- **Foundation Models**: Pre-training, fine-tuning, RL, and evaluation-grade curation
- **Agent Systems**: Clean tool traces, structure context, de-identification, and quality gating
- **RAG & Analytics**: Extraction, normalization, semantic chunking, deduplication, and data profiling


### 3. Production-Ready Performance
- **Scale**: Process 70B samples in 2h on 50 Ray nodes (6400 cores)
- **Efficiency**: Deduplicate 5TB in 2.8h using 1280 cores
- **Optimization**: Automatic OP fusion (2-10x speedup), adaptive parallelism, CUDA acceleration, robustness
- **Observability**: Built-in tracing for debugging, auditing, and iterative improvement

> *⭐ If Data-Juicer saved you time or improved your data work, please consider starring the repo.* It helps more people discover the project and keeps you notified of new releases and features.

---

## 📰 News

<details open>
<summary>[2026-02-02] Release v1.4.6: <b>Copilot, Video Bytes I/O & Ray Tracing </b></summary>

- 🤖 *Q&A Copilot* —  Now live on our [Doc Site](https://datajuicer.github.io/data-juicer/en/main/index.html) | [DingTalk](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?) | [Discord](https://discord.gg/ngQbB9hEVK). Feel free to ask anything related to Data-Juicer ecosystem!  
    - Check 🤖 [Data-Juicer Agents](https://github.com/datajuicer/data-juicer-agents/blob/main) | 📃 [Deploy-ready codes](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot) | 🎬[ More demos](https://github.com/datajuicer/data-juicer-agents/blob/main/qa-copilot/DEMO.md) for more details.
- 🎬 *Video Bytes I/O* — Direct bytes processing for video pipelines  
- 🫆 *Ray Mode Tracer* — Track changed samples in distributed processing  
- 🐳 *Enhancements & fixes* — refreshed Docker image, small perf boosts, GitHub Insights traffic workflow, Ray compatibility updates, and bug/doc fixes.
</details>

<details open>
<summary>[2026-01-15] Release v1.4.5: <b>20+ New OPs, Ray vLLM Pipelines & Sphinx Docs Upgrade</b> </summary>

- *Embodied-AI OPs*: added/enhanced mappers for video captioning (VLM), video object segmentation (YOLOE+SAM2), video depth estimation (viz + point cloud), human pose (MMPose), image tagging (VLM), single-image 3D body mesh recovery (SAM 3D Body), plus *S3 upload/download*.
- *New Pipeline OP*: compose multiple OPs into one pipeline; introduced *Ray + vLLM* pipelines for LLM/VLM inference.
- *Docs upgrade*: moved to a unified *Sphinx-based* documentation build/deploy workflow with isolated theme/architecture repo.
- *Enhancements & fixes*: dependency updates, improved Ray deduplication and S3 loading, OpenAI Responses API support, tracer consistency, Docker base updated to CUDA 12.6.3 + Ubuntu 24.04 + Py3.11, and multiple bug fixes. 

</details>

<details>
<summary>[2025-12-01] Release v1.4.4: <b>NeurIPS’25 Spotlight, 6 New Video/MM OPs & S3 I/O</b> </summary>

- NeurIPS'25 **Spotlight** for Data-Juicer 2.0
- *Repo split*: sandbox/recipes/agents moved to standalone repos
- *S3 I/O* added to loader/exporter
- *6 new video & multimodal OPs* (character detection, VGGT, whole-body pose, hand reconstruction) + docs/Ray/video I/O improvements and bug fixes
</details>

View [All Release](https://github.com/datajuicer/data-juicer/releases) and [News Archive](docs/news.md)

---

## 🔌 Users & Ecosystems
> The below list focuses on *developer-facing integration and usages* in *alphabetical order*.  
> Missing your project / name? Feel free to [open a PR](https://github.com/datajuicer/data-juicer/pulls) or [reach out](#contributing--community).

Data-Juicer plugs into your existing stack and evolves with community contributions:

### Extensions
- **[data-juicer-agents](https://github.com/datajuicer/data-juicer-agents)** — DJ Copilot and agentic workflows  
- **[data-juicer-hub](https://github.com/datajuicer/data-juicer-hub)** — Community recipes and best practices  
- **[data-juicer-sandbox](https://github.com/datajuicer/data-juicer-sandbox)** — Data-model co-development with feedback loops  


### Frameworks & Platforms
[AgentScope](https://github.com/agentscope-ai/agentscope) · [Apache Arrow](https://github.com/apache/arrow) · [Apache HDFS](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsUserGuide.html) · [Apache Hudi](https://hudi.apache.org/) · [Apache Iceberg](https://iceberg.apache.org/) · [Apache Paimon](https://paimon.apache.org/) · [Alibaba PAI](https://www.alibabacloud.com/en/product/machine-learning?_p_lc=1) · [Delta Lake](https://delta.io/) · [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) · [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) · [Eval-Scope](https://github.com/modelscope/evalscope) · [Huawei Ascend](https://www.huawei.com/en/products/cloud-computing-dc/atlas/ascend) · [Hugging Face](https://huggingface.co/) · [LanceDB](https://lancedb.github.io/lance/) · [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) · [ModelScope](https://modelscope.cn/) · [ModelScope Swift](https://github.com/modelscope/ms-swift) · [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) · [Ray](https://docs.ray.io/) · [RM-Gallery](https://github.com/modelscope/RM-Gallery) · [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) · [Volcano Engine](https://www.volcengine.com/)

### Industry
Alibaba Group, Ant Group, BYD Auto, ByteDance, DTSTACK, JD.com, NVIDIA, OPPO, Xiaohongshu, Xiaomi, Ximalaya, and more.

### Academia
CAS, Nanjing University, Peking University, RUC, Tsinghua University, UCAS, Zhejiang University, and more.


###  Contributing & Community
We believe in *building together*. Whether you're fixing a typo, crafting a new operator, or sharing a breakthrough recipe, every contribution shapes the future of data processing.

We welcome contributions at all levels: 
- **[Good First Issues](https://github.com/datajuicer/data-juicer/labels/good%20first%20issue)** — Add operators, improve docs, report issues, or fix bugs
- **[Developer Guide](https://datajuicer.github.io/data-juicer/en/main/docs/DeveloperGuide.html)** — Optimize engines, add features, or enhance core infrastructure
- **[DJ-Hub](https://github.com/datajuicer/data-juicer-hub)** — Share knowledge: recipes, papers, and best practices
- **Connect**: [Slack](https://join.slack.com/t/data-juicer/shared_invite/zt-23zxltg9d-Z4d3EJuhZbCLGwtnLWWUDg) · [DingTalk](https://qr.dingtalk.com/action/joingroup?code=v1,k1,N78tgW54U447gJP5aMC95B6qgQhlkVQS4+dp7qQq6MpuRVJIwrSsXmL8oFqU5ajJ&_dt_no_comment=1&origin=11?) · [Discord](https://discord.gg/ngQbB9hEVK)

| Discord | DingTalk |
|:---:|:---:|
| <img src="https://gw.alicdn.com/imgextra/i1/O1CN011Oj8CB1f8Bw5JpgJA_!!6000000003961-0-tps-762-769.jpg" width="100"> | <img src="https://gw.alicdn.com/imgextra/i3/O1CN01bBPoaX1EwZsiYudtd_!!6000000000416-2-tps-656-660.png" width="100"> |


Data-Juicer is made possible by the users and community:
- **Initiated by**: Alibaba Tongyi Lab  
- **Co-developed with**: Alibaba Cloud PAI, Anyscale (Ray team), Sun Yat-sen University, NVIDIA (NeMo team), and [contributors worldwide](https://github.com/datajuicer/data-juicer/graphs/contributors)
- **Inspired by**: Apache Arrow, Ray, Hugging Face Datasets, BLOOM, RedPajama-Data, ...

---


## Documentation

For detailed documentation, please see [here](https://datajuicer.github.io/data-juicer/en/main/docs_index.html).

**Quick Links:**
- **[operator zoo](https://datajuicer.github.io/data-juicer/en/main/docs/Operators.html)** — Browse 200+ operators with examples
- **[data-juicer-hub](https://github.com/datajuicer/data-juicer-hub)** — Community-driven recipes and best practices
- **[developer guide](https://datajuicer.github.io/data-juicer/en/main/docs/DeveloperGuide.html)** — Build your own code and contribute to DJ 
- **[data-juicer-cookbook](https://datajuicer.github.io/data-juicer/en/main/docs/tutorial/DJ-Cookbook.html)** — resource archive
- **[awesome_llm_data](https://datajuicer.github.io/data-juicer/en/main/docs/awesome_llm_data)** —  “Awesome List” for data-model co-development


---

## 📄 License & Attribution

Data-Juicer is released under the [Apache License 2.0](LICENSE).  
Attribution is appreciated: please use our [badge](https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/assets/DJ-Org-Logo.jpeg), or text as "This project uses Data-Juicer: https://github.com/datajuicer".

---

## 📖 Citation
If you find Data-Juicer useful in your work, please cite:

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

