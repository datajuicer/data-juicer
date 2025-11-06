# DataJuicer 智能体

基于 [AgentScope](https://github.com/agentscope-ai/agentscope) 和 [Data-Juicer (DJ)](https://github.com/datajuicer/data-juicer) 构建的数据处理多智能体系统。该项目展示了如何利用大模型的自然语言理解能力，让非专家用户也能轻松使用 Data-Juicer 的强大数据处理能力。

## 🎯 为什么需要 DataJuicer Agent？

在大模型研发和应用的实际工作中，**数据处理仍然是一个高成本、低效率、难复现的环节**。很多团队花在数据分析、清洗、合成等阶段的时间，往往超过模型训练、需求对齐、应用功能开发。

我们希望通过智能体技术，把开发者从繁琐的脚本拼凑中解放出来，让数据研发更接近"所想即所得"的体验。

**数据直接定义了模型能力的上限**。真正决定模型表现的，是数据的**质量、多样性、有害性控制、任务匹配度**等多个维度。优化数据，本质上就是在优化模型本身。而要高效地做这件事，我们需要一套系统化的工具。

DataJuicer Agent 正是为支撑**数据与模型协同优化**这一新范式而设计的智能协作系统。

## 📋 目录

- [🎯 为什么需要 DataJuicer Agent？](#-为什么需要-datajuicer-agent)
- [📋 目录](#-目录)
- [这个智能体做了什么？](#这个智能体做了什么)
- [架构](#架构)
  - [多智能体路由架构](#多智能体路由架构)
  - [两种集成方式](#两种集成方式)
- [快速开始](#快速开始)
  - [系统要求](#系统要求)
  - [安装](#安装)
  - [配置](#配置)
  - [使用](#使用)
- [智能体介绍](#智能体介绍)
  - [数据处理智能体](#数据处理智能体)
  - [代码开发智能体](#代码开发智能体)
- [高级功能](#高级功能)
  - [算子检索](#算子检索)
    - [检索模式](#检索模式)
    - [使用](#使用-1)
  - [MCP 智能体](#mcp-智能体)
    - [MCP 服务器类型](#mcp-服务器类型)
    - [配置](#配置-1)
    - [使用方法](#使用方法)
- [定制化与扩展](#定制化与扩展)
  - [自定义 Prompts](#自定义-prompts)
  - [更换模型](#更换模型)
  - [扩展新智能体](#扩展新智能体)
- [Roadmap](#roadmap)
  - [Data-Juicer 问答智能体 (演示可用)](#data-juicer-问答智能体-演示可用)
  - [交互式数据分析与可视化智能体 (开发中)](#交互式数据分析与可视化智能体-开发中)
  - [其它方向](#其它方向)
  - [常见问题](#常见问题)
  - [优化建议](#优化建议)
- [相关资源](#相关资源)

## 这个智能体做了什么？

Data-Juicer (DJ) 是一个**覆盖大模型数据全生命周期的开源处理系统**，提供四个核心能力：

- **全栈算子库（DJ-OP）**：近 200 个高性能、可复用的多模态算子，覆盖文本、图像、音视频
- **高性能引擎（DJ-Core）**：基于 Ray 构建，支持 TB 级数据、万核分布式计算，具备算子融合与多粒度容错
- **协同开发平台（DJ-Sandbox）**：引入 A/B Test 与 Scaling Law 思想，用小规模实验驱动大规模优化
- **自然语言交互层（DJ-Agents）**：通过 Agent 技术，让开发者用对话方式构建数据流水线

DataJuicer Agent 不是一个简单的问答机器人，而是一个**数据处理的智能协作者**。具体来说，它能：

- **智能查询**：根据自然语言描述，自动匹配最合适的算子（从近200个算子中精准定位）
- **自动化流程**：描述数据处理需求，自动生成 Data-Juicer YAML 配置并执行
- **自定义扩展**：帮助用户开发自定义算子，无缝集成到本地环境

**我们的目标是：让开发者专注于"做什么"，而不是"怎么做"**。

## 架构

### 多智能体路由架构

DataJuicer Agent 采用**多智能体路由架构**，这是系统可扩展性的关键。当用户输入一个自然语言请求，首先由 **Router Agent** 进行任务分诊，判断这是标准的数据处理任务，还是需要开发新能力的定制需求。

```
用户查询
    ↓
Router Agent (任务分诊)
    ├── 标准数据处理任务 → Data Processing Agent (DJ Agent)
    │   ├── 预览数据样本（确认字段名和数据格式）
    │   ├── query_dj_operators (基于语义匹配算子)
    │   ├── 生成 YAML 配置文件
    │   └── execute_safe_command (执行 dj-process, dj-analyze)
    │
    └── 自定义算子开发 → Code Development Agent (DJ Dev Agent)
        ├── get_basic_files (获取基类和注册机制)
        ├── get_operator_example (获取相似算子示例)
        ├── 生成符合规范的算子代码
        └── 本地集成（注册到用户指定路径）
```

### 两种集成方式

Agent 与 DataJuicer 的集成有两种方式，以适应不同使用场景：

- **绑定工具模式**：Agent 调用 DataJuicer 的命令行工具（如 `dj-analyze`、`dj-process`），兼容现有用户习惯，迁移成本低
- **绑定 MCP 模式**：Agent 直接调用 DataJuicer 的 MCP（Model Context Protocol）接口，无需生成中间 YAML 文件，直接运行算子或数据菜谱，性能更优

这两种方式由 Agent 根据任务复杂度和性能需求自动选择，确保灵活性与效率兼得。

## 快速开始

### 系统要求

- Python 3.10+
- 有效的 DashScope API 密钥
- 可选：Data-Juicer 源码（用于自定义算子开发）

### 安装

```bash
# 推荐使用uv
uv pip install -r requirements.txt
```

或

```bash
pip install -r requirements.txt
```

### 配置

1. **设置 API 密钥**

```bash
export DASHSCOPE_API_KEY="your-dashscope-key"
```

2. **可选：配置 Data-Juicer 路径（用于自定义算子开发）**

```bash
export DATA_JUICER_PATH="your-data-juicer-path"
```

> **提示**：也可以在运行时通过对话设置，例如：
> - "帮我设置 DataJuicer 路径：/path/to/data-juicer"
> - "帮我更新 DataJuicer 路径：/path/to/data-juicer"

### 使用

通过 `-u` 或 `--use_studio` 参数选择运行方式：

```bash
# 使用 AgentScope Studio 的交互式界面（请先安装并启动 AgentScope Studio）
python main.py --use_studio True

# 或直接使用命令行模式（默认）
python main.py
```

注：

AgentScope Studio 通过 npm 安装：

```bash
npm install -g @agentscope/studio
```

使用以下命令启动 Studio：

```bash
as_studio
```

## 智能体介绍

### 数据处理智能体

负责与 Data-Juicer 交互，执行实际的数据处理任务。支持从自然语言描述自动推荐算子、生成配置并执行。

**工作流程：**

当用户说："我的数据保存在 xxx，请清理其中文本长度小于5、图片大小小于10MB的条目"，Agent 并不会盲目执行，而是按步骤推进：

1. **数据预览**：预览前 5–10 个数据样本，确认字段名和数据格式——这是避免配置错误的关键一步
2. **算子检索**：调用 `query_dj_operators` 工具，基于语义匹配合适的算子
3. **参数决策**：LLM 自主决定全局参数（如 dataset_path、export_path）和算子具体配置
4. **配置生成**：生成标准的 YAML 配置文件
5. **执行处理**：调用 `dj-process` 命令执行实际处理

整个过程既自动化，又具备可解释性。用户可以在任何环节介入干预，确保结果符合预期。

**典型用途：**
- **数据清洗**：去重、移除低质量样本、格式标准化
- **多模态处理**：同时处理文本、图像、视频数据
- **批量转换**：格式转换、数据增强、特征提取

<details>
<summary>查看完整示例日志（from AgentScope Studio）</summary>
<img src="assets/dj_agent_image.png" width="100%">
</details>

**示例执行流程：**

用户输入："The data in ./data/demo-dataset-images.jsonl, remove samples with text field length less than 5 and image size less than 100Kb..."

Agent 执行步骤：
1. 调用 `query_dj_operators`，精准返回两个算子：`text_length_filter` 和 `image_size_filter`
2. 用 `view_text_file` 工具预览原始数据，确认字段确实是 'text' 和 'image'
3. 生成 YAML 配置，并通过 `write_text_file` 保存到临时路径
4. 调用 `execute_safe_command` 执行 `dj-process`，返回结果路径

整个过程没有人工干预，但每一步都可追溯、可验证。**这正是我们追求的"自动化但不失控"的数据处理体验**。

### 代码开发智能体

当内置算子无法满足需求时，传统做法是：查文档、抄代码、调参数、写测试——整个过程可能耗时数小时。

Operator Development Agent 的目标，是将这个过程压缩到几分钟，并保证代码质量。默认使用 `qwen3-coder-480b-a35b-instruct` 模型驱动。

**工作流程：**

当用户提出："帮我创建一个将单词倒序排列的算子，并生成单元测试文件"，Router 会将其路由至 DJ Dev Agent。

该 Agent 的执行流程分为四步：

1. **算子检索**：查找功能相似的现有算子作为参考
2. **获取模板**：拉取基类文件和典型示例，确保代码风格一致
3. **生成代码**：基于用户提供的函数原型，生成符合 DataJuicer 规范的算子类
4. **本地集成**：将新算子注册到用户指定的本地代码库路径

整个过程将模糊需求转化为可运行、可测试、可复用的模块。

**生成内容：**

- **实现算子**：创建算子类文件，继承 Mapper/Filter 基类，使用 `@OPERATORS.register_module` 装饰器注册
- **更新注册**：修改 `__init__.py`，将新类加入 `__all__` 列表
- **编写测试**：生成覆盖多种场景的单元测试，包括边缘 case，确保鲁棒性

**典型用途：**
- **开发领域特定的过滤或转换算子**
- **集成自有的数据处理逻辑**
- **为特定场景扩展 Data-Juicer 能力**

<details>
<summary>查看完整示例日志（from AgentScope Studio）</summary>
<img src="assets/dj_dev_agent_image.png" width="100%">
</details>

## 高级功能

### 算子检索

算子检索是 Agent 能否精准工作的核心。DJ 智能体实现了一个智能算子检索工具，通过独立的 LLM 查询环节从 Data-Juicer 的近200个算子中快速找到最相关的算子。这是数据处理智能体和代码开发智能体能够准确运行的关键组件。

我们没有采用单一方案，而是提供了三种模式，通过 `-r` 参数灵活选择：

#### 检索模式

**LLM 检索 (默认)**
- 使用 Qwen-Turbo 从语义层面理解用户需求，适合复杂、模糊的描述
- 提供详细的匹配理由和相关性评分
- Token 消耗较高，但匹配精度最高

**向量检索 (vector)**
- 基于 DashScope 文本嵌入 + FAISS 相似度搜索
- 速度快，适合批量任务或快速原型
- 无需调用 LLM，成本更低

**自动模式 (auto)**
- 优先尝试 LLM 检索，失败时自动降级到向量检索

#### 使用

通过 `-r` 或 `--retrieve_mode` 参数指定检索模式：

```bash
python main.py --retrieve_mode vector
```

更多参数说明见 `python main.py --help`

### MCP 智能体

除了命令行，DataJuicer 还原生支持 MCP 服务，这是提升性能的重要手段。MCP 服务可直接通过原生接口获取算子信息、执行数据处理，易于迁移和集成，无需单独的 LLM 查询和命令行调用。

#### MCP 服务器类型

Data-Juicer 提供两类 MCP：

**Recipe-Flow MCP（数据菜谱）**
- 提供 `get_data_processing_ops` 和 `run_data_recipe` 两个工具
- 通过算子类型、适用模态等标签进行检索，**无需调用 LLM 或向量模型**
- 适合标准化、高频场景，性能更优

**Granular-Operators MCP（细粒度算子）**
- 将每个内置算子包装为独立工具，调用即运行
- 默认返回所有算子，但可通过环境变量控制可见范围
- 适合精细化控制，构建完全定制化的数据处理管道

这意味着，在某些场景下，Agent 的调用路径可以比手动写 YAML *更短、更快、更直接*。

详细信息请参考：[Data-Juicer MCP 服务文档](https://datajuicer.github.io/data-juicer/en/main/docs/DJ_service.html#mcp-server)

> **注意**：Data-Juicer MCP 服务器目前处于早期开发阶段，功能和工具可能会随着持续开发而变化。

#### 配置

在 `configs/mcp_config.json` 中配置服务地址：

```json
{
    "mcpServers": {
        "DJ_recipe_flow": {
            "url": "http://127.0.0.1:8080/sse"
        }
    }
}
```

#### 使用方法

启用 MCP 智能体替代 DJ 智能体：

```bash
# 启用 MCP 智能体和开发智能体
python main.py --available_agents [dj_mcp,dj_dev]

# 或使用简写
python main.py -a [dj_mcp,dj_dev]
```


## 定制化与扩展

### 自定义 Prompts

所有 Agent 的系统提示词都定义在 `prompts.py` 文件中。

### 更换模型

你可以在 `main.py` 中为不同 Agent 指定不同模型。例如：
- 主 Agent 使用 `qwen-max` 处理复杂推理
- 开发 Agent 使用 `qwen3-coder-480b-a35b-instruct` 优化代码生成质量

同时，Formatter 和 Memory 也可替换。这种设计让系统既能开箱即用，又能适配企业级需求。

### 扩展新智能体

DataJuicer Agent 是一个开放框架。核心在于 `agents2toolkit` 函数——它能将任意 Agent 自动包装为 Router 可调用的工具。

只需将你的 Agent 实例加入 `agents` 列表，Router 就会在运行时动态生成对应工具，并根据任务语义自动路由。

这意味着，你可以基于此框架，快速构建领域专属的数据智能体。

*扩展性，是我们设计的重要原则*。

## Roadmap

Data-Juicer 智能体生态系统正在快速扩展，以下是当前正在开发或计划中的新智能体：

### Data-Juicer 问答智能体 (演示可用)

为用户提供关于 Data-Juicer 算子、概念和最佳实践的详细解答。

<video controls width="100%" height="auto" playsinline>
    <source src="https://github.com/user-attachments/assets/a8392691-81cf-4a25-94da-967dcf92c685" type="video/mp4">
    您的浏览器不支持视频标签。
</video>

### 交互式数据分析与可视化智能体 (开发中)

我们正在构建更高级的**人机协同数据优化工作流**，引入人类反馈：
- 用户可查看统计、归因分析以及可视化结果
- 动态编辑菜谱，批准或拒绝建议
- 底层由 `dj.analyzer`（数据分析）、`dj.attributor`（效果归因）、`dj.sandbox`（实验管理）共同支撑
- 支持基于验证任务的闭环优化

### 其它方向

- **数据处理智能体 Benchmarking**：量化不同 Agent 在准确性、效率、鲁棒性上的表现
- **数据"体检报告" & 数据智能推荐**：自动诊断数据问题并推荐优化方案
- **Router Agent 增强**：更无感丝滑，譬如当缺乏算子时→代码开发Agent→数据处理agent
- **MCP 进一步优化**：内嵌 LLM，用户可直接使用 MCP 链接自己本地环境如IDE，获得目前数据处理 agent 类似的体验
- **面向知识库、RAG 的数据智能体**
- **更好的处理方案自动生成**：更少 token 用量，更高效，更优质处理结果
- **数据工作流模版复用及自动调优**：基于 DataJuicer 社区数据菜谱
- ......

### 常见问题

**Q: 如何获取 DashScope API 密钥？**
A: 访问 [DashScope 官网](https://dashscope.aliyun.com/) 注册账号并申请 API 密钥。

**Q: 为什么算子检索失败？**
A: 请检查网络连接和 API 密钥配置，或尝试切换到向量检索模式。

**Q: 如何调试自定义算子？**
A: 确保 Data-Juicer 路径配置正确，并查看代码开发智能体提供的示例代码。

**Q: MCP 服务连接失败怎么办？**
A: 检查 MCP 服务器是否正在运行，确认配置文件中的 URL 地址正确。

**Q: 报错requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: http://localhost: 3000/trpc/pushMessage**
A: 请检查是否agentscope studio已经成功拉起。尝试先`npm install -g @agentscope/studio`下载agentscope studio，然后`as_studio`启动。

### 优化建议

- 对于大规模数据处理，建议使用DataJuicer提供的分布式模式
- 合理设置批处理大小以平衡内存使用和处理速度
- 更多进阶数据处理（合成、Data-Model Co-Development）等特性能力请参考DataJuicer[文档页](https://datajuicer.github.io/data-juicer/zh_CN/main/index_ZH)

---

## 相关资源
- DataJuicer 已经被用于大量通义和阿里云内外部用户，背后也衍生了多项研究。所有代码持续维护增强中。

*欢迎访问 GitHub，Star、Fork、提 Issue，以及加入社区共建！*
- **项目地址**：
  - [AgentScope](https://github.com/agentscope-ai/agentscope)
  - [DataJuicer](https://github.com/datajuicer/data-juicer)

**贡献指南**：欢迎提交 Issue 和 Pull Request 来改进 agentscope、DataJuicer Agent 及 DataJuicer。如果您在使用过程中遇到问题或有功能建议，请随时联系我们。
