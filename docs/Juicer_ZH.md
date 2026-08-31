# Juicer：自然语言数据精炼模型

**Juicer** 是基于 [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 构建的数据精炼模型（MoE 架构：35B 总参数 / 3B 激活参数）。它能将自然语言描述的清洗指令、过滤规则和语义标注需求转化为结构化输出——严格的标记文本或规范的 JSON。

Juicer **不是**通用聊天模型，而是专为数据精炼工作流设计，支持本地部署以处理敏感数据。

---

## 资源链接

| 资源 | 链接 |
|------|------|
| HuggingFace 模型 | [datajuicer/Juicer-35B-A3B](https://huggingface.co/datajuicer/Juicer-35B-A3B) |
| ModelScope 模型 | [Data-Juicer/Juicer-35B-A3B](https://www.modelscope.cn/models/Data-Juicer/Juicer-35B-A3B) |
| Juicer Playground | [data-juicer-hub/juicer_playground](https://github.com/datajuicer/data-juicer-hub/tree/main/juicer_playground) |

---

## 快速开始

### 0. 准备启动脚本

下文的 `serve.sh`、`requirements.txt` 和 `app.py` 来自独立的 data-juicer-hub 仓库，不是本仓库中的同名文件。在已按该仓库部署说明准备好推理环境和模型权重后，进入其 Playground 目录：

```bash
git clone https://github.com/datajuicer/data-juicer-hub.git
cd data-juicer-hub/juicer_playground
```

### 1. 部署模型

Juicer 可作为 OpenAI 兼容端点提供服务。在单张 H20（96 GB）上：

```bash
export MODEL_ID=/path/to/juicer-model
bash serve.sh --model "$MODEL_ID" --port 8000
```

### 2. 启动 Playground

[Juicer Playground](https://github.com/datajuicer/data-juicer-hub/tree/main/juicer_playground) 提供交互式界面，可试用配方、浏览展示用例、对比 Juicer 与基座模型：

```bash
pip install -r requirements.txt
export JUICER_BASE_URL=http://localhost:8000/v1
python app.py
# 打开 http://localhost:7860
```

---

## 评测

Juicer 在 [CDR-Bench](https://github.com/lukahhcm/data-juicer-hub/tree/CDR-Bench) 上进行评测，涵盖原子映射/过滤、组合工作流、顺序敏感管道和语义任务（PII、幻觉、量规、安全）。

---

## 了解更多

完整的部署选项（vLLM、SGLang、Transformers）、展示用例、集成代码和 AB 对比设置，请访问 [Juicer Playground README](https://github.com/datajuicer/data-juicer-hub/tree/main/juicer_playground)。
