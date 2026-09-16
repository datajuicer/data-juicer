# Juicer: Natural-Language Data Refinement Model

**Juicer** is a data-refinement model built on [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (MoE architecture: 35B total parameters / 3B activated). It turns natural-language cleaning instructions, filtering rules, and semantic-tagging requirements into structured outputs—strict tagged text or canonical JSON.

Juicer is designed for data-refinement workflows and supports local deployment to process sensitive data in your own environment.

---

## Resources

| Resource | Link |
|----------|------|
| HuggingFace Model | [datajuicer/Juicer-35B-A3B](https://huggingface.co/datajuicer/Juicer-35B-A3B) |
| ModelScope Model | [Data-Juicer/Juicer-35B-A3B](https://www.modelscope.cn/models/Data-Juicer/Juicer-35B-A3B) |
| Juicer Playground | [data-juicer-hub/juicer_playground](https://github.com/datajuicer/data-juicer-hub/tree/main/juicer_playground) |

---

## Quick Start

### 0. Get the launch scripts

The `juicer_playground` directory in data-juicer-hub provides `serve.sh`, `requirements.txt`, and `app.py`. Clone the repository and enter its Playground directory:

```bash
git clone https://github.com/datajuicer/data-juicer-hub.git
cd data-juicer-hub/juicer_playground
```

### 1. Deploy the model

Prepare the inference environment and model weights using the [Playground deployment instructions](https://github.com/datajuicer/data-juicer-hub/tree/main/juicer_playground#1-start-the-model). Juicer can be served as an OpenAI-compatible endpoint. On a single H20 (96 GB):

```bash
export MODEL_ID=/path/to/juicer-model
bash serve.sh --model "$MODEL_ID" --port 8000
```

### 2. Launch the Playground

The [Juicer Playground](https://github.com/datajuicer/data-juicer-hub/tree/main/juicer_playground) provides an interactive UI to try recipes, browse showcase cases, and compare Juicer against the base model:

```bash
pip install -r requirements.txt
export JUICER_BASE_URL=http://localhost:8000/v1
python app.py
# open http://localhost:7860
```

---

## Evaluation

Juicer is evaluated on [CDR-Bench](https://github.com/lukahhcm/data-juicer-hub/tree/CDR-Bench), covering atomic mappers/filters, compositional workflows, order-sensitive pipelines, and semantic tasks (PII, hallucination, rubric, safety).

---

## Learn More

For full deployment options (vLLM, SGLang, Transformers), showcase cases, integration code, and AB comparison setup, visit the [Juicer Playground README](https://github.com/datajuicer/data-juicer-hub/tree/main/juicer_playground).
