# Agent training-dataset recipes（训练数据价值）

可组合DJ YAML，把 agent 交互数据 从 **分析审计洞察** 延伸到 **training-ready 数据**。与全量配方
[`agent_interaction_quality_analysis.yaml`](agent_interaction_quality_analysis.yaml)（偏 stats / dialog_* / 报告）并列使用。

**字段 → 训练用途**（归因链、jq 示例、局限）：[`TRAIN_DATA_FIELD_GUIDE.md`](TRAIN_DATA_FIELD_GUIDE.md)。

**两次导出对比**（验证迭代）：`python demos/agent/scripts/diff_agent_exports.py --help`

---

## 两条推荐路径

| 路径 | 适用 | 顺序 |
|------|------|------|
| **A · 原始交互数据优先** | 手里是 raw JSONL，希望一条龙打上 lineage、PK、噪声与工具标签 | **R1** → **R2** → **R3** |
| **B · 已跑全量分析** | 已有 `agent_interaction_quality_analysis.yaml` 的 `processed.jsonl`（含 `dialog_*`、`stats.llm_*`），只需补 **跨模型 cohort + id 去重** 再接价值栈 | **[``agent_interaction_quality_analysis.yaml``](agent_interaction_quality_analysis.yaml)** → **R0** → **R2** → **R3** |

路径 B 下：将 **`R2_train_data_stack.yaml`** 的 `dataset_path` 指到 **`R0_bridge_from_analysis.yaml`** 的 `export_path`（默认 `./outputs/agent_train_data_R0/from_analysis.jsonl`）。  
路径 B 要求导出里已有与 R1 一致的 **`agent_lineage_*`**；若缺，请先在 **原始 JSONL** 上跑一段与 `R1_initial_filter.yaml` 相同的 `agent_dialog_normalize_mapper`（`lineage_extra_keys` 等），再跑分析或 R0。

**语义去重**：仅在「单模型、同 query」语料上启用 `R1_initial_filter.yaml` 里注释的 `document_minhash_deduplicator`；否则跨模型 PK 行会被折叠。

**无同一 `<query, env>` / lineage id 时的 cohort**：在 `agent_cross_model_pair_mapper` 上设置  
`group_key_mode: normalized_query`（归一化 query 精确同组）或 `simhash_lsh`（近重复 query + 可选 `extra_group_text_key` 把 env 摘要拼进 SimHash 输入）。详见算子 docstring 与 [`TRAIN_DATA_FIELD_GUIDE.md`](TRAIN_DATA_FIELD_GUIDE.md)。

---

## Pipeline 阶段一览

| Stage | File | Role |
|--------|------|------|
| **R0** | `R0_bridge_from_analysis.yaml` | 在 **全量分析导出** 上追加 **跨模型 pairwise** + **按 id 精确去重**；输出 `outputs/agent_train_data_R0/from_analysis.jsonl`。 |
| **R1** | `R1_initial_filter.yaml` | 从 raw 做 normalize、跨模型、去重、sys_log/harness、usage、tool 类型等。输出 `outputs/agent_train_data_R1/processed.jsonl`。 |
| **R2** | `R2_train_data_stack.yaml` | Taxonomy、learnable value、bad-case signals；可选 `agent_insight_llm_mapper`。输入：R0 或 R1 导出。输出 `outputs/agent_train_data_R2/processed.jsonl`。 |
| **R3** | `R3_post_process.yaml` | PII（regex + LLM）、copyright/HTML、**安全 + 蒸馏 + 改写提示**（API），再写 `meta.agent_training_card`。训练数据集输出：`outputs/agent_train_data_R3/train_data.jsonl`。 |
| **R3 CPU** | `R3_post_process_cpu_only.yaml` | 无远程 LLM；确定性清洗 + training card（冒烟 / 离线）。 |

**硬约束**

- **跨模型 pairing 必须早于** 对 `query` 的语义去重（见 R1 头注释）。
- **R3** 要求每行已有 `text`、`messages`（通常来自分析或 R1 的 `agent_dialog_normalize_mapper`）。

---

## Running

```bash
# 路径 A
dj-process --config demos/agent/recipes/R1_initial_filter.yaml
dj-process --config demos/agent/recipes/R2_train_data_stack.yaml
dj-process --config demos/agent/recipes/R3_post_process.yaml

# 路径 B（先改 R0 / R2 的 dataset_path / export_path）
dj-process --config demos/agent/recipes/agent_interaction_quality_analysis.yaml   # 已有可跳过
dj-process --config demos/agent/recipes/R0_bridge_from_analysis.yaml
dj-process --config demos/agent/recipes/R2_train_data_stack.yaml           # dataset_path → R0 输出
dj-process --config demos/agent/recipes/R3_post_process.yaml
```

**Offline / CI**（无 API）：

```bash
dj-process --config demos/agent/recipes/R3_post_process_cpu_only.yaml
```

凭证与性能：[`../QUICKSTART_BAD_CASE.md`](../QUICKSTART_BAD_CASE.md)、[`../PERFORMANCE_LLM.md`](../PERFORMANCE_LLM.md)。

---

## R3 full stack（强模型 + 训练数据集产物）

`R3_post_process.yaml` 顺序概要：

1. `pii_redaction_mapper` — 确定性脱敏。  
2. `pii_llm_suspect_mapper` — LLM 复核；对外分发建议 `redaction_mode: whole_field`。  
3. `clean_copyright_mapper` / `clean_html_mapper`。  
4. `agent_safety_gate_mapper` → `meta.agent_training_safety_gate`；蒸馏在 `require_safety_gate_ok: true` 时依赖 `ok`。  
5. `agent_distill_trajectory_mapper` → `meta.agent_distilled_trajectory`（默认 gold/silver）。  
6. `agent_rewrite_hint_mapper` → `meta.agent_rewrite_hints`（默认 bronze）。  
7. `agent_training_card_mapper` → **`meta.agent_training_card` 为 JSON 字符串**（整段 `json.loads` 后使用；内部数值已 JSON 安全化）。

成本：PII / 改写可用轻模型；安全 / 蒸馏建议强模型；在 YAML 中替换 `api_model` / `api_or_hf_model` 即可。

---

## Meta / stats 约定

| Key | Writer |
|-----|--------|
| `agent_training_safety_gate` | `agent_safety_gate_mapper` |
| `agent_distilled_trajectory` | `agent_distill_trajectory_mapper` |
| `agent_rewrite_hints` | `agent_rewrite_hint_mapper` |
| `agent_training_card` | `agent_training_card_mapper`（**JSON 字符串**） |

解析后：`learnable_value_json` 仍为 JSON 字符串；`safety_gate_ok` 为 **`true` / `false` / `unknown`**；`llm_*_score` 缺省 **`-1.0`**。

`tool_success_ratio == -1.0` 表示无 success+fail 工具分母（无比率）。`total_tokens` 等 usage 字段为 **int**（缺省 `0`）。

`agent_error_taxonomy` 的 **`evidence` 叶子均为字符串**（避免 Arrow null/float 冲突）。`agent_training_card_mapper` 在运行时会强制 **`self.turbo = True`**（非 batched `map`），以便在极宽 `__dj__meta__` 上追加 `agent_training_card`。

详见 `MetaKeys`：`data_juicer/utils/constant.py`。
