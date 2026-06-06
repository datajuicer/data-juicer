# Training-Ready 字段指南（Agent JSONL）

面向 *「从分析审计转向训练数据价值」*：下列字段如何在 **SFT / Preference（DPO 类）/ 过滤 / RM** 中使用，以及读取时注意点。  
导出列名一般为 `__dj__meta__`（或配置中的 `meta`）；下文用 *meta* 指该对象。

更完整的 bad-case / insight 语义见 [`../BAD_CASE_INSIGHTS.md`](../BAD_CASE_INSIGHTS.md)；配方顺序见 [`README.md`](README.md)。

---

## 1. 推荐归因链顺序

1. **噪声**：`meta.agent_sys_log_noise` · `meta.agent_harness_noise` — 先判断红/黄是否来自系统日志/harness内在问题，避免把 infra 当模型缺陷。  
2. **对照能力**：`stats` 中 `llm_analysis_score` / `llm_quality_score` / `llm_difficulty_score` 及 `stats.*_record`（JSON 字符串）— 基模末轮质量。  
3. **对话轴**：`meta.dialog_*`、`meta.agent_trace_coherence`、`meta.agent_tool_relevance` — 与 §5b 分析配方一致。  
4. **跨模型 / 近似同题 PK**：`meta.agent_cross_model_pair` — 同 cohort 内 `delta_to_best`、`has_pairwise_contrast`；`match_basis` 标明分组依据：`exact_pair_key`（同 lineage id）、`normalized_query`（归一化 query 完全一致）、`simhash_lsh`（query+可选附加文本的 SimHash + LSH + Hamming，近似同题）。**跨代回归**可在同一 cohort 内比较 `my_version` vs `best_version`（无需单独算子）。  
5. **能力分桶**：`meta.agent_error_taxonomy` — `buckets.*.severity` 与 **`evidence` 叶子（均为字符串）**；解析数值时请 `float(s)` 或按需 `json.loads`。  
6. **ROI 分层**：`meta.agent_learnable_value`（标量）、`meta.agent_training_dataset_tier` / `meta.agent_learnable_value_tier`。  
7. **人审优先级**：`meta.agent_bad_case_signals`、`meta.agent_bad_case_tier`。  
8. **教师模型 + 数据卡片**（R3）：`agent_training_safety_gate`、`agent_distilled_trajectory`、`agent_rewrite_hints`、`agent_training_card`。

---

## 2. 字段 → 训练用途建议

| 字段 | 典型用途 | 说明 |
|------|----------|------|
| `text` / `query` / `response` / `dialog_history` | SFT 主语料 | R3 前做 PII/HTML/copyright；长轨迹注意截断与许可。 |
| `meta.agent_training_dataset_tier` | **训练数据集分桶** | `gold` / `silver` / `bronze` / `drop`；可与 `hard_drop_recommended` 联用。 |
| `meta.agent_learnable_value` | **排序加权** | 标量分数；高优先入训或复制多 epoch。 |
| `meta.agent_cross_model_pair` | **Preference / RM** | 多模型 + 近似同题 cohort；看 `match_basis` 理解分组可信度。`simhash_lsh` 有误合并风险，可把关键 env 拼进 `extra_group_text_key` 或收紧 `simhash_max_hamming`。 |
| `meta.agent_error_taxonomy` | **课程学习 / 按能力混桶** | 按 `reasoning` / `tool_use` 等 severity 混比例；evidence 为字符串。 |
| `meta.agent_bad_case_tier` + `agent_bad_case_signals` | **人审队列、报告对齐** | 与 HTML 报告同源；高置信子集可先审。 |
| `meta.tool_success_ratio` | **工具链质量门控** | **`-1.0`** 表示无 success+fail 工具轮（无比率）；比较前先筛 `>= 0`。 |
| `meta.total_tokens` 等 | **成本 / 长度分层** | 全为 int；缺省为 `0`。 |
| `meta.agent_training_safety_gate` | **训前合规** | 蒸馏默认 `require_safety_gate_ok: true` 时依赖 `ok`。 |
| `meta.agent_distilled_trajectory` | **SFT 目标补充** | 教师对轨迹的 `distilled_final_reply` 等；注意 API 成本与闭源策略。 |
| `meta.agent_rewrite_hints` | **改写 / 二次标注** | 低 tier 的结构化提示，不直接替代 messages。 |
| `meta.agent_training_card` | **Dataset card / 交接** | **整段为 JSON 字符串**：`card = json.loads(row["__dj__meta__"]["agent_training_card"])`；内含 `training_dataset_tier`、`safety_gate_ok`（`true`/`false`/`unknown`）、`llm_*_score`（缺省 `-1.0`）、`learnable_value_json`（内层 JSON 字符串）。 |

---

## 3. 验证两次跑数（diff）

```bash
python demos/agent/scripts/diff_agent_exports.py \
  --before ./outputs/agent_train_data_R2/processed.jsonl \
  --after ./outputs/agent_train_data_R3/train_data.jsonl \
  --meta-keys agent_training_dataset_tier agent_training_card \
  --stats-keys llm_analysis_score
```

## 4. jq 示例（按 tier 导出路径）

以下假设每行 JSON 顶层有 `__dj__meta__`（若你的导出用 `meta`，请替换键名）。

```bash
# gold + 非 hard_drop，且安全门为 true（若已跑 R3 安全门）
jq -c 'select(
  (.__dj__meta__.agent_training_dataset_tier? == "gold")
  and (.__dj__meta__.agent_error_taxonomy.hard_drop_recommended? != true)
  and ((.__dj__meta__.agent_training_safety_gate.ok?) // true)
)' train_data.jsonl > gold_sft_candidates.jsonl
```

```bash
# 解析 training_card 一行看推荐用法
jq -r '.__dj__meta__.agent_training_card | fromjson | .recommended_usage' train_data.jsonl | head
```

---

## 5. 与两条接线路径的关系

| 路径 | 顺序 |
|------|------|
| **A · 从原始交互轨迹** | `R1_initial_filter` → `R2_train_data_stack` → `R3_*` |
| **B · 从全量分析导出** | `demos/agent/recipes/agent_interaction_quality_analysis.yaml` → **`R0_bridge_from_analysis`** → `R2_train_data_stack` → `R3_*` |

### 5.1 路径 A（仅 R1）与全量分析配方的信号差

路径 A **有意不写** 分析 YAML 里 §4（对话意图/情感/主题等）、§5b（末轮 LLM 质检 `meta.dialog_*` 等）、§8（`llm_quality_score` / `llm_analysis_score` / `llm_difficulty_score` 及 `*_record`）、`agent_skill_insight_mapper`、分析侧 `agent_bad_case_signal_mapper` / `agent_insight_llm_mapper` 等。  
R1 只保证 **规范化 + lineage、跨模型 cohort、去重、sys_log/harness、用量与工具标签**。

对下游的影响（算子实现上的「有则用、无则弱」）：

- **`agent_error_taxonomy_mapper`**：会读 `stats` 里 LLM 记录与 `meta` 里已有 `dialog_*`；路径 A 下多数维度 **缺失**，taxonomy 桶与 severity **偏保守、信息量少**。
- **`agent_learnable_value_mapper`**：`weight_difficulty` 依赖 **`stats.llm_difficulty_score`**；无该字段时难度项 **视为 0**；`weight_generalizability` 依赖 `meta` 里 intent/topic 标签，路径 A 下常 **无**，该项贡献弱。
- **R2 的 `agent_bad_case_signal_mapper`**：`signal_on_llm_analysis_low` / `signal_on_llm_text_quality_low` 等依赖 **`stats.llm_*`**；路径 A 下通常 **不触发**（与 `exclude_if_sys_log_or_harness_noise` 等组合时，仍以 sys_log/harness、tool、learnable tier 为主）。

需要与分析报告 **同密度** 的 bad-case / taxonomy / learnable 排序时，请走 **路径 B**，或在 R1 与 R2 之间 **插入** `agent_interaction_quality_analysis.yaml` 中对应 LLM 段落（见该文件 §4 / §5b / §8 注释与 `demos/agent/PERFORMANCE_LLM.md`）。

路径 B 下请将 **`R2_train_data_stack.yaml`** 的 `dataset_path` 改为 R0 的 `export_path`。

---

## 6. 训练前定量验证（不依赖训练）

可用 `transition_report.py` 对同一批 `id` 的 before / after 导出做离线对比：

```bash
python demos/agent/scripts/transition_report.py \
  --before ./outputs/agent_train_data_R2/processed.jsonl \
  --after ./outputs/agent_train_data_R3/train_data.jsonl \
  --require-safety-ok \
  --out-json ./outputs/agent_train_data_R3/transition_report.json
```

默认输出要点：

- 红/黄/绿迁移矩阵（before → after）
- `net_green_gain`（红黄升绿概率 - 绿降黄红概率）
- `harmful_promotion_rate`（Red→Green 中仍疑似风险样本比例）
- `high/medium` signal burden 均值变化
- `agent_learnable_value` 与 quality/hardness proxy 的 Spearman（排序一致性）
- top-k（10/20/30%）绿色覆盖率、hard precision、hardness lift
- （可选）stage retention：`raw->R1->R2->R3` 的 kept/dropped 规模与 drop profile（red/yellow/green、noise、hard_drop）

建议把该报告作为每轮配方迭代的 gate，并固定抽样审计 `Red→Green` 与 `Green→Red`。

> 说明：若要比较 **R3 蒸馏/改写后的文本语义质量**，建议在 R3 输出后跑一条轻量复评链（至少 quality/analysis + bad_case + learnable）。否则 R3 后的 quality 相关分数仍可能是改写前语义。

## 7. 已知局限 / TODO

- **跨代回归**、**session 时间线合并**、**流式末包空检测** 尚未单独算子化；当前用 lineage + 手工规则或后续 mapper 扩展。  
- **蒸馏**当前为「教师阅读当前轨迹」生成目标，不自动从 PK 赢面模型 **拷贝** 异模型轨迹。  
- **data-model co-dev 迭代** via DJ-Sandbox & DJ-Agents
- **better judege**：将 `agent_learnable_value` 从规则线性分桶升级为可校准/可学习排序器，并引入更强评审信号（或人工抽样）作为对照，避免弱 judge 上限锁定。
