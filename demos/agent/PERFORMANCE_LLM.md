# Agent 流水线里 LLM 算子：加速与超参

全量 `agent_interaction_quality_analysis.yaml` 里 **LLM 调用次数 ≈（样本数 × 算子个数）**（部分算子内部还有多重循环）。下面按「并行」「减少工作量」「减少输出」三类说明。

## 1. 并行（多进程发 API）

| 配置 | 作用 | 注意 |
|------|------|------|
| **`np`**（顶层，默认如 `4`） | 每个算子 `dataset.map` 的 worker 数上限；与 **`auto_op_parallelism: true`**（默认）一起时，`process_utils.calculate_np` 还会按算子再估一并发度，日志里常见 `num_proc=4` 但实际可能看到更高建议值 | 过大易 **429 / 限流**；可先 `8~16` 试，按服务商配额调 |
| **单算子 `num_proc`** | 在 YAML 里给某个 op 写 `num_proc: 8`，会 **cap** 自动并行（见 `OP.runtime_np()`：`min(自动估值, num_proc)`） | 只对「慢算子」放宽，避免全局 `np` 过大 |
| **`turbo: true`**（顶层 config） | `NestedDataset.map` 对某些非 batched 算子走 **`batched: false`** 路径（见 `dj_dataset.py`），官方说明偏向 batch=1 时的吞吐 | 可与 `np` 组合试用，以实测为准 |
| **`executor_type: ray`** | 分布式_executor，适合集群与大数据集 | 本地单机未必更快，需额外 Ray 环境 |

**API 模型**：当前菜谱多是 **远程 HTTP**，加速主要来自 **多进程并发请求**，受 **带宽、时延、QPS** 限制，不是线性随 `np` 增长。

## 2. 减少「每个算子」的工作量

| 超参 | 常见位置 | 建议 |
|------|-----------|------|
| **`try_num`** | `dialog_*_mapper`、`extract_entity_attribute_mapper`、`relation_identity_mapper` 等 | 默认 often `3`；调试/跑通可 **`1~2`**（失败重试变少，总耗时可明显下降；解析不稳时别太低） |
| **`max_round`** | `dialog_intent_detection_mapper` 等 | 默认 `10`；改 **`3~5`** 可缩短 prompt、略降质量 |
| **`query_entities` × `query_attributes`** | `extract_entity_attribute_mapper` | 每个 `(实体, 属性)` **各 1 次 API/样本**；列表越短越快 |
| **换小/快模型** | `api_model` / `api_or_hf_model` | 例如技能归纳里 **`qwen3.5-27B` → `qwen-turbo`** 做开发迭代 |
| **注释整条 op** | 非当前分析必需的 dialog 维度等 | 开发时关掉几条 mapper，等价成倍减少请求 |

### 2.1 `dialog_*_mapper` 为什么能到「每样本 ~100s」

- **导出 / formatter 的长度截断通常不作用在 `dialog_history` 上**。Agent 流水线里 `response` 往往含 **整段 tool 轨迹**（多段 assistant + `[Tool result]`），**单轮即可数万～十几万字符**，全额进入 `dialog_*` 的 user prompt。
- 四个算子（intent / sentiment / topic / sentiment_intensity）对 **``dialog_history`` 里每一轮** 各打 **1 次** API；实现上还会在末尾 **再 append 一轮** `(query_key, response_key)`，因此 **单用户轮次** 的数据也常见 **≥2 次调用 / 算子 / 样本**。
- **日志里 `num_proc=5` 与 `100s/ examples`**：多为 **单进程内串行的多轮调用 + 超大输入 + 重试** 叠加，不是「只发了一条消息」。

**优先手段（已写入默认菜谱示例）：**

| 手段 | 说明 |
|------|------|
| **`max_response_chars_for_prompt` / `max_query_chars_for_prompt`** | 仅截断 **送入 LLM 的** query/response 字符串，**不改**样本落盘内容；缓解超长 tool 日志 |
| **`history_tool_result_max_chars` / `history_max_assistant_trace_chars`**（`agent_dialog_normalize_mapper`） | **写回** `dialog_history` / `text` / `query` / `response`：超长时对单条 tool 或整段助手侧做 **首尾保留 + 中段省略标记**（`meta.agent_dialog_history_compressed`），可与上项叠加；代码默认单 tool **10000**（与旧版 `[:10000]` 同量级）、整段助手侧默认关闭；**`0` = 不限制** |
| **`max_round`** | 控制拼进 prompt 的历史块数（每轮对应 4 段模板文本） |
| **`try_num`** | 解析失败时的重试次数 |
| **`sampling_params.max_tokens`** | 限制模型侧输出长度 |

开发阶段可再关掉 2～3 个 `dialog_*` 算子，只保留 sentiment 或 intent。

## 3. 减少 token（更快、更便宜）

| 超参 | 说明 |
|------|------|
| **`sampling_params`** | 多数 API LLM 算子支持在 YAML 里写，例如 `sampling_params: { max_tokens: 256 }`（具体键名随提供商，DJ 会做部分归一，见 `model_utils.update_sampling_params`） |
| **`query_preview_max_chars` / `response_preview_max_chars`** | **`agent_insight_llm_mapper`**：已控制 evidence 长度；可再 **调小** 降 insight 单次耗时 |
| **`run_for_tiers`** | **`agent_insight_llm_mapper`**：已默认只跑 `high_precision` + `watchlist`；可改为 **`["high_precision"]`** 或暂关该算子 |

`llm_analysis_filter` / `llm_quality_score_filter` / `llm_difficulty_score_filter` 的 **结构化输出** 若把 `max_tokens` 压太低，可能解析失败 → `try_num` 反扑，需折中。

## 4. 迭代开发时的推荐组合

1. 使用 **`demos/agent/minimal_configs/09_bad_case_smoke.yaml`** 或自剪一版「只保留必要 LLM」的 YAML。  
2. 顶层 **`np: 8`**（或按配额），必要时 **`turbo: true`** 试跑对比。  
3. Dialog 一类：**`max_round: 4`**，**`try_num: 1`**。  
4. **`agent_skill_insight_mapper`**：开发改用 **`qwen-turbo`**。  
5. 全量跑稳定后再把 `try_num`、模型、实体列表调回生产值。

## 5. 与本仓库脚本的衔接

后处理脚本与 **`keep_stats_in_res_ds`** 无关加速；瓶颈仍在 **dj-process 阶段 LLM**。可先 **`dataset` 抽样**（config 里 `dataset.max_sample_num` 等）缩短单次实验周期。

更底层的缓存/检查点：**`use_checkpoint: true`** 便于反复改后半段算子时不重头算（注意改前半段算子参数会 invalidate，需查官方 checkpoint 说明）。
