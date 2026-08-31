# agent_bad_case_signal_mapper

Attach structured bad-case *signals* and a conservative *tier* to each sample.

Design goal: **precision over recall** for the ``high_precision`` tier.

**Upstream coverage** (when present in the pipeline):

- ``meta``: ``tool_*``, ``usage`` tokens, ``primary_tool_type``, ``dominant_tool_types``,
  ``dialog_intent_labels``, ``dialog_topic_labels``, ``dialog_sentiment_labels``,
  ``agent_turn_count``, lineage keys.
- ``stats``: ``llm_analysis_*``, ``llm_quality_*``, ``llm_difficulty_*``,
  ``text_len``, ``num_words``, ``perplexity``, ``lang_score``.
- ``meta``: optional ``dialog_*`` / ``agent_trace_coherence`` /
  ``agent_tool_relevance`` records (1–5 scores from lightweight LLM mappers).

Each signal group can be toggled via constructor flags. ``high`` weight feeds
``high_precision`` tier (with config); ``medium`` feeds ``watchlist`` only.

**Tool-heavy agent runs:** use ``min_tool_fail_count_for_signal`` to avoid
treating a single exploratory tool error (common before recovery) as strong
bad-case evidence.

**P-percentile calibration** (optional): set ``auto_calibrate_thresholds`` and
``calibration_json_path`` to a JSON file produced by
``demos/agent/scripts/compute_percentile_thresholds.py --write-calibration``.
Per-sample thresholds merge ``default`` with ``by_request_model`` using
``meta.agent_request_model``. When ``calibration_manual_overrides_auto`` is
true (default), explicit ``max_total_tokens`` / ``max_latency_ms`` / perplexity
settings in YAML override the file; set it false to prefer calibration.

为每个样本附加结构化的 bad-case *信号* 和保守的 *层级*。

设计目标：对于 ``high_precision`` 层级，**精确率优先于召回率**。

**上游覆盖范围**（当存在于 pipeline 中时）：

- ``meta``：``tool_*``、``usage`` token、``primary_tool_type``、``dominant_tool_types``、
  ``dialog_intent_labels``、``dialog_topic_labels``、``dialog_sentiment_labels``、
  ``agent_turn_count``、lineage keys。
- ``stats``：``llm_analysis_*``、``llm_quality_*``、``llm_difficulty_*``、
  ``text_len``、``num_words``、``perplexity``、``lang_score``。
- ``meta``：可选的 ``dialog_*`` / ``agent_trace_coherence`` /
  ``agent_tool_relevance`` 记录（来自轻量级 LLM mapper 的 1-5 分评分）。

每个信号组均可通过构造函数 flag 进行切换。``high`` 权重馈入
``high_precision`` 层级（通过 config 配置）；``medium`` 仅馈入 ``watchlist``。

**工具密集型 agent 运行：** 使用 ``min_tool_fail_count_for_signal`` 以避免
将单次探索性工具错误（在恢复前很常见）视为强烈的
bad-case 证据。

**P 百分位数校准**（可选）：将 ``auto_calibrate_thresholds`` 和
``calibration_json_path`` 设置为由
``demos/agent/scripts/compute_percentile_thresholds.py --write-calibration`` 生成的 JSON 文件。
逐样本阈值使用 ``meta.agent_request_model`` 将 ``default`` 与 ``by_request_model`` 合并。
当 ``calibration_manual_overrides_auto`` 为 true（默认值）时，YAML 中显式的 ``max_total_tokens`` / ``max_latency_ms`` / perplexity
设置将覆盖该文件；将其设为 false 则优先使用校准值。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `query_key` | str | `'query'` |  |
| `response_key` | str | `'response'` |  |
| `signal_on_tool_fail` | bool | `True` |  |
| `min_tool_fail_count_for_signal` | int | `1` |  |
| `signal_on_low_tool_success_ratio` | bool | `True` |  |
| `tool_success_ratio_max_for_signal` | float | `0.499` |  |
| `min_tool_rounds_for_ratio_signal` | int | `2` |  |
| `signal_on_suspect_empty_response` | bool | `True` |  |
| `min_query_len_for_empty_check` | int | `80` |  |
| `max_response_len_for_empty_check` | int | `20` |  |
| `max_total_tokens` | Optional[int] | `None` |  |
| `max_latency_ms` | Optional[int] | `None` |  |
| `calibration_json_path` | Optional[str] | `None` |  |
| `auto_calibrate_thresholds` | bool | `False` |  |
| `calibration_manual_overrides_auto` | bool | `True` |  |
| `auto_enable_perplexity_from_calibration` | bool | `True` |  |
| `signal_on_llm_analysis_low` | bool | `True` |  |
| `llm_analysis_score_max_for_bad` | float | `0.28` |  |
| `llm_analysis_discard_must_be_strict` | bool | `True` |  |
| `high_precision_llm_analysis_discard_threshold` | float | `0.24` |  |
| `signal_on_llm_text_quality_low` | bool | `True` |  |
| `llm_text_quality_score_max_for_bad` | float | `0.28` |  |
| `llm_text_quality_discard_must_be_strict` | bool | `True` |  |
| `high_precision_llm_text_quality_discard_threshold` | float | `0.24` |  |
| `signal_on_negative_sentiment_hint` | bool | `False` |  |
| `negative_sentiment_substrings` | Optional[List[str]] | `None` |  |
| `signal_on_high_perplexity` | bool | `False` |  |
| `perplexity_high_threshold` | float | `800.0` |  |
| `signal_hard_query_poor_reply` | bool | `False` |  |
| `hard_query_difficulty_min` | float | `0.72` |  |
| `poor_reply_quality_max` | float | `0.36` |  |
| `high_precision_on_tool_fail_alone` | bool | `True` |  |
| `min_medium_signals_for_watchlist` | int | `2` |  |
| `signal_on_low_dialog_quality_meta` | bool | `True` |  |
| `dialog_quality_low_score_threshold` | float | `2.0` |  |
| `min_dialog_quality_low_axes_for_signal` | int | `1` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/agent_bad_case_signal_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_agent_bad_case_signal_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)