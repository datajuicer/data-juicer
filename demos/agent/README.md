# Agent quality & bad-case docs

**English:** This folder documents a **recipe-driven** pipeline for agent logs
(JSONL with `messages` + `choices` / `response_choices`). Core operators live
under `data_juicer/ops/mapper/` (e.g. `agent_dialog_normalize_mapper`,
`agent_bad_case_signal_mapper`, `dialog_*` quality mappers); default prompts
and flattened dialog labels are **English-first** for upstream reuse—override
`user_label` / `assistant_label` on the normalize mapper or subclass mappers
for other locales.

---

# Agent 质检 / bad-case 文档索引

| 需求 | 文档 |
|------|------|
| **只出 HTML 报告**（已有 `processed.jsonl`） | [`BAD_CASE_REPORT.md`](BAD_CASE_REPORT.md) |
| **怎么跑 smoke / full / 校准 / 常见问题** | [`QUICKSTART_BAD_CASE.md`](QUICKSTART_BAD_CASE.md) |
| **分层逻辑、信号表、insight 字段、jq** | [`BAD_CASE_INSIGHTS.md`](BAD_CASE_INSIGHTS.md) |
| **后处理 Python 脚本参数** | [`scripts/README.md`](scripts/README.md) |
| **LLM 算子加速** | [`PERFORMANCE_LLM.md`](PERFORMANCE_LLM.md) |

端到端配置：**`agent_interaction_quality_analysis.yaml`**；无 API 冒烟：**`minimal_configs/09_bad_case_smoke.yaml`**。

---

## 维护指南

| 项 | 说明 |
|----|------|
| **pre-commit** | `pre-commit run --all-files`。若 `build-op-doc` 提示 `Operator document is updated`，需将 **`docs/Operators.md`**（及 `docs/operators/` 若生成）纳入同一提交；可按需再跑一轮 hook 直至 `build-op-doc` 通过。 |
| **算子文档** | 新增/改名 mapper 后，以 `python .pre-commit-hooks/build_op_doc.py` 更新表格；线上算子表见仓库根目录 [`docs/Operators.md`](../docs/Operators.md)。 |
| **信号 ↔ 报告** | 若增加 `agent_bad_case_signals` 的 `code`，请同步 [`scripts/bad_case_signal_support.py`](scripts/bad_case_signal_support.py)，以便 HTML 归因表一致。 |
| **单测（示例）** | `pytest tests/ops/mapper/test_agent_bad_case_signal_mapper.py tests/ops/mapper/test_agent_insight_llm_mapper.py tests/ops/mapper/test_usage_counter_mapper.py tests/utils/test_agent_output_locale.py tests/demos/agent/test_generate_bad_case_report_smoke.py -q` |
| **根 README** | 与本目录的交叉链接见 [仓库 README「Documentation」](../README.md#documentation) 中的 Agent quality 条目。 |
