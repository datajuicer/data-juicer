# Bad case 流水线：一键运行与端到端指南

姊妹篇：**方法论** → `[BAD_CASE_INSIGHTS.md](BAD_CASE_INSIGHTS.md)`；**仅报告命令** → `[BAD_CASE_REPORT.md](BAD_CASE_REPORT.md)`。`[README.md](README.md)` 为全文档索引。

## 只想看报告（最少步骤）

1. 跑完 `dj-process`（或 `bash … run_bad_case_pipeline.sh smoke|full`）。
2. `**bash demos/agent/scripts/run_bad_case_pipeline.sh report ./outputs/.../processed.jsonl`**
  → 打开生成的 `**processed_bad_case_report.html**`。说明见 `**BAD_CASE_REPORT.md**`。

## 前置条件

- 在**仓库根目录**使用含 `dj-process`环境，例如：
  ```bash
  cd /path/to/data-juicer
  uv venv && source .venv/bin/activate
  uv pip install -e ".[ai_services]"   # 全量菜谱需要 LLM；仅 smoke 可只装 -e .
  ```
- 确认：`which dj-process` 指向当前环境，而非 Homebrew 全局旧包（见 `minimal_configs/README.md`）。

## 一键命令（推荐）

在仓库根目录执行：


| 目的                                                    | 命令                                                                                                              |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **HTML 报告**（校验 + 图 + 表）                               | `bash demos/agent/scripts/run_bad_case_pipeline.sh report ./outputs/agent_quality/processed.jsonl`              |
| **冒烟**（无 API）：pipeline + 后处理 + **自动报告**               | `bash demos/agent/scripts/run_bad_case_pipeline.sh smoke`（`SKIP_AUTO_REPORT=1` 可关掉末段报告）                         |
| **仅后处理**（verify / 分位数 / cohort / slice，不调 matplotlib） | `bash … postprocess …`                                                                                          |
| **写 P95 校准 JSON**（第二遍跑 mapper 前）                      | `bash demos/agent/scripts/run_bad_case_pipeline.sh calibrate-and-slice ./outputs/agent_quality/processed.jsonl` |
| **全量菜谱**（多 LLM，贵/慢）                                   | `bash demos/agent/scripts/run_bad_case_pipeline.sh full`                                                        |
| **单元测试**（不跑 dj-process）                               | `bash demos/agent/scripts/run_bad_case_pipeline.sh unittest`                                                    |


输出位置可用环境变量覆盖：`SMOKE_OUT`、`FULL_OUT`、`CAL_OUT`（见脚本内 `usage`）。

## 端到端两条路径

### A. 本地快速验证（与 BAD_CASE_INSIGHTS 对齐，无第 10 步 LLM 洞察）

1. `bash demos/agent/scripts/run_bad_case_pipeline.sh smoke`
  - 配置：`demos/agent/minimal_configs/09_bad_case_smoke.yaml`  
  - 导出：`./outputs/agent_bad_case_smoke/processed.jsonl`  
  - 检查：`__dj__meta__.agent_bad_case_tier` / `agent_bad_case_signals`（或合并 `_stats.jsonl` 后脚本读取）
2. 用 `jq` 抽查（见 `BAD_CASE_INSIGHTS.md` 文末）。

### B. 完整分析（含 LLM 评估 + 可选校准 + insight）

1. 配置 API：按项目惯例设置各 LLM 算子所需环境变量 / 密钥。
2. `bash demos/agent/scripts/run_bad_case_pipeline.sh full`
  - 或：`dj-process --config demos/agent/agent_interaction_quality_analysis.yaml`
3. **（可选）两阶段校准**：
  - 第一轮导出后：`bash demos/agent/scripts/run_bad_case_pipeline.sh calibrate-and-slice ./outputs/agent_quality/processed.jsonl`  
  - 在 YAML 的 `agent_bad_case_signal_mapper` 中打开 `auto_calibrate_thresholds` 与 `calibration_json_path`（脚本会打印示例路径）。  
  - 对**同结构**数据再跑第二轮 `dj-process`。
4. 阅读 `__dj__meta__.agent_insight_llm`（第 10 步成功时）；校验可加：
  `python demos/agent/scripts/verify_bad_case_export.py --input ... --require-insight`

## 脚本目录说明

详见 `**demos/agent/scripts/README.md`**（各 Python 脚本参数与示例）。

## LLM 算子慢、想加速？

见 `**demos/agent/PERFORMANCE_LLM.md**`：`np` / `num_proc` / `turbo`、`try_num`、`max_round`、`sampling_params`、换小模型、收窄 `agent_insight_llm_mapper` 等。

## 导出里的 meta 键名

- 主字段为 `**__dj__meta__**`（不是 `meta`），除非设置 `**keep_stats_in_res_ds: true**`，否则默认不会出现在主 `processed.jsonl` 里，而在 `**processed_stats.jsonl**`。  
- `demos/agent/scripts/` 下分析脚本会**自动**把同目录的 `*_stats.jsonl` 按行合并后再读；菜谱 **09** 与 **agent_interaction_quality_analysis.yaml** 已设 `keep_stats_in_res_ds: true`，便于 `jq` 单文件查看。

## 常见问题

- **smoke 导出 0 条、verify 报 `got 0`**：若 YAML 里含 `language_id_score_filter` 且 `lang: "zh"`，FastText 标签可能不是字面 `"zh"`，会把 demo 全过滤掉。`09_bad_case_smoke.yaml` 已去掉该算子；全量 `agent_interaction_quality_analysis.yaml` 仍保留，大语料上一般正常。  
- `**lid.176.bin` 下载失败**：全量菜谱里的 `language_id_score_filter` 需要语言模型；可先跑 `01_normalize_only` 或换网络/手动放到 `~/.cache/data_juicer/models/`（见 `minimal_configs/README.md`）。  
- **full 中途 API 失败**：导出可能不完整；可用 `verify_bad_case_export.py`（不加 `--require-insight`）检查 bad-case 字段是否已写出。  
- **分位数为空**：确认 `copy_lineage_fields: true` 与 `usage_counter_mapper` 已跑，`meta.total_tokens` / `meta.agent_request_model` 存在。

