# Agent pipeline 后分析脚本

**HTML 报告一行命令** → [`../BAD_CASE_REPORT.md`](../BAD_CASE_REPORT.md)。**smoke / full / postprocess / calibrate / unittest** → [`../QUICKSTART_BAD_CASE.md`](../QUICKSTART_BAD_CASE.md)（命令总表）；环境变量 `PYTHON`、`SMOKE_OUT`、`FULL_OUT`、`CAL_OUT` 与 `bash …/run_bad_case_pipeline.sh --help` 同见该页。

---

## Python 脚本一览

在仓库根目录执行（或把路径换成你的 `export_path` 输出 jsonl）。

| 脚本 | 作用 |
|------|------|
| `generate_bad_case_report.py` | **自助报告**：tier、**字段→信号归因表**、high / medium（附录）分层图、cohort → 单页 HTML |
| `bad_case_signal_support.py` | 报告内归因表静态数据（signal code → 上游 meta/stats / 算子） |
| `dj_export_row.py` | 解析 `__dj__meta__` / `__dj__stats__`，并与同级 `*_stats.jsonl` 按行合并（供其它脚本使用） |
| `verify_bad_case_export.py` | 检查是否含 `agent_bad_case_tier` / `agent_bad_case_signals`（自动合并 `_stats.jsonl`） |
| `compute_percentile_thresholds.py` | 按 `agent_request_model` 汇总 token / 延迟 / `stats.perplexity`；控制台分位数，或 `--write-calibration` 生成 JSON |
| `analyze_bad_case_cohorts.py` | 按 model / pt / tier 汇总；可选 `--out-csv`、`--pandas-priority` |
| `slice_export_by_tier.py` | 按 `meta.agent_bad_case_tier`（及可选 `--model`）导出子集 jsonl |

依赖：`python>=3.8`；`analyze_bad_case_cohorts.py` 的 pandas 路径可选（无则标准库聚合）。

---

## 手动分步示例

```bash
python demos/agent/scripts/verify_bad_case_export.py \
  --input ./outputs/agent_bad_case_smoke/processed.jsonl

python demos/agent/scripts/compute_percentile_thresholds.py \
  --input ./outputs/agent_quality/processed.jsonl

python demos/agent/scripts/compute_percentile_thresholds.py \
  --input ./outputs/agent_quality/processed.jsonl \
  --write-calibration ./outputs/agent_quality/bad_case_calibration.json \
  --calibration-percentile 95

python demos/agent/scripts/compute_percentile_thresholds.py \
  --input ./outputs/agent_quality/processed.jsonl \
  --by-pt

python demos/agent/scripts/analyze_bad_case_cohorts.py \
  --input ./outputs/agent_quality/processed.jsonl \
  --out-csv ./outputs/agent_quality/cohort_summary.csv

python demos/agent/scripts/slice_export_by_tier.py \
  --input ./outputs/agent_quality/processed.jsonl \
  --tier high_precision \
  --output ./outputs/agent_quality/high_precision_only.jsonl
```

## 与端到端流程的对应关系

1. **`dj-process`**（smoke 用 `minimal_configs/09_bad_case_smoke.yaml`，全量用 `agent_interaction_quality_analysis.yaml`）  
2. **`verify_bad_case_export.py`** — 确认 bad-case 字段已落盘  
3. **`compute_percentile_thresholds.py`** — 看分布或写校准 JSON  
4. **`analyze_bad_case_cohorts.py`** — cohort 报表  
5. **`slice_export_by_tier.py`** — 导出待人工复核子集  

第 10 步 LLM 洞察仅在全量菜谱中启用；smoke 配置已关闭 `signal_on_llm_*`，避免依赖 API。

补充：

- `run_bad_case_pipeline.sh report`（以及 `smoke` / `full` 末尾自动报告）默认附加 **`--bilingual`** 与 **`--llm-summary`**。关闭：`--no-bilingual` / `--no-llm-summary`，或环境变量 `BAD_CASE_REPORT_BILINGUAL=0` / `BAD_CASE_REPORT_LLM=0`。
- 页首导读若出现 **`The read operation timed out`**：为 DashScope/OpenAI 兼容接口读超时；可调大 **`BAD_CASE_REPORT_LLM_TIMEOUT_SEC`**（默认 120）或 **`--llm-timeout-sec`**，超时会自动重试 1 次。
- 也可直接调用 `generate_bad_case_report.py --bilingual --llm-summary ...`。
- **产出 / 报告语言**：菜谱里各 LLM 算子可设 **`preferred_output_lang: zh`** 或 **`en`**（见 `data_juicer/utils/agent_output_locale.py`）；`agent_insight_llm_mapper` 会把选用语言写入 **`meta.agent_pipeline_output_lang`**。生成 HTML 时用 **`--report-lang auto|zh|en`**（`auto` 时读环境变量 **`BAD_CASE_REPORT_LANG`** 或上述 meta），仅影响报告里**分档展示名**与 `<html lang>` 等；全文 i18n 可逐步扩展。
