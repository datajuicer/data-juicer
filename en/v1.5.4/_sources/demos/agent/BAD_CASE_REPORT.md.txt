# Bad case 自助报告（简化入口）

面向：**跑完 `dj-process` 后只想看汇总与图**，不必记一串脚本。完整跑法见 [`QUICKSTART_BAD_CASE.md`](QUICKSTART_BAD_CASE.md)，文档索引见 [`README.md`](README.md)。

- 产出报告示例

<div align="center">
  <video src="https://gist.github.com/user-attachments/assets/df3687ab-72fc-4a15-ba2f-2448312b4405" width="100%" controls loop></video>
</div>


## 一行命令

在仓库根目录（替换为你的 `processed.jsonl`）：

```bash
bash demos/agent/scripts/run_bad_case_pipeline.sh report ./outputs/agent_quality/processed.jsonl
```

默认在同目录生成 **`processed_bad_case_report.html`**（若只传 jsonl，输出路径可省略第二个参数）。

浏览器打开该 HTML 即可：**tier 表（中文展示名 + 机器枚举）**、**Tier 分层说明**、**字段→信号归因表**、**样本钻取**（展开含 **meta/stats 快照**、**每条 signal 对应上游字段与取值**；另有 `query` / `response` / JSON）；图含 high/medium 信号条形图、按模型堆叠图、cohort 表；若有第 10 步 insight 会列 **强怀疑档（high_precision）** 的 headline。**medium** 信号在页内标注为附录类启发式。

> 钻取区 `#n` 为页内锚点，复制地址栏 `#fragment` 便于展陈与对日志。normalize 阶段会把稳定 id 与下标写入 `meta`（见 `agent_dialog_normalize_mapper`：`agent_request_id`、`agent_last_user_msg_idx`、`agent_last_assistant_msg_idx`）。

页底 **「Advanced / debugging」** 折叠区指向方法论与分步脚本（校准、jq、切片等）。静态归因定义见 **`demos/agent/scripts/bad_case_signal_support.py`**。

## 与 `smoke` / `full` 的关系

- `smoke`、`full` 跑完后会 **自动生成** 同目录下的 `*_bad_case_report.html`（除非设置 **`SKIP_AUTO_REPORT=1`**）。
- 仅需重出报告、不重跑 pipeline 时：用上面的 **`report`** 子命令。

## 直接调 Python（可选）

```bash
python demos/agent/scripts/generate_bad_case_report.py \
  --input ./outputs/agent_quality/processed.jsonl \
  --output ./outputs/agent_quality/my_report.html \
  --title "我的质检报告"
```

- **`--no-charts`**：只有表格（无 matplotlib 环境时也可用）。
- **`--sample-headlines 0`**：不展示 insight 摘录。
- **`--drilldown-limit N`**：钻取卡片数量上限（默认 48）；设为 **0** 则整段不输出。

## 仍是「进阶」的内容（刻意不收进报告页）

见 **`BAD_CASE_INSIGHTS.md`**（分层逻辑）、**`PERFORMANCE_LLM.md`**、**`scripts/README.md`**（`compute_percentile_thresholds`、`slice_export_by_tier`、校准 JSON 等）。
