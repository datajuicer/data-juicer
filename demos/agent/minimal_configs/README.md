# Agent 流水线最小可运行配置（便于逐项调试）

与 `agent_interaction_quality_analysis.yaml` 同目录，本目录下为拆分后的 minimal 配置，方便单测式逐个排查。

- **数据**: `demos/local/demo-agent-data-content.jsonl`（顶层有 messages、response_choices、id 等，**无 text**）
- **输出**: `./outputs/agent_minimal/<name>.jsonl`
- **text_keys**: 所有 agent 配置里已设 `text_keys: "id"`，以便 formatter 用已有字段通过校验；第一个 op `agent_dialog_normalize_mapper` 会从 messages/choices 写出 `text`。

## 环境与依赖（参考 docs/DeveloperGuide.md）

- `jsonlines` 在 **Core dependencies**（`pyproject.toml` 的 `dependencies`）里，安装项目时本应一起装上。
- 若出现 `ModuleNotFoundError: No module named 'jsonlines'`：
  - 先装齐核心依赖：`uv pip install -e .`（只装 core，不含 extras）。
  - 再按需加 extras：`uv pip install -e ".[ai_services]"` 等。
  - 或单独补装：`uv pip install jsonlines`。
- 建议先用**标准 demo** 验证环境：  
  `dj-process --config demos/process_simple/process.yaml`  
  能跑通再试本目录下带 agent 算子的配置。

## 运行方式（在仓库根目录）

```bash
# 00 获取demo数据集
bash demos/agent/scripts/fetch_demo_data.sh

# 01 仅 normalize，无 LLM
dj-process --config demos/agent/minimal_configs/01_normalize_only.yaml

# 02 normalize + PII
dj-process --config demos/agent/minimal_configs/02_normalize_pii.yaml

# 03 加 clean
dj-process --config demos/agent/minimal_configs/03_clean.yaml

# 04 usage + tool 标签，无 LLM
dj-process --config demos/agent/minimal_configs/04_usage_tool.yaml

# 05 若干 filter，无 LLM
dj-process --config demos/agent/minimal_configs/05_filters.yaml

# 06 单个 dialog mapper（需 LLM）
dj-process --config demos/agent/minimal_configs/06_one_dialog_mapper.yaml

# 07 实体/关系/关键词（需 LLM；已配 query_entities/query_attributes、source_entity/target_entity 以产出实体与关系）
dj-process --config demos/agent/minimal_configs/07_entity_keyword.yaml

# 08 单个 LLM filter
dj-process --config demos/agent/minimal_configs/08_one_llm_filter.yaml

# 09 bad case 信号 + 分层（无 LLM；对齐 BAD_CASE_INSIGHTS 本地验证）
dj-process --config demos/agent/minimal_configs/09_bad_case_smoke.yaml

# 或一键：后处理 + 校验 + 分位数 + cohort（见 demos/agent/QUICKSTART_BAD_CASE.md）
bash demos/agent/scripts/run_bad_case_pipeline.sh smoke
```

## 配置说明摘要

- **09**：在 **04+05** 基础上增加 `copy_lineage_fields` 与 `agent_bad_case_signal_mapper`，并关闭依赖 LLM 评估的 `signal_on_llm_*`，用于本地验证 `meta.agent_bad_case_tier` / `agent_bad_case_signals`。
- **06**：dialog_intent/topic/sentiment 等 mapper 会对多轮标签做**去重**（保持顺序），避免重复意图标签。
- **07**：extract_entity_attribute 需配置 `query_entities`、`query_attributes`；relation_identity 需配置 `source_entity`、`target_entity`，否则实体/关系多为空。当前 07 已配「用户/助手/任务」与「目标/行为」、「用户→助手」关系。
- **agent_skill_insights**：完整 recipe 中在 agent_tool_type_mapper 后增加 `agent_skill_insight_mapper`，用 LLM 将 agent_tool_types + agent_skill_types 归纳为 3～5 条**具体**能力短语（中文约 10 字/条、带对象与动作；避免「文件读写、处理、操作」式空洞标签），写入 `meta.agent_skill_insights`，比仅用正则的 skill_types 更有区分度。

## 建议调试顺序

1. 先跑 **01**，确认大 yaml 的「单 op」写法是否被 parser 接受。
2. 若 01 通过，再跑 **02、03、04、05**，确认无 LLM 段是否都正常。
3. 跑 **09** 可验证 `agent_bad_case_signal_mapper` 与导出字段，无需 API。
4. 再跑 **06、07、08**，确认带 LLM 的 op 是否因配置或 parser 报错。

若某一编号报错，可基本定位到是「某 op 或某组合」导致；再对比大 yaml 中该段的写法即可缩小范围。

## 运行环境注意

- **务必在仓库的 editable 环境中运行**（例如 `uv venv && source .venv/bin/activate` 后 `uv pip install -e ".[ai_services]"`），再用 `dj-process`。若用 Homebrew 等全局安装的 `dj-process`（路径如 `/opt/homebrew/bin/dj-process`），会找不到 `jsonlines` 等依赖，且不会用到本仓库对 config 的修复。
- `demos/process_simple` 会下载 `lid.176.bin`（语言识别模型）；若网络失败，可手动下载到 `~/.cache/data_juicer/models/` 或先试本目录 01（仅 normalize，不依赖该模型）。
