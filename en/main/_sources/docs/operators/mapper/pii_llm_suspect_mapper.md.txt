# pii_llm_suspect_mapper

LLM audit (and optional redaction) for possibly missed PII.

Writes JSON to ``meta[result_key]`` (default ``MetaKeys.pii_llm_suspect``).
Set ``redaction_mode`` to ``evidence`` or ``whole_field`` to also modify
``inspect_keys`` string fields (and ``messages`` when listed). Place
**after** ``pii_redaction_mapper``.

Use ``gate_mode="heuristic"`` to call the API only when cheap patterns
suggest residual risk (long digit runs, @, secret-like keywords, etc.).

**Pre-LLM extensions** (still no API cost unless you enable spaCy):

- ``heuristic_name_rules`` (default True): contextual CJK / English name
  cues so person-heavy text is not skipped when the base heuristic fires
  only on digits and secrets.
- ``spacy_ner_models``: optional list of spaCy pipeline names (e.g.
  ``["zh_core_web_sm", "en_core_web_sm"]``) so one job loads both and
  runs NER on the same text prefix until a ``PERSON`` / ``PER`` hit.
- ``spacy_ner_model``: legacy single name; merged after ``spacy_ner_models``
  (deduped). Install with ``python -m spacy download <name>``.
- ``spacy_auto_download`` (default True): if the pipeline is missing, run
  spaCy's downloader before ``spacy.load`` (needs network, uses pip).
  Disable in air-gapped jobs or set env ``PII_SPACY_AUTO_DOWNLOAD=0``.

针对可能遗漏的 PII 进行 LLM 审计（及可选的脱敏处理）。

将 JSON 写入 ``meta[result_key]``（默认为 ``MetaKeys.pii_llm_suspect``）。将 ``redaction_mode`` 设置为 ``evidence`` 或 ``whole_field`` 以同时修改 ``inspect_keys`` 字符串字段（以及列出时的 ``messages``）。请放置在 ``pii_redaction_mapper`` **之后**。

使用 ``gate_mode="heuristic"`` 仅在低成本模式提示存在残余风险（长串数字、@、类似密钥的关键字等）时调用 API。

**LLM 前扩展**（除非启用 spaCy，否则仍无 API 成本）：

- ``heuristic_name_rules``（默认 True）：上下文 CJK / 英文姓名线索，确保当基础启发式规则仅因数字和密钥触发时，不会跳过包含大量人名的文本。
- ``spacy_ner_models``：可选的 spaCy pipeline 名称列表（例如 ``["zh_core_web_sm", "en_core_web_sm"]``），以便单个任务同时加载两者，并在同一文本前缀上运行 NER，直到命中 ``PERSON`` / ``PER``。
- ``spacy_ner_model``：旧版单一名称；在 ``spacy_ner_models`` 之后合并（去重）。使用 ``python -m spacy download <name>`` 安装。
- ``spacy_auto_download``（默认 True）：如果缺少 pipeline，则在 ``spacy.load`` 之前运行 spaCy 的下载器（需要网络，使用 pip）。在物理隔离的任务中禁用，或设置环境变量 ``PII_SPACY_AUTO_DOWNLOAD=0``。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | str | `'qwen-turbo'` |  |
| `inspect_keys` | Optional[List[str]] | `None` |  |
| `messages_key` | Optional[str] | `'messages'` |  |
| `max_messages_for_prompt` | PositiveInt | `4` |  |
| `max_chars_per_field` | PositiveInt | `6000` |  |
| `max_chars_messages_excerpt` | PositiveInt | `8000` |  |
| `gate_mode` | str | `'heuristic'` |  |
| `result_key` | str | `'pii_llm_suspect'` |  |
| `raw_key` | str | `'pii_llm_suspect_raw'` |  |
| `overwrite` | bool | `False` |  |
| `api_endpoint` | Optional[str] | `None` |  |
| `response_path` | Optional[str] | `None` |  |
| `system_prompt` | Optional[str] | `None` |  |
| `preferred_output_lang` | str | `'zh'` |  |
| `try_num` | PositiveInt | `2` |  |
| `model_params` | Optional[Dict] | `None` |  |
| `sampling_params` | Optional[Dict] | `None` |  |
| `text_key` | str | `'text'` |  |
| `heuristic_name_rules` | bool | `True` |  |
| `spacy_ner_model` | Optional[str] | `None` |  |
| `spacy_ner_models` | Optional[List[str]] | `None` |  |
| `spacy_ner_max_chars` | PositiveInt | `4000` |  |
| `spacy_auto_download` | bool | `True` |  |
| `redaction_mode` | str | `'none'` |  |
| `redaction_placeholder` | str | `'[LLM_PII_SUSPECT_REDACTED]'` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/pii_llm_suspect_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_pii_llm_suspect_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)