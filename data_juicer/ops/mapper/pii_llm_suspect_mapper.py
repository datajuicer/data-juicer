# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Optional LLM PII audit; optional aggressive redaction after regex mappers.

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.ops.mapper.dialog_llm_input_utils import clip_text_for_dialog_prompt
from data_juicer.utils.agent_output_locale import normalize_preferred_output_lang
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "pii_llm_suspect_mapper"

_SPACY_PIPELINE_LOCKS: Dict[str, threading.Lock] = {}


def _spacy_pipeline_lock(name: str) -> threading.Lock:
    if name not in _SPACY_PIPELINE_LOCKS:
        _SPACY_PIPELINE_LOCKS[name] = threading.Lock()
    return _SPACY_PIPELINE_LOCKS[name]


def _resolve_spacy_auto_download_flag(requested: bool) -> bool:
    """Env ``PII_SPACY_AUTO_DOWNLOAD`` overrides when set to true/false."""
    raw = os.environ.get("PII_SPACY_AUTO_DOWNLOAD", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return bool(requested)


def ensure_spacy_pipeline_installed(model_name: str, *, auto_download: bool) -> None:
    """If ``auto_download`` and pipeline missing, run ``spacy.cli.download`` (needs network)."""
    if not auto_download or not model_name:
        return
    try:
        import spacy.util

        if spacy.util.is_package(model_name):
            return
    except Exception:
        pass
    try:
        from spacy.cli import download

        logger.info(
            "pii_llm_suspect_mapper: downloading spaCy pipeline {} (pip/network)",
            repr(model_name),
        )
        download(model_name)
    except Exception as e:
        logger.warning(
            "pii_llm_suspect_mapper: spaCy download({}) failed: {} — "
            "install manually e.g. python -m spacy download {}",
            repr(model_name),
            e,
            model_name,
        )


DEFAULT_REDACTION_PLACEHOLDER = "[LLM_PII_SUSPECT_REDACTED]"

DEFAULT_SYSTEM_PROMPT = (
    "你是数据合规审计助手。下面给出若干字段的截断文本（可能已部分脱敏）。"
    "请只识别**仍可能残留的敏感信息**（如真实人名、未脱敏手机号/证件片段、"
    "内网地址、密钥片段、可识别工号、仍未规则脱敏的 URL/IPv4/MAC、"
    "JWT/PEM/credential 片段等）。不要编造；不确定则不要列出。"
    "必须只输出一个 JSON 对象，不要 markdown，不要解释。Schema:\n"
    '{"suspected":[{"field":"字段名","category":"类型简述",'
    '"evidence":"从输入中摘录的极短原文片段≤48字符"}],'
    '"likely_clean":true或false}\n'
    "若无可疑项，suspected 用 []，likely_clean 用 true。"
)

DEFAULT_SYSTEM_PROMPT_EN = (
    "You are a data-compliance auditor. Below are truncated field excerpts "
    "(possibly partially redacted). Identify **only** residual sensitive "
    "data (real names, phone/ID fragments, intranet hosts, secret snippets, "
    "identifiable employee IDs, raw URLs/IPv4/MAC not yet redacted, "
    "JWT/PEM/credential fragments, etc.). Do not invent; if unsure, omit.\n"
    "Output exactly one JSON object, no markdown. Schema:\n"
    '{"suspected":[{"field":"field_key","category":"short_type",'
    '"evidence":"verbatim snippet from input, ≤48 chars"}],'
    '"likely_clean":true or false}\n'
    "Use suspected=[] and likely_clean=true when nothing suspicious."
)


def _extract_json_object(text: str) -> Optional[dict]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        frag = s[start : end + 1]  # noqa: E203
        return json.loads(frag)
    except json.JSONDecodeError:
        return None


def _heuristic_trigger(text: str) -> bool:
    """Cheap pre-filter: skip LLM when nothing looks like residual PII."""
    if not text or len(text.strip()) < 8:
        return False
    if re.search(r"\d{6,}", text):
        return True
    if "@" in text:
        return True
    secret_pat = r"(?i)(?:^|[^a-z0-9_])(?:" r"api[_-]?key|apikey|secret|token|password|passwd|credential|bearer)\b"
    if re.search(secret_pat, text):
        return True
    intranet = r"(?i)\b(?:内网|局域网|vpn|ssh\s|rdp://|mysql://|redis://)\b"
    if re.search(intranet, text):
        return True
    # Long alphanumeric runs (possible tokens / hashes)
    if re.search(r"[A-Za-z0-9_-]{24,}", text):
        return True
    return False


# High-precision contextual cues (no extra deps). Intentionally conservative to limit false API calls.
_NAME_LIKE_RULE_PAT = re.compile(
    r"(?:"
    r"(?:^|[\s，,。.!！?？;；:：])"
    r"(?:我叫|叫我|更名为|姓名\s*[:：\s为是]|真名\s*[:：\s为是]|"
    r"联系人\s*[:：\s为]?|负责人\s*[:：\s为]?|经办人\s*[:：\s为]?|"
    r"用户(?:姓名|名叫)\s*[:：\s为]?|(?:致|尊敬的)\s*)"
    # Name may be followed by more hanzi (e.g. 我叫张三想咨询) without punctuation.
    r"[\u4e00-\u9fff]{2,4}(?=$|[\s，,。.!！?？;；:：]|[\u4e00-\u9fff])"
    r"|"
    r"[\u4e00-\u9fff]{2,4}(?:先生|女士|老师|大夫|经理|主任|院长|博士|教授|同志)(?=[\s，,。.!！?？;；]|$)"
    r"|"
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
    r")",
    re.UNICODE,
)


def _name_like_rule_trigger(text: str) -> bool:
    """Rule-based cues that real person names may appear (esp. missed by ``_heuristic_trigger``)."""
    if not text or len(text.strip()) < 4:
        return False
    return _NAME_LIKE_RULE_PAT.search(text) is not None


def _normalize_spacy_ner_model_names(
    spacy_ner_model: Optional[str],
    spacy_ner_models: Optional[List[str]],
) -> List[str]:
    """Deduplicated pipeline names; list order first, then legacy ``spacy_ner_model``."""
    out: List[str] = []
    seen: Set[str] = set()
    if spacy_ner_models:
        for x in spacy_ner_models:
            s = str(x).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    one = (spacy_ner_model or "").strip()
    if one and one not in seen:
        out.append(one)
    return out


def _collect_messages_excerpt(
    messages: Any,
    max_messages: int,
    max_chars: int,
) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    parts: List[str] = []
    n = 0
    for msg in reversed(messages):
        if n >= max_messages:
            break
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        content = msg.get("content")
        chunk = ""
        if isinstance(content, str):
            chunk = content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    t = block.get("text") or block.get("content")
                    if isinstance(t, str):
                        chunk += t + "\n"
                elif isinstance(block, str):
                    chunk += block + "\n"
        chunk = chunk.strip()
        if chunk:
            parts.append(f"[{role}]{chunk[:500]}")
            n += 1
    joined = "\n".join(reversed(parts))
    return clip_text_for_dialog_prompt(
        joined,
        max_chars,
        "messages excerpt truncated",
    )


def _build_field_payload(
    sample: dict,
    keys: List[str],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for k in keys:
        if k not in sample:
            continue
        v = sample[k]
        if isinstance(v, str) and v.strip():
            out.append((k, v))
    return out


def _canonical_field_for_redaction(
    field: str,
    inspect_keys: List[str],
    messages_key: Optional[str],
) -> Optional[str]:
    f = (field or "").strip()
    if f in inspect_keys:
        return f
    if messages_key and f == f"{messages_key}_excerpt":
        return messages_key
    return None


def _redact_string_substrings(
    text: str,
    snippets: List[str],
    placeholder: str,
) -> str:
    if not text or not snippets:
        return text
    # Longer snippets first so we do not leave broken partial matches first.
    uniq = []
    seen = set()
    for s in sorted(set(snippets), key=len, reverse=True):
        s = (s or "").strip()
        if len(s) < 2 or s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    out = text
    for s in uniq:
        if s in out:
            out = out.replace(s, placeholder)
    return out


def _redact_messages_substrings(
    messages: Any,
    snippets: List[str],
    placeholder: str,
) -> None:
    if not isinstance(messages, list) or not snippets:
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = _redact_string_substrings(
                content,
                snippets,
                placeholder,
            )
        elif isinstance(content, list):
            for i, block in enumerate(content):
                if isinstance(block, dict):
                    for k in ("text", "content"):
                        if k in block and isinstance(block[k], str):
                            block[k] = _redact_string_substrings(
                                block[k],
                                snippets,
                                placeholder,
                            )
                elif isinstance(block, str):
                    content[i] = _redact_string_substrings(
                        block,
                        snippets,
                        placeholder,
                    )


def _replace_messages_with_placeholder(
    messages: Any,
    placeholder: str,
) -> None:
    """In-place: single user turn with placeholder text (keeps list shape)."""
    if not isinstance(messages, list):
        return
    messages.clear()
    messages.append({"role": "user", "content": placeholder})


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class PiiLlmSuspectMapper(Mapper):
    """LLM audit (and optional redaction) for possibly missed PII.

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
    """

    def __init__(
        self,
        api_model: str = "qwen-turbo",
        *,
        inspect_keys: Optional[List[str]] = None,
        messages_key: Optional[str] = "messages",
        max_messages_for_prompt: PositiveInt = 4,
        max_chars_per_field: PositiveInt = 6000,
        max_chars_messages_excerpt: PositiveInt = 8000,
        gate_mode: str = "heuristic",
        result_key: str = MetaKeys.pii_llm_suspect,
        raw_key: str = MetaKeys.pii_llm_suspect_raw,
        overwrite: bool = False,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        preferred_output_lang: str = "zh",
        try_num: PositiveInt = 2,
        model_params: Optional[Dict] = None,
        sampling_params: Optional[Dict] = None,
        text_key: str = "text",
        heuristic_name_rules: bool = True,
        spacy_ner_model: Optional[str] = None,
        spacy_ner_models: Optional[List[str]] = None,
        spacy_ner_max_chars: PositiveInt = 4000,
        spacy_auto_download: bool = True,
        redaction_mode: str = "none",
        redaction_placeholder: str = DEFAULT_REDACTION_PLACEHOLDER,
        **kwargs,
    ):
        super().__init__(text_key=text_key, **kwargs)
        self.inspect_keys = list(
            inspect_keys or ["text", "query", "response"],
        )
        self.messages_key = messages_key
        self.max_messages_for_prompt = int(max_messages_for_prompt)
        self.max_chars_per_field = int(max_chars_per_field)
        self.max_chars_messages_excerpt = int(max_chars_messages_excerpt)
        self.gate_mode = (gate_mode or "heuristic").strip().lower()
        self.result_key = result_key
        self.raw_key = raw_key
        self.overwrite = bool(overwrite)
        self.preferred_output_lang = normalize_preferred_output_lang(preferred_output_lang)
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif self.preferred_output_lang == "zh":
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT_EN
        self.try_num = try_num
        self.sampling_params = dict(sampling_params or {})
        self.model_key = prepare_model(
            model_type="api",
            model=api_model,
            endpoint=api_endpoint,
            response_path=response_path,
            **(model_params or {}),
        )
        rm = (redaction_mode or "none").strip().lower()
        if rm not in ("none", "evidence", "whole_field"):
            logger.warning(
                "pii_llm_suspect_mapper: unknown redaction_mode=%r, " "using none",
                redaction_mode,
            )
            rm = "none"
        self.redaction_mode = rm
        rp = redaction_placeholder or DEFAULT_REDACTION_PLACEHOLDER
        self.redaction_placeholder = str(rp)
        self.heuristic_name_rules = bool(heuristic_name_rules)
        self.spacy_ner_model_names = _normalize_spacy_ner_model_names(
            spacy_ner_model,
            spacy_ner_models,
        )
        self.spacy_ner_max_chars = int(spacy_ner_max_chars)
        self.spacy_auto_download = _resolve_spacy_auto_download_flag(
            bool(spacy_auto_download),
        )
        self._spacy_nlp_by_name: Dict[str, Any] = {}
        self._spacy_load_failed: Set[str] = set()

    def _get_spacy_nlp(self, model_name: str) -> Any:
        if model_name in self._spacy_load_failed:
            return None
        if model_name in self._spacy_nlp_by_name:
            return self._spacy_nlp_by_name[model_name]
        lock = _spacy_pipeline_lock(model_name)
        with lock:
            if model_name in self._spacy_nlp_by_name:
                return self._spacy_nlp_by_name[model_name]
            if model_name in self._spacy_load_failed:
                return None
            ensure_spacy_pipeline_installed(
                model_name,
                auto_download=self.spacy_auto_download,
            )
            try:
                import spacy

                nlp = spacy.load(model_name)
                self._spacy_nlp_by_name[model_name] = nlp
                return nlp
            except Exception as e:  # pragma: no cover - env specific
                logger.warning(
                    "pii_llm_suspect_mapper: spacy.load(%r) failed (%s); " "skipping this model for the NER gate.",
                    model_name,
                    e,
                )
                self._spacy_load_failed.add(model_name)
                return None

    def _spacy_person_trigger(self, text: str) -> bool:
        if not self.spacy_ner_model_names:
            return False
        snippet = (text or "")[: self.spacy_ner_max_chars]
        if len(snippet.strip()) < 2:
            return False
        for model_name in self.spacy_ner_model_names:
            nlp = self._get_spacy_nlp(model_name)
            if nlp is None:
                continue
            try:
                doc = nlp(snippet)
            except Exception as e:  # pragma: no cover
                logger.debug(
                    "pii_llm_suspect_mapper spacy infer (%s): %s",
                    model_name,
                    e,
                )
                continue
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "PER"):
                    return True
        return False

    def _heuristic_allow_llm(self, user_block: str) -> bool:
        """If True, heuristic gate opens and the LLM may run."""
        if _heuristic_trigger(user_block):
            return True
        if self.heuristic_name_rules and _name_like_rule_trigger(user_block):
            return True
        if self._spacy_person_trigger(user_block):
            return True
        return False

    def _assemble_user_block(self, sample: dict) -> Tuple[str, List[str]]:
        lines: List[str] = []
        fields_used: List[str] = []
        for fname, raw in _build_field_payload(sample, self.inspect_keys):
            clipped = clip_text_for_dialog_prompt(
                raw,
                self.max_chars_per_field,
                f"{fname} truncated",
            )
            lines.append(f"### field:{fname}\n{clipped}")
            fields_used.append(fname)
        if self.messages_key and self.messages_key in sample:
            ex = _collect_messages_excerpt(
                sample[self.messages_key],
                self.max_messages_for_prompt,
                self.max_chars_messages_excerpt,
            )
            if ex.strip():
                lines.append(f"### field:{self.messages_key}_excerpt\n{ex}")
                fields_used.append(f"{self.messages_key}_excerpt")
        return "\n\n".join(lines), fields_used

    def _apply_redaction(self, sample: dict, cleaned: List[dict]) -> None:
        if self.redaction_mode == "none":
            return
        ph = self.redaction_placeholder
        by_field: Dict[str, List[str]] = {}
        fields_whole: Set[str] = set()
        for item in cleaned:
            cf = _canonical_field_for_redaction(
                str(item.get("field", "")),
                self.inspect_keys,
                self.messages_key,
            )
            if not cf:
                continue
            fields_whole.add(cf)
            ev = str(item.get("evidence", "")).strip()
            if ev:
                by_field.setdefault(cf, []).append(ev)

        if self.redaction_mode == "whole_field":
            for cf in fields_whole:
                if cf not in sample:
                    continue
                if self.messages_key and cf == self.messages_key:
                    m = sample.get(self.messages_key)
                    if isinstance(m, list):
                        _replace_messages_with_placeholder(m, ph)
                elif isinstance(sample.get(cf), str):
                    sample[cf] = ph
            return

        if self.redaction_mode == "evidence":
            for cf, snippets in by_field.items():
                if not snippets or cf not in sample:
                    continue
                if self.messages_key and cf == self.messages_key:
                    _redact_messages_substrings(
                        sample[self.messages_key],
                        snippets,
                        ph,
                    )
                elif isinstance(sample.get(cf), str):
                    sample[cf] = _redact_string_substrings(
                        sample[cf],
                        snippets,
                        ph,
                    )

    def process_single(self, sample, rank=None):
        meta = sample.get(Fields.meta)
        if not isinstance(meta, dict):
            sample[Fields.meta] = {}
            meta = sample[Fields.meta]

        if not self.overwrite and self.result_key in meta:
            return sample

        user_block, fields_used = self._assemble_user_block(sample)
        if not user_block.strip():
            meta[self.result_key] = {
                "skipped": True,
                "reason": "no_inspectable_text",
                "fields_inspected": [],
            }
            return sample

        gated = self.gate_mode == "heuristic" and not self._heuristic_allow_llm(
            user_block,
        )
        if gated:
            meta[self.result_key] = {
                "skipped": True,
                "reason": "heuristic_gate",
                "fields_inspected": fields_used,
                "likely_clean": True,
            }
            return sample

        user_msg = "以下文本可能已部分正则脱敏。请按系统说明只输出 JSON。\n\n" + user_block
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        sp = {**self.sampling_params}
        sp.setdefault("max_tokens", 512)
        sp.setdefault("temperature", 0.1)

        raw = ""
        for attempt in range(self.try_num):
            try:
                client = get_model(self.model_key, rank=rank)
                raw = client(messages, **sp)
                if raw and isinstance(raw, str) and raw.strip():
                    break
            except Exception as e:
                logger.warning(
                    "pii_llm_suspect_mapper attempt %s: %s",
                    attempt + 1,
                    e,
                )

        if not raw:
            meta[self.result_key] = {
                "error": "empty_llm_response",
                "fields_inspected": fields_used,
            }
            return sample

        parsed = _extract_json_object(raw)
        if parsed is None:
            meta[self.raw_key] = raw[:8000]
            meta[self.result_key] = {
                "error": "json_parse_failed",
                "fields_inspected": fields_used,
            }
            return sample

        suspected = parsed.get("suspected")
        if not isinstance(suspected, list):
            suspected = []
        cleaned: List[dict] = []
        for item in suspected[:32]:
            if not isinstance(item, dict):
                continue
            cleaned.append(
                {
                    "field": str(item.get("field", ""))[:64],
                    "category": str(item.get("category", ""))[:120],
                    "evidence": str(item.get("evidence", ""))[:80],
                }
            )

        default_clean = len(cleaned) == 0
        likely_clean = bool(parsed.get("likely_clean", default_clean))
        meta[self.result_key] = {
            "suspected": cleaned,
            "likely_clean": likely_clean,
            "fields_inspected": fields_used,
            "redaction_mode": self.redaction_mode,
            "redaction_applied": self.redaction_mode != "none",
        }
        self._apply_redaction(sample, cleaned)
        return sample
