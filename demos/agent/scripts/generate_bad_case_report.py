#!/usr/bin/env python3
"""HTML 报告：分档统计、信号图、队列表、case study 样例（可展开）、可选 LLM 页首导读。

- 图表使用中文字体配置，避免中文显示为方框。
- 默认生成 **两份 HTML**：PII 安全版（``--output``）与同目录 ``*_pii_audit.html``（审计版）；
  Case study 页内在两档（强怀疑 / 待观察）间 **各约一半** 配额，并按与 Insight 相同的
  **字符 TF-IDF + MiniBatchKMeans** 语义簇做轮询抽样；全量写入对应 ``*_drilldown_full.jsonl``。
- 含 **对话形态与用量** 小节：``messages`` 长度、用户轮数、``text`` 字符数、choices 长度、
  token（meta / stats）与耗时（``agent_total_cost_time_ms`` 等）汇总表及可选直方图。
- ``--llm-summary``：调用 OpenAI 兼容接口（默认读 ``DASHSCOPE_API_KEY`` / ``OPENAI_API_KEY``）。
- 导读 HTTP 读超时默认 120s，可用 ``--llm-timeout-sec`` 或环境变量 ``BAD_CASE_REPORT_LLM_TIMEOUT_SEC``；超时自动重试 1 次。
- **默认单遍流式扫描**（适合百万行量级，避免整表 ``list`` OOM）：tier / 信号 / cohort / 宏观标签等计数为全量精确；
  Case study 与 Insight 仅从每档 **优先保留的有界候选池** 生成（``--drill-retention-cap``、``--insight-retention-cap``）；
  对话指标直方图 / 分位数对每指标 **贮样近似**（``--dialog-metric-samples``）。小数据若要整表进内存可显式加 ``--eager-load-all-rows``（大表易 OOM）。
- Case study 展开区展示 **query**、**message**（``messages`` 数组则按角色分条排版；纯文本若含多轮
  ``User:/Assistant:`` 等也会分段）、**response**（末轮/规范化字段）；``*_drilldown_full.jsonl`` 仍含
  ``message`` 与 ``response`` 字符串字段。

示例::

  python demos/agent/scripts/generate_bad_case_report.py \\
    --input ./outputs/agent_quality/processed.jsonl \\
    --output ./outputs/agent_quality/bad_case_report.html \\
    --llm-summary

  # 多批 jsonl 按顺序读入（可与各自 *_stats.jsonl 对齐），无需先 cat 合并：

  python demos/agent/scripts/generate_bad_case_report.py \\
    --input ./batch_a/processed.jsonl \\
    --input ./batch_b/processed.jsonl \\
    --output ./merged_view_report.html
"""

from __future__ import annotations

import argparse
import base64
import heapq
import html
import io
import json
import os
import random
import re
import socket
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

# HTML 分档展示语言（main 中根据 --report-lang / 环境变量 / meta 设定）
_REPORT_LOCALE = "zh"

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_bad_case_cohorts import aggregate_cohort_stdlib, load_merged_rows  # noqa: E402
from bad_case_signal_support import (  # noqa: E402
    DIALOG_QUALITY_SCORE_META_KEYS,
    SIGNAL_SUPPORT_ROWS,
)
from dj_export_row import get_dj_meta, get_dj_stats, iter_merged_export_rows  # noqa: E402

# 流式报告：控制常驻内存的「详情」行数；全量计数仍逐行扫描。
DEFAULT_DRILL_RETENTION_CAP = 48_000
DEFAULT_INSIGHT_RETENTION_CAP = 14_000
DEFAULT_DIALOG_METRIC_SAMPLES = 200_000

# 机器枚举（jsonl / jq）不变；页面与图例用中文降低误解（原 high_precision ≠「高精度模型」）
TIER_LABEL_ZH: Dict[str, str] = {
    "high_precision": "强怀疑（主证据）",
    "watchlist": "待观察（弱证据）",
    "none": "未标记",
}
TIER_LABEL_EN: Dict[str, str] = {
    "high_precision": "High suspicion (primary evidence)",
    "watchlist": "Watchlist (weak evidence)",
    "none": "Unlabeled",
}


def set_report_locale(lang: str) -> None:
    """Set HTML tier display language: ``zh`` or ``en``."""
    global _REPORT_LOCALE
    s = (lang or "zh").strip().lower()
    _REPORT_LOCALE = "en" if s.startswith("en") else "zh"


def _normalize_report_lang_token(value: Optional[str]) -> str:
    if not value or not str(value).strip():
        return "zh"
    s = str(value).strip().lower()
    return "en" if s.startswith("en") else "zh"


def infer_report_locale(
    rows: List[dict],
    arg_mode: str,
    env_lang: Optional[str],
) -> str:
    """Resolve report UI language: explicit zh/en, or auto from env / meta."""
    am = (arg_mode or "auto").strip().lower()
    if am == "en":
        return "en"
    if am == "zh":
        return "zh"
    if env_lang and str(env_lang).strip():
        return _normalize_report_lang_token(env_lang)
    for row in rows[:120]:
        meta = get_dj_meta(row)
        v = meta.get("agent_pipeline_output_lang")
        if v:
            return _normalize_report_lang_token(str(v))
    return "zh"


def _row_messages_str(row: dict) -> str:
    """Prefer root ``messages`` (common in agent exports); fall back to ``response``."""
    m = row.get("messages")
    if m is None:
        return str(row.get("response") or "")
    if isinstance(m, str):
        return m
    return str(m)


def _extract_openai_style_content_text(content: Any) -> str:
    """Flatten ``message.content`` (str / multimodal list / nested dict)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text") is not None:
                    parts.append(str(block.get("text") or ""))
                elif block.get("text") is not None:
                    parts.append(str(block.get("text") or ""))
                elif block.get("content") is not None:
                    parts.append(_extract_openai_style_content_text(block.get("content")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(x for x in parts if x).strip()
    if isinstance(content, dict):
        for k in ("text", "value", "content"):
            if k in content and content[k] not in (None, ""):
                return _extract_openai_style_content_text(content[k])
        return ""
    return str(content).strip()


def _tool_calls_short_summary(msg: dict) -> str:
    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list) or not tcs:
        return ""
    names: List[str] = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif tc.get("name"):
            names.append(str(tc["name"]))
    if not names:
        return "[tool_calls]"
    tail = "…]" if len(names) > 12 else "]"
    return "[tool_calls: " + ", ".join(names[:12]) + tail


def _dict_message_turn_plain(msg: dict) -> str:
    body = _extract_openai_style_content_text(msg.get("content"))
    extra = _tool_calls_short_summary(msg)
    if body and extra:
        return body + "\n" + extra
    return body or extra


def _normalize_dialog_role_token(raw: str) -> str:
    t = (raw or "").strip()
    tl = t.lower()
    if tl in ("user", "assistant", "system", "tool"):
        return tl
    return {"用户": "user", "助手": "assistant", "系统": "system", "工具": "tool"}.get(t, "other")


def _role_badge_label(role_key: str, *, lang_en: bool) -> str:
    r = (role_key or "").strip().lower()
    if lang_en:
        if not r or r == "other":
            return "?"
        return r.capitalize()
    return {
        "user": "用户",
        "assistant": "助手",
        "tool": "工具",
        "system": "系统",
        "other": "?",
    }.get(r, r or "?")


def _messages_list_to_conv_html(msgs: List[Any], *, lang_en: bool) -> Optional[str]:
    if not isinstance(msgs, list) or not msgs:
        return None
    if not all(isinstance(m, dict) for m in msgs):
        return None
    chunks: List[str] = []
    for msg in msgs:
        role_lc = str(msg.get("role") or "").strip().lower()
        rkey = role_lc if role_lc in ("user", "assistant", "tool", "system") else "other"
        if rkey == "other" and msg.get("role") is not None:
            label = html.escape(str(msg.get("role")).strip() or "?")
        else:
            label = html.escape(_role_badge_label(rkey, lang_en=lang_en))
        raw = _dict_message_turn_plain(msg)
        body = (
            html.escape(raw)
            if raw
            else "<span class='conv-empty'>(empty)</span>"
        )
        rcls = rkey if rkey in ("user", "assistant", "tool", "system") else "other"
        esc_r = html.escape(rcls)
        chunks.append(
            f'<div class="conv-turn conv-{esc_r}">'
            f'<div class="conv-role">{label}</div>'
            f'<div class="conv-body">{body}</div></div>'
        )
    return '<div class="conv-thread">' + "".join(chunks) + "</div>"


def _plain_text_multi_turn_html(text: str, *, lang_en: bool) -> Optional[str]:
    """Detect ``User:`` / ``Assistant:`` (or CN labels) at line starts; build same layout as messages[]."""
    s = text.strip()
    if len(s) < 8:
        return None
    rx = re.compile(r"(?m)^\s*(User|Assistant|System|用户|助手|系统)\s*:\s*")
    matches = list(rx.finditer(s))
    if len(matches) < 2:
        return None
    chunks: List[str] = []
    for i, m in enumerate(matches):
        role_key = _normalize_dialog_role_token(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        segment = s[start:end].strip()
        rcls = role_key if role_key in ("user", "assistant", "tool", "system") else "other"
        label = html.escape(_role_badge_label(role_key, lang_en=lang_en))
        body = (
            html.escape(segment)
            if segment
            else "<span class='conv-empty'>(empty)</span>"
        )
        esc_r = html.escape(rcls)
        chunks.append(
            f'<div class="conv-turn conv-{esc_r}">'
            f'<div class="conv-role">{label}</div>'
            f'<div class="conv-body">{body}</div></div>'
        )
    return '<div class="conv-thread">' + "".join(chunks) + "</div>"


def _drill_message_panel_html(row: dict, *, lang_en: bool) -> str:
    """HTML for case-study ``message`` panel (formatted thread or ``<pre>`` fallback)."""
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        conv = _messages_list_to_conv_html(msgs, lang_en=lang_en)
        if conv:
            return f'<div class="conv-scroll">{conv}</div>'
    if isinstance(msgs, str) and msgs.strip():
        alt = _plain_text_multi_turn_html(msgs, lang_en=lang_en)
        if alt:
            return f'<div class="conv-scroll">{alt}</div>'
        return f'<pre class="drill-pre drill-pre--msg">{html.escape(msgs)}</pre>'
    root_msg = row.get("message")
    if isinstance(root_msg, str) and root_msg.strip():
        alt = _plain_text_multi_turn_html(root_msg, lang_en=lang_en)
        if alt:
            return f'<div class="conv-scroll">{alt}</div>'
        return f'<pre class="drill-pre drill-pre--msg">{html.escape(root_msg)}</pre>'
    fb = _row_messages_str(row)
    return f'<pre class="drill-pre drill-pre--msg">{html.escape(fb)}</pre>'


def _iter_all_merged_rows(paths: List[str]) -> Iterator[Tuple[int, dict]]:
    """Chain ``iter_merged_export_rows`` for every input path; yield (global_index, row)."""
    g = 0
    for p in paths:
        for _, row in iter_merged_export_rows(p):
            yield g, row
            g += 1


def _heap_push_bounded_min_sort(
    heap: List[Tuple[int, int, int, dict]],
    sort_key: Tuple[int, int, int],
    row: dict,
    max_size: int,
) -> None:
    """Keep at most ``max_size`` rows with **smallest** lexicographic ``sort_key`` (best-first)."""
    if max_size <= 0:
        return
    inv = (-sort_key[0], -sort_key[1], -sort_key[2], row)
    if len(heap) < max_size:
        heapq.heappush(heap, inv)
        return
    if inv > heap[0]:
        heapq.heapreplace(heap, inv)


def _heap_to_sorted_rows(heap: List[Tuple[int, int, int, dict]]) -> List[dict]:
    items = sorted(heap, key=lambda t: (-t[0], -t[1], -t[2]))
    return [t[3] for t in items]


def _dialog_series_reservoir_add(
    series: Dict[str, List[float]],
    seen: Dict[str, int],
    row: dict,
    cap: int,
    rng: random.Random,
) -> None:
    """Append numeric dialog-shape samples with per-key reservoir cap (approximate distribution)."""
    if cap <= 0:
        return
    meta = get_dj_meta(row)
    stats = get_dj_stats(row)
    msgs = _row_messages_list(row)
    if msgs is not None:
        k = "messages_len"
        v = float(len(msgs))
        _reservoir_append(series, seen, k, v, cap, rng)
        k = "user_turns_in_messages"
        _reservoir_append(series, seen, k, float(_count_user_turns_in_messages(msgs)), cap, rng)
        k = "tool_touch_msgs"
        _reservoir_append(series, seen, k, float(_count_tool_touch_messages(msgs)), cap, rng)
    atc = _coerce_int(meta.get("agent_turn_count"))
    if atc is not None:
        _reservoir_append(series, seen, "agent_turn_count", float(atc), cap, rng)
    text = row.get("text")
    if isinstance(text, str) and text:
        _reservoir_append(series, seen, "text_chars", float(len(text)), cap, rng)
    cl = _choices_list_len(row)
    if cl is not None:
        _reservoir_append(series, seen, "choices_len", float(cl), cap, rng)
    pt = _meta_or_stats_int(meta, stats, "prompt_tokens")
    if pt is not None:
        _reservoir_append(series, seen, "prompt_tokens", float(pt), cap, rng)
    ct = _meta_or_stats_int(meta, stats, "completion_tokens")
    if ct is not None:
        _reservoir_append(series, seen, "completion_tokens", float(ct), cap, rng)
    tt = _meta_or_stats_int(meta, stats, "total_tokens")
    if tt is not None:
        _reservoir_append(series, seen, "total_tokens", float(tt), cap, rng)
    lat = _latency_ms_from_row(row, meta)
    if lat is not None:
        _reservoir_append(series, seen, "latency_ms", float(lat), cap, rng)


def _reservoir_append(
    series: Dict[str, List[float]],
    seen: Dict[str, int],
    key: str,
    val: float,
    cap: int,
    rng: random.Random,
) -> None:
    if key not in series:
        series[key] = []
        seen[key] = 0
    seen[key] += 1
    n = seen[key]
    lst = series[key]
    if len(lst) < cap:
        lst.append(val)
        return
    j = rng.randint(1, n)
    if j <= cap:
        lst[j - 1] = val


def _finalize_cohort_rows(
    tier_counts: DefaultDict[Tuple[str, str, str], int],
    signal_counts: DefaultDict[Tuple[str, str, str], Counter],
) -> List[dict]:
    """Same table shape as ``aggregate_cohort_stdlib`` output."""
    models_pts = {(m, p) for (m, p, _) in tier_counts}
    out: List[dict] = []
    for model, pt in sorted(models_pts):
        for tier in ("high_precision", "watchlist", "none"):
            c = tier_counts.get((model, pt, tier), 0)
            top_ct = signal_counts.get((model, pt, tier), Counter())
            top = ", ".join(f"{k}:{v}" for k, v in top_ct.most_common(5))
            out.append(
                {
                    "agent_request_model": model,
                    "agent_pt": pt,
                    "tier": tier,
                    "count": c,
                    "top_signal_codes": top,
                }
            )
    return out


@dataclass
class StreamedReportSnapshot:
    n_rows: int
    tier_cnt: Counter
    high_c: Counter
    med_c: Counter
    model_tier: Dict[str, Counter]
    cohort_rows: List[dict]
    macro_counters: Dict[str, Counter]
    skill_stats: Tuple[int, int, int]
    dialog_series: Dict[str, List[float]]
    locale_probe_rows: List[dict]
    hp_drill_pool: List[dict]
    wl_drill_pool: List[dict]
    hp_insight_pool: List[dict]
    wl_insight_pool: List[dict]
    n_hp_total: int
    n_wl_total: int
    insight_headline_hp: int
    insight_headline_wl: int
    insight_pii_omitted_hp: int
    insight_pii_omitted_wl: int
    drill_pool_truncated: bool


def _stream_scan_report_inputs(
    paths: List[str],
    *,
    row_limit: Optional[int],
    drill_cap: int,
    insight_cap: int,
    dialog_sample_cap: int,
    rng: random.Random,
) -> StreamedReportSnapshot:
    tier_cnt: Counter = Counter()
    high_c: Counter = Counter()
    med_c: Counter = Counter()
    model_tier: DefaultDict[str, Counter] = defaultdict(Counter)
    tier_sig_counts: DefaultDict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    tier_only_counts: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
    macro_tool_primary: Counter = Counter()
    macro_tool_all: Counter = Counter()
    macro_skill_insight: Counter = Counter()
    macro_skill_type: Counter = Counter()
    macro_intent: Counter = Counter()
    macro_topic: Counter = Counter()
    macro_sentiment: Counter = Counter()
    si_rows = 0
    si_with = 0
    si_slots = 0
    dialog_series: Dict[str, List[float]] = {}
    dialog_seen: Dict[str, int] = {}
    locale_probe: deque = deque(maxlen=120)
    hp_drill_h: List[Tuple[int, int, int, dict]] = []
    wl_drill_h: List[Tuple[int, int, int, dict]] = []
    hp_insight_h: List[Tuple[int, int, int, dict]] = []
    wl_insight_h: List[Tuple[int, int, int, dict]] = []
    n_hp = n_wl = 0
    ih_hp = ih_wl = 0
    pii_om_hp = pii_om_wl = 0
    n_rows = 0
    for g, row in _iter_all_merged_rows(paths):
        if row_limit is not None and n_rows >= row_limit:
            break
        if not isinstance(row, dict):
            n_rows += 1
            continue
        n_rows += 1
        if len(locale_probe) < 120:
            locale_probe.append(row)
        meta = get_dj_meta(row)
        tier = str(meta.get("agent_bad_case_tier", "none"))
        tier_cnt[tier] += 1
        model = str(meta.get("agent_request_model") or "_unknown")
        model_tier[model][tier] += 1
        for s in meta.get("agent_bad_case_signals") or []:
            if not isinstance(s, dict) or not s.get("code"):
                continue
            code = str(s["code"])
            w = s.get("weight")
            if w == "high":
                high_c[code] += 1
            elif w == "medium":
                med_c[code] += 1
        pt_k = str(meta.get("agent_pt", "_unknown"))
        key3 = (model, pt_k, tier)
        tier_only_counts[key3] += 1
        sigs = meta.get("agent_bad_case_signals") or []
        if isinstance(sigs, list):
            for s in sigs:
                if isinstance(s, dict) and s.get("code"):
                    tier_sig_counts[key3][str(s["code"])] += 1
        ptt = meta.get("primary_tool_type")
        if ptt is not None:
            s = str(ptt).strip()
            if s:
                macro_tool_primary[s] += 1
        _counter_add_row_unique_labels(macro_tool_all, _labels_from_meta_list(meta, "agent_tool_types"))
        _counter_add_row_unique_labels(macro_skill_insight, _skill_insight_labels_from_meta(meta))
        _counter_add_row_unique_labels(macro_skill_type, _labels_from_meta_list(meta, "agent_skill_types"))
        _counter_add_row_unique_labels(macro_intent, _labels_from_meta_list(meta, "dialog_intent_labels"))
        _counter_add_row_unique_labels(macro_topic, _labels_from_meta_list(meta, "dialog_topic_labels"))
        _counter_add_row_unique_labels(macro_sentiment, _labels_from_meta_list(meta, "dialog_sentiment_labels"))
        labs = _skill_insight_labels_from_meta(meta)
        si_rows += 1
        if labs:
            si_with += 1
            si_slots += len(labs)
        _dialog_series_reservoir_add(
            dialog_series, dialog_seen, row, dialog_sample_cap, rng
        )
        st = _insight_sort_tuple(row, g)
        if tier == "high_precision":
            n_hp += 1
            _heap_push_bounded_min_sort(hp_drill_h, st, row, drill_cap)
        elif tier == "watchlist":
            n_wl += 1
            _heap_push_bounded_min_sort(wl_drill_h, st, row, drill_cap)
        ins = meta.get("agent_insight_llm") or {}
        if isinstance(ins, dict) and (ins.get("headline") or "").strip():
            if tier == "high_precision":
                ih_hp += 1
                if _exclude_insight_card_for_pii_row(row, ins):
                    pii_om_hp += 1
                _heap_push_bounded_min_sort(hp_insight_h, st, row, insight_cap)
            elif tier == "watchlist":
                ih_wl += 1
                if _exclude_insight_card_for_pii_row(row, ins):
                    pii_om_wl += 1
                _heap_push_bounded_min_sort(wl_insight_h, st, row, insight_cap)
    cohort_rows = _finalize_cohort_rows(tier_only_counts, tier_sig_counts)
    macro_counters = {
        "tool_primary": macro_tool_primary,
        "tool_all": macro_tool_all,
        "skill_insight": macro_skill_insight,
        "skill_type": macro_skill_type,
        "intent": macro_intent,
        "topic": macro_topic,
        "sentiment": macro_sentiment,
    }
    hp_drill = _heap_to_sorted_rows(hp_drill_h)
    wl_drill = _heap_to_sorted_rows(wl_drill_h)
    hp_ins = _heap_to_sorted_rows(hp_insight_h)
    wl_ins = _heap_to_sorted_rows(wl_insight_h)
    drill_trunc = (n_hp > len(hp_drill)) or (n_wl > len(wl_drill))
    return StreamedReportSnapshot(
        n_rows=n_rows,
        tier_cnt=tier_cnt,
        high_c=high_c,
        med_c=med_c,
        model_tier=dict(model_tier),
        cohort_rows=cohort_rows,
        macro_counters=macro_counters,
        skill_stats=(si_rows, si_with, si_slots),
        dialog_series=dialog_series,
        locale_probe_rows=list(locale_probe),
        hp_drill_pool=hp_drill,
        wl_drill_pool=wl_drill,
        hp_insight_pool=hp_ins,
        wl_insight_pool=wl_ins,
        n_hp_total=n_hp,
        n_wl_total=n_wl,
        insight_headline_hp=ih_hp,
        insight_headline_wl=ih_wl,
        insight_pii_omitted_hp=pii_om_hp,
        insight_pii_omitted_wl=pii_om_wl,
        drill_pool_truncated=drill_trunc,
    )


def _tier_zh(machine: str) -> str:
    if _REPORT_LOCALE == "en":
        return TIER_LABEL_EN.get(str(machine), str(machine))
    return TIER_LABEL_ZH.get(str(machine), str(machine))


def _fmt_evidence_val(val: object, maxlen: int = 200) -> str:
    if val is None:
        return "—"
    s = str(val).strip()
    if len(s) > maxlen:
        return s[:maxlen] + "…"
    return s


def _is_llm_read_timeout(err: BaseException) -> bool:
    """urllib / socket 读超时常见为 URLError(reason=timeout) 或文案含 timed out。"""
    if isinstance(err, TimeoutError):
        return True
    if isinstance(err, socket.timeout):
        return True
    low = str(err).lower()
    if "timed out" in low or "timeout" in low:
        return True
    reason = getattr(err, "reason", None)
    if reason is not None and reason is not err:
        return _is_llm_read_timeout(reason)
    return False


def _dialog_quality_axis_cell(meta: dict, key: str) -> str:
    """Format one axis record under meta[key] for evidence table."""
    rec = meta.get(key)
    if rec is None:
        return "—"
    if not isinstance(rec, dict):
        return _fmt_evidence_val(rec, 220)
    if rec.get("skipped"):
        r = (rec.get("reason") or "").strip()
        return f"skipped{f' ({r})' if r else ''}"
    err = rec.get("error")
    if err is not None and err != "":
        return f"error={_fmt_evidence_val(err, 140)}"
    sc = rec.get("score")
    reason = (rec.get("reason") or "").strip()
    parts: List[str] = []
    if sc is not None:
        parts.append(f"score={sc}")
    else:
        parts.append("(no score)")
    if reason:
        parts.append(f"reason={reason[:220]}")
    return "; ".join(parts)


def _evidence_rows_for_signal(
    code: str,
    row: dict,
    meta: dict,
    stats: dict,
    sig: Optional[dict] = None,
) -> List[Tuple[str, str]]:
    """Map signal code → (field_path, display_value) for report tables."""
    rows: List[Tuple[str, str]] = []

    def add(k: str, v: object) -> None:
        rows.append((k, _fmt_evidence_val(v)))

    if code == "tool_message_error_pattern":
        add("meta.tool_success_count", meta.get("tool_success_count"))
        add("meta.tool_fail_count", meta.get("tool_fail_count"))
        add("meta.tool_unknown_count", meta.get("tool_unknown_count"))
        add("meta.tool_success_ratio", meta.get("tool_success_ratio"))
        add("算子", "tool_success_tagger_mapper（regex 扫 role=tool）")
    elif code == "low_tool_success_ratio":
        add("meta.tool_success_ratio", meta.get("tool_success_ratio"))
        add("meta.tool_success_count", meta.get("tool_success_count"))
        add("meta.tool_fail_count", meta.get("tool_fail_count"))
    elif code == "llm_agent_analysis_eval_low":
        add("stats.llm_analysis_score", stats.get("llm_analysis_score"))
        rec = stats.get("llm_analysis_record")
        if isinstance(rec, dict):
            add("stats.llm_analysis_record.recommendation", rec.get("recommendation"))
        add("算子", "llm_analysis_filter")
    elif code == "llm_reply_quality_eval_low":
        add("stats.llm_quality_score", stats.get("llm_quality_score"))
        add("算子", "llm_quality_score_filter")
    elif code == "suspect_empty_or_trivial_final_response":
        add("sample.query 长度", len(row.get("query") or ""))
        add("sample.message 长度", len(_row_messages_str(row)))
        add("sample.response 长度", len(str(row.get("response") or "")))
        add("说明", "优先展示 message；response 为兼容字段（agent_dialog_normalize 末轮）")
    elif code == "high_token_usage":
        add("meta.total_tokens", meta.get("total_tokens"))
        add("算子", "usage_counter_mapper")
    elif code == "high_latency_ms":
        add("meta.agent_total_cost_time_ms", meta.get("agent_total_cost_time_ms"))
        add("算子", "agent_dialog_normalize_mapper.copy_lineage_fields")
    elif code == "negative_sentiment_label_hint":
        add("meta.dialog_sentiment_labels", meta.get("dialog_sentiment_labels"))
        add("算子", "dialog_sentiment_detection_mapper")
    elif code == "high_perplexity":
        add("stats.perplexity", stats.get("perplexity"))
        add("算子", "perplexity_filter")
    elif code == "hard_query_low_reply_quality_conjunction":
        add("stats.llm_difficulty_score", stats.get("llm_difficulty_score"))
        add("stats.llm_quality_score", stats.get("llm_quality_score"))
        add("算子", "llm_difficulty_score_filter ∩ llm_quality_score_filter")
    elif code == "dialog_turn_quality_meta_low":
        if sig and sig.get("detail"):
            add("signal.detail", sig.get("detail"))
        add(
            "说明",
            "各轴 1–5 分与 reason 如下；触发本信号时 detail.axes 中轴通常 score≤threshold。",
        )
        for axis_key in DIALOG_QUALITY_SCORE_META_KEYS:
            add(f"meta.{axis_key}", _dialog_quality_axis_cell(meta, axis_key))
        add(
            "算子",
            "dialog_*_mapper / agent_trace_coherence_mapper / agent_tool_relevance_mapper",
        )
    else:
        add("(见归因总表)", "本信号上游字段未逐项绑定，可查 meta / stats 全文")
    # de-dup keys keeping first
    seen = set()
    out: List[Tuple[str, str]] = []
    for k, v in rows:
        if k in seen:
            continue
        seen.add(k)
        out.append((k, v))
    return out


def _signal_evidence_tables_html(signals: List[dict], row: dict) -> str:
    """Per-signal small tables: code + upstream fields + values."""
    meta = get_dj_meta(row)
    stats = get_dj_stats(row)
    blocks = []
    for sig in signals:
        if not isinstance(sig, dict):
            continue
        code = str(sig.get("code") or "")
        if not code:
            continue
        w = html.escape(str(sig.get("weight") or ""))
        det = html.escape(_fmt_evidence_val(sig.get("detail"), 300))
        erows = _evidence_rows_for_signal(code, row, meta, stats, sig)
        body = "".join(
            f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(v)}</td></tr>"
            for k, v in erows
        )
        sig_thead = (
            "<thead><tr><th scope='col'>字段路径</th>"
            "<th scope='col'>取值 / 说明</th></tr></thead>"
        )
        blocks.append(
            f"<div class='sig-evidence'><div class='sig-evidence-h'>"
            f"<code>{html.escape(code)}</code> "
            f"<span class='wtag'>weight={w}</span> "
            f"<span class='det'>{det}</span></div>"
            f"<table class='inner sig-evidence-table'>{sig_thead}<tbody>{body}</tbody></table></div>"
        )
    if not blocks:
        return "<p class='note'>本样本无结构化信号。</p>"
    return "<div class='sig-evidence-wrap'>" + "".join(blocks) + "</div>"


def _fmt_snapshot_cell(v: object) -> str:
    """Format meta/stats scalars for the global snapshot (cleaner floats than raw JSON)."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        # trim float noise e.g. 0.27999999999999997 → 0.28
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return _fmt_evidence_val(v, 220)


# 与下方「各信号 ↔ 上游字段」小表结构对齐：末尾一行「算子」说明常见写入来源
_SNAPSHOT_UPSTREAM_OPS = (
    "tool_success_tagger_mapper（meta.tool_success_*）；"
    "usage_counter_mapper（meta.total_tokens）；"
    "agent_dialog_normalize_mapper（meta.agent_turn_count 等 lineage）；"
    "llm_analysis_filter（stats.llm_analysis_score）；"
    "llm_quality_score_filter（stats.llm_quality_score）；"
    "llm_difficulty_score_filter（stats.llm_difficulty_score）"
)


def _global_evidence_snapshot_html(row: dict) -> str:
    """Snapshot of key meta/stats for quick sanity check."""
    meta = get_dj_meta(row)
    stats = get_dj_stats(row)
    pairs: List[Tuple[str, str]] = [
        ("meta.tool_success_count", _fmt_snapshot_cell(meta.get("tool_success_count"))),
        ("meta.tool_fail_count", _fmt_snapshot_cell(meta.get("tool_fail_count"))),
        ("meta.tool_unknown_count", _fmt_snapshot_cell(meta.get("tool_unknown_count"))),
        ("meta.tool_success_ratio", _fmt_snapshot_cell(meta.get("tool_success_ratio"))),
        ("meta.total_tokens", _fmt_snapshot_cell(meta.get("total_tokens"))),
        ("meta.agent_turn_count", _fmt_snapshot_cell(meta.get("agent_turn_count"))),
        ("stats.llm_analysis_score", _fmt_snapshot_cell(stats.get("llm_analysis_score"))),
        ("stats.llm_quality_score", _fmt_snapshot_cell(stats.get("llm_quality_score"))),
        ("stats.llm_difficulty_score", _fmt_snapshot_cell(stats.get("llm_difficulty_score"))),
        ("算子", _SNAPSHOT_UPSTREAM_OPS),
    ]
    rows_html: List[str] = []
    for k, v in pairs:
        tr_cls = " class='snap-ops'" if k == "算子" else ""
        rows_html.append(
            f"<tr{tr_cls}><td><code>{html.escape(k)}</code></td>"
            f"<td>{html.escape(v)}</td></tr>"
        )
    body = "".join(rows_html)
    thead = (
        "<thead><tr><th scope='col'>字段路径</th>"
        "<th scope='col'>取值 / 说明</th></tr></thead>"
    )
    return (
        "<section class='snap'><h4>关键 meta / stats 快照</h4>"
        "<p class='note'>与归因链对照；缺失项为 <code>—</code>（可能未跑对应算子或未写入导出）。"
        "本表为跨信号汇总，逐条信号的上游字段见下方各 <code>sig-evidence</code> 小表。</p>"
        f"<table class='inner snap-table'>{thead}<tbody>{body}</tbody></table></section>"
    )


def _configure_matplotlib_cjk(plt_mod) -> None:
    """Avoid CJK in chart titles/labels rendering as tofu (□) in embedded PNGs."""
    plt_mod.rcParams.update(
        {
            "font.sans-serif": [
                "PingFang SC",
                "Hiragino Sans GB",
                "Heiti SC",
                "Songti SC",
                "STHeiti",
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "Source Han Sans SC",
                "WenQuanYi Zen Hei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def _get_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _configure_matplotlib_cjk(plt)
        return plt
    except ImportError:  # pragma: no cover
        return None


def _fig_to_data_uri(fig, plt_mod) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt_mod.close(fig)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _tier_counts(rows: List[dict]) -> Counter:
    c: Counter = Counter()
    for row in rows:
        meta = get_dj_meta(row)
        c[str(meta.get("agent_bad_case_tier", "none"))] += 1
    return c


def _coerce_int(val: object) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _coerce_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _meta_or_stats_int(meta: dict, stats: dict, *keys: str) -> Optional[int]:
    for d in (meta, stats):
        for k in keys:
            v = _coerce_int(d.get(k))
            if v is not None:
                return v
    return None


def _row_messages_list(row: dict) -> Optional[List]:
    m = row.get("messages")
    return m if isinstance(m, list) else None


def _count_user_turns_in_messages(messages: List) -> int:
    return sum(
        1
        for msg in messages
        if isinstance(msg, dict) and str(msg.get("role") or "").lower() == "user"
    )


def _count_tool_touch_messages(messages: List) -> int:
    n = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("tool_calls"):
            n += 1
        if str(msg.get("role") or "").lower() == "tool":
            n += 1
    return n


def _choices_list_len(row: dict) -> Optional[int]:
    for k in ("response_choices", "choices"):
        v = row.get(k)
        if isinstance(v, list):
            return len(v)
    return None


def _latency_ms_from_row(row: dict, meta: dict) -> Optional[float]:
    v = _coerce_float(meta.get("agent_total_cost_time_ms"))
    if v is not None:
        return v
    return _coerce_float(row.get("total_cost_time"))


def _gather_dialog_shape_series(rows: List[dict]) -> Dict[str, List[float]]:
    """Per-row numeric series for dialogue / usage shape (best-effort over DJ export shapes)."""
    keys = (
        "messages_len",
        "user_turns_in_messages",
        "agent_turn_count",
        "text_chars",
        "choices_len",
        "tool_touch_msgs",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_ms",
    )
    series: Dict[str, List[float]] = {k: [] for k in keys}
    for row in rows:
        meta = get_dj_meta(row)
        stats = get_dj_stats(row)
        msgs = _row_messages_list(row)
        if msgs is not None:
            series["messages_len"].append(float(len(msgs)))
            series["user_turns_in_messages"].append(float(_count_user_turns_in_messages(msgs)))
            series["tool_touch_msgs"].append(float(_count_tool_touch_messages(msgs)))
        atc = _coerce_int(meta.get("agent_turn_count"))
        if atc is not None:
            series["agent_turn_count"].append(float(atc))
        text = row.get("text")
        if isinstance(text, str) and text:
            series["text_chars"].append(float(len(text)))
        cl = _choices_list_len(row)
        if cl is not None:
            series["choices_len"].append(float(cl))
        pt = _meta_or_stats_int(meta, stats, "prompt_tokens")
        ct = _meta_or_stats_int(meta, stats, "completion_tokens")
        tt = _meta_or_stats_int(meta, stats, "total_tokens")
        if pt is not None:
            series["prompt_tokens"].append(float(pt))
        if ct is not None:
            series["completion_tokens"].append(float(ct))
        if tt is not None:
            series["total_tokens"].append(float(tt))
        lat = _latency_ms_from_row(row, meta)
        if lat is not None:
            series["latency_ms"].append(lat)
    return series


def _series_summary(vals: List[float]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = s[n // 2]
    p90_idx = min(n - 1, max(0, int(0.9 * (n - 1))))
    return {
        "n": float(n),
        "min": s[0],
        "max": s[-1],
        "avg": sum(s) / n,
        "median": float(mid),
        "p90": s[p90_idx],
    }


def _fmt_metric_cell(x: Optional[float], *, as_int: bool = False) -> str:
    if x is None:
        return "—"
    if as_int:
        return str(int(round(x)))
    if abs(x) >= 1000 or x == int(x):
        return str(int(round(x)))
    return f"{x:.2f}"


def _chart_numeric_histogram(
    plt_mod,
    values: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> Optional[str]:
    if plt_mod is None or len(values) < 3:
        return None
    vals = [float(x) for x in values if x == x]
    if len(vals) < 3:
        return None
    n_bins = min(48, max(8, len(vals) // 4))
    fig, ax = plt_mod.subplots(figsize=(7.2, 3.4))
    ax.hist(vals, bins=n_bins, color="#2e7d9a", edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _dialog_shape_metrics_section_html(
    rows: List[dict],
    plt_mod,
    *,
    no_charts: bool,
    html_lang: str,
    series_override: Optional[Dict[str, List[float]]] = None,
    sampled_note: str = "",
) -> str:
    """HTML: messages / turns / tokens / latency summary + optional histograms."""
    en = str(html_lang).lower().startswith("en")
    series = (
        series_override
        if series_override is not None
        else _gather_dialog_shape_series(rows)
    )
    labels = {
        "messages_len": (
            "messages list length (count of items)"
            if en
            else "messages 列表长度（条数）"
        ),
        "user_turns_in_messages": (
            "User turns (role==user in messages)"
            if en
            else "用户轮数（messages 中 role=user 条数）"
        ),
        "agent_turn_count": (
            "meta.agent_turn_count (dialog rounds from normalize)"
            if en
            else "meta.agent_turn_count（规范化写入的对话轮数）"
        ),
        "text_chars": (
            "Flattened text field length (chars)"
            if en
            else "扁平 text 字段字符数"
        ),
        "choices_len": (
            "response_choices / choices list length"
            if en
            else "response_choices 或 choices 列表长度"
        ),
        "tool_touch_msgs": (
            "Tool-touching messages (tool_calls or role=tool)"
            if en
            else "含工具轨迹的消息条数（tool_calls 或 role=tool）"
        ),
        "prompt_tokens": (
            "prompt_tokens (meta / stats)"
            if en
            else "prompt_tokens（meta 或 __dj__stats__）"
        ),
        "completion_tokens": (
            "completion_tokens (meta / stats)"
            if en
            else "completion_tokens（meta 或 stats）"
        ),
        "total_tokens": (
            "total_tokens (meta / stats; usage_counter / upstream)"
            if en
            else "total_tokens（meta 或 stats；用量算子等）"
        ),
        "latency_ms": (
            "Latency ms (meta.agent_total_cost_time_ms or row.total_cost_time)"
            if en
            else "耗时毫秒（meta.agent_total_cost_time_ms 或行上 total_cost_time）"
        ),
    }
    order = [
        "messages_len",
        "user_turns_in_messages",
        "agent_turn_count",
        "text_chars",
        "choices_len",
        "tool_touch_msgs",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_ms",
    ]
    th_m = "Metric" if en else "指标"
    th_n = "n" if en else "样本数"
    th_lo = "min" if en else "最小"
    th_hi = "max" if en else "最大"
    th_avg = "avg" if en else "均值"
    th_med = "median" if en else "中位"
    th_p90 = "p90" if en else "p90"
    rows_html: List[str] = []
    chart_blocks: List[str] = []
    hist_specs = (
        ("messages_len", "messages_len", "Messages / row"),
        ("user_turns_in_messages", "user_turns", "User turns / row"),
        ("total_tokens", "total_tokens", "total_tokens / row"),
        ("latency_ms", "latency", "Latency (ms) / row"),
    )
    for key in order:
        vals = series.get(key) or []
        summ = _series_summary(vals)
        if not summ:
            continue
        lab = labels[key]
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(lab)}</td>"
            f"<td>{_fmt_metric_cell(summ.get('n'), as_int=True)}</td>"
            f"<td>{_fmt_metric_cell(summ.get('min'))}</td>"
            f"<td>{_fmt_metric_cell(summ.get('max'))}</td>"
            f"<td>{_fmt_metric_cell(summ.get('avg'))}</td>"
            f"<td>{_fmt_metric_cell(summ.get('median'))}</td>"
            f"<td>{_fmt_metric_cell(summ.get('p90'))}</td>"
            "</tr>"
        )
    if not rows_html:
        empty = (
            "No dialogue / token / latency fields detected on loaded rows."
            if en
            else "当前载入的样本上未检测到 messages 长度、token 或耗时等可用字段。"
        )
        return (
            "<section class='dialog-metrics-sec' id='sec-dialog-metrics'>"
            f"<h2>{html.escape('Dialogue shape & usage' if en else '对话形态与用量')}</h2>"
            f"<p class='note'>{html.escape(empty)}</p></section>"
        )

    intro = (
        "Counts and distributions are computed on the same rows as the rest of this report "
        "(respects <code>--limit</code>). Token fields are read from <code>meta</code> then "
        "<code>__dj__stats__</code> when present."
        if en
        else "统计范围与本报告其它章节一致（受 <code>--limit</code> 影响）。"
        "token 类字段优先读 <code>meta</code>，缺失时再读 <code>__dj__stats__</code>。"
    )
    if sampled_note.strip():
        intro = f"{intro} {sampled_note.strip()}"
    h2 = "Dialogue shape & token / latency" if en else "对话形态与 token / 时延"
    y_ct = "Rows" if en else "样本数"

    if plt_mod is not None and not no_charts:
        for series_key, alt_slug, en_title in hist_specs:
            vals = series.get(series_key) or []
            if len(vals) < 5:
                continue
            title = en_title if en else (
                "messages 条数/行" if series_key == "messages_len"
                else "用户轮数/行" if series_key == "user_turns_in_messages"
                else "total_tokens/行" if series_key == "total_tokens"
                else "耗时(ms)/行"
            )
            xlabel = (
                "Messages" if series_key == "messages_len"
                else "User turns" if series_key == "user_turns_in_messages"
                else "total_tokens" if series_key == "total_tokens"
                else "ms"
            ) if en else (
                "条数" if series_key == "messages_len"
                else "轮数" if series_key == "user_turns_in_messages"
                else "tokens" if series_key == "total_tokens"
                else "ms"
            )
            uri = _chart_numeric_histogram(plt_mod, vals, title, xlabel, y_ct)
            if uri:
                chart_blocks.append(
                    f"<h4 class='chart-sub'>{html.escape(title)}</h4>"
                    f"<p class='note'><img src='{uri}' alt='{html.escape(alt_slug)}'/></p>"
                )

    charts_part = (
        "<details class='dialog-metrics-charts' open>"
        f"<summary>{html.escape('Histograms' if en else '分布直方图（默认展开）')}</summary>"
        + ("".join(chart_blocks) if chart_blocks else f"<p class='note'>{html.escape('No chart (need ≥5 points per metric or matplotlib off).' if en else '暂无图（每项至少 5 个有效点，或未启用图表）。')}</p>")
        + "</details>"
    )

    return (
        "<section class='dialog-metrics-sec' id='sec-dialog-metrics'>"
        f"<h2>{html.escape(h2)}</h2>"
        f"<p class='note'>{intro}</p>"
        "<table class='inner'><thead><tr>"
        f"<th>{html.escape(th_m)}</th><th>{html.escape(th_n)}</th>"
        f"<th>{html.escape(th_lo)}</th><th>{html.escape(th_hi)}</th>"
        f"<th>{html.escape(th_avg)}</th><th>{html.escape(th_med)}</th><th>{html.escape(th_p90)}</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows_html)}</tbody></table>"
        f"{charts_part}"
        "</section>"
    )


def _signal_counts_by_weight(rows: List[dict]) -> Tuple[Counter, Counter]:
    high_c: Counter = Counter()
    med_c: Counter = Counter()
    for row in rows:
        meta = get_dj_meta(row)
        for s in meta.get("agent_bad_case_signals") or []:
            if not isinstance(s, dict) or not s.get("code"):
                continue
            code = str(s["code"])
            w = s.get("weight")
            if w == "high":
                high_c[code] += 1
            elif w == "medium":
                med_c[code] += 1
    return high_c, med_c


# --- 报告「双视角」信号分桶（Insight 卡片与图表区共用；交叉类出现在两视角）---
_PURE_BASE_MODEL_SIGNALS: FrozenSet[str] = frozenset(
    {
        "llm_agent_analysis_eval_low",
        "llm_reply_quality_eval_low",
        "suspect_empty_or_trivial_final_response",
        "hard_query_low_reply_quality_conjunction",
        "high_perplexity",
    }
)
_PURE_AGENT_STACK_SIGNALS: FrozenSet[str] = frozenset(
    {
        "tool_message_error_pattern",
        "low_tool_success_ratio",
        "high_token_usage",
        "high_latency_ms",
    }
)
_CROSS_READ_SIGNALS: FrozenSet[str] = frozenset(
    {
        "dialog_turn_quality_meta_low",
        "negative_sentiment_label_hint",
    }
)


_SIGNAL_ROLE_ORDER = {"primary": 0, "structured": 1, "appendix": 2}
_SIGNAL_ROLE_LABEL_ZH = {
    "primary": "主证据",
    "structured": "结构化·轴分（建议重点读）",
    "appendix": "附录·启发式",
}


def _attribution_table_html() -> str:
    parts = []
    rows_sorted = sorted(
        SIGNAL_SUPPORT_ROWS,
        key=lambda r: (
            _SIGNAL_ROLE_ORDER.get(str(r.get("role")), 9),
            str(r.get("code", "")),
        ),
    )
    for r in rows_sorted:
        rk = str(r.get("role", "appendix"))
        role = _SIGNAL_ROLE_LABEL_ZH.get(rk, rk)
        parts.append(
            "<tr>"
            f"<td><code>{html.escape(r['code'])}</code></td>"
            f"<td>{html.escape(role)}</td>"
            f"<td>{html.escape(str(r['weight_hint']))}</td>"
            f"<td>{html.escape(str(r['upstream']))}</td>"
            "</tr>"
        )
    thead = (
        "<thead><tr><th>信号代码</th><th>证据角色</th><th>典型权重</th>"
        "<th>上游字段与算子</th></tr></thead>"
    )
    foot = (
        "<p class='note'><strong>证据角色：</strong>"
        "<code>主证据</code> 多对应 high 权重或强分层依据；"
        "<code>结构化·轴分</code> 来自 §5b 多轴 1–5 质检 meta 汇总（多为 medium，"
        "但与 discard/低分主证据同读更有解释力）；"
        "<code>附录·启发式</code> 多为弱提示，需结合业务判断。</p>"
    )
    return f"<table>{thead}<tbody>{''.join(parts)}</tbody></table>{foot}"


def _insight_fields_legend_html(*, html_lang: str = "zh-CN") -> str:
    """Semantics for badges / JSON fields emitted by agent_insight_llm_mapper."""
    en = str(html_lang).lower().startswith("en")
    rows_pr = [
        (
            "P0",
            "最高优先级人工复核；仅在极强、可复核证据下使用（默认应稀少）。",
            "Highest manual-review priority; use sparingly for the strongest, "
            "auditable evidence only.",
        ),
        (
            "P1",
            "建议优先排期复核。",
            "Schedule review soon.",
        ),
        (
            "P2",
            "常规复核或抽样即可。",
            "Normal / sampled review.",
        ),
        (
            "P3",
            "低优先级；多为信息性或与主问题弱相关。",
            "Low priority; informational or weakly related.",
        ),
    ]
    rows_al = [
        (
            "aligned",
            "数值类 stats 与 LLM 文字 rationale 方向一致、可互相印证。",
            "Numeric stats and qualitative rationales agree.",
        ),
        (
            "mixed",
            "部分一致、部分存疑或证据不足，需在样例里对照 query/response/meta。",
            "Partly consistent; verify against per-sample fields.",
        ),
        (
            "conflict",
            "数字与文字判断明显矛盾，或 insight 与 bad-case 信号解读冲突；"
            "建议结合下文 case study 核对原始 meta/stats。",
            "Clear tension between numbers and text; drill into raw meta/stats.",
        ),
    ]
    pr_body = "".join(
        "<tr>"
        f"<td><code>{html.escape(k)}</code></td>"
        f"<td>{html.escape(zh)}</td>"
        f"<td lang='en'>{html.escape(en)}</td>"
        "</tr>"
        for k, zh, en in rows_pr
    )
    al_body = "".join(
        "<tr>"
        f"<td><code>{html.escape(k)}</code></td>"
        f"<td>{html.escape(zh)}</td>"
        f"<td lang='en'>{html.escape(en)}</td>"
        "</tr>"
        for k, zh, en in rows_al
    )
    th3 = (
        "<thead><tr><th>取值</th><th>中文说明</th>"
        "<th lang='en'>English</th></tr></thead>"
    )
    fold_sum = (
        "Insight badge / narrative semantics — click to collapse"
        if en
        else "Insight 徽标与 narrative 字段（可收起）"
    )
    fold_body = (
        "<p class='note' lang='en'>Tokens below are produced as English enums in JSON; "
        "cards show the same strings as pink/blue badges.</p>"
        "<h3><code>human_review_priority</code>（卡片粉色徽标）</h3>"
        f"<table class='inner'>{th3}<tbody>{pr_body}</tbody></table>"
        "<h3><code>narrative_alignment</code>（卡片蓝色徽标）</h3>"
        "<p class='note'>约定取值：<code>aligned</code> | <code>mixed</code> | "
        "<code>conflict</code>（与 <code>agent_insight_llm_mapper</code> 的 schema 一致；"
        "若模型输出其它字符串，以 JSON 原文为准）。</p>"
        f"<table class='inner'>{th3}<tbody>{al_body}</tbody></table>"
    )
    return (
        "<section class='insight-legend' id='sec-insight-fields'>"
        "<details class='report-fold insight-fold' open>"
        f"<summary>{html.escape(fold_sum)}</summary>"
        f"<div class='report-fold-body'>{fold_body}</div>"
        "</details></section>"
    )


def _reading_scenarios_html(*, html_lang: str = "zh-CN") -> str:
    """Two audience tracks: base-model tuning vs agent/tools/product tuning."""
    en = str(html_lang).lower().startswith("en")
    if en:
        return (
            "<section class='reading-scenarios' id='sec-audience'>"
            "<h2>Reading perspectives</h2>"
            "<p class='note'>One report, two common <em>reading weights</em>—not orthogonal silos: "
            "native agentic runs mix model quality and tooling. Use the "
            "<a href='#sec-charts'>Charts &amp; lens tabs</a> to re-order attention on the "
            "<strong>same batch</strong>; see also "
            "<a href='#sec-signal-cluster'>signal clustering</a>.</p>"
            "<div class='audience-grid'>"
            "<div class='audience-card'>"
            "<h3>2.1 Base-model performance</h3>"
            "<ul>"
            "<li><a href='#sec-cohort'>Model × date × tier table</a>; "
            "<a href='#sec-charts'>Charts</a> (shared stacked tier) + "
            "<strong>Lens A</strong> tab for signal bars</li>"
            "<li><code>stats.llm_analysis_score</code>, <code>llm_quality_score</code>, "
            "<code>llm_difficulty_score</code> (see snapshot + signal evidence)</li>"
            "<li>§5b axes in meta (<code>dialog_*</code>, <code>agent_trace_coherence</code>): "
            "<a href='#sec-insight-fields'>Insight fields</a>, "
            "signal <code>dialog_turn_quality_meta_low</code></li>"
            "<li><a href='#sec-insights-hp'>Insight cards (high suspicion)</a> · "
            "<a href='#sec-insights-wl'>watchlist</a> — headline / P0–P3 vs scores</li>"
            "<li><a href='#sec-cases'>Case studies</a>: empty or truncated <code>response</code>, "
            "LLM eval rationales</li>"
            "</ul>"
            "</div>"
            "<div class='audience-card'>"
            "<h3>2.2 Agent stack, tools, product</h3>"
            "<ul>"
            "<li>Tools: <code>meta.tool_*</code>, signals "
            "<code>tool_message_error_pattern</code>, <code>low_tool_success_ratio</code> — "
            "prefer <strong>Lens B</strong> under <a href='#sec-charts'>Charts &amp; tabs</a></li>"
            "<li>Skills / usage / latency: <code>agent_skill_insights</code>, "
            "<code>total_tokens</code>, <code>agent_total_cost_time_ms</code></li>"
            "<li>Trace fit: <code>agent_trace_coherence</code>, "
            "<code>agent_tool_relevance</code> (and tool-type meta)</li>"
            "<li>PII / redaction: check <code>pii_*</code> mappers and root "
            "<code>query</code>/<code>response</code> in exported JSON.</li>"
            "<li><a href='#sec-cases'>Case studies</a>: tool traces and per-signal evidence tables</li>"
            "</ul>"
            "</div>"
            "</div>"
            "</section>"
        )
    return (
        "<section class='reading-scenarios' id='sec-audience'>"
        "<h2>分场景阅读指引</h2>"
        "<p class='note'>两类读法对应<strong>先读什么、多信哪类证据</strong>，不是两套互斥数据；"
        "原生 Agent 场景里基模与链路往往一起出问题。"
        "<a href='#sec-charts'>图表区</a>的两个视角 Tab 与 <a href='#sec-signal-cluster'>信号聚类表</a>"
        "帮你先看批次全貌再下钻单条。</p>"
        "<div class='audience-grid'>"
        "<div class='audience-card'>"
        "<h3>2.1 关心基模（生成质量 / 指令遵循 / 难度）</h3>"
        "<ul>"
        "<li><a href='#sec-cohort'>按模型 × 日期 × 分档 队列明细</a>；"
        "<a href='#sec-charts'>图表区</a> 内共用堆叠分档图，并打开 <strong>视角 A</strong> Tab 看偏基模/末轮系信号柱图</li>"
        "<li><code>stats.llm_analysis_score</code>、<code>llm_quality_score</code>、"
        "<code>llm_difficulty_score</code>（快照表与各信号证据小表）</li>"
        "<li>§5b 末轮/轨迹 LLM 轴分（<code>meta.dialog_*</code>、<code>agent_trace_coherence</code> 等）与 "
        "信号 <code>dialog_turn_quality_meta_low</code>；对照 "
        "<a href='#sec-insight-fields'>Insight 字段说明</a></li>"
        "<li><a href='#sec-insights-hp'>单条 Insight（强怀疑）</a> · "
        "<a href='#sec-insights-wl'>（待观察）</a>： headline、P0–P3 与分数量化是否一致</li>"
        "<li><a href='#sec-cases'>Case study</a>：空/极短回复、<code>llm_*_record</code> 里的 discard 与 rationale</li>"
        "</ul>"
        "</div>"
        "<div class='audience-card'>"
        "<h3>2.2 关心 Agent 链路 / 工具 / 产品与数据出口</h3>"
        "<ul>"
        "<li>工具链：<code>meta.tool_success_count</code> / <code>tool_fail_count</code> / "
        "<code>tool_success_ratio</code>；信号 <code>tool_message_error_pattern</code>（high）、"
        "<code>low_tool_success_ratio</code>（medium）— 图表区优先看 <strong>视角 B</strong> Tab</li>"
        "<li>能力归纳与用量：<code>meta.agent_skill_insights</code>、<code>total_tokens</code>、"
        "<code>agent_total_cost_time_ms</code>（latency 等信号若菜谱开启）</li>"
        "<li>轨迹与工具相关性：<code>agent_trace_coherence</code>、<code>agent_tool_relevance</code> "
        "及工具类型相关 meta</li>"
        "<li>脱敏与合规：在导出 JSON 里核对 <code>pii_*</code> 与根字段 "
        "<code>query</code>/<code>response</code> 是否一致。</li>"
        "<li><a href='#sec-cases'>Case study</a>：各信号 ↔ 上游字段，以及 <code>messages</code> 工具轨迹</li>"
        "</ul>"
        "</div>"
        "</div>"
        "</section>"
    )


def _model_tier_matrix(rows: List[dict]) -> Dict[str, Counter]:
    m: DefaultDict[str, Counter] = defaultdict(Counter)
    for row in rows:
        meta = get_dj_meta(row)
        model = str(meta.get("agent_request_model") or "_unknown")
        tier = str(meta.get("agent_bad_case_tier", "none"))
        m[model][tier] += 1
    return dict(m)


# Stacked ``request_model`` figure: top K by row count; remaining models merged into one bar.
MODEL_STACK_CHART_TOP_N = 8


def _model_tier_sorted_pairs(
    model_tier: Dict[str, Counter],
) -> List[Tuple[str, Counter]]:
    return sorted(
        model_tier.items(),
        key=lambda kv: (-sum(kv[1].values()), kv[0]),
    )


def _json_pretty(obj: object) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except TypeError:  # pragma: no cover
        return str(obj)


def _human_priority_rank(token: str) -> int:
    """P0 most urgent first → P3; unknown last."""
    u = (token or "").strip().upper()
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    return order.get(u, 50)


def _narrative_alignment_rank(token: str) -> int:
    """aligned → mixed → conflict (increasing tension); unknown last."""
    v = (token or "").strip().lower()
    order = {"aligned": 0, "mixed": 1, "conflict": 2}
    return order.get(v, 50)


def _insight_sort_tuple(row: dict, row_index: int) -> Tuple[int, int, int]:
    meta = get_dj_meta(row)
    ins = meta.get("agent_insight_llm") or {}
    if not isinstance(ins, dict):
        ins = {}
    pr = _human_priority_rank(str(ins.get("human_review_priority") or ""))
    ar = _narrative_alignment_rank(str(ins.get("narrative_alignment") or ""))
    return (pr, ar, row_index)


# Redaction placeholders (pii_llm_suspect / pii_redaction mappers).
_PII_INSIGHT_PLACEHOLDERS: Tuple[str, ...] = (
    "[LLM_PII_SUSPECT_REDACTED]",
    "[PATH_REDACTED]",
    "[EMAIL_REDACTED]",
    "[REDACTED]",  # generic secret placeholder from pii_redaction_mapper
    "[ID_REDACTED]",
    "[PHONE_REDACTED]",
    "[ID_CARD_REDACTED]",
    "[CHANNEL_ID_REDACTED]",
    "[URL_REDACTED]",
    "[IP_REDACTED]",
    "[JWT_REDACTED]",
    "[PEM_REDACTED]",
    "[MAC_REDACTED]",
)


def _row_text_has_pii_redaction_placeholder(row: dict) -> bool:
    parts: List[str] = []
    for k in ("query", "response", "text"):
        v = row.get(k)
        if isinstance(v, str):
            parts.append(v)
    parts.append(_row_messages_str(row))
    msgs = row.get("messages")
    if isinstance(msgs, str):
        parts.append(msgs)
    elif isinstance(msgs, (list, dict)):
        try:
            parts.append(json.dumps(msgs, ensure_ascii=False))
        except TypeError:
            parts.append(str(msgs))
    blob = "\n".join(parts)
    return any(ph in blob for ph in _PII_INSIGHT_PLACEHOLDERS)


def _pii_llm_suspect_meta_flags_row(meta: dict) -> bool:
    rec = meta.get("pii_llm_suspect")
    if not isinstance(rec, dict):
        return False
    if rec.get("error"):
        return True
    suspected = rec.get("suspected")
    if isinstance(suspected, list) and len(suspected) > 0:
        return True
    if rec.get("likely_clean") is False:
        return True
    if rec.get("redaction_applied") is True:
        return True
    return False


def _insight_llm_text_suggests_pii(ins: dict) -> bool:
    chunks: List[str] = []
    h = ins.get("headline")
    if isinstance(h, str) and h.strip():
        chunks.append(h)
    for c in ins.get("root_causes") or []:
        if isinstance(c, dict):
            for k in ("factor", "rationale_one_line"):
                v = c.get(k)
                if isinstance(v, str) and v.strip():
                    chunks.append(v)
    if not chunks:
        return False
    blob = "\n".join(chunks)
    if re.search(r"\bpii\b", blob, re.IGNORECASE):
        return True
    for needle in ("脱敏", "敏感信息", "个人隐私", "残留敏感", "疑似隐私"):
        if needle in blob:
            return True
    blob_l = blob.lower()
    for needle in ("redaction", "redacted"):
        if needle in blob_l:
            return True
    return False


def _exclude_insight_card_for_pii_row(row: dict, ins: dict) -> bool:
    """True when the row carries PII-audit / redaction signals (safe reports omit these)."""
    meta = get_dj_meta(row)
    if _pii_llm_suspect_meta_flags_row(meta):
        return True
    if _row_text_has_pii_redaction_placeholder(row):
        return True
    if isinstance(ins, dict) and _insight_llm_text_suggests_pii(ins):
        return True
    return False


def _row_has_pii_audit_signals(row: dict) -> bool:
    meta = get_dj_meta(row)
    ins = meta.get("agent_insight_llm") or {}
    if not isinstance(ins, dict):
        ins = {}
    return _exclude_insight_card_for_pii_row(row, ins)


def _filter_drill_pool_rows(rows: List[dict], *, exclude_pii: bool) -> List[dict]:
    if not exclude_pii:
        return list(rows)
    return [r for r in rows if not _row_has_pii_audit_signals(r)]


def _ordered_bad_case_row_lists(
    rows: List[dict],
    *,
    exclude_pii: bool,
) -> Tuple[List[dict], List[dict]]:
    """Split bad-case rows into (high_precision ordered, watchlist ordered)."""
    hp: List[Tuple[int, int, int, dict]] = []
    wl: List[Tuple[int, int, int, dict]] = []
    for i, row in enumerate(rows):
        if exclude_pii and _row_has_pii_audit_signals(row):
            continue
        meta = get_dj_meta(row)
        tier = str(meta.get("agent_bad_case_tier", "none"))
        if tier == "high_precision":
            pr, ar, _ = _insight_sort_tuple(row, i)
            hp.append((pr, ar, i, row))
        elif tier == "watchlist":
            pr, ar, _ = _insight_sort_tuple(row, i)
            wl.append((pr, ar, i, row))
    hp.sort(key=lambda x: (x[0], x[1], x[2]))
    wl.sort(key=lambda x: (x[0], x[1], x[2]))
    return [t[3] for t in hp], [t[3] for t in wl]


def _balanced_tier_caps(n_hp: int, n_wl: int, display_cap: int) -> Tuple[int, int]:
    """Target ~half high_precision and half watchlist; spill unused quota to the other tier."""
    if display_cap <= 0:
        return 0, 0
    want_hp = display_cap // 2
    want_wl = display_cap - want_hp
    hp_take = min(want_hp, n_hp)
    wl_take = min(want_wl, n_wl)
    deficit = display_cap - hp_take - wl_take
    if deficit > 0:
        more_hp = min(deficit, n_hp - hp_take)
        hp_take += more_hp
        deficit -= more_hp
    if deficit > 0:
        wl_take += min(deficit, n_wl - wl_take)
    return hp_take, wl_take


def _row_headline_or_snippet_for_audit_cluster(row: dict) -> str:
    meta = get_dj_meta(row)
    ins = meta.get("agent_insight_llm") or {}
    if isinstance(ins, dict):
        h = (ins.get("headline") or "").strip()
        if h:
            return h
    q = str(row.get("query") or "")[:400]
    r = str(_row_messages_str(row))[:600]
    blob = (q + "\n" + r).strip()
    return blob if blob else " "


def _diversify_rows_by_semantic_clusters(
    rows_priority: List[dict],
    limit: int,
    max_k: int,
) -> List[dict]:
    """Round-robin pick across MiniBatchKMeans clusters (same featurizer as Insight semantic block)."""
    if limit <= 0 or not rows_priority:
        return []
    if limit >= len(rows_priority):
        return list(rows_priority)
    texts = [_row_headline_or_snippet_for_audit_cluster(r) for r in rows_priority]
    labels = _tfidf_kmeans_labels(texts, max_k)
    if labels is None:
        return rows_priority[:limit]
    by_lab: DefaultDict[int, List[dict]] = defaultdict(list)
    for r, lab in zip(rows_priority, labels):
        by_lab[int(lab)].append(r)
    clusters = sorted(by_lab.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    queues: List[List[dict]] = [list(lst) for _, lst in clusters]
    out: List[dict] = []
    while len(out) < limit and any(queues):
        for q in queues:
            if len(out) >= limit:
                break
            if q:
                out.append(q.pop(0))
    if len(out) < limit:
        seen = {id(x) for x in out}
        for r in rows_priority:
            if len(out) >= limit:
                break
            if id(r) not in seen:
                out.append(r)
                seen.add(id(r))
    return out


def _pick_balanced_diverse_drill_rows(
    rows: List[dict],
    display_n: int,
    *,
    exclude_pii: bool,
    semantic_max_k: int,
    use_diversify: bool,
) -> List[dict]:
    hp_rows, wl_rows = _ordered_bad_case_row_lists(rows, exclude_pii=exclude_pii)
    hp_take, wl_take = _balanced_tier_caps(len(hp_rows), len(wl_rows), display_n)
    if not use_diversify:
        return hp_rows[:hp_take] + wl_rows[:wl_take]
    hp_pick = _diversify_rows_by_semantic_clusters(hp_rows, hp_take, semantic_max_k)
    wl_pick = _diversify_rows_by_semantic_clusters(wl_rows, wl_take, semantic_max_k)
    return hp_pick + wl_pick


def _merge_drill_all_entries(
    hp_rows: List[dict],
    wl_rows: List[dict],
    limit: Optional[int],
) -> List[dict]:
    out: List[dict] = []
    for r in hp_rows:
        out.append(_row_to_drill_entry(r))
        if limit is not None and len(out) >= limit:
            return out
    for r in wl_rows:
        out.append(_row_to_drill_entry(r))
        if limit is not None and len(out) >= limit:
            break
    return out


def _build_drill_all_entries(
    rows: List[dict],
    cap_export: Optional[int],
    *,
    exclude_pii: bool,
) -> List[dict]:
    hp_rows, wl_rows = _ordered_bad_case_row_lists(rows, exclude_pii=exclude_pii)
    return _merge_drill_all_entries(hp_rows, wl_rows, cap_export)


def _request_id_to_drill_anchor(drill: List[dict]) -> Dict[str, str]:
    """Map request_id → fragment id for in-page drill cards.

    First occurrence wins.
    """
    out: Dict[str, str] = {}
    for i, d in enumerate(drill):
        rid = str(d.get("request_id") or "").strip()
        if rid and rid not in out:
            out[rid] = f"#bc-drill-{i}"
    return out


def _row_to_drill_entry(row: dict, *, lang_en: bool = False) -> dict:
    meta = get_dj_meta(row)
    tier = str(meta.get("agent_bad_case_tier", ""))
    rid = (
        meta.get("agent_request_id")
        or row.get("request_id")
        or row.get("trace_id")
        or row.get("id")
    )
    rid_s = str(rid).strip() if rid is not None else ""
    u_idx = meta.get("agent_last_user_msg_idx")
    a_idx = meta.get("agent_last_assistant_msg_idx")
    signals = meta.get("agent_bad_case_signals") or []
    insight = meta.get("agent_insight_llm") or {}
    meta_subset = {
        k: meta[k]
        for k in (
            "agent_request_id",
            "agent_last_user_msg_idx",
            "agent_last_assistant_msg_idx",
            "agent_request_model",
            "agent_pt",
            "agent_bad_case_tier",
            "agent_turn_count",
            "agent_total_cost_time_ms",
        )
        if k in meta
    }
    return {
        "tier": tier,
        "tier_label_zh": _tier_zh(tier),
        "request_id": rid_s,
        "u_idx": u_idx,
        "a_idx": a_idx,
        "model": str(meta.get("agent_request_model") or ""),
        "pt": str(meta.get("agent_pt") or ""),
        "query": row.get("query") or "",
        "message": _row_messages_str(row),
        "message_html": _drill_message_panel_html(row, lang_en=lang_en),
        "response": row.get("response") or "",
        "signals_json": _json_pretty(signals),
        "insight_json": _json_pretty(insight),
        "meta_json": _json_pretty(meta_subset),
        "evidence_snapshot_html": _global_evidence_snapshot_html(row),
        "signal_evidence_html": _signal_evidence_tables_html(signals, row),
    }


def _drill_export_payload(d: dict) -> dict:
    """JSONL-friendly row (drop pre-rendered HTML)."""
    skip = frozenset({"evidence_snapshot_html", "signal_evidence_html", "message_html"})
    return {k: v for k, v in d.items() if k not in skip}


def _write_drilldown_jsonl(path: Path, entries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in entries:
            f.write(json.dumps(_drill_export_payload(d), ensure_ascii=False, default=str) + "\n")


def _idx_badge(u_idx: object, a_idx: object) -> str:
    parts = []
    if u_idx is not None:
        parts.append(f"user_idx={u_idx}")
    if a_idx is not None:
        parts.append(f"asst_idx={a_idx}")
    return " · ".join(parts) if parts else "—"


def _drilldown_section_html(
    drill: List[dict],
    *,
    total_count: int,
    export_rel: Optional[str] = None,
    html_lang: str = "zh-CN",
    use_display_diversify: bool = True,
    scale_note_html: str = "",
) -> str:
    """Case study 清单：页内为两档配额 + 可选语义分桶抽样，全量可另存 jsonl。"""
    en = str(html_lang).lower().startswith("en")
    title = (
        "Case study (high_precision / watchlist, expandable)"
        if en
        else "Case study（强怀疑 / 待观察，可展开详情）"
    )
    if not drill and total_count == 0:
        empty = (
            "No high_precision or watchlist rows in this batch, or this section was disabled."
            if en
            else "本批没有 <code>high_precision</code>（强怀疑）或 "
            "<code>watchlist</code>（待观察）命中样本；或已在命令行关闭本段。"
        )
        return (
            f"<h2 id='sec-cases'>{html.escape(title)}</h2>"
            f"<p class='note'>{html.escape(empty)}</p>"
        )
    shown = len(drill)
    scale_block = scale_note_html or ""
    extra_note = ""
    if total_count > shown:
        strat = (
            "Per-tier ~half quota + semantic cluster round-robin (char TF-IDF + "
            "MiniBatchKMeans, same as Insight); within tier: P0→P3, aligned→conflict."
            if use_display_diversify and en
            else (
                "两档各约一半配额，并对每档按与 Insight 相同的字符 TF-IDF + MiniBatchKMeans "
                "语义簇做轮询抽样；档内仍为 P0→P3、aligned→mixed→conflict。"
                if use_display_diversify
                else (
                    "Per-tier ~half quota; within tier: P0→P3, aligned→conflict."
                    if en
                    else "两档各约一半配额；档内按 P0→P3、aligned→mixed→conflict 截取。"
                )
            )
        )
        head = (
            f"<p class='note'><strong>页内展示 {shown} 条</strong>（{strat}）；"
            f"本批本版导出条件共 <strong>{total_count}</strong> 条。"
            if not en
            else f"<p class='note'><strong>{shown}</strong> rows in-page ({strat}); "
            f"<strong>{total_count}</strong> rows in this report’s export cohort."
        )
        extra_note = head
        if export_rel:
            ext = (
                f" Full export: "
                f"<a href='{html.escape(export_rel)}'><code>{html.escape(export_rel)}</code></a> "
                f"(JSON Lines)."
                if en
                else (
                    f" 全量请打开："
                    f"<a href='{html.escape(export_rel)}'><code>{html.escape(export_rel)}</code></a> "
                    f"（JSON Lines）。"
                )
            )
            extra_note += ext
        extra_note += "</p>"
    elif export_rel and total_count > 0:
        extra_note = (
            f"<p class='note'>本批共 <strong>{total_count}</strong> 条，已写入 "
            f"<a href='{html.escape(export_rel)}'><code>{html.escape(export_rel)}</code></a>。</p>"
            if not en
            else f"<p class='note'><strong>{total_count}</strong> rows written to "
            f"<a href='{html.escape(export_rel)}'><code>{html.escape(export_rel)}</code></a>.</p>"
        )
    cards = []
    for i, d in enumerate(drill):
        anchor = f"bc-drill-{i}"
        tier_m = html.escape(d["tier"])
        tier_show = html.escape(d.get("tier_label_zh") or _tier_zh(d["tier"]))
        tier_cls = "tier-hp" if d["tier"] == "high_precision" else "tier-wl"
        rid = html.escape(d["request_id"] or "—")
        idx_txt = html.escape(_idx_badge(d["u_idx"], d["a_idx"]))
        model = html.escape(d["model"] or "—")
        pt = html.escape(d["pt"] or "—")
        ev_snap = d.get("evidence_snapshot_html") or ""
        sig_ev = d.get("signal_evidence_html") or ""
        msg_panel = d.get("message_html") or (
            f'<pre class="drill-pre drill-pre--msg">'
            f"{html.escape(d.get('message') or '')}</pre>"
        )
        rsp_txt = d.get("response")
        rsp_txt = rsp_txt if isinstance(rsp_txt, str) else (str(rsp_txt) if rsp_txt is not None else "")
        cards.append(
            f'<div class="drill-card" id="{anchor}">'
            '<div class="drill-summary">'
            f'<span class="tier-tag {tier_cls}" title="机器值 {tier_m}">{tier_show}</span> '
            f'<span class="tier-mach"><code>{tier_m}</code></span> '
            f'<a class="anchor-link" href="#{anchor}" title="锚点">#{i}</a> '
            f"<code class=\"rid\" title=\"request_id / trace / id\">{rid}</code> "
            f'<span class="idx" title="messages 中下标（0-based）">{idx_txt}</span> '
            f'<span class="cohort-mini">{model} · {pt}</span> '
            '<button type="button" class="drill-toggle" aria-expanded="false">'
            "展开字段</button>"
            "</div>"
            '<div class="drill-body" hidden>'
            f"{ev_snap}"
            "<h4 class='ev-h'>各信号 ↔ 上游字段与取值</h4>"
            f"{sig_ev}"
            '<div class="field-grid">'
            "<section><h4>query</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['query'])}</pre></section>"
            "<section class='drill-field-wide'><h4>message</h4>"
            f"{msg_panel}</section>"
            "<section class='drill-field-wide'><h4>response</h4>"
            f"<pre class=\"drill-pre drill-pre--response\">{html.escape(rsp_txt)}</pre></section>"
            "<section><h4>agent_bad_case_signals（JSON）</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['signals_json'])}</pre></section>"
            "<section><h4>agent_insight_llm</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['insight_json'])}</pre></section>"
            "<section><h4>meta（钻取子集）</h4>"
            f"<pre class=\"drill-pre\">{html.escape(d['meta_json'])}</pre></section>"
            "</div></div></div>"
        )
    intro = (
        "<p class='note'>列表顺序：<strong>强怀疑区块在前，待观察区块在后</strong>。"
        "点「展开字段」查看快照与各信号证据、<strong>query</strong>、<strong>message</strong>（多轮 "
        "<code>messages</code> 会按角色分条排版）、<strong>response</strong>（末轮/规范化字段）；"
        "<code>#编号</code> 可分享锚点；与上文 Insight 中 <code>request_id</code> 对应。</p>"
        if not en
        else (
            "<p class='note'>Order: <strong>high_precision block</strong>, then <strong>watchlist</strong>. "
            "Expand for evidence tables, <strong>query</strong>, <strong>message</strong> "
            "(multi-turn <code>messages[]</code> rendered by role), and <strong>response</strong>; "
            "<code>#</code> is the in-page anchor; <code>request_id</code> matches Insight cards.</p>"
        )
    )
    block = (
        f"<h2 id='sec-cases'>{html.escape(title)}</h2>"
        f"{scale_block}{extra_note}{intro}"
        '<div class="drill-list">'
        f"{''.join(cards)}</div>"
    )
    return block


def _build_llm_digest_compact(
    n_rows: int,
    tier_cnt: Counter,
    high_c: Counter,
    med_c: Counter,
    cohort_rows: List[dict],
) -> str:
    """Minimal text for page-top LLM (shorter latency / tokens)."""
    hp = int(tier_cnt.get("high_precision", 0))
    wl = int(tier_cnt.get("watchlist", 0))
    nn = int(tier_cnt.get("none", 0))
    lines = [
        f"n={n_rows} high_precision={hp} watchlist={wl} none={nn}",
    ]
    if high_c:
        top_h = high_c.most_common(8)
        lines.append("high_signals: " + ", ".join(f"{c}:{n}" for c, n in top_h))
    if med_c:
        top_m = med_c.most_common(4)
        lines.append("med_signals: " + ", ".join(f"{c}:{n}" for c, n in top_m))
    ranked = sorted(
        [r for r in cohort_rows if int(r.get("count") or 0) > 0],
        key=lambda r: -int(r.get("count") or 0),
    )[:6]
    if ranked:
        bits = []
        for r in ranked:
            sig = str(r.get("top_signal_codes") or "")
            if len(sig) > 42:
                sig = sig[:42] + "…"
            bits.append(
                f"{r.get('agent_request_model', '')}|pt={r.get('agent_pt', '')}|"
                f"{r.get('tier', '')}|n={r.get('count')}|{sig}"
            )
        lines.append("top_cohorts: " + " / ".join(bits))
    return "\n".join(lines)


def _rule_based_exec_summary(
    n_rows: int,
    tier_cnt: Counter,
    high_c: Counter,
    med_c: Counter,
) -> str:
    hp = int(tier_cnt.get("high_precision", 0))
    wl = int(tier_cnt.get("watchlist", 0))
    nn = int(tier_cnt.get("none", 0))
    lines = [
        "【结论摘要】",
        f"- 本批合计 {n_rows} 条；其中强怀疑（主证据）{hp} 条、待观察（弱证据）{wl} 条、未标记 {nn} 条。",
    ]
    if high_c:
        bits = [f"{c}（{n} 次）" for c, n in high_c.most_common(5)]
        lines.append("- high 权重信号出现较多的有：" + "；".join(bits) + "。")
    lines.extend(
        [
            "",
            "【后续分析建议】",
            "- 结合下方「按模型堆叠图」与 cohort 表，看强怀疑是否集中在少数模型或日期桶。",
            "- 对 high 权重信号做共现统计，避免单条启发式误杀。",
            "- 强怀疑档建议优先人工复核；待观察档宜抽样或与业务规则对照。",
            "",
            "【阅读提示】",
            "- 「强怀疑」多为结构化主证据；「待观察」多为弱信号组合，请勿混读。",
        ]
    )
    return "\n".join(lines)


def _fetch_exec_summary_llm(
    digest: str,
    *,
    model: str,
    api_key: str,
    api_base: str,
    timeout_sec: int = 120,
) -> Optional[str]:
    """OpenAI-compatible ``/v1/chat/completions`` (DashScope 兼容模式、OpenAI 等).

    读超时（含 ``The read operation timed out``）时自动重试 1 次；仍失败则返回 None 并打印 WARNING。
    """
    url = api_base.rstrip("/") + "/chat/completions"
    system = (
        "你是数据分析顾问。只根据用户给的若干行批次汇总写报告页首短导读；"
        "禁止编造未出现的数字、档位名或信号 code；表述简练。"
    )
    user = (
        "以下为同一批次的极简汇总（非逐条日志）。请用纯中文、纯文本 output，分三块，块间空一行：\n"
        "【结论摘要】3～5行，每行以「- 」，基于 n= / high_precision= / high_signals。\n"
        "【后续分析建议】2～4行，每行「- 」，可执行即可。\n"
        "【阅读提示】1～2行，区分强怀疑(high_precision)与待观察(watchlist)。\n\n"
        f"{digest}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 768,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        req_headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=req_headers,
    )
    body: Optional[dict] = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except json.JSONDecodeError as e:
            print(f"WARNING: LLM 导读响应非 JSON: {e}", file=sys.stderr)
            return None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            if attempt == 0 and _is_llm_read_timeout(e):
                print(
                    f"WARNING: LLM 导读读超时（{timeout_sec}s），重试一次…",
                    file=sys.stderr,
                )
                continue
            print(
                f"WARNING: LLM 导读请求失败: {e} "
                f"（timeout={timeout_sec}s；可调 --llm-timeout-sec 或 BAD_CASE_REPORT_LLM_TIMEOUT_SEC）",
                file=sys.stderr,
            )
            return None
    if body is None:
        return None
    try:
        return str(body["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError):
        print(f"WARNING: LLM 导读返回格式异常: {str(body)[:800]}", file=sys.stderr)
        return None


def _exec_summary_section_html(body_text: str, source_note: str) -> str:
    return (
        "<section class='exec-summary' id='sec-guide'>"
        "<h2>报告导读（结论与后续分析）</h2>"
        f"<p class='note'>{html.escape(source_note)}</p>"
        f"<pre class='exec-summary-pre'>{html.escape(body_text)}</pre>"
        "</section>"
    )


def _report_toc_links(html_lang: str = "zh-CN") -> List[Tuple[str, str]]:
    """(href, label) for in-page navigation."""
    en = str(html_lang).lower().startswith("en")
    if en:
        return [
            ("#sec-guide", "Guide"),
            ("#sec-audience", "Perspectives"),
            ("#sec-counts", "Counts by tier"),
            ("#sec-tiers", "Tier semantics"),
            ("#sec-charts", "Charts & tabs"),
            ("#sec-dialog-metrics", "Dialogue & tokens"),
            ("#sec-macro-dist", "Tags & tools"),
            ("#sec-signal-cluster", "Signal clustering"),
            ("#sec-insight-fields", "Insight fields"),
            ("#sec-insights-hp", "Insight (high suspicion)"),
            ("#sec-insights-wl", "Insight (watchlist)"),
            ("#sec-cases", "Case study"),
            ("#sec-attrib", "Signal attribution"),
            ("#sec-cohort", "Cohort table"),
        ]
    return [
        ("#sec-guide", "导读"),
        ("#sec-audience", "分场景阅读"),
        ("#sec-counts", "各档条数"),
        ("#sec-tiers", "分档说明"),
        ("#sec-charts", "图表与 Tab"),
        ("#sec-dialog-metrics", "对话与用量"),
        ("#sec-macro-dist", "标签/工具分布"),
        ("#sec-signal-cluster", "信号聚类"),
        ("#sec-insight-fields", "Insight 字段"),
        ("#sec-insights-hp", "Insight（强怀疑）"),
        ("#sec-insights-wl", "Insight（待观察）"),
        ("#sec-cases", "Case study"),
        ("#sec-attrib", "归因表"),
        ("#sec-cohort", "队列明细"),
    ]


def _report_toc_html(variant: str = "bar", *, html_lang: str = "zh-CN") -> str:
    """Top bar (bar), sticky sidebar (side), or compact one-liner (mini) for section ends."""
    links = _report_toc_links(html_lang)
    if variant == "side":
        lis = "".join(
            f'<li><a href="{html.escape(h)}">{html.escape(t)}</a></li>' for h, t in links
        )
        return (
            '<nav class="report-toc report-toc--side" aria-label="页面目录">'
            "<strong>快速跳转</strong>"
            f"<ul>{lis}</ul></nav>"
        )
    if variant == "mini":
        inner = " · ".join(
            f'<a href="{html.escape(h)}">{html.escape(t)}</a>' for h, t in links
        )
        return (
            '<nav class="report-toc report-toc--mini" aria-label="本节结束·快速跳转">'
            f"<p class='note'><strong>快速跳转：</strong> {inner}</p></nav>"
        )
    inner = " · ".join(
        f'<a href="{html.escape(h)}">{html.escape(t)}</a>' for h, t in links
    )
    return (
        '<nav class="report-toc report-toc--bar" aria-label="页面内锚点">'
        f"<p class='note'><strong>快速跳转：</strong> {inner}</p></nav>"
    )


def _row_signal_code_weights(meta: dict) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in meta.get("agent_bad_case_signals") or []:
        if isinstance(s, dict) and s.get("code"):
            w = str(s.get("weight") or "").strip()
            out.append((str(s["code"]), w or "—"))
    return out


def _insight_card_lens_html(meta: dict, *, html_lang: str) -> str:
    """Per-row dual lens: which hooks to read first for base-turn vs tools/stack."""
    en = str(html_lang).lower().startswith("en")
    entries = _row_signal_code_weights(meta)
    codes_a = _PURE_BASE_MODEL_SIGNALS | _CROSS_READ_SIGNALS
    codes_b = _PURE_AGENT_STACK_SIGNALS | _CROSS_READ_SIGNALS

    if not entries:
        if en:
            inner = (
                "No <code>agent_bad_case_signals</code> on this row — see the case-study section for snapshots."
            )
        else:
            inner = html.escape(
                "本行未挂 bad-case 信号；可到下文章节「case study」看快照与证据。"
            )
        return f"<div class='ins-lenses'><p class='note ins-lens-empty'>{inner}</p></div>"

    def _fmt_items(items: List[Tuple[str, str]]) -> str:
        lis = []
        for c, w in items:
            cx = ""
            if c in _CROSS_READ_SIGNALS:
                cx = (
                    f" <span class='lens-cross'>({'cross' if en else '交叉'})</span>"
                )
            wt = html.escape(w)
            lis.append(f"<li><code>{html.escape(c)}</code> · {wt}{cx}</li>")
        return "<ul class='lens-sig'>" + "".join(lis) + "</ul>"

    def _pick(codeset: Set[str]) -> List[Tuple[str, str]]:
        return [(c, w) for c, w in entries if c in codeset]

    a_items = _pick(codes_a)
    b_items = _pick(codes_b)
    other = [(c, w) for c, w in entries if c not in codes_a and c not in codes_b]

    title_a = "Lens A · base / final turn" if en else "视角 A · 基模与末轮"
    title_b = "Lens B · tools / stack cost" if en else "视角 B · 工具与链路"
    hint_a = (
        "Start from LLM scores, discard rationales, §5b turn axes; cross-tagged rows also need a tools check."
        if en
        else "先盯 LLM 分、discard、末轮多轴；标「交叉」时请对照工具/用量是否拖累末轮。"
    )
    hint_b = (
        "Start from tool pattern, success ratio, tokens/latency; cross-tagged rows feed back to turn quality."
        if en
        else "先盯工具成败、token、时延；「交叉」类需回头看重生成与末轮打分。"
    )
    miss_a = (
        "No A-lens-typical codes — still scan drilldown / Lens B."
        if en
        else "未挂视角 A 典型信号（可对照视角 B 或下文 case study）。"
    )
    miss_b = (
        "No B-lens-typical codes — still scan drilldown / Lens A."
        if en
        else "未挂视角 B 典型信号（可对照视角 A 或 case study 里的 tool meta）。"
    )

    def _col(title: str, hint: str, items: List[Tuple[str, str]], miss: str) -> str:
        body = _fmt_items(items) if items else f"<p class='lens-miss'>{html.escape(miss)}</p>"
        return (
            "<div class='ins-lens'>"
            f"<strong>{html.escape(title)}</strong>"
            f"<p class='lens-hint'>{html.escape(hint)}</p>"
            f"{body}"
            "</div>"
        )

    head = "Dual-lens hints (same bucketing as charts above)" if en else "双视角（与上方图表、聚类表同一套信号分桶）"
    other_html = ""
    if other:
        bits = []
        for c, w in other:
            bits.append(f"<code>{html.escape(c)}</code> · {html.escape(w)}")
        ot = "Other (see attribution):" if en else "其它信号（见归因表）："
        other_html = (
            f"<p class='note lens-other'><strong>{html.escape(ot)}</strong> "
            f"{' · '.join(bits)}</p>"
        )

    return (
        "<div class='ins-lenses'>"
        f"<div class='ins-lenses-head'>{html.escape(head)}</div>"
        "<div class='ins-lenses-grid'>"
        f"{_col(title_a, hint_a, a_items, miss_a)}"
        f"{_col(title_b, hint_b, b_items, miss_b)}"
        "</div>"
        f"{other_html}"
        "</div>"
    )


def _skill_insight_coverage_stats(rows: List[dict]) -> Tuple[int, int, int]:
    """Return (n_rows, rows_with_any_skill_insight, sum of label slots per row)."""
    n = len(rows)
    with_lab = 0
    slots = 0
    for row in rows:
        meta = get_dj_meta(row)
        labs = _skill_insight_labels_from_meta(meta)
        if labs:
            with_lab += 1
            slots += len(labs)
    return n, with_lab, slots


def _insight_cluster_key(text: str) -> str:
    t = " ".join((text or "").split())
    if not t:
        return ""
    if all(ord(c) < 128 for c in t):
        return t.casefold()
    return t


def _model_family_key(model: str) -> str:
    m = (model or "").strip().lower()
    if not m:
        return "unknown"
    # Substring checks on model id / endpoint strings; keep broader vendors last.
    if any(k in m for k in ("qwen", "tongyi", "dashscope", "通义")):
        return "qwen"
    if any(
        k in m
        for k in (
            "gpt",
            "openai",
            "o1-preview",
            "o1-mini",
            "o3-",
            "chatgpt",
        )
    ):
        return "gpt"
    if "claude" in m or "anthropic" in m:
        return "claude"
    if "gemini" in m or "google" in m:
        return "gemini"
    if "deepseek" in m:
        return "deepseek"
    if any(k in m for k in ("kimi", "moonshot", "月之暗面")):
        return "kimi"
    if any(k in m for k in ("glm", "chatglm", "zhipu", "智谱", "cogview", "codegeex")):
        return "glm"
    if any(k in m for k in ("minimax", "abab", "海螺")):
        return "minimax"
    return "other"


def _model_family_tab_label(fk: str, *, en: bool) -> str:
    labels = {
        "qwen": ("Qwen / 通义", "Qwen / Tongyi"),
        "gpt": ("GPT / OpenAI", "GPT / OpenAI"),
        "claude": ("Claude", "Claude"),
        "gemini": ("Gemini", "Gemini"),
        "deepseek": ("DeepSeek", "DeepSeek"),
        "kimi": ("Kimi / Moonshot", "Kimi / Moonshot"),
        "glm": ("GLM / 智谱", "GLM / Zhipu"),
        "minimax": ("MiniMax", "MiniMax"),
        "other": ("其它模型", "Other models"),
        "unknown": ("未标注模型", "Unknown model"),
    }
    zh, eg = labels.get(fk, ("其它", "Other"))
    return eg if en else zh


def _normalize_model_tab_bucket_key(model: str) -> str:
    """Bucket key for Insight model tabs: full label, versions kept (3.5 vs 2.5, etc.)."""
    s = " ".join(str(model or "").split())
    if not s:
        return "unknown"
    if all(ord(c) < 128 for c in s):
        return s.casefold()
    return s


_INSIGHT_TAB_LABEL_MAX = 56


def _truncate_insight_tab_caption(s: str, max_len: int = _INSIGHT_TAB_LABEL_MAX) -> str:
    t = s.strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _insight_model_tab_bucket(row: dict, *, group: str) -> str:
    meta = get_dj_meta(row)
    raw = str(meta.get("agent_request_model") or "")
    if group == "family":
        return _model_family_key(raw)
    return _normalize_model_tab_bucket_key(raw)


def _insight_tab_bucket_from_chart_model_key(chart_key: str, *, group: str) -> str:
    """Map ``_model_tier_matrix`` key (uses ``_unknown``) to the same bucket as Insight tabs."""
    raw = "" if chart_key == "_unknown" else str(chart_key)
    if group == "family":
        return _model_family_key(raw)
    return _normalize_model_tab_bucket_key(raw)


def _insight_tab_batch_volume_by_bucket(
    model_tier: Dict[str, Counter],
    *,
    group: str,
) -> Counter:
    vol: Counter = Counter()
    for ck, tc in model_tier.items():
        bk = _insight_tab_bucket_from_chart_model_key(ck, group=group)
        vol[bk] += sum(tc.values())
    return vol


def _insight_model_tab_caption(bucket: str, *, group: str, en: bool) -> str:
    if group == "family":
        return _model_family_tab_label(bucket, en=en)
    if bucket == "unknown":
        return _model_family_tab_label("unknown", en=en)
    return bucket


def _collect_tier_insight_candidates(
    rows: List[dict],
    tier: str,
    *,
    hide_pii: bool,
) -> Tuple[List[dict], int]:
    """Rows with headline in tier, sorted; ``hide_pii`` drops PII-flagged rows (count in int)."""
    scored: List[Tuple[Tuple[int, int, int], dict]] = []
    omitted = 0
    for i, row in enumerate(rows):
        meta = get_dj_meta(row)
        if str(meta.get("agent_bad_case_tier", "")) != tier:
            continue
        ins = meta.get("agent_insight_llm") or {}
        if not isinstance(ins, dict):
            ins = {}
        hl = (ins.get("headline") or "").strip()
        if not hl:
            continue
        if hide_pii and _exclude_insight_card_for_pii_row(row, ins):
            omitted += 1
            continue
        scored.append((_insight_sort_tuple(row, i), row))
    scored.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))
    return [row for _, row in scored], omitted


def _insight_card_html(
    row: dict,
    anchor_by_rid: Dict[str, str],
    *,
    en: bool,
) -> str:
    meta = get_dj_meta(row)
    ins = meta.get("agent_insight_llm") or {}
    if not isinstance(ins, dict):
        ins = {}
    hl = (ins.get("headline") or "").strip()
    rid = str(
        meta.get("agent_request_id")
        or row.get("request_id")
        or row.get("trace_id")
        or row.get("id")
        or ""
    ).strip()
    model = str(meta.get("agent_request_model") or "")
    pr = (ins.get("human_review_priority") or "").strip()
    align = (ins.get("narrative_alignment") or "").strip()
    audit = (ins.get("audit_notes") or "").strip()
    facets = ins.get("viz_facets") or []
    if not isinstance(facets, list):
        facets = [str(facets)]
    facets_s = "、".join(str(x) for x in facets[:12] if x)
    causes = ins.get("root_causes") or []
    cause_lis = []
    if isinstance(causes, list):
        for c in causes[:5]:
            if not isinstance(c, dict):
                cause_lis.append(f"<li>{html.escape(str(c))}</li>")
                continue
            factor = html.escape(str(c.get("factor") or ""))
            conf = html.escape(str(c.get("confidence") or ""))
            r1 = html.escape(str(c.get("rationale_one_line") or "")[:280])
            cited = c.get("cited_fields") or []
            if not isinstance(cited, list):
                cited = [str(cited)]
            cf = html.escape(", ".join(str(x) for x in cited[:8]))
            if cf:
                cite_html = (
                    f' <span class="cite">Fields: {cf}</span>'
                    if en
                    else f' <span class="cite">依据字段: {cf}</span>'
                )
            else:
                cite_html = ""
            if en:
                cause_lis.append(
                    f"<li><strong>{factor}</strong> (conf {conf}) — {r1}{cite_html}</li>"
                )
            else:
                cause_lis.append(
                    f"<li><strong>{factor}</strong>（置信 {conf}）— {r1}{cite_html}</li>"
                )
    causes_html = "<ul class='causes'>" + "".join(cause_lis) + "</ul>" if cause_lis else ""
    drill_href = anchor_by_rid.get(rid) if rid else None
    if drill_href:
        ttxt = (
            "→ Case study for this row (snapshots &amp; fields)"
            if en
            else "→ 本条 case study（快照与字段）"
        )
        to_drill = (
            f"<a class='to-drill' href='{html.escape(drill_href, quote=True)}'>"
            f"{ttxt}</a>"
        )
    else:
        ttxt = "→ Case study section" if en else "→ case study 章节"
        to_drill = (
            f"<a class='to-drill to-drill-fallback' href='#sec-cases'>{ttxt}</a>"
        )
    meta_line = (
        f"<span class='ins-meta'><code>{html.escape(rid or '—')}</code> · "
        f"{html.escape(model or '—')} · {to_drill}</span>"
    )
    badges = []
    if pr:
        badges.append(f"<span class='ins-badge pr'>{html.escape(pr)}</span>")
    if align:
        badges.append(f"<span class='ins-badge al'>{html.escape(align)}</span>")
    badge_html = " ".join(badges)
    audit_html = (
        f"<p class='ins-audit'><strong>{'Audit notes' if en else '审阅备注'}</strong>："
        f"{html.escape(audit)}</p>"
        if audit
        else ""
    )
    facets_fold = ""
    if facets_s:
        sumy = "建议制图维度（可展开）" if not en else "Viz facet suggestions (expand)"
        facets_body = f"<p class='ins-facets'>{html.escape(facets_s)}</p>"
        facets_fold = (
            f"<details class='insight-extra'><summary>{html.escape(sumy)}</summary>"
            f"<div class='insight-extra-body'>{facets_body}</div></details>"
        )
    lens_html = _insight_card_lens_html(meta, html_lang="en" if en else "zh-CN")
    return (
        "<div class='insight-card'>"
        f"<div class='insight-h'>{html.escape(hl)} {badge_html}</div>"
        f"{meta_line}"
        f"{causes_html}"
        f"{lens_html}"
        f"{audit_html}"
        f"{facets_fold}"
        "</div>"
    )


def _collect_headline_audit_cluster_stats(
    candidates: List[dict],
) -> Tuple[Counter, Dict[str, str], Counter, Dict[str, str]]:
    hc: Counter = Counter()
    h_example: Dict[str, str] = {}
    ac: Counter = Counter()
    a_example: Dict[str, str] = {}
    for row in candidates:
        meta = get_dj_meta(row)
        ins = meta.get("agent_insight_llm") or {}
        if not isinstance(ins, dict):
            ins = {}
        hl = (ins.get("headline") or "").strip()
        if hl:
            k = _insight_cluster_key(hl)
            if k:
                hc[k] += 1
                h_example.setdefault(k, hl)
        au = (ins.get("audit_notes") or "").strip()
        if au:
            k2 = _insight_cluster_key(au)
            if k2:
                ac[k2] += 1
                a_example.setdefault(k2, au[:200])
    return hc, h_example, ac, a_example


def _insight_exact_cluster_tables_only_html(
    hc: Counter,
    h_example: Dict[str, str],
    ac: Counter,
    a_example: Dict[str, str],
    *,
    en: bool,
    top_n: int,
) -> str:
    if not hc and not ac:
        return ""
    th_c = "Rows" if en else "条数"
    th_x = "Representative text" if en else "代表性原文"
    parts: List[str] = [
        "<p class='note'>"
        + (
            "Keys collapse ASCII case and extra spaces; not semantic clustering."
            if en
            else "合并规则：英文大小写不敏感、空白压紧；非语义聚类，仅便于看重复表述。"
        )
        + "</p>"
    ]
    if hc:
        h4 = "Headline" if en else "卡片标题 headline"
        parts.append(f"<h4>{html.escape(h4)}</h4>")
        parts.append("<table class='inner insight-cluster'><thead><tr>")
        parts.append(f"<th>{th_c}</th><th>{th_x}</th></tr></thead><tbody>")
        for k, n in hc.most_common(top_n):
            ex = html.escape(h_example.get(k, k)[:220])
            parts.append(f"<tr><td>{int(n)}</td><td>{ex}</td></tr>")
        parts.append("</tbody></table>")
    if ac:
        h4a = "Audit notes" if en else "审阅备注 audit_notes"
        parts.append(f"<h4>{html.escape(h4a)}</h4>")
        parts.append("<table class='inner insight-cluster'><thead><tr>")
        parts.append(f"<th>{th_c}</th><th>{th_x}</th></tr></thead><tbody>")
        for k, n in ac.most_common(top_n):
            ex = html.escape(a_example.get(k, k)[:220])
            parts.append(f"<tr><td>{int(n)}</td><td>{ex}</td></tr>")
        parts.append("</tbody></table>")
    return "".join(parts)


def _tfidf_kmeans_labels(texts: List[str], max_k: int) -> Optional[List[int]]:
    n = len(texts)
    if n < 4:
        return None
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return None
    try:
        k = max(2, min(int(max_k), n // 2, n - 1))
        if k < 2:
            return None
        max_df = 1.0 if n < 12 else 0.95
        vec = TfidfVectorizer(
            max_features=8000,
            analyzer="char",
            ngram_range=(1, 3),
            min_df=1,
            max_df=max_df,
        )
        x = vec.fit_transform(texts)
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=0,
            batch_size=min(256, n),
            n_init=3,
        )
        lab = km.fit_predict(x)
        return lab.tolist()
    except Exception:
        return None


def _semantic_cluster_subsection_html(
    texts: List[str],
    labels: List[int],
    *,
    en: bool,
    top_n: int,
    title_en: str,
    title_zh: str,
) -> str:
    title = title_en if en else title_zh
    th_c = "Rows" if en else "条数"
    th_x = (
        "Representative (shortest in cluster)"
        if en
        else "代表性原文（簇内最短）"
    )
    buckets: DefaultDict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[int(lab)].append(i)
    ordered = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    rows: List[str] = []
    for _lab, idxs in ordered[:top_n]:
        rep_i = min(idxs, key=lambda ii: len(texts[ii]))
        ex = html.escape(texts[rep_i][:220])
        rows.append(f"<tr><td>{len(idxs)}</td><td>{ex}</td></tr>")
    tb = "".join(rows)
    return (
        f"<h4>{html.escape(title)}</h4>"
        "<table class='inner insight-cluster'><thead><tr>"
        f"<th>{th_c}</th><th>{th_x}</th></tr></thead><tbody>{tb}</tbody></table>"
    )


def _insight_semantic_cluster_block_html(
    candidates: List[dict],
    *,
    en: bool,
    top_n: int,
    max_k: int,
) -> str:
    summary = (
        "Semantic clustering (char TF-IDF + MiniBatchKMeans, open by default)"
        if en
        else "语义聚类（字符 TF-IDF + MiniBatchKMeans，默认展开）"
    )
    try:
        import sklearn  # noqa: F401
    except ImportError:
        msg = (
            "Semantic clustering needs <code>scikit-learn</code>. "
            "Install with <code>pip install scikit-learn</code> and re-run the report; "
            "the exact-normalized table below remains available."
            if en
            else "语义聚类依赖 <code>scikit-learn</code>，请执行 "
            "<code>pip install scikit-learn</code> 后重新生成报告；"
            "下方精确合并表仍可用。"
        )
        return (
            "<details class='insight-cluster insight-cluster-semantic' open>"
            f"<summary>{html.escape(summary)}</summary>"
            f"<p class='note'>{msg}</p></details>"
        )
    h_texts: List[str] = []
    a_texts: List[str] = []
    for row in candidates:
        meta = get_dj_meta(row)
        ins = meta.get("agent_insight_llm") or {}
        if not isinstance(ins, dict):
            ins = {}
        hl = (ins.get("headline") or "").strip()
        if hl:
            h_texts.append(hl)
        au = (ins.get("audit_notes") or "").strip()
        if au:
            a_texts.append(au)
    body_parts: List[str] = []
    if len(h_texts) >= 4:
        h_lab = _tfidf_kmeans_labels(h_texts, max_k)
        if h_lab is not None:
            body_parts.append(
                _semantic_cluster_subsection_html(
                    h_texts,
                    h_lab,
                    en=en,
                    top_n=top_n,
                    title_en="Headline",
                    title_zh="卡片标题 headline",
                )
            )
    if len(a_texts) >= 4:
        a_lab = _tfidf_kmeans_labels(a_texts, max_k)
        if a_lab is not None:
            body_parts.append(
                _semantic_cluster_subsection_html(
                    a_texts,
                    a_lab,
                    en=en,
                    top_n=top_n,
                    title_en="Audit notes",
                    title_zh="审阅备注 audit_notes",
                )
            )
    if not body_parts:
        msg = (
            "Not enough headline/audit lines (need ≥4 per sub-table) to build semantic clusters."
            if en
            else "本档 headline 或审阅备注有效条数不足 4，未生成语义簇表。"
        )
        inner = f"<p class='note'>{html.escape(msg)}</p>"
    else:
        foot = (
            "Char n-gram TF-IDF + MiniBatchKMeans; exploratory only, not human topic labels."
            if en
            else "基于字符 n-gram 的 TF-IDF 与 MiniBatchKMeans，仅供浏览探索，不等价于人工主题归类。"
        )
        inner = "".join(body_parts) + f"<p class='note'>{html.escape(foot)}</p>"
    return (
        "<details class='insight-cluster insight-cluster-semantic' open>"
        f"<summary>{html.escape(summary)}</summary>{inner}</details>"
    )


def _insight_headline_audit_cluster_html(
    candidates: List[dict],
    *,
    en: bool,
    top_n: int = 18,
    use_semantic_cluster: bool = True,
    semantic_max_k: int = 15,
) -> str:
    if not candidates:
        return ""
    hc, h_example, ac, a_example = _collect_headline_audit_cluster_stats(candidates)
    exact_tables = _insight_exact_cluster_tables_only_html(
        hc, h_example, ac, a_example, en=en, top_n=top_n
    )
    if not use_semantic_cluster:
        if not exact_tables:
            return ""
        h3 = (
            "Headline and audit clustering (this tier, exact-normalized)"
            if en
            else "本档 Headline / 审阅备注聚类（归一化后精确合并）"
        )
        return (
            "<div class='insight-cluster'>"
            f"<h3>{html.escape(h3)}</h3>{exact_tables}</div>"
        )
    sem = _insight_semantic_cluster_block_html(
        candidates, en=en, top_n=top_n, max_k=semantic_max_k
    )
    parts: List[str] = ["<div class='insight-clusters'>", sem]
    if exact_tables:
        sum_exact = (
            "Exact-normalized headline / audit merge (expand)"
            if en
            else "精确归一：Headline / 审阅备注合并（点击展开）"
        )
        parts.append(
            "<details class='insight-cluster insight-cluster-exact'>"
            f"<summary>{html.escape(sum_exact)}</summary>{exact_tables}</details>"
        )
    parts.append("</div>")
    return "".join(parts)


def _insight_model_tabs_html(
    sec_id: str,
    candidates: List[dict],
    limit: int,
    model_tab_limit: int,
    anchor_by_rid: Dict[str, str],
    *,
    en: bool,
    model_tab_group: str = "full",
    model_tier: Optional[Dict[str, Counter]] = None,
) -> str:
    if not candidates:
        return ""
    group = "family" if model_tab_group == "family" else "full"
    buckets: DefaultDict[str, List[dict]] = defaultdict(list)
    for row in candidates:
        bk = _insight_model_tab_bucket(row, group=group)
        buckets[bk].append(row)
    batch_vol: Optional[Counter] = None
    if model_tier:
        batch_vol = _insight_tab_batch_volume_by_bucket(model_tier, group=group)
    tab_keys = list(buckets.keys())
    if batch_vol is not None:
        tab_order = sorted(
            tab_keys,
            key=lambda bk: (-int(batch_vol.get(bk, 0)), bk),
        )
    else:
        tab_order = sorted(tab_keys, key=lambda bk: (-len(buckets[bk]), bk))
    list_html = "".join(
        _insight_card_html(r, anchor_by_rid, en=en) for r in candidates[:limit]
    )
    if len(tab_order) <= 1:
        return f"<div class='insight-list'>{list_html}</div>"

    if group == "family":
        aria = "Insight cards by model family" if en else "按模型族分组的 Insight 卡片"
    else:
        aria = (
            "Insight cards by full agent_request_model"
            if en
            else "按完整 agent_request_model 分组的 Insight 卡片"
        )
    parts: List[str] = [
        "<div class='insight-tab-host'>",
        f"<p class='note'>{html.escape(aria)}</p>",
        f"<div class='scenario-tabs' role='tablist' aria-label='{html.escape(aria)}'>",
    ]
    panels: List[str] = []

    def panel_block(pid: str, rows_slice: List[dict], *, active: bool) -> None:
        hid = "" if active else " hidden"
        cls = "scenario-panel active" if active else "scenario-panel"
        inner = "".join(_insight_card_html(r, anchor_by_rid, en=en) for r in rows_slice)
        panels.append(
            f"<div id='{html.escape(pid, quote=True)}' class='{cls}' "
            f"role='tabpanel'{hid}><div class='insight-list'>{inner}</div></div>"
        )

    pid_all = f"{sec_id}-panel-all"
    shown_all = candidates[:limit]
    parts.append(
        f"<button type='button' class='scenario-tab active' role='tab' "
        f"aria-selected='true' aria-controls='{html.escape(pid_all, quote=True)}' "
        f"id='{html.escape(sec_id + '-tab-all', quote=True)}' "
        f"data-panel='{html.escape(pid_all, quote=True)}'>"
        f"{html.escape(('All' if en else '全部') + f' ({len(shown_all)}/{len(candidates)})')}"
        f"</button>"
    )
    for i, bk in enumerate(tab_order):
        rows_b = buckets[bk]
        if not rows_b:
            continue
        shown = rows_b[:model_tab_limit]
        pid = f"{sec_id}-panel-m{i}"
        tab_id = f"{sec_id}-tab-m{i}"
        cap_full = _insight_model_tab_caption(bk, group=group, en=en)
        cap_short = _truncate_insight_tab_caption(cap_full)
        tab_label = f"{cap_short} ({len(shown)}/{len(rows_b)})"
        title_attr = f" title='{html.escape(cap_full, quote=True)}'"
        parts.append(
            f"<button type='button' class='scenario-tab' role='tab' "
            f"aria-selected='false' aria-controls='{html.escape(pid, quote=True)}' "
            f"id='{html.escape(tab_id, quote=True)}' "
            f"data-panel='{html.escape(pid, quote=True)}'{title_attr}>"
            f"{html.escape(tab_label)}</button>"
        )
    parts.append("</div>")
    panel_block(pid_all, shown_all, active=True)
    for i, bk in enumerate(tab_order):
        rows_b = buckets[bk]
        if not rows_b:
            continue
        shown = rows_b[:model_tab_limit]
        pid = f"{sec_id}-panel-m{i}"
        panel_block(pid, shown, active=False)
    parts.extend(panels)
    parts.append("</div>")
    return "".join(parts)


def _insight_section_rich_html(
    rows: List[dict],
    tier: str,
    limit: int,
    model_tab_limit: int,
    use_model_tabs: bool,
    anchor_by_rid: Dict[str, str],
    *,
    html_lang: str = "zh-CN",
    use_semantic_cluster: bool = True,
    semantic_max_k: int = 15,
    model_tab_group: str = "full",
    model_tier: Optional[Dict[str, Counter]] = None,
    hide_pii: bool = False,
    use_display_diversify: bool = True,
    insight_candidate_pool: Optional[List[dict]] = None,
    insight_pii_omitted_full_count: Optional[int] = None,
) -> str:
    """Insight cards from ``agent_insight_llm`` plus headline/audit clusters.

    Cards shown in-page use semantic-cluster round-robin when enabled (same TF-IDF
    + MiniBatchKMeans as the open cluster block). ``hide_pii`` removes PII-flagged
    rows (safe report variant).
    """
    en = str(html_lang).lower().startswith("en")
    tier_zh = _tier_zh(tier)
    sec_id = (
        "sec-insights-hp"
        if tier == "high_precision"
        else "sec-insights-wl"
        if tier == "watchlist"
        else "sec-insights"
    )
    if insight_candidate_pool is not None:
        candidates = []
        for r in insight_candidate_pool:
            meta = get_dj_meta(r)
            if str(meta.get("agent_bad_case_tier", "")) != tier:
                continue
            ins = meta.get("agent_insight_llm") or {}
            if not isinstance(ins, dict):
                ins = {}
            if not (ins.get("headline") or "").strip():
                continue
            if hide_pii and _exclude_insight_card_for_pii_row(r, ins):
                continue
            candidates.append(r)
        omitted_pii = (
            int(insight_pii_omitted_full_count or 0) if hide_pii else 0
        )
    else:
        candidates, omitted_pii = _collect_tier_insight_candidates(
            rows, tier, hide_pii=hide_pii
        )
    if not candidates and omitted_pii == 0:
        return (
            f"<section class='insight-sec' id='{sec_id}'>"
            f"<h2>单条 Insight 摘录（{html.escape(tier_zh)}）</h2>"
            "<p class='note'>本档下暂无带 <code>headline</code> 的 "
            "<code>meta.agent_insight_llm</code>（可能未跑 insight 算子，或解析失败）。</p></section>"
        )
    if hide_pii and not candidates and omitted_pii > 0:
        msg = (
            f"{omitted_pii} headline row(s) in this tier omitted here (PII audit / redaction); "
            "open the PII audit HTML or export."
            if en
            else (
                f"本档 {omitted_pii} 条带 headline 的样本因 PII 审计/脱敏信号未在此页展示；"
                "请打开同目录 PII 审计版 HTML 或导出。"
            )
        )
        return (
            f"<section class='insight-sec' id='{sec_id}'>"
            f"<h2>单条 Insight 摘录（{html.escape(tier_zh)}）</h2>"
            f"<p class='note'>{html.escape(msg)}</p></section>"
        )
    cluster_html = _insight_headline_audit_cluster_html(
        candidates,
        en=en,
        use_semantic_cluster=use_semantic_cluster,
        semantic_max_k=semantic_max_k,
    )
    div_note = ""
    if use_display_diversify and len(candidates) >= 4:
        div_note += (
            "<p class='note'>Card sample uses semantic-cluster round-robin (same method as the "
            "open cluster block below) for audit diversity.</p>"
            if en
            else "<p class='note'>以下卡片按语义簇轮询抽样展示（与下方默认展开的聚类同源），"
            "减轻与 case study 的简单前缀重复。</p>"
        )
    if hide_pii and omitted_pii > 0:
        div_note += (
            f"<p class='note'>{omitted_pii} PII-flagged headline row(s) omitted here "
            "(see PII audit report).</p>"
            if en
            else f"<p class='note'>另有 <strong>{omitted_pii}</strong> 条含 PII 线索的 headline 未在此安全版展示"
            "（详见 PII 审计版）。</p>"
        )
    display_rows = (
        _diversify_rows_by_semantic_clusters(candidates, limit, semantic_max_k)
        if use_display_diversify
        else candidates[:limit]
    )
    if use_model_tabs:
        n_chart_models = len(model_tier) if model_tier else 0
        chart_tail_en = (
            (
                "Sub-tab order matches the stacked request_model chart (descending batch "
                f"volume per bucket). The chart shows the top {MODEL_STACK_CHART_TOP_N} "
                "models by requests and merges the rest."
            )
            if n_chart_models > MODEL_STACK_CHART_TOP_N
            else (
                "Sub-tab order matches the stacked request_model chart (descending batch "
                "volume per bucket), same models as in the figure."
            )
        )
        chart_tail_zh = (
            (
                "子 Tab 顺序与同页「按请求模型堆叠」一致，按<strong>全批次</strong>各桶请求量降序；"
                f"图中为 Top {MODEL_STACK_CHART_TOP_N} 高频模型，其余合并为一柱。"
            )
            if n_chart_models > MODEL_STACK_CHART_TOP_N
            else (
                "子 Tab 顺序与同页「按请求模型堆叠」一致，按<strong>全批次</strong>各桶请求量降序；"
                "图中为本批全部请求模型。"
            )
        )
        if str(model_tab_group).strip().lower() == "family":
            intro = (
                "<p class='note'>Sorted P0→P3 then aligned→mixed→conflict. "
                "Each card adds <strong>dual-lens cues</strong> from this row’s signals; "
                "same bucketing as <a href='#sec-charts'>charts tabs</a>. "
                "<a href='#sec-insight-fields'>Badge semantics</a>. "
                "Sub-tabs group by model <em>family</em> (parsed from "
                f"<code>agent_request_model</code>). {chart_tail_en}</p>"
                if en
                else "<p class='note'>排序：P0→P3，同档 aligned→mixed→conflict。"
                "卡片中的<strong>双视角</strong>由本行信号生成，分桶规则与"
                "<a href='#sec-charts'>图表</a>、"
                "<a href='#sec-signal-cluster'>聚类表</a>一致。"
                "<a href='#sec-insight-fields'>徽标含义</a>。"
                "下方按 <code>agent_request_model</code> 解析出的<strong>模型族</strong>分 Tab 抽样展示；"
                f"{chart_tail_zh}</p>"
            )
        else:
            intro = (
                "<p class='note'>Sorted P0→P3 then aligned→mixed→conflict. "
                "Each card adds <strong>dual-lens cues</strong> from this row’s signals; "
                "same bucketing as <a href='#sec-charts'>charts tabs</a>. "
                "<a href='#sec-insight-fields'>Badge semantics</a>. "
                "Sub-tabs use the full <code>agent_request_model</code> string "
                f"(versions such as 3.5 vs 2.5 stay separate). {chart_tail_en}</p>"
                if en
                else "<p class='note'>排序：P0→P3，同档 aligned→mixed→conflict。"
                "卡片中的<strong>双视角</strong>由本行信号生成，分桶规则与"
                "<a href='#sec-charts'>图表</a>、"
                "<a href='#sec-signal-cluster'>聚类表</a>一致。"
                "<a href='#sec-insight-fields'>徽标含义</a>。"
                "下方按 <code>agent_request_model</code> <strong>完整型号</strong>分 Tab（如 3.5 与 2.5 不同桶）抽样展示；"
                f"{chart_tail_zh}</p>"
            )
    else:
        intro = (
            "<p class='note'>Sorted P0→P3 then aligned→mixed→conflict. "
            "Each card adds <strong>dual-lens cues</strong> from this row’s signals; "
            "same bucketing as <a href='#sec-charts'>charts tabs</a>. "
            "<a href='#sec-insight-fields'>Badge semantics</a>.</p>"
            if en
            else "<p class='note'>排序：P0→P3，同档 aligned→mixed→conflict。"
            "卡片中的<strong>双视角</strong>由本行信号生成，分桶规则与"
            "<a href='#sec-charts'>图表</a>、"
            "<a href='#sec-signal-cluster'>聚类表</a>一致。"
            "<a href='#sec-insight-fields'>徽标含义</a>。</p>"
        )
    if use_model_tabs:
        body_html = _insight_model_tabs_html(
            sec_id,
            display_rows,
            limit,
            max(1, model_tab_limit),
            anchor_by_rid,
            en=en,
            model_tab_group=model_tab_group,
            model_tier=model_tier,
        )
    else:
        body_html = "".join(
            _insight_card_html(r, anchor_by_rid, en=en) for r in display_rows[:limit]
        )
        body_html = f"<div class='insight-list'>{body_html}</div>"
    return (
        f"<section class='insight-sec' id='{sec_id}'>"
        f"<h2>单条 Insight 摘录（{html.escape(tier_zh)}）</h2>"
        f"{intro}"
        f"{div_note}"
        f"{cluster_html}"
        f"{body_html}</section>"
    )


def _insight_sections_html(
    rows: List[dict],
    per_tier_limit: int,
    model_tab_limit: int,
    use_model_tabs: bool,
    anchor_by_rid: Dict[str, str],
    *,
    html_lang: str = "zh-CN",
    use_semantic_cluster: bool = True,
    semantic_max_k: int = 15,
    model_tab_group: str = "full",
    model_tier: Optional[Dict[str, Counter]] = None,
    hide_pii: bool = False,
    use_display_diversify: bool = True,
    insight_hp_pool: Optional[List[dict]] = None,
    insight_wl_pool: Optional[List[dict]] = None,
    insight_pii_omitted_hp: Optional[int] = None,
    insight_pii_omitted_wl: Optional[int] = None,
) -> str:
    """Render insight sections for both high_precision and watchlist tiers."""
    blocks = [
        _insight_section_rich_html(
            rows,
            "high_precision",
            per_tier_limit,
            model_tab_limit,
            use_model_tabs,
            anchor_by_rid,
            html_lang=html_lang,
            use_semantic_cluster=use_semantic_cluster,
            semantic_max_k=semantic_max_k,
            model_tab_group=model_tab_group,
            model_tier=model_tier,
            hide_pii=hide_pii,
            use_display_diversify=use_display_diversify,
            insight_candidate_pool=insight_hp_pool,
            insight_pii_omitted_full_count=insight_pii_omitted_hp,
        ),
        _insight_section_rich_html(
            rows,
            "watchlist",
            per_tier_limit,
            model_tab_limit,
            use_model_tabs,
            anchor_by_rid,
            html_lang=html_lang,
            use_semantic_cluster=use_semantic_cluster,
            semantic_max_k=semantic_max_k,
            model_tab_group=model_tab_group,
            model_tier=model_tier,
            hide_pii=hide_pii,
            use_display_diversify=use_display_diversify,
            insight_candidate_pool=insight_wl_pool,
            insight_pii_omitted_full_count=insight_pii_omitted_wl,
        ),
    ]
    return "".join(blocks)


def _signal_counter_subset(c: Counter, codes: Set[str]) -> Counter:
    return Counter({k: int(v) for k, v in c.items() if k in codes and int(v) > 0})


def _reading_lens_cell(code: str, *, en: bool) -> str:
    if code in _CROSS_READ_SIGNALS:
        text = (
            "Cross-read: base + agent"
            if en
            else "交叉（基模与 Agent 链并读，权重自调）"
        )
    elif code in _PURE_BASE_MODEL_SIGNALS:
        text = "Base-model · turn quality first" if en else "偏基模/综合评估/末轮"
    elif code in _PURE_AGENT_STACK_SIGNALS:
        text = "Agent · tools · budget first" if en else "偏 Agent/工具/用量时延"
    else:
        text = "Other / pipeline-defined" if en else "其它（以归因表为准）"
    return text


def _signal_cluster_table_html(
    high_c: Counter,
    med_c: Counter,
    *,
    html_lang: str,
) -> str:
    en = str(html_lang).lower().startswith("en")
    keys = sorted(set(high_c.keys()) | set(med_c.keys()), key=lambda k: (-(high_c[k] + med_c[k]), k))
    if not keys:
        return (
            "<p class='note'>"
            + ("No signals in this batch." if en else "本批未写入 bad-case 信号。")
            + "</p>"
        )
    th_sig = "Signal <code>code</code>" if en else "信号 <code>code</code>"
    th_hi = "high" if en else "high 权重"
    th_med = "medium" if en else "medium"
    th_tot = "Total" if en else "合计"
    th_lens = "Reading emphasis" if en else "解读侧重（非正交）"
    rows_html = []
    for code in keys:
        hi = int(high_c.get(code, 0))
        md = int(med_c.get(code, 0))
        tot = hi + md
        rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(code)}</code></td>"
            f"<td>{hi}</td><td>{md}</td><td>{tot}</td>"
            f"<td>{html.escape(_reading_lens_cell(code, en=en))}</td>"
            "</tr>"
        )
    cap_note = (
        "<p class='note'>"
        + (
            html.escape(
                "Per-sample occurrence counts; emphasis = reading order, not silos (base vs agent overlap)."
            )
            if en
            else (
                "计数为样本上信号出现次数；「解读侧重」=阅读顺序，两类视角<strong>非正交</strong>，"
                "交叉类会出现在两个 Tab 中。"
            )
        )
        + "</p>"
    )
    return (
        cap_note
        + "<table class='signal-cluster'><thead><tr>"
        f"<th>{th_sig}</th><th>{th_hi}</th><th>{th_med}</th><th>{th_tot}</th><th>{th_lens}</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows_html)}</tbody></table>"
    )


def _build_charts_section_html(
    *,
    chart_tier: Optional[str],
    chart_model: Optional[str],
    high_c: Counter,
    med_c: Counter,
    chart_full_high: Optional[str],
    chart_full_med: Optional[str],
    plt_mod,
    no_charts: bool,
    html_lang: str,
    macro_distribution_html: str = "",
) -> str:
    """Shared overview + macro tag distributions + signal cluster + tabbed lens charts."""
    en = str(html_lang).lower().startswith("en")
    cluster = _signal_cluster_table_html(high_c, med_c, html_lang=html_lang)

    codes_a = _PURE_BASE_MODEL_SIGNALS | _CROSS_READ_SIGNALS
    codes_b = _PURE_AGENT_STACK_SIGNALS | _CROSS_READ_SIGNALS
    hi_a = _signal_counter_subset(high_c, codes_a)
    hi_b = _signal_counter_subset(high_c, codes_b)
    md_a = _signal_counter_subset(med_c, codes_a)
    md_b = _signal_counter_subset(med_c, codes_b)

    chart_base_h = chart_base_m = chart_agent_h = chart_agent_m = None
    if plt_mod is not None and not no_charts:
        if hi_a:
            chart_base_h = _chart_signals(
                hi_a,
                plt_mod,
                (
                    "Lens A — high (base / turn / LLM eval)"
                    if en
                    else "视角 A — high 主证据（偏基模·末轮·综合评估）"
                ),
                "#c0392b",
            )
        if md_a:
            chart_base_m = _chart_signals(
                md_a,
                plt_mod,
                (
                    "Lens A — medium (same filter)"
                    if en
                    else "视角 A — medium（同视角筛选；含交叉类）"
                ),
                "#2980b9",
            )
        if hi_b:
            chart_agent_h = _chart_signals(
                hi_b,
                plt_mod,
                (
                    "Lens B — high (tools / usage / latency)"
                    if en
                    else "视角 B — high 主证据（偏工具·成功率·用量时延）"
                ),
                "#c0392b",
            )
        if md_b:
            chart_agent_m = _chart_signals(
                md_b,
                plt_mod,
                (
                    "Lens B — medium (same filter)"
                    if en
                    else "视角 B — medium（同视角筛选；含交叉类）"
                ),
                "#2980b9",
            )

    def _img_block(title_h3: str, uri: Optional[str], alt: str) -> str:
        if not uri:
            return ""
        return (
            f"<h4 class='chart-sub'>{html.escape(title_h3)}</h4>"
            f"<p class='note'><img src='{uri}' alt='{html.escape(alt)}'/></p>"
        )

    overview_parts: List[str] = []
    if chart_tier:
        overview_parts.append(
            "<h3>"
            + ("Tier distribution" if en else "各分档条数（批次总览·共用）")
            + "</h3>"
            f"<p class='note'><img src='{chart_tier}' alt='tier'/></p>"
        )
    if chart_model:
        overview_parts.append(
            "<h3>"
            + ("Stacked by request_model" if en else "按请求模型堆叠（批次总览·共用）")
            + "</h3>"
            f"<p class='note'><img src='{chart_model}' alt='model'/></p>"
        )
    overview_html = "\n".join(overview_parts) if overview_parts else (
        f"<p class='note'>{'No tier/model charts.' if en else '无分档/模型图。'}</p>"
        if not no_charts
        else ""
    )

    def _panel_lens_a() -> str:
        parts = [
            "<p class='note'><strong>"
            + (
                "Focus: LLM scores, discard, §5b turn axes; cross-tagged signals also in Tab B."
                if en
                else "优先看：LLM 分、discard、末轮多轴；「交叉」信号请与 Tab B 对照。"
            )
            + "</strong></p>"
        ]
        parts.append(
            _img_block(
                "high" if en else "high 主证据",
                chart_base_h,
                "lens-a-high",
            )
        )
        parts.append(
            _img_block(
                "medium" if en else "medium 弱证据",
                chart_base_m,
                "lens-a-med",
            )
        )
        if not chart_base_h and not chart_base_m:
            parts.append(
                "<p class='note'>"
                + (
                    "No high/medium signals in this lens for this batch."
                    if en
                    else "本批在视角 A 过滤下暂无 high/medium 信号柱图。"
                )
                + "</p>"
            )
        return "\n".join(parts)

    def _panel_lens_b() -> str:
        parts = [
            "<p class='note'><strong>"
            + (
                "Focus: tools, success ratio, tokens/latency; cross-check turn scores (agentic coupling)."
                if en
                else "优先看：工具、成功率、token/时延；末轮分与链路常耦合，两 Tab 连着读。"
            )
            + "</strong></p>"
        ]
        parts.append(
            _img_block("high" if en else "high 主证据", chart_agent_h, "lens-b-high")
        )
        parts.append(
            _img_block("medium" if en else "medium 弱证据", chart_agent_m, "lens-b-med")
        )
        if not chart_agent_h and not chart_agent_m:
            parts.append(
                "<p class='note'>"
                + (
                    "No high/medium signals in this lens for this batch."
                    if en
                    else "本批在视角 B 过滤下暂无 high/medium 信号柱图。"
                )
                + "</p>"
            )
        return "\n".join(parts)

    details_inner = ""
    if chart_full_high or chart_full_med:
        d_h = (
            f"<h4>{'All high signals (unfiltered)' if en else '全量 high（未按视角过滤）'}</h4>"
            f"<p class='note'><img src='{chart_full_high}' alt='all high'/></p>"
            if chart_full_high
            else ""
        )
        d_m = (
            f"<h4>{'All medium signals (unfiltered)' if en else '全量 medium（未按视角过滤）'}</h4>"
            f"<p class='note'><img src='{chart_full_med}' alt='all med'/></p>"
            if chart_full_med
            else ""
        )
        details_inner = (
            f"<details class='charts-global'><summary>"
            f"{html.escape('Global signal charts (unfiltered)' if en else '全局信号图（未按视角过滤，供对照）')}"
            f"</summary>{d_h}{d_m}</details>"
        )

    h2 = "Charts & signal lenses (tabs)" if en else "图表与双视角 Tab"
    merged_lead_en = (
        "Same batch: tier/model charts → "
        "<a href='#sec-dialog-metrics'>dialogue / tokens / latency</a> → "
        "<a href='#sec-macro-dist'>tools &amp; semantic tags</a> → "
        "<a href='#sec-signal-cluster'>clustering</a> → two tabs that <em>re-weight</em> reads. "
        "Stacks = sample counts; signal bars = in-batch occurrences. "
        "Per row: Insight cards and case-study section below."
    )
    merged_lead_zh = (
        "阅读顺序：分档/模型总览 → <a href='#sec-dialog-metrics'>对话与 token/时延</a> → "
        "<a href='#sec-macro-dist'>工具与语义标签</a> → "
        "<a href='#sec-signal-cluster'>信号聚类</a> → 两个 Tab 切换视角。"
        "堆叠与分档柱为<strong>样本数</strong>；信号柱为<strong>出现次数</strong>。"
        "单条核对请看 Insight 卡片与下文的 case study。"
    )
    if no_charts or plt_mod is None:
        merged_lead_en += " No raster charts (use table + tab text)."
        merged_lead_zh += " 未生成柱图时仅表与 Tab 文字。"

    merged_intro = "<p class='note'>" + (merged_lead_en if en else merged_lead_zh) + "</p>"

    tab_a = "Lens A — base / turn quality first" if en else "视角 A — 更关注基模与末轮质量"
    tab_b = "Lens B — agent / tools first" if en else "视角 B — 更关注 Agent 链路与工具"

    return f"""<section class='charts-section' id='sec-charts'>
<h2>{html.escape(h2)}</h2>
{merged_intro}
{overview_html}
{macro_distribution_html}
<h3 id='sec-signal-cluster'>{'Signal clustering' if en else '信号聚类与解读侧重'}</h3>
{cluster}
<div class='scenario-tabs' role='tablist' aria-label="{html.escape('Chart lenses' if en else '图表视角')}">
<button type='button' class='scenario-tab active' role='tab' aria-selected='true' aria-controls='panel-lens-a' id='tab-lens-a' data-panel='panel-lens-a'>{html.escape(tab_a)}</button>
<button type='button' class='scenario-tab' role='tab' aria-selected='false' aria-controls='panel-lens-b' id='tab-lens-b' data-panel='panel-lens-b'>{html.escape(tab_b)}</button>
</div>
<div id='panel-lens-a' class='scenario-panel active' role='tabpanel' aria-labelledby='tab-lens-a'>
{_panel_lens_a()}
</div>
<div id='panel-lens-b' class='scenario-panel' role='tabpanel' aria-labelledby='tab-lens-b' hidden>
{_panel_lens_b()}
</div>
{details_inner}
</section>"""


def _chart_tier_bar(tier_cnt: Counter, plt_mod) -> Optional[str]:
    if plt_mod is None:
        return None
    order = ("high_precision", "watchlist", "none")
    labels = [t for t in order if tier_cnt.get(t, 0) > 0]
    if not labels:
        labels = list(tier_cnt.keys())
    vals = [tier_cnt[t] for t in labels]
    labels_zh = [f"{_tier_zh(t)}\n({t})" for t in labels]
    fig, ax = plt_mod.subplots(figsize=(7, 3.5))
    colors = {
        "high_precision": "#c0392b",
        "watchlist": "#f39c12",
        "none": "#95a5a6",
    }
    bars = ax.bar(
        labels_zh,
        vals,
        color=[colors.get(x, "#3498db") for x in labels],
    )
    try:
        ax.bar_label(bars, labels=[str(int(v)) for v in vals], padding=3, fontsize=10)
    except AttributeError:  # matplotlib < 3.4
        pass
    ax.set_title("分档样本数（柱顶数字为条数；中文为展示名）")
    ax.set_ylabel("条数")
    plt_mod.setp(ax.xaxis.get_majorticklabels(), rotation=12, ha="center")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _chart_signals(
    sig_cnt: Counter,
    plt_mod,
    title: str,
    color: str = "#2980b9",
    top_n: int = 14,
) -> Optional[str]:
    if plt_mod is None or not sig_cnt:
        return None
    items = sig_cnt.most_common(top_n)
    labels = [x[0] for x in items]
    vals = [x[1] for x in items]
    fig, ax = plt_mod.subplots(figsize=(7, max(2.5, 0.35 * len(labels))))
    labels_r = labels[::-1]
    vals_r = vals[::-1]
    bars = ax.barh(labels_r, vals_r, color=color)
    try:
        ax.bar_label(bars, labels=[str(int(v)) for v in vals_r], padding=3, fontsize=9)
    except AttributeError:
        pass
    ax.set_title(title)
    ax.set_xlabel("条数")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _chart_by_model(model_tier: Dict[str, Counter], plt_mod) -> Optional[str]:
    if plt_mod is None or len(model_tier) == 0:
        return None
    ordered = _model_tier_sorted_pairs(model_tier)
    top_n = MODEL_STACK_CHART_TOP_N
    other_label = "其他（其余模型）"
    tiers = ("high_precision", "watchlist", "none")
    if len(ordered) <= top_n:
        models = [m for m, _ in ordered]
        per_model: Dict[str, Counter] = {m: model_tier[m] for m in models}
        title = "按 request_model 堆叠分档（每段数字为该档占比）"
    else:
        models = [m for m, _ in ordered[:top_n]]
        per_model = {m: model_tier[m] for m in models}
        other_c = Counter()
        for m, c in ordered[top_n:]:
            other_c.update(c)
        models.append(other_label)
        per_model[other_label] = other_c
        title = (
            f"按 request_model 堆叠分档（Top {top_n} 请求量 + 其余合并；每段数字为该档占比）"
        )
    fig, ax = plt_mod.subplots(figsize=(max(6, len(models) * 0.9), 3.8))
    n = len(models)
    x = list(range(n))
    w = 0.65
    bottom = [0.0] * n
    colors = {"high_precision": "#c0392b", "watchlist": "#f39c12", "none": "#bdc3c7"}
    _tier_legend_zh = {
        "high_precision": "高置信异常样本（Bad Case）",
        "watchlist": "待观察样本（不完美交互）",
        "none": "正常样本（高质量交互）",
    }
    # Pre-compute per-model totals for ratio calculation
    model_totals = [sum(per_model[m].values()) for m in models]
    # Collect all (i, yc, label, v, col_total) for deferred label rendering
    # to allow visibility filtering based on bar height proportion
    label_tasks: List[tuple] = []  # (xi, yc, text, v, col_total)
    for tier in tiers:
        vs = [float(per_model[m].get(tier, 0)) for m in models]
        if not any(vs):
            continue
        leg = _tier_legend_zh.get(tier, tier)
        ax.bar(x, vs, bottom=bottom, label=leg, color=colors[tier], width=w)
        for i, v in enumerate(vs):
            if v > 0:
                yc = bottom[i] + v / 2.0
                label_tasks.append((x[i], yc, v, model_totals[i]))
        bottom = [b + v for b, v in zip(bottom, vs)]
    # Determine figure height in data units for min-height threshold
    col_max = max(bottom) if bottom else 1.0
    # Minimum segment height (as fraction of tallest column) to show label inline;
    # segments shorter than this get their label placed just above the segment.
    MIN_RATIO_FOR_INLINE = 0.08
    for xi, yc, v, col_total in label_tasks:
        ratio = v / col_total if col_total > 0 else 0.0
        label_text = f"{ratio:.0%}"
        seg_height_frac = v / col_max if col_max > 0 else 0.0
        if seg_height_frac >= MIN_RATIO_FOR_INLINE:
            # Segment tall enough: render label centred inside the bar segment
            ax.text(
                xi,
                yc,
                label_text,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        else:
            # Segment too short: render small label just above the segment top,
            # slightly offset to avoid overlap with neighbouring text
            seg_top = yc + v / 2.0
            ax.text(
                xi,
                seg_top + col_max * 0.012,
                label_text,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#555",
            )
    # Annotate the total sample count above each stacked bar
    for i, total in enumerate(model_totals):
        if total > 0:
            ax.text(
                x[i],
                bottom[i] + col_max * 0.015,
                str(int(total)),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#333",
            )
    # Extend y-axis upper limit so the top-bar annotations are not clipped
    ax.set_ylim(0, col_max * 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title(title)
    ax.set_ylabel("样本数目")
    ax.legend()
    plt_mod.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _normalize_list_label(s: str) -> str:
    """Collapse whitespace so the same phrase from LLM maps to one bucket."""
    return " ".join(str(s).strip().split())


def _labels_from_meta_list(meta: dict, key: str) -> List[str]:
    raw = meta.get(key)
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for x in raw:
        s = _normalize_list_label(str(x))
        if s:
            out.append(s)
    return list(dict.fromkeys(out))


# Align with agent_skill_insight_mapper: split glued phrases without re-running the pipeline.
_SKILL_INSIGHT_SPLIT_RE = re.compile(r"[,，、;；]+")


def _skill_insight_labels_from_meta(meta: dict) -> List[str]:
    """Expand ``agent_skill_insights`` list entries on CN/EN punctuation (same as mapper)."""
    raw = meta.get("agent_skill_insights")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for x in raw:
        for piece in _SKILL_INSIGHT_SPLIT_RE.split(str(x)):
            s = _normalize_list_label(piece)
            if s:
                out.append(s)
    return list(dict.fromkeys(out))


def _counter_add_row_unique_labels(c: Counter, labels: List[str]) -> None:
    for lab in dict.fromkeys(labels):
        c[lab] += 1


def _compute_macro_tag_counters(rows: List[dict]) -> Dict[str, Counter]:
    """Aggregate tag counts per sample row.

    primary_tool_type: at most +1 per row. List-like meta: +1 per distinct
    label within the row.
    """
    tool_primary: Counter = Counter()
    tool_all: Counter = Counter()
    skill_insight: Counter = Counter()
    skill_type: Counter = Counter()
    intent: Counter = Counter()
    topic: Counter = Counter()
    sentiment: Counter = Counter()
    for row in rows:
        meta = get_dj_meta(row)
        pt = meta.get("primary_tool_type")
        if pt is not None:
            s = str(pt).strip()
            if s:
                tool_primary[s] += 1
        _counter_add_row_unique_labels(
            tool_all, _labels_from_meta_list(meta, "agent_tool_types")
        )
        _counter_add_row_unique_labels(
            skill_insight,
            _skill_insight_labels_from_meta(meta),
        )
        _counter_add_row_unique_labels(
            skill_type,
            _labels_from_meta_list(meta, "agent_skill_types"),
        )
        _counter_add_row_unique_labels(
            intent,
            _labels_from_meta_list(meta, "dialog_intent_labels"),
        )
        _counter_add_row_unique_labels(
            topic,
            _labels_from_meta_list(meta, "dialog_topic_labels"),
        )
        _counter_add_row_unique_labels(
            sentiment,
            _labels_from_meta_list(meta, "dialog_sentiment_labels"),
        )
    return {
        "tool_primary": tool_primary,
        "tool_all": tool_all,
        "skill_insight": skill_insight,
        "skill_type": skill_type,
        "intent": intent,
        "topic": topic,
        "sentiment": sentiment,
    }


def _chart_macro_bar(
    cnt: Counter,
    plt_mod,
    title: str,
    color: str,
    xlabel: str,
    top_n: int = 18,
) -> Optional[str]:
    if plt_mod is None or not cnt:
        return None
    items = cnt.most_common(top_n)
    labels = [x[0] for x in items]
    vals = [x[1] for x in items]
    fig, ax = plt_mod.subplots(figsize=(7.2, max(2.8, 0.32 * len(labels))))
    labels_r = labels[::-1]
    vals_r = vals[::-1]
    bars = ax.barh(labels_r, vals_r, color=color)
    try:
        ax.bar_label(
            bars,
            labels=[str(int(v)) for v in vals_r],
            padding=2,
            fontsize=8,
        )
    except AttributeError:
        pass
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _wordcloud_font_path() -> Optional[str]:
    for p in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if os.path.isfile(p):
            return p
    return None


def _wordcloud_from_counter(
    cnt: Counter,
    plt_mod,
    suptitle: str,
) -> Optional[str]:
    try:
        from wordcloud import WordCloud
    except ImportError:  # pragma: no cover
        return None
    if plt_mod is None or not cnt:
        return None
    freq = {str(k): int(v) for k, v in cnt.most_common(250) if str(k).strip()}
    if not freq:
        return None
    fp = _wordcloud_font_path()
    try:
        wc = WordCloud(
            width=920,
            height=380,
            background_color="white",
            font_path=fp,
            max_words=90,
            colormap="viridis",
            relative_scaling=0.35,
            prefer_horizontal=0.85,
        ).generate_from_frequencies(freq)
    except (ValueError, OSError):
        return None
    fig, ax = plt_mod.subplots(figsize=(9.2, 3.9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return _fig_to_data_uri(fig, plt_mod)


def _macro_inner_table_html(cnt: Counter, *, en: bool, top: int = 15) -> str:
    if not cnt:
        return ""
    th = "Label" if en else "标签"
    thc = "Rows" if en else "样本数"
    rows_html = []
    for lab, n in cnt.most_common(top):
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(lab))}</td>"
            f"<td>{int(n)}</td>"
            "</tr>"
        )
    thead = f"<thead><tr><th>{th}</th><th>{thc}</th></tr></thead>"
    return (
        f"<table class='macro-mini'>{thead}"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _build_macro_distribution_section(
    rows: List[dict],
    plt_mod,
    no_charts: bool,
    html_lang: str,
    *,
    precomputed_counters: Optional[Dict[str, Counter]] = None,
    precomputed_skill_stats: Optional[Tuple[int, int, int]] = None,
) -> str:
    en = str(html_lang).lower().startswith("en")
    if precomputed_counters is not None and precomputed_skill_stats is not None:
        counters = precomputed_counters
        si_n_tot, si_n_with, si_slots = precomputed_skill_stats
    else:
        counters = _compute_macro_tag_counters(rows)
        si_n_tot, si_n_with, si_slots = _skill_insight_coverage_stats(rows)
    if not any(counters[k] for k in counters):
        empty = (
            "No tool/skill/intent/topic/sentiment tags in meta for this batch."
        )
        if not en:
            empty = (
                "本批 meta 中未找到工具/技能/意图/主题/情绪等标签（可能未跑对应 mapper）。"
            )
        h3t = "Tag overview" if en else "宏观：工具与语义标签分布"
        return (
            f"<h3 id='sec-macro-dist'>{html.escape(h3t)}</h3>"
            f"<p class='note'>{html.escape(empty)}</p>"
        )

    intro_en = (
        "Counts are <strong>per sample</strong>: a row adds +1 to each "
        "<em>distinct</em> label in that row (lists). Primary tool is one "
        "value per row. Word clouds use the same frequencies."
    )
    intro_zh = (
        "计数按<strong>样本</strong>：列表字段中同一行每种标签只计 1 次；"
        "<code>primary_tool_type</code> 每行最多 1 个主类。"
        "词云与柱状图同源频次。"
    )
    intro = intro_en if en else intro_zh
    specs = [
        (
            "tool_primary",
            "Primary tool (meta.primary_tool_type)",
            "主工具类型（meta.primary_tool_type）",
            "#2b6cb0",
        ),
        (
            "tool_all",
            "Tool names (meta.agent_tool_types, row-unique)",
            "工具出现（meta.agent_tool_types，每行去重）",
            "#3182ce",
        ),
        (
            "skill_insight",
            "Skill insights (meta.agent_skill_insights)",
            "能力归纳（meta.agent_skill_insights）",
            "#38a169",
        ),
        (
            "skill_type",
            "Skill types (meta.agent_skill_types)",
            "技能类型（meta.agent_skill_types）",
            "#48bb78",
        ),
        (
            "intent",
            "Intent labels (meta.dialog_intent_labels)",
            "意图（meta.dialog_intent_labels）",
            "#805ad5",
        ),
        (
            "topic",
            "Topic labels (meta.dialog_topic_labels)",
            "主题（meta.dialog_topic_labels）",
            "#6b46c1",
        ),
        (
            "sentiment",
            "Sentiment labels (meta.dialog_sentiment_labels)",
            "情绪（meta.dialog_sentiment_labels）",
            "#c05621",
        ),
    ]
    xlabel = "Rows (with label)" if en else "含该标签的样本数"
    h3m = "Tag & tool overview" if en else "宏观：工具与语义标签分布"
    parts: List[str] = [
        f"<h3 id='sec-macro-dist'>{html.escape(h3m)}</h3>"
        f"<p class='note'>{intro}</p>",
    ]
    for key, title_en, title_zh, color in specs:
        cnt = counters.get(key) or Counter()
        if not cnt:
            continue
        title = title_en if en else title_zh
        parts.append("<div class='macro-dim'>")
        parts.append(f"<h4>{html.escape(title)}</h4>")
        if key == "skill_insight":
            if en:
                cov = (
                    f"This report loaded <strong>{si_n_tot}</strong> rows; "
                    f"<strong>{si_n_with}</strong> have non-empty "
                    f"<code>meta.agent_skill_insights</code>; "
                    f"<strong>{si_slots}</strong> label slots total "
                    "(deduped per row; list items are also split on "
                    "<code>,，、;；</code> like the mapper, so old JSONL "
                    "need not be reprocessed). If this is far below row count, "
                    "many rows likely skipped the mapper (no tools/skills), "
                    "API failure, or you used <code>--limit</code>."
                )
            else:
                cov = (
                    f"本报告载入 <strong>{si_n_tot}</strong> 行；其中 "
                    f"<strong>{si_n_with}</strong> 行 "
                    f"<code>meta.agent_skill_insights</code> 非空；"
                    f"标签条数合计 <strong>{si_slots}</strong>（每行内短语已去重；"
                    "若明显少于总行数，请核对是否使用 <code>--limit</code>、"
                    "checkpoint 是否跳过该算子、或大量行无 tool/skill / API 失败。"
                )
            parts.append(f"<p class='note macro-si-cov'>{cov}</p>")
        if no_charts or plt_mod is None:
            parts.append(_macro_inner_table_html(cnt, en=en))
        else:
            bar_uri = _chart_macro_bar(cnt, plt_mod, title, color, xlabel)
            if bar_uri:
                alt_b = html.escape(f"{key}-bar")
                parts.append(
                    f"<p class='note macro-bar'>"
                    f"<img src='{bar_uri}' alt='{alt_b}'/></p>"
                )
            wc_uri = _wordcloud_from_counter(cnt, plt_mod, title)
            if wc_uri:
                sum_wc = "Word cloud" if en else "词云（与柱状图同源计数）"
                alt_w = html.escape(f"{key}-wc")
                parts.append(
                    f"<details class='macro-wc'>"
                    f"<summary>{html.escape(sum_wc)}</summary>"
                    f"<p class='note'><img src='{wc_uri}' alt='{alt_w}'/></p>"
                    "</details>"
                )
            if not bar_uri and not wc_uri:
                parts.append(_macro_inner_table_html(cnt, en=en))
        parts.append("</div>")
    return "\n".join(parts)


def _pii_audit_report_path(safe_output: Path) -> Path:
    """Sibling path for the full PII-audit HTML next to the safe ``--output``."""
    return safe_output.with_name(f"{safe_output.stem}_pii_audit{safe_output.suffix}")


def _report_variant_banner_html(
    *,
    variant: str,
    html_lang: str,
    wrote_audit_sibling: bool,
) -> str:
    en = str(html_lang).lower().startswith("en")
    if variant == "safe":
        msg = (
            "This is the <strong>PII-safe</strong> report: Insight cards and in-page Case study "
            "exclude rows flagged by PII audit / redaction heuristics."
            if en
            else "本页为 <strong>PII 安全版</strong>：单条 Insight 与 Case study 页内展示已排除 "
            "含 PII 审计或脱敏线索的样本。"
        )
        if wrote_audit_sibling:
            msg += (
                " Open <code>*_pii_audit.html</code> in the same folder for the audit view."
                if en
                else " 内部分析请打开同目录生成的 <code>*_pii_audit.html</code>（PII 审计版）。"
            )
    else:
        msg = (
            "This is the <strong>PII audit</strong> report: Insight and Case study include "
            "PII-flagged rows for internal review only — do not share externally."
            if en
            else "本页为 <strong>PII 审计版</strong>：Insight 与 Case study 保留含 PII 线索的样本，"
            "仅供内部分析，请勿对外转发。"
        )
    return f"<p class='note report-variant-banner'>{msg}</p>"


def _data_source_summary_for_meta(paths: List[str], *, html_lang: str) -> str:
    """One-line meta for page header; full paths for multi-input go to bottom <details>."""
    resolved = [str(Path(p).resolve()) for p in paths]
    en = html_lang == "en"
    if len(resolved) == 1:
        esc = html.escape(resolved[0])
        return f"Data file <code>{esc}</code>" if en else f"数据文件 <code>{esc}</code>"
    n = len(resolved)
    if en:
        return (
            f"Data sources: <strong>{n}</strong> jsonl files (concatenated in order); "
            f"<a href=\"#sec-data-provenance\">full paths</a> in the collapsible section "
            "at the bottom (audit trail)."
        )
    return (
        f"数据源 <strong>{n}</strong> 个 jsonl（按顺序拼接）；"
        f"<a href=\"#sec-data-provenance\">完整路径</a>见页底折叠区（审计溯源）。"
    )


def _data_source_provenance_block(paths: List[str], *, html_lang: str) -> str:
    """Bottom collapsible ordered list of input paths; empty when only one file."""
    if len(paths) <= 1:
        return ""
    resolved = [str(Path(p).resolve()) for p in paths]
    en = html_lang == "en"
    items = "".join(f"<li><code>{html.escape(p)}</code></li>" for p in resolved)
    summary = "Input file paths (audit trail)" if en else "数据来源路径（审计溯源）"
    note = (
        "Order matches <code>--input</code> when this report was generated."
        if en
        else "顺序与生成报告时的 <code>--input</code> 指定顺序一致。"
    )
    return (
        "<details class='report-fold audit-provenance' id='sec-data-provenance'>"
        f"<summary>{html.escape(summary)}</summary>"
        "<div class='report-fold-body'>"
        f"<p class='note'>{html.escape(note)}</p>"
        "<ol class='provenance-list'>"
        f"{items}</ol></div></details>"
    )


def _html_page(
    title: str,
    input_paths: List[str],
    n_rows: int,
    tier_cnt: Counter,
    cohort_rows: List[dict],
    attribution_table: str,
    exec_summary_html: str,
    charts_section_html: str,
    dialog_metrics_html: str,
    drilldown_html: str,
    insight_legend_html: str,
    insight_section_html: str,
    reading_scenarios_html: str,
    bilingual_header: str = "",
    *,
    html_lang: str = "zh-CN",
    variant_banner: str = "",
) -> str:
    gen_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tier_rows = "".join(
        "<tr>"
        f"<td>{html.escape(_tier_zh(k))}</td>"
        f"<td><code>{html.escape(k)}</code></td>"
        f"<td>{v}</td>"
        "</tr>"
        for k, v in sorted(tier_cnt.items(), key=lambda x: -x[1])
    )
    cohort_lines = []
    for r in cohort_rows:
        if not r.get("count") and not r.get("top_signal_codes"):
            continue
        tm = str(r.get("tier", "") or "")
        cohort_lines.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('agent_request_model', '')))}</td>"
            f"<td>{html.escape(str(r.get('agent_pt', '')))}</td>"
            f"<td>{html.escape(_tier_zh(tm))} <code>{html.escape(tm)}</code></td>"
            f"<td>{int(r.get('count') or 0)}</td>"
            f"<td>{html.escape(str(r.get('top_signal_codes', '')))}</td>"
            "</tr>"
        )

    toc_bar = _report_toc_html("bar", html_lang=html_lang)
    toc_side = _report_toc_html("side", html_lang=html_lang)
    toc_mini = _report_toc_html("mini", html_lang=html_lang)

    css = (
        "body{font-family:'PingFang SC','Hiragino Sans GB','Microsoft YaHei',"
        "'Noto Sans SC',system-ui,sans-serif;margin:24px;}"
        ".page-shell{display:flex;align-items:flex-start;gap:20px;"
        "max-width:1320px;margin:0 auto;}"
        ".page-main{flex:1;min-width:0;max-width:1100px;}"
        ".report-toc--side{flex:0 0 200px;position:sticky;top:12px;"
        "align-self:flex-start;font-size:0.86rem;border:1px solid #e8e8e8;"
        "border-radius:10px;padding:12px 14px;background:#fafbfd;"
        "max-height:calc(100vh - 24px);overflow:auto;line-height:1.45;}"
        ".report-toc--side strong{display:block;margin-bottom:6px;font-size:0.9rem;}"
        ".report-toc--side ul{list-style:none;padding:0;margin:0;}"
        ".report-toc--side li{margin:2px 0;}"
        ".report-toc--side a{color:#1565c0;text-decoration:none;}"
        ".report-toc--side a:hover{text-decoration:underline;}"
        ".report-toc--mini{margin:1rem 0 1.25rem;padding-top:8px;"
        "border-top:1px dashed #ddd;}"
        ".report-toc--mini .note{margin:0;font-size:0.84rem;}"
        "a.to-drill{font-size:0.84rem;white-space:nowrap;}"
        "a.to-drill-fallback{color:#666;}"
        "@media(max-width:960px){"
        ".page-shell{flex-direction:column;}"
        ".report-toc--side{position:relative;top:0;max-height:none;width:100%;"
        "flex:1 1 auto;}}"
        "h1{font-size:1.35rem;}.meta{color:#555;font-size:0.9rem;}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}"
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left;}"
        "th{background:#f4f4f4;}img{max-width:100%;height:auto;}"
        "ul{line-height:1.5;}.model{color:#555;font-size:0.85rem;}"
        "details{margin-top:2rem;padding:12px;background:#fafafa;border:1px solid #eee;}"
        "details.report-fold{margin-top:0.75rem;padding:8px 12px;background:#fafbfc;"
        "border:1px solid #e5e7eb;border-radius:8px;}"
        "summary{cursor:pointer;font-weight:600;}"
        ".note{color:#666;font-size:0.88rem;}"
        ".drill-list{display:flex;flex-direction:column;gap:10px;margin:1rem 0;}"
        ".drill-card{border:1px solid #ddd;border-radius:8px;background:#fff;"
        "box-shadow:0 1px 2px rgba(0,0,0,0.04);}"
        ".drill-summary{display:flex;flex-wrap:wrap;align-items:center;gap:8px;"
        "padding:10px 12px;background:#fafafa;border-radius:8px 8px 0 0;"
        "border-bottom:1px solid #eee;}"
        ".drill-body{padding:12px;border-radius:0 0 8px 8px;}"
        ".tier-tag{font-size:0.75rem;padding:2px 8px;border-radius:999px;"
        "font-weight:600;}"
        ".tier-hp{background:#fadbd8;color:#922b21;}"
        ".tier-wl{background:#fdebd0;color:#9c640c;}"
        ".rid{font-size:0.85rem;background:#f4f6f8;padding:2px 6px;border-radius:4px;}"
        ".idx{font-size:0.82rem;color:#555;}"
        ".cohort-mini{font-size:0.82rem;color:#666;}"
        ".anchor-link{font-weight:600;margin-right:4px;}"
        "button.drill-toggle{margin-left:auto;font-size:0.85rem;cursor:pointer;"
        "padding:6px 12px;border:1px solid #bbb;border-radius:6px;background:#fff;}"
        "button.drill-toggle:hover{background:#f0f0f0;}"
        ".field-grid{display:grid;gap:14px;}"
        "@media(min-width:900px){.field-grid{grid-template-columns:1fr 1fr;}}"
        ".field-grid section{margin:0;}"
        ".field-grid h4{margin:0 0 6px;font-size:0.9rem;color:#333;}"
        "pre.drill-pre{margin:0;white-space:pre-wrap;word-break:break-word;"
        "max-height:320px;overflow:auto;font-size:0.82rem;line-height:1.45;"
        "background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:6px;}"
        "pre.drill-pre--msg,pre.drill-pre--response{max-height:420px;}"
        "@media(min-width:900px){.field-grid section.drill-field-wide{grid-column:1/-1;}}"
        ".conv-scroll{max-height:420px;overflow:auto;border-radius:6px;border:1px solid #333;"
        "background:#252526;}"
        ".conv-thread{display:flex;flex-direction:column;gap:10px;padding:10px 12px;}"
        ".conv-turn{border-radius:6px;border-left:4px solid #666;padding:8px 12px;"
        "background:#1e1e1e;}"
        ".conv-user{border-left-color:#3794ff;}"
        ".conv-assistant{border-left-color:#89d185;}"
        ".conv-tool{border-left-color:#cca700;}"
        ".conv-system{border-left-color:#9d9d9d;}"
        ".conv-other{border-left-color:#888;}"
        ".conv-role{font-size:0.72rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.04em;color:#858585;margin-bottom:5px;}"
        ".conv-body{font-size:0.84rem;line-height:1.5;color:#d4d4d4;white-space:pre-wrap;"
        "word-break:break-word;}"
        ".conv-empty{color:#888;font-style:italic;font-size:0.82rem;}"
        ".tier-mach{font-size:0.78rem;color:#888;}"
        ".tier-legend li{margin:6px 0;}"
        "table.inner{font-size:0.82rem;margin:0.5rem 0;width:100%;}"
        "table.inner td:first-child{white-space:nowrap;width:38%;vertical-align:top;}"
        ".insight-legend table.inner{table-layout:fixed;}"
        ".insight-legend table.inner td:first-child{width:6.5em;max-width:8em;"
        "white-space:nowrap;vertical-align:top;}"
        ".insight-legend table.inner td:nth-child(2),.insight-legend table.inner "
        "td:nth-child(3){width:auto;white-space:normal;word-break:break-word;}"
        ".sig-evidence-wrap{display:flex;flex-direction:column;gap:10px;margin:0.8rem 0;}"
        ".sig-evidence{border:1px solid #e0e0e0;border-radius:6px;padding:8px 10px;background:#fcfcfc;}"
        ".sig-evidence-h{margin-bottom:6px;font-size:0.88rem;}"
        ".sig-evidence .wtag{color:#555;font-size:0.8rem;margin-left:6px;}"
        ".sig-evidence .det{color:#666;font-size:0.8rem;margin-left:6px;}"
        ".snap table.inner{margin-top:4px;}"
        ".snap-table thead th,.sig-evidence-table thead th{font-size:0.8rem;padding:5px 8px;}"
        ".snap-table td:last-child{word-break:break-word;line-height:1.45;}"
        ".snap-table tr.snap-ops td{font-size:0.82rem;color:#444;}"
        ".snap-table tr.snap-ops code{font-weight:600;}"
        "h4.ev-h{margin:12px 0 6px;font-size:0.95rem;}"
        ".exec-summary-pre{margin:0.5rem 0 0;white-space:pre-wrap;word-break:"
        "break-word;line-height:1.55;font-size:0.92rem;background:#f7f9fc;border:1px solid #e2e8f0;"
        "border-radius:8px;padding:14px 16px;}"
        ".insight-list{display:flex;flex-direction:column;gap:12px;margin:1rem 0;}"
        ".insight-card{border:1px solid #e0e0e0;border-radius:8px;padding:12px 14px;"
        "background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04);}"
        ".insight-h{font-weight:600;font-size:1rem;margin-bottom:6px;line-height:1.4;}"
        ".ins-meta{display:block;font-size:0.85rem;color:#555;margin-bottom:8px;}"
        ".ins-badge{font-size:0.75rem;padding:2px 8px;border-radius:6px;margin-left:6px;}"
        ".ins-badge.pr{background:#fce4ec;color:#880e4f;}"
        ".ins-badge.al{background:#e3f2fd;color:#0d47a1;}"
        "ul.causes{margin:6px 0 0 1rem;padding:0;font-size:0.88rem;line-height:1.45;}"
        "ul.causes li{margin:4px 0;}"
        ".cite{color:#666;font-size:0.82rem;}"
        ".ins-facets,.ins-audit{font-size:0.86rem;margin:8px 0 0;color:#444;}"
        "section.insight-sec{margin-top:2rem;}"
        "section.reading-scenarios{margin-top:1.25rem;}"
        ".audience-grid{display:grid;gap:14px;margin:0.75rem 0;}"
        "@media(min-width:820px){.audience-grid{grid-template-columns:1fr 1fr;}}"
        ".audience-card{border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;"
        "background:#f8fafc;}"
        ".audience-card h3{margin:0 0 8px;font-size:0.98rem;color:#1a365d;}"
        ".audience-card ul{margin:0;padding-left:1.1rem;font-size:0.88rem;line-height:1.55;}"
        ".report-footnote{border-left:3px solid #64b5f6;padding:8px 12px;margin:10px 0 0;"
        "background:#f5f9ff;border-radius:0 6px 6px 0;}"
        ".charts-section{margin-top:1rem;}"
        ".charts-section>.note a{color:#1565c0;}"
        ".signal-cluster{font-size:0.86rem;}"
        ".signal-cluster code{font-size:0.82rem;word-break:break-word;}"
        ".scenario-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px;"
        "border-bottom:1px solid #e0e0e0;}"
        ".scenario-tab{font-size:0.88rem;padding:8px 14px;border:1px solid transparent;"
        "border-bottom:none;border-radius:8px 8px 0 0;background:#f5f5f5;cursor:pointer;color:#333;}"
        ".scenario-tab:hover{background:#eee;}"
        ".scenario-tab.active{background:#fff;border-color:#e0e0e0;border-bottom:1px solid #fff;"
        "margin-bottom:-1px;font-weight:600;color:#1565c0;}"
        ".scenario-panel{display:none;margin-top:8px;padding-top:4px;}"
        ".scenario-panel.active{display:block;}"
        ".chart-sub{margin:10px 0 4px;font-size:0.95rem;}"
        ".charts-global{margin-top:1rem;}"
        ".macro-dim{margin:14px 0;padding:10px 12px;border:1px solid #e2e8f0;border-radius:8px;"
        "background:#fafcfe;}"
        ".macro-dim h4{margin:0 0 8px;font-size:0.95rem;color:#2d3748;}"
        ".macro-bar img{max-width:100%;}"
        ".macro-mini{font-size:0.84rem;width:100%;max-width:520px;}"
        ".macro-mini th,.macro-mini td{border:1px solid #e2e8f0;padding:4px 8px;text-align:left;}"
        ".macro-mini th{background:#edf2f7;}"
        ".macro-wc{margin-top:6px;border:1px solid #eee;border-radius:6px;padding:6px 10px;"
        "background:#fff;}"
        ".macro-si-cov{font-size:0.84rem;margin:4px 0 8px;}"
        ".insight-clusters{margin:12px 0;}"
        ".insight-clusters>details.insight-cluster{margin:10px 0;padding:10px 12px;"
        "border:1px solid #e2e8f0;border-radius:8px;background:#f8fafc;}"
        ".insight-clusters>details.insight-cluster>summary{cursor:pointer;font-weight:600;"
        "font-size:0.95rem;color:#2d3748;list-style:none;}"
        ".insight-clusters>details.insight-cluster>summary::-webkit-details-marker{display:none;}"
        ".insight-cluster{margin:12px 0;padding:10px 12px;border:1px solid #e2e8f0;"
        "border-radius:8px;background:#f8fafc;}"
        ".insight-cluster h3{margin:0 0 8px;font-size:1rem;color:#2d3748;}"
        ".insight-cluster h4{margin:10px 0 6px;font-size:0.92rem;color:#4a5568;}"
        ".insight-tab-host{margin-top:10px;}"
        ".macro-wc summary{cursor:pointer;font-weight:600;font-size:0.88rem;}"
        ".report-fold>summary{cursor:pointer;font-weight:600;font-size:0.95rem;list-style:none;}"
        ".report-fold>summary::-webkit-details-marker{display:none;}"
        ".report-fold-body{margin-top:8px;}"
        "ol.provenance-list{margin:8px 0;padding-left:1.25rem;font-size:0.88rem;"
        "line-height:1.55;word-break:break-word;}"
        "ol.provenance-list code{font-size:0.86rem;}"
        ".audit-provenance{margin-top:1.5rem;}"
        ".tier-fold ul.tier-legend{margin:0.4rem 0;padding-left:1.15rem;}"
        ".insight-fold{margin:1rem 0;}"
        ".ins-lenses{margin-top:10px;padding-top:8px;border-top:1px dashed #e0e0e0;}"
        ".ins-lenses-head{font-size:0.8rem;color:#555;margin-bottom:6px;}"
        ".ins-lenses-grid{display:grid;gap:10px;}"
        "@media(min-width:720px){.ins-lenses-grid{grid-template-columns:1fr 1fr;}}"
        ".ins-lens{font-size:0.84rem;border:1px solid #e8eef5;border-radius:8px;"
        "padding:8px 10px;background:#fafcff;line-height:1.45;}"
        ".ins-lens strong{display:block;font-size:0.88rem;margin-bottom:4px;color:#1a237e;}"
        ".ins-lens .lens-hint{margin:0 0 6px;color:#555;font-size:0.82rem;}"
        ".ins-lens .lens-sig{margin:0;padding-left:1rem;font-size:0.82rem;}"
        ".ins-lens .lens-miss{margin:0;font-size:0.82rem;color:#777;}"
        ".ins-lens .lens-cross{color:#6a1b9a;font-size:0.78rem;}"
        ".lens-other{font-size:0.82rem;margin:8px 0 0;}"
        ".insight-extra{margin-top:8px;border:1px solid #eee;border-radius:8px;padding:6px 10px;"
        "background:#fbfbfb;font-size:0.86rem;}"
        ".insight-extra summary{cursor:pointer;font-weight:600;}"
        ".insight-extra-body{margin-top:6px;}"
    )
    thead = (
        "<thead><tr><th>请求模型</th><th>日期桶 (pt)</th>"
        "<th>分档（中文 / 枚举）</th><th>条数</th><th>常见信号代码</th></tr></thead>"
    )
    tier_legend = (
        "<details class='report-fold tier-fold' id='sec-tiers'>"
        "<summary>分档含义：high_precision / watchlist / none（点击展开）</summary>"
        "<div class='report-fold-body'>"
        "<ul class='tier-legend'>"
        "<li><strong>强怀疑</strong> — <code>high_precision</code>："
        "多为主证据；若开启 <code>high_precision_on_tool_fail_alone</code> 也可能仅凭 tool 失败进入。</li>"
        "<li><strong>待观察</strong> — <code>watchlist</code>："
        "弱/启发式提示，不宜直接等同劣质样本。</li>"
        "<li><strong>未标记</strong> — <code>none</code>。</li>"
        "</ul>"
        "<p class='note'><code>high_precision</code> 在此指「优先复核」，与模型精度无关。</p>"
        "</div></details>"
    )
    adv = (
        "<p>进阶说明见 <code>demos/agent/BAD_CASE_INSIGHTS.md</code>、"
        "<code>demos/agent/scripts/README.md</code>。</p>"
    )

    input_sources_summary_html = _data_source_summary_for_meta(
        input_paths, html_lang=html_lang
    )
    input_sources_provenance_html = _data_source_provenance_block(
        input_paths, html_lang=html_lang
    )

    return f"""<!DOCTYPE html>
<html lang="{html.escape(html_lang)}"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>{css}</style>
</head><body>
<h1>{html.escape(title)}</h1>
{variant_banner}
<div class="meta">
  生成时间 {html.escape(gen_at)} ·
  {input_sources_summary_html}<br/>
  本报告载入 <strong>{n_rows}</strong> 条样本（若使用 --limit 则仅为其子集）
</div>
{bilingual_header}
{toc_bar}
<div class="page-shell">
{toc_side}
<main class="page-main">
{exec_summary_html}
{reading_scenarios_html}
<h2 id='sec-counts'>各分档条数</h2>
<table><thead><tr><th>展示名</th><th>机器枚举</th><th>条数</th></tr></thead>
<tbody>{tier_rows}</tbody></table>
{tier_legend}
{charts_section_html}
{dialog_metrics_html}
{insight_legend_html}
{insight_section_html}
{drilldown_html}
<h2 id='sec-attrib'>信号归因对照表</h2>
<p class='note'>信号 <code>code</code> 与上游 meta / stats / 算子的对应关系；可与 case study 中的证据表对照。</p>
{attribution_table}
<h2 id='sec-cohort'>按模型 × 日期 × 分档 的队列明细</h2>
<table>{thead}<tbody>{"".join(cohort_lines)}</tbody></table>
{toc_mini}
</main>
</div>
{input_sources_provenance_html}
<details>
<summary>进阶 / 调试资源</summary>
{adv}
</details>
<script>
(function () {{
  document.querySelectorAll("button.drill-toggle").forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      var card = btn.closest(".drill-card");
      if (!card) return;
      var body = card.querySelector(".drill-body");
      if (!body) return;
      var open = body.hasAttribute("hidden");
      if (open) {{
        body.removeAttribute("hidden");
        btn.textContent = "收起字段";
        btn.setAttribute("aria-expanded", "true");
      }} else {{
        body.setAttribute("hidden", "");
        btn.textContent = "展开字段";
        btn.setAttribute("aria-expanded", "false");
      }}
    }});
  }});
  document.querySelectorAll('.scenario-tabs').forEach(function (tablist) {{
    var tabs = tablist.querySelectorAll('.scenario-tab');
    tabs.forEach(function (tab) {{
      tab.addEventListener('click', function () {{
        var pid = tab.getAttribute('data-panel');
        if (!pid) return;
        tabs.forEach(function (t) {{
          var on = t === tab;
          t.classList.toggle('active', on);
          t.setAttribute('aria-selected', on ? 'true' : 'false');
        }});
        var host = tablist.parentElement;
        if (!host) return;
        host.querySelectorAll('.scenario-panel').forEach(function (pnl) {{
          var show = pnl.id === pid;
          pnl.classList.toggle('active', show);
          if (show) pnl.removeAttribute('hidden');
          else pnl.setAttribute('hidden', '');
        }});
      }});
    }});
  }});
}})();
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        action="append",
        dest="inputs",
        required=True,
        metavar="JSONL",
        help="processed.jsonl（可重复；多文件按顺序拼接载入，各自仍可合并同名 *_stats.jsonl）",
    )
    ap.add_argument("--output", required=True, help="Output .html path")
    ap.add_argument(
        "--title",
        default="智能体交互 · Bad-case 分析报告",
        help="HTML title / H1",
    )
    ap.add_argument(
        "--bilingual",
        action="store_true",
        help="Add a bilingual (ZH/EN) header hint block to the HTML report",
    )
    ap.add_argument(
        "--report-lang",
        choices=["auto", "zh", "en"],
        default="auto",
        help="报告分档等展示语言：zh|en|auto（auto 时先试 BAD_CASE_REPORT_LANG，再试 meta.agent_pipeline_output_lang）",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max rows to read")
    ap.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip matplotlib figures (table-only HTML)",
    )
    ap.add_argument(
        "--sample-headlines",
        type=int,
        default=10,
        help="Max high_precision insight cards (0=off)",
    )
    ap.add_argument(
        "--insight-model-tab-limit",
        type=int,
        default=8,
        metavar="N",
        help="每个模型子 Tab 内最多展示几条 Insight（需未加 --no-insight-model-tabs）",
    )
    ap.add_argument(
        "--insight-model-tab-group",
        choices=("full", "family"),
        default="full",
        help="Insight 子 Tab 分桶：full=完整 agent_request_model（默认，区分 3.5/2.5 等）；"
        "family=按厂商族合并",
    )
    ap.add_argument(
        "--no-insight-model-tabs",
        action="store_true",
        help="关闭 Insight 节按模型的子 Tab（不再分桶，仅单一列表）",
    )
    ap.add_argument(
        "--no-insight-semantic-cluster",
        action="store_true",
        help="关闭 Insight 节默认的语义聚类（TF-IDF+KMeans），仅保留精确归一合并表",
    )
    ap.add_argument(
        "--report-pii-variants",
        choices=("both", "safe", "audit"),
        default="both",
        help="默认 both：写出 PII 安全版（--output）与同目录 *_pii_audit.html；"
        "safe|audit 仅生成一种；audit 时路径即 --output",
    )
    ap.add_argument(
        "--no-audit-display-diversify",
        action="store_true",
        help="关闭 Insight 卡片与 Case study 页内的语义簇轮询抽样（改回按优先级顺序截取）",
    )
    ap.add_argument(
        "--drilldown-limit",
        type=int,
        default=-1,
        help="0=关闭 case study 整节；>0=仅导出/保留前 N 条（截断全量清单）；默认 -1=不截断导出",
    )
    ap.add_argument(
        "--drilldown-display-max",
        type=int,
        default=50,
        help="Case study 页内最多展开条数（默认两档各约一半；语义簇轮询抽样，其余见 jsonl）",
    )
    ap.add_argument(
        "--no-drilldown-export",
        action="store_true",
        help="不写 *_drilldown_full.jsonl（仍可按 display-max 展示页内卡片）",
    )
    ap.add_argument(
        "--llm-summary",
        action="store_true",
        help="调用 OpenAI 兼容接口生成页首导读（需环境变量 API Key）",
    )
    ap.add_argument(
        "--llm-model",
        default=os.environ.get("BAD_CASE_REPORT_LLM_MODEL", "qwen3.5-plus"),
        help="页首导读所用模型（默认 qwen3.5-plus 或环境变量 BAD_CASE_REPORT_LLM_MODEL）",
    )
    ap.add_argument(
        "--llm-api-base",
        default=os.environ.get(
            "OPENAI_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        help="Chat Completions base URL（需含 /v1，实际请求 .../chat/completions）",
    )
    ap.add_argument(
        "--llm-api-key",
        default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        help="API Key（默认读环境变量 DASHSCOPE_API_KEY 或 OPENAI_API_KEY）",
    )
    ap.add_argument(
        "--llm-timeout-sec",
        type=int,
        default=int(os.environ.get("BAD_CASE_REPORT_LLM_TIMEOUT_SEC", "120")),
        metavar="SEC",
        help="页首导读 HTTP 读超时秒数（默认 120 或环境变量 BAD_CASE_REPORT_LLM_TIMEOUT_SEC）",
    )
    ap.add_argument(
        "--eager-load-all-rows",
        action="store_true",
        help="整表读入内存再统计（小数据可用；百万级以上易 OOM。默认：单遍流式扫描 + 有界池）",
    )
    ap.add_argument(
        "--drill-retention-cap",
        type=int,
        default=DEFAULT_DRILL_RETENTION_CAP,
        metavar="N",
        help="流式模式下每档（强怀疑/待观察）最多保留 N 条进入 case study 候选池（默认 48000）",
    )
    ap.add_argument(
        "--insight-retention-cap",
        type=int,
        default=DEFAULT_INSIGHT_RETENTION_CAP,
        metavar="N",
        help="流式模式下每档最多保留 N 条带 headline 的 Insight 候选（默认 14000）",
    )
    ap.add_argument(
        "--dialog-metric-samples",
        type=int,
        default=DEFAULT_DIALOG_METRIC_SAMPLES,
        metavar="N",
        help="对话/token/时延直方图与分位数每指标最多贮样 N 行（默认 200000，全量扫描计数）",
    )
    args = ap.parse_args()

    snap: Optional[StreamedReportSnapshot] = None
    rows: List[dict]
    if args.eager_load_all_rows:
        rows = load_merged_rows(args.inputs, args.limit)
        if not rows:
            print("ERROR: no rows loaded; check --input path(s).", file=sys.stderr)
            return 2
    else:
        rows = []
        rng = random.Random(31)
        snap = _stream_scan_report_inputs(
            list(args.inputs),
            row_limit=args.limit,
            drill_cap=max(1, int(args.drill_retention_cap)),
            insight_cap=max(1, int(args.insight_retention_cap)),
            dialog_sample_cap=max(0, int(args.dialog_metric_samples)),
            rng=rng,
        )
        if snap.n_rows <= 0:
            print("ERROR: no rows loaded; check --input path(s).", file=sys.stderr)
            return 2
        if snap.drill_pool_truncated:
            print(
                "NOTE: 大规模流式模式：case study / drill 导出仅从每档优先保留的候选池中选取；"
                f"强怀疑 {snap.n_hp_total} 条 / 待观察 {snap.n_wl_total} 条（池上限各 {args.drill_retention_cap}）。"
                "可调 --drill-retention-cap / --insight-retention-cap。",
                file=sys.stderr,
            )

    n_rows = len(rows) if rows else (snap.n_rows if snap else 0)

    resolved_report_lang = infer_report_locale(
        rows if rows else (snap.locale_probe_rows if snap else []),
        str(args.report_lang),
        os.environ.get("BAD_CASE_REPORT_LANG"),
    )
    set_report_locale(resolved_report_lang)
    html_page_lang = "en" if resolved_report_lang == "en" else "zh-CN"

    if snap is None:
        tier_cnt = _tier_counts(rows)
        high_c, med_c = _signal_counts_by_weight(rows)
        model_tier = _model_tier_matrix(rows)
        cohort = aggregate_cohort_stdlib(rows)
    else:
        tier_cnt = snap.tier_cnt
        high_c, med_c = snap.high_c, snap.med_c
        model_tier = snap.model_tier
        cohort = snap.cohort_rows
    att_html = _attribution_table_html()

    chart_tier = chart_sig_high = chart_sig_med = chart_model = None
    plt_mod = None
    if not args.no_charts:
        plt_mod = _get_plt()
    if plt_mod is not None:
        chart_tier = _chart_tier_bar(tier_cnt, plt_mod)
        if high_c:
            chart_sig_high = _chart_signals(
                high_c,
                plt_mod,
                "本批 high 权重主证据信号（柱末数字为条数）",
                "#c0392b",
            )
        if med_c:
            chart_sig_med = _chart_signals(
                med_c,
                plt_mod,
                "附录：medium 启发式 / 弱证据信号（柱末数字为条数）",
                "#2980b9",
            )
        if len(model_tier) >= 1:
            chart_model = _chart_by_model(model_tier, plt_mod)

    rule_summary = _rule_based_exec_summary(n_rows, tier_cnt, high_c, med_c)
    digest_llm = _build_llm_digest_compact(n_rows, tier_cnt, high_c, med_c, cohort)
    llm_summary: Optional[str] = None
    if args.llm_summary:
        key = (args.llm_api_key or "").strip()
        if not key:
            print("WARNING: --llm-summary 已开启但未配置 API Key。", file=sys.stderr)
        llm_summary = _fetch_exec_summary_llm(
            digest_llm,
            model=args.llm_model,
            api_key=key,
            api_base=args.llm_api_base,
            timeout_sec=max(5, int(args.llm_timeout_sec)),
        )

    body_summary = llm_summary or rule_summary
    summary_note = (
        "以下由大模型根据上方同批次聚合摘要生成，数字请务必与下方表格交叉核对。"
        if llm_summary
        else "以下由离线规则根据当前批次统计即时生成。若需更自然的表述，可加参数 --llm-summary 并配置 API Key。"
    )
    exec_summary_html = _exec_summary_section_html(body_summary, summary_note)

    if snap is not None:
        macro_html = _build_macro_distribution_section(
            [],
            plt_mod,
            args.no_charts,
            html_page_lang,
            precomputed_counters=snap.macro_counters,
            precomputed_skill_stats=snap.skill_stats,
        )
        dm_note = (
            "大规模模式下对话指标直方图/分位数基于每指标贮样近似（见 --dialog-metric-samples）。"
            if html_page_lang != "en"
            else "Large-batch mode: histograms/quantiles use per-metric reservoir sampling "
            "(see --dialog-metric-samples)."
        )
        dialog_metrics_html = _dialog_shape_metrics_section_html(
            [],
            plt_mod,
            no_charts=args.no_charts,
            html_lang=html_page_lang,
            series_override=snap.dialog_series,
            sampled_note=dm_note,
        )
    else:
        macro_html = _build_macro_distribution_section(
            rows, plt_mod, args.no_charts, html_page_lang
        )
        dialog_metrics_html = _dialog_shape_metrics_section_html(
            rows,
            plt_mod,
            no_charts=args.no_charts,
            html_lang=html_page_lang,
        )
    charts_section_html = _build_charts_section_html(
        chart_tier=chart_tier,
        chart_model=chart_model,
        high_c=high_c,
        med_c=med_c,
        chart_full_high=chart_sig_high,
        chart_full_med=chart_sig_med,
        plt_mod=plt_mod,
        no_charts=args.no_charts,
        html_lang=html_page_lang,
        macro_distribution_html=macro_html,
    )

    out_path = Path(args.output)
    use_div = not args.no_audit_display_diversify
    insight_semantic_k = 15

    variants: List[Tuple[str, Path]] = []
    rpv = str(args.report_pii_variants).strip().lower()
    if rpv in ("both", "safe"):
        variants.append(("safe", out_path))
    if rpv in ("both", "audit"):
        audit_p = out_path if rpv == "audit" else _pii_audit_report_path(out_path)
        variants.append(("audit", audit_p))
    wrote_both = rpv == "both"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bilingual_header = ""
    if args.bilingual:
        bilingual_header = (
            "<p class='note' lang='en'><strong>Bilingual report:</strong> "
            "Explanatory prose is primarily Chinese for readability. "
            "Machine enums remain English (<code>high_precision</code>, "
            "<code>watchlist</code>, <code>none</code>, signal "
            "<code>code</code>s, JSON keys) for scripting and handoff. "
            "Insight badges use English tokens (<code>P0</code>–<code>P3</code>, "
            "<code>aligned</code>/<code>mixed</code>/<code>conflict</code>); "
            "see the <a href='#sec-insight-fields'>field semantics</a> table.</p>"
        )

    for variant, target_html in variants:
        hide_pii = variant == "safe"
        drill_html = ""
        export_rel: Optional[str] = None
        drill_show: List[dict] = []
        if args.drilldown_limit != 0:
            cap_export = args.drilldown_limit if args.drilldown_limit > 0 else None
            display_n = max(0, args.drilldown_display_max)
            scale_note = ""
            if snap is not None:
                total_drill = snap.n_hp_total + snap.n_wl_total
                hp_f = _filter_drill_pool_rows(snap.hp_drill_pool, exclude_pii=hide_pii)
                wl_f = _filter_drill_pool_rows(snap.wl_drill_pool, exclude_pii=hide_pii)
                drill_all = _merge_drill_all_entries(hp_f, wl_f, cap_export)
                combined = hp_f + wl_f
                if snap.drill_pool_truncated:
                    cap = int(args.drill_retention_cap)
                    if str(html_page_lang).lower().startswith("en"):
                        scale_note = (
                            "<p class='note'>This batch has "
                            f"<strong>{total_drill}</strong> high_precision+watchlist rows; "
                            "in-page cards and drill JSONL are drawn from a "
                            f"<strong>priority-retained pool of up to {cap}</strong> per tier "
                            "(streaming mode; tune <code>--drill-retention-cap</code>).</p>"
                        )
                    else:
                        scale_note = (
                            "<p class='note'>本批强怀疑+待观察合计 <strong>"
                            f"{total_drill}</strong> 条；页内与 drill 导出仅来自每档最多保留 "
                            f"<strong>{cap}</strong> 条的优先池（流式模式，可调 "
                            "<code>--drill-retention-cap</code>）。</p>"
                        )
                if display_n:
                    pick_rows = _pick_balanced_diverse_drill_rows(
                        combined,
                        display_n,
                        exclude_pii=False,
                        semantic_max_k=insight_semantic_k,
                        use_diversify=use_div,
                    )
                    drill_show = [
                        _row_to_drill_entry(
                            r,
                            lang_en=str(html_page_lang).lower().startswith("en"),
                        )
                        for r in pick_rows
                    ]
            else:
                drill_all = _build_drill_all_entries(
                    rows, cap_export, exclude_pii=hide_pii
                )
                total_drill = len(drill_all)
                if display_n:
                    pick_rows = _pick_balanced_diverse_drill_rows(
                        rows,
                        display_n,
                        exclude_pii=hide_pii,
                        semantic_max_k=insight_semantic_k,
                        use_diversify=use_div,
                    )
                    drill_show = [
                        _row_to_drill_entry(
                            r,
                            lang_en=str(html_page_lang).lower().startswith("en"),
                        )
                        for r in pick_rows
                    ]
            if not args.no_drilldown_export and drill_all:
                export_path = target_html.with_name(target_html.stem + "_drilldown_full.jsonl")
                _write_drilldown_jsonl(export_path, drill_all)
                export_rel = export_path.name
                print(f"Wrote drilldown export {export_path.resolve()}")
            drill_html = _drilldown_section_html(
                drill_show,
                total_count=total_drill,
                export_rel=export_rel,
                html_lang=html_page_lang,
                use_display_diversify=use_div,
                scale_note_html=scale_note,
            )

        anchor_by_rid = _request_id_to_drill_anchor(drill_show)
        insight_section_html = ""
        if args.sample_headlines > 0:
            if snap is not None:
                insight_section_html = _insight_sections_html(
                    rows if rows else snap.locale_probe_rows,
                    args.sample_headlines,
                    max(1, int(args.insight_model_tab_limit)),
                    not args.no_insight_model_tabs,
                    anchor_by_rid,
                    html_lang=html_page_lang,
                    use_semantic_cluster=not args.no_insight_semantic_cluster,
                    semantic_max_k=insight_semantic_k,
                    model_tab_group=str(args.insight_model_tab_group),
                    model_tier=model_tier,
                    hide_pii=hide_pii,
                    use_display_diversify=use_div,
                    insight_hp_pool=snap.hp_insight_pool,
                    insight_wl_pool=snap.wl_insight_pool,
                    insight_pii_omitted_hp=snap.insight_pii_omitted_hp,
                    insight_pii_omitted_wl=snap.insight_pii_omitted_wl,
                )
            else:
                insight_section_html = _insight_sections_html(
                    rows,
                    args.sample_headlines,
                    max(1, int(args.insight_model_tab_limit)),
                    not args.no_insight_model_tabs,
                    anchor_by_rid,
                    html_lang=html_page_lang,
                    use_semantic_cluster=not args.no_insight_semantic_cluster,
                    semantic_max_k=insight_semantic_k,
                    model_tab_group=str(args.insight_model_tab_group),
                    model_tier=model_tier,
                    hide_pii=hide_pii,
                    use_display_diversify=use_div,
                )

        variant_banner = _report_variant_banner_html(
            variant=variant,
            html_lang=html_page_lang,
            wrote_audit_sibling=wrote_both,
        )
        page_title = args.title
        if wrote_both:
            page_title = (
                f"{args.title}（PII 安全版）"
                if variant == "safe"
                else f"{args.title}（PII 审计版）"
            )

        page = _html_page(
            page_title,
            list(args.inputs),
            n_rows,
            tier_cnt,
            cohort,
            att_html,
            exec_summary_html,
            charts_section_html,
            dialog_metrics_html,
            drill_html,
            _insight_fields_legend_html(html_lang=html_page_lang),
            insight_section_html,
            _reading_scenarios_html(html_lang=html_page_lang),
            bilingual_header=bilingual_header,
            html_lang=html_page_lang,
            variant_banner=variant_banner,
        )
        target_html.write_text(page, encoding="utf-8")
        print(f"Wrote {target_html.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
