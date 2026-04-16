# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fuse deterministic + optional LLM-eval signals into a conservative bad-case
# triage for human review (precision-oriented by default).

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys

OP_NAME = "agent_bad_case_signal_mapper"
logger = logging.getLogger(__name__)

# Lightweight 1–5 turn/trace judges (dialog_* / agent_* quality LLM mappers).
_DIALOG_QUALITY_SCORE_META_KEYS = (
    MetaKeys.dialog_memory_consistency,
    MetaKeys.dialog_coreference,
    MetaKeys.dialog_topic_shift,
    MetaKeys.dialog_error_recovery,
    MetaKeys.dialog_clarification_quality,
    MetaKeys.dialog_proactivity,
    MetaKeys.dialog_non_repetition,
    MetaKeys.agent_trace_coherence,
    MetaKeys.agent_tool_relevance,
)

_calibration_missing_path_warned: Optional[str] = None


def _load_calibration_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not str(path).strip():
        return None
    ap = os.path.abspath(os.path.expanduser(str(path).strip()))
    if not os.path.isfile(ap):
        global _calibration_missing_path_warned
        if _calibration_missing_path_warned != ap:
            logger.warning(
                "agent_bad_case_signal_mapper: calibration_json_path is not a file "
                "(%s); auto thresholds disabled for this run.",
                ap,
            )
            _calibration_missing_path_warned = ap
        return None
    try:
        with open(ap, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "agent_bad_case_signal_mapper: failed to load calibration JSON %s: %s",
            ap,
            e,
        )
        return None
    if not isinstance(data, dict):
        return None
    return data


def _normalize_recommendation(record: Any) -> str:
    # record may be a JSON string (serialized by _normalize_record) or a dict
    if isinstance(record, str):
        if not record:
            return ""
        try:
            import json as _json

            record = _json.loads(record)
        except Exception:
            return ""
    if not isinstance(record, dict):
        return ""
    r = record.get("recommendation")
    if isinstance(r, (list, tuple, np.ndarray)) and len(r) > 0:
        r = r[0]
    if hasattr(r, "item"):
        try:
            r = r.item()
        except Exception:
            pass
    if r is None:
        return ""
    return str(r).strip().strip('"').lower()


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentBadCaseSignalMapper(Mapper):
    """Attach structured bad-case *signals* and a conservative *tier* to each sample.

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
    """

    def __init__(
        self,
        query_key: str = "query",
        response_key: str = "response",
        # --- tool path ---
        signal_on_tool_fail: bool = True,
        # Agent runs often include exploratory tool errors (e.g. wrong path)
        # before recovery; require this many pattern-matched **error** tool
        # messages before emitting ``tool_message_error_pattern``.
        min_tool_fail_count_for_signal: int = 1,
        signal_on_low_tool_success_ratio: bool = True,
        tool_success_ratio_max_for_signal: float = 0.499,
        min_tool_rounds_for_ratio_signal: int = 2,
        # --- empty response heuristic ---
        signal_on_suspect_empty_response: bool = True,
        min_query_len_for_empty_check: int = 80,
        max_response_len_for_empty_check: int = 20,
        # --- cost / latency (optional absolute thresholds) ---
        max_total_tokens: Optional[int] = None,
        max_latency_ms: Optional[int] = None,
        # --- optional P-percentile calibration (see demos/agent/scripts/compute_percentile_thresholds.py) ---
        calibration_json_path: Optional[str] = None,
        auto_calibrate_thresholds: bool = False,
        # When True (default): explicit max_* / perplexity_high_threshold in YAML win over JSON.
        calibration_manual_overrides_auto: bool = True,
        # If JSON row has perplexity_high_threshold, enable that signal even when
        # signal_on_high_perplexity is False (unless manual override supplies threshold).
        auto_enable_perplexity_from_calibration: bool = True,
        # --- llm_analysis_filter (agent scenario eval) ---
        signal_on_llm_analysis_low: bool = True,
        llm_analysis_score_max_for_bad: float = 0.28,
        llm_analysis_discard_must_be_strict: bool = True,
        high_precision_llm_analysis_discard_threshold: float = 0.24,
        # --- llm_quality_score_filter (reply quality dims) ---
        signal_on_llm_text_quality_low: bool = True,
        llm_text_quality_score_max_for_bad: float = 0.28,
        llm_text_quality_discard_must_be_strict: bool = True,
        high_precision_llm_text_quality_discard_threshold: float = 0.24,
        # --- dialog tags (weak signals → medium only) ---
        signal_on_negative_sentiment_hint: bool = False,
        negative_sentiment_substrings: Optional[List[str]] = None,
        # --- text stats from filters ---
        signal_on_high_perplexity: bool = False,
        perplexity_high_threshold: float = 800.0,
        # --- difficulty × quality conjunction (off by default) ---
        signal_hard_query_poor_reply: bool = False,
        hard_query_difficulty_min: float = 0.72,
        poor_reply_quality_max: float = 0.36,
        # --- tier composition ---
        high_precision_on_tool_fail_alone: bool = True,
        min_medium_signals_for_watchlist: int = 2,
        # --- dialog / trace 1–5 meta (no extra LLM; reads upstream mapper output) ---
        signal_on_low_dialog_quality_meta: bool = True,
        dialog_quality_low_score_threshold: float = 2.0,
        min_dialog_quality_low_axes_for_signal: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_key = query_key
        self.response_key = response_key
        self.signal_on_tool_fail = signal_on_tool_fail
        self.min_tool_fail_count_for_signal = max(1, int(min_tool_fail_count_for_signal))
        self.signal_on_low_tool_success_ratio = signal_on_low_tool_success_ratio
        self.tool_success_ratio_max_for_signal = tool_success_ratio_max_for_signal
        self.min_tool_rounds_for_ratio_signal = min_tool_rounds_for_ratio_signal
        self.signal_on_suspect_empty_response = signal_on_suspect_empty_response
        self.min_query_len_for_empty_check = min_query_len_for_empty_check
        self.max_response_len_for_empty_check = max_response_len_for_empty_check
        self.max_total_tokens = max_total_tokens
        self.max_latency_ms = max_latency_ms
        self.calibration_json_path = calibration_json_path
        self.auto_calibrate_thresholds = bool(auto_calibrate_thresholds)
        self.calibration_manual_overrides_auto = bool(calibration_manual_overrides_auto)
        self.auto_enable_perplexity_from_calibration = bool(auto_enable_perplexity_from_calibration)
        self._calibration: Optional[Dict[str, Any]] = None
        if self.auto_calibrate_thresholds and self.calibration_json_path:
            self._calibration = _load_calibration_json(self.calibration_json_path)
            if self._calibration is not None:
                logger.info(
                    "agent_bad_case_signal_mapper: loaded calibration (percentile=%s) from %s",
                    self._calibration.get("percentile", "n/a"),
                    os.path.abspath(os.path.expanduser(str(self.calibration_json_path))),
                )
        self.signal_on_llm_analysis_low = signal_on_llm_analysis_low
        self.llm_analysis_score_max_for_bad = llm_analysis_score_max_for_bad
        self.llm_analysis_discard_must_be_strict = llm_analysis_discard_must_be_strict
        self.high_precision_llm_analysis_discard_threshold = high_precision_llm_analysis_discard_threshold
        self.signal_on_llm_text_quality_low = signal_on_llm_text_quality_low
        self.llm_text_quality_score_max_for_bad = llm_text_quality_score_max_for_bad
        self.llm_text_quality_discard_must_be_strict = llm_text_quality_discard_must_be_strict
        self.high_precision_llm_text_quality_discard_threshold = high_precision_llm_text_quality_discard_threshold
        self.signal_on_negative_sentiment_hint = signal_on_negative_sentiment_hint
        self.negative_sentiment_substrings = negative_sentiment_substrings or [
            "负面",
            "negative",
            "angry",
            "沮丧",
            "不满",
            "frustrated",
            "disappointed",
            "unhappy",
        ]
        self.signal_on_high_perplexity = signal_on_high_perplexity
        self.perplexity_high_threshold = perplexity_high_threshold
        self.signal_hard_query_poor_reply = signal_hard_query_poor_reply
        self.hard_query_difficulty_min = hard_query_difficulty_min
        self.poor_reply_quality_max = poor_reply_quality_max
        self.high_precision_on_tool_fail_alone = high_precision_on_tool_fail_alone
        self.min_medium_signals_for_watchlist = min_medium_signals_for_watchlist
        self.signal_on_low_dialog_quality_meta = bool(signal_on_low_dialog_quality_meta)
        self.dialog_quality_low_score_threshold = float(dialog_quality_low_score_threshold)
        self.min_dialog_quality_low_axes_for_signal = max(1, int(min_dialog_quality_low_axes_for_signal))

    def _resolve_calibration_row(self, meta: dict) -> Dict[str, Any]:
        if not self.auto_calibrate_thresholds or not self._calibration:
            return {}
        default = self._calibration.get("default")
        if not isinstance(default, dict):
            default = {}
        bym = self._calibration.get("by_request_model")
        if not isinstance(bym, dict):
            bym = {}
        model = str(meta.get(MetaKeys.agent_request_model) or "").strip()
        row = dict(default)
        if model and model in bym and isinstance(bym[model], dict):
            row.update(bym[model])
        return row

    def _effective_max_total_tokens(self, meta: dict) -> Optional[int]:
        cal_v: Optional[int] = None
        if self.auto_calibrate_thresholds and self._calibration:
            raw = self._resolve_calibration_row(meta).get("max_total_tokens")
            if raw is not None:
                try:
                    cal_v = int(raw)
                except (TypeError, ValueError):
                    pass
        if self.calibration_manual_overrides_auto:
            if self.max_total_tokens is not None:
                return self.max_total_tokens
            return cal_v
        if cal_v is not None:
            return cal_v
        return self.max_total_tokens

    def _effective_max_latency_ms(self, meta: dict) -> Optional[int]:
        cal_v: Optional[int] = None
        if self.auto_calibrate_thresholds and self._calibration:
            raw = self._resolve_calibration_row(meta).get("max_latency_ms")
            if raw is not None:
                try:
                    cal_v = int(raw)
                except (TypeError, ValueError):
                    pass
        if self.calibration_manual_overrides_auto:
            if self.max_latency_ms is not None:
                return self.max_latency_ms
            return cal_v
        if cal_v is not None:
            return cal_v
        return self.max_latency_ms

    def _effective_perplexity(self, meta: dict) -> Tuple[bool, float]:
        """Return (signal_on, threshold)."""
        row = self._resolve_calibration_row(meta) if self.auto_calibrate_thresholds and self._calibration else {}
        cal_th: Optional[float] = None
        raw = row.get("perplexity_high_threshold")
        if raw is not None:
            try:
                cal_th = float(raw)
            except (TypeError, ValueError):
                pass

        want_signal = self.signal_on_high_perplexity or (
            cal_th is not None
            and self.auto_enable_perplexity_from_calibration
            and self.auto_calibrate_thresholds
            and self._calibration is not None
        )
        if not want_signal:
            return False, float(self.perplexity_high_threshold)

        if self.calibration_manual_overrides_auto and self.signal_on_high_perplexity:
            return True, float(self.perplexity_high_threshold)

        if not self.calibration_manual_overrides_auto and cal_th is not None:
            return True, cal_th

        if cal_th is not None and (self.signal_on_high_perplexity or self.auto_enable_perplexity_from_calibration):
            return True, cal_th

        if self.signal_on_high_perplexity:
            return True, float(self.perplexity_high_threshold)

        return False, float(self.perplexity_high_threshold)

    def _append(
        self,
        signals: List[dict],
        code: str,
        detail: str,
        weight: str,
    ) -> None:
        signals.append({"code": code, "detail": detail, "weight": weight})

    def _llm_eval_signal(
        self,
        stats: dict,
        signals: List[dict],
        score_key: str,
        record_key: str,
        score_max: float,
        discard_strict: bool,
        high_thresh: float,
        code: str,
    ) -> None:
        score = stats.get(score_key)
        record = stats.get(record_key)
        rec_norm = _normalize_recommendation(record)
        if score is None:
            return
        if float(score) > score_max:
            return
        strict = (not discard_strict) or (rec_norm == "discard")
        if not strict:
            return
        w = "high" if float(score) <= high_thresh and rec_norm == "discard" else "medium"
        self._append(
            signals,
            code,
            f"score={score}, recommendation={rec_norm or 'n/a'}",
            w,
        )

    def process_single(self, sample: dict) -> dict:
        meta = sample.setdefault(Fields.meta, {})
        stats = sample.get(Fields.stats) or {}
        signals: List[dict] = []

        fail_count = int(meta.get(MetaKeys.tool_fail_count) or 0)
        if self.signal_on_tool_fail and fail_count >= self.min_tool_fail_count_for_signal:
            self._append(
                signals,
                "tool_message_error_pattern",
                f"tool_fail_count={fail_count}",
                "high",
            )

        succ = int(meta.get(MetaKeys.tool_success_count) or 0)
        rounds = succ + fail_count
        ratio = meta.get(MetaKeys.tool_success_ratio)
        if (
            self.signal_on_low_tool_success_ratio
            and rounds >= self.min_tool_rounds_for_ratio_signal
            and ratio is not None
            and float(ratio) <= self.tool_success_ratio_max_for_signal
        ):
            self._append(
                signals,
                "low_tool_success_ratio",
                f"ratio={ratio}, success={succ}, fail={fail_count}",
                "medium",
            )

        q = (sample.get(self.query_key) or "").strip()
        r = (sample.get(self.response_key) or "").strip()
        if (
            self.signal_on_suspect_empty_response
            and len(q) >= self.min_query_len_for_empty_check
            and len(r) <= self.max_response_len_for_empty_check
        ):
            self._append(
                signals,
                "suspect_empty_or_trivial_final_response",
                f"query_len={len(q)}, response_len={len(r)}",
                "medium",
            )

        eff_max_tok = self._effective_max_total_tokens(meta)
        total_tokens = meta.get(MetaKeys.total_tokens)
        if eff_max_tok is not None and total_tokens is not None and int(total_tokens) > eff_max_tok:
            self._append(
                signals,
                "high_token_usage",
                f"total_tokens={total_tokens}",
                "medium",
            )

        eff_max_lat = self._effective_max_latency_ms(meta)
        latency = meta.get(MetaKeys.agent_total_cost_time_ms)
        if eff_max_lat is not None and latency is not None and int(latency) > eff_max_lat:
            self._append(
                signals,
                "high_latency_ms",
                f"total_cost_time_ms={latency}",
                "medium",
            )

        if self.signal_on_llm_analysis_low:
            self._llm_eval_signal(
                stats,
                signals,
                StatsKeys.llm_analysis_score,
                StatsKeys.llm_analysis_record,
                self.llm_analysis_score_max_for_bad,
                self.llm_analysis_discard_must_be_strict,
                self.high_precision_llm_analysis_discard_threshold,
                "llm_agent_analysis_eval_low",
            )

        if self.signal_on_llm_text_quality_low:
            self._llm_eval_signal(
                stats,
                signals,
                StatsKeys.llm_quality_score,
                StatsKeys.llm_quality_record,
                self.llm_text_quality_score_max_for_bad,
                self.llm_text_quality_discard_must_be_strict,
                self.high_precision_llm_text_quality_discard_threshold,
                "llm_reply_quality_eval_low",
            )

        if self.signal_on_negative_sentiment_hint:
            labels = meta.get(MetaKeys.dialog_sentiment_labels)
            if isinstance(labels, list) and labels:
                blob = " ".join(str(x).lower() for x in labels)
                if any(s.lower() in blob for s in self.negative_sentiment_substrings):
                    self._append(
                        signals,
                        "negative_sentiment_label_hint",
                        f"labels={labels[:6]}",
                        "medium",
                    )

        ppl_on, ppl_th = self._effective_perplexity(meta)
        if ppl_on:
            ppl = stats.get(StatsKeys.perplexity)
            if ppl is not None and float(ppl) >= ppl_th:
                self._append(
                    signals,
                    "high_perplexity",
                    f"perplexity={ppl}",
                    "medium",
                )

        if self.signal_hard_query_poor_reply:
            d = stats.get(StatsKeys.llm_difficulty_score)
            qs = stats.get(StatsKeys.llm_quality_score)
            if (
                d is not None
                and qs is not None
                and float(d) >= self.hard_query_difficulty_min
                and float(qs) <= self.poor_reply_quality_max
            ):
                self._append(
                    signals,
                    "hard_query_low_reply_quality_conjunction",
                    f"difficulty={d}, llm_quality_score={qs}",
                    "medium",
                )

        if self.signal_on_low_dialog_quality_meta:
            lows: List[str] = []
            th = float(self.dialog_quality_low_score_threshold)
            for k in _DIALOG_QUALITY_SCORE_META_KEYS:
                rec = meta.get(k)
                if not isinstance(rec, dict):
                    continue
                if rec.get("skipped") or rec.get("error"):
                    continue
                sc = rec.get("score")
                if sc is None:
                    continue
                try:
                    fv = float(sc)
                except (TypeError, ValueError):
                    continue
                if fv <= th:
                    lows.append(k)
            if len(lows) >= self.min_dialog_quality_low_axes_for_signal:
                self._append(
                    signals,
                    "dialog_turn_quality_meta_low",
                    f"axes={lows}, threshold={th}",
                    "medium",
                )

        meta[MetaKeys.agent_bad_case_signals] = signals

        mediums = [s for s in signals if s.get("weight") == "medium"]

        tool_fail_high = any(
            s.get("code") == "tool_message_error_pattern" and s.get("weight") == "high" for s in signals
        )
        llm_high = any(
            s.get("code")
            in (
                "llm_agent_analysis_eval_low",
                "llm_reply_quality_eval_low",
            )
            and s.get("weight") == "high"
            for s in signals
        )

        tier = "none"
        if tool_fail_high:
            tier = "high_precision" if self.high_precision_on_tool_fail_alone else "watchlist"
        elif llm_high:
            tier = "high_precision"
        elif len(mediums) >= self.min_medium_signals_for_watchlist:
            tier = "watchlist"
        elif len(signals) == 1 and signals[0].get("weight") == "medium":
            tier = "watchlist"

        meta[MetaKeys.agent_bad_case_tier] = tier
        return sample
