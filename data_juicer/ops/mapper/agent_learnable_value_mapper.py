# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single scalar learnable-value score + tier for
# training-data prioritization (no LLM).

from __future__ import annotations

from typing import Any

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys

OP_NAME = "agent_learnable_value_mapper"


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentLearnableValueMapper(Mapper):
    """Write training-value scalar and tier meta fields."""

    # Key logic (rules only; no extra LLM calls in this op):
    # - difficulty: stats.llm_difficulty_score (normalized to [0,1])
    # - taxonomy focus: max(bucket severity) - mean(bucket severity)
    # - completeness: turns + tool chain integrity + compression flag
    # - generalizability: intent/topic label density per turn
    # - cross-model signal: agent_cross_model_pair.delta_to_best
    #
    # Note (high-ceiling): keep this mapper as cheap coarse ranking, but add a
    # post-hoc calibration/learning layer (or stronger judge + sampled human
    # audit) so the ranking is less bottlenecked by weak upstream judges.
    # Note (r3-semantic-shift): if R3 rewrite/distill changes response semantics,
    # run a lightweight re-scoring chain for quality-related upstream signals.

    def __init__(
        self,
        weight_difficulty: float = 0.35,
        weight_taxonomy_focus: float = 0.25,
        weight_completeness: float = 0.20,
        weight_generalizability: float = 0.10,
        weight_cross_model: float = 0.10,
        tier_gold: float = 0.70,
        tier_silver: float = 0.50,
        tier_bronze: float = 0.30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wd = float(weight_difficulty)
        self.wt = float(weight_taxonomy_focus)
        self.wc = float(weight_completeness)
        self.wg = float(weight_generalizability)
        self.wx = float(weight_cross_model)
        self.tier_gold = float(tier_gold)
        self.tier_silver = float(tier_silver)
        self.tier_bronze = float(tier_bronze)

    def process_single(self, sample):
        meta = sample.setdefault(Fields.meta, {})
        stats = sample.get(Fields.stats) or {}
        if not isinstance(stats, dict):
            stats = {}

        difficulty = _f(stats.get(StatsKeys.llm_difficulty_score))
        if difficulty > 1.0:
            difficulty = min(1.0, difficulty / 5.0)

        tax = meta.get(MetaKeys.agent_error_taxonomy) or {}
        buckets = tax.get("buckets") if isinstance(tax, dict) else None
        severities = []
        if isinstance(buckets, dict):
            for _name, b in buckets.items():
                if isinstance(b, dict) and b.get("severity") is not None:
                    severities.append(_f(b["severity"]))
        focus = 0.0
        if severities:
            mx = max(severities)
            mn = sum(severities) / len(severities)
            focus = max(0.0, mx - mn)

        turns = _f(meta.get(MetaKeys.agent_turn_count))
        compressed = bool(meta.get(MetaKeys.agent_dialog_history_compressed))
        chain_ok = meta.get(MetaKeys.agent_tool_chain_complete)
        completeness = 0.0
        if turns >= 2.0 and chain_ok is not False and not compressed:
            completeness = 1.0
        elif turns >= 1.0 and chain_ok is not False:
            completeness = 0.65
        else:
            completeness = 0.35

        intents = meta.get(MetaKeys.dialog_intent_labels) or []
        topics = meta.get(MetaKeys.dialog_topic_labels) or []
        labels = set()
        if isinstance(intents, list):
            labels.update(str(x).strip() for x in intents if str(x).strip())
        if isinstance(topics, list):
            labels.update(str(x).strip() for x in topics if str(x).strip())
        gen = min(1.0, len(labels) / max(turns, 1.0)) if turns else 0.0

        pair = meta.get(MetaKeys.agent_cross_model_pair) or {}
        cross = 0.35
        if isinstance(pair, dict) and pair.get("has_pairwise_contrast"):
            delta = pair.get("delta_to_best")
            if delta is not None:
                cross = min(1.0, max(0.0, _f(delta) * 2.0))

        lv = self.wd * difficulty + self.wt * focus + self.wc * completeness + self.wg * gen + self.wx * cross
        lv = max(0.0, min(1.0, float(lv)))

        tax_hard = bool(tax.get("hard_drop_recommended")) if isinstance(tax, dict) else False
        tier = "drop"
        if tax_hard:
            tier = "drop"
        elif lv >= self.tier_gold:
            tier = "gold"
        elif lv >= self.tier_silver:
            tier = "silver"
        elif lv >= self.tier_bronze:
            tier = "bronze"

        meta[MetaKeys.agent_learnable_value] = lv
        meta[MetaKeys.agent_learnable_value_tier] = tier
        meta[MetaKeys.agent_training_dataset_tier] = tier
        return sample
