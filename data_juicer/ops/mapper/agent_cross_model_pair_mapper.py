# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dataset-level cross-model / cross-version pairing for the same logical sample
# (e.g. same ``sample_id`` with different ``tag.model``). Writes per-row
# ``meta.agent_cross_model_pair`` for downstream learnable-value / preference data.
#
# This operator **must see the full table at once** (grouping is global). The HF
# executor uses a custom ``run()``; the Ray executor detects
# :attr:`REQUIRES_FULL_DATASET_PASS` and runs :meth:`apply_full_dataset_annotations`
# after ``take_all`` (see ``data_juicer.core.data.ray_dataset``). Very large
# datasets may OOM; shard upstream or run on a node with enough RAM.
#
# When there is **no shared lineage id**, set ``group_key_mode`` to
# ``normalized_query`` (exact match after normalizing ``query``) or
# ``simhash_lsh`` (near-duplicate ``query`` + optional env text via SimHash + LSH +
# Hamming threshold) so cross-model / **cross-version** rows can still form cohorts.

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import free_models

OP_NAME = "agent_cross_model_pair_mapper"


def _lineage_meta_key(extra_key: str) -> str:
    """Must match ``lineage_meta_key`` in ``agent_dialog_normalize_mapper``."""
    return "agent_lineage_" + str(extra_key).strip().replace(".", "_")


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_query_text(s: Any) -> str:
    t = (s or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _simhash_features(text: str, ngram: int = 3) -> List[str]:
    if not text:
        return []
    if len(text) <= ngram:
        return [text]
    return [text[i : i + ngram] for i in range(len(text) - ngram + 1)]


def _simhash_64(text: str, ngram: int = 3) -> int:
    """64-bit SimHash for near-duplicate grouping (not cryptographic)."""
    vec = [0] * 64
    for feat in _simhash_features(text, ngram):
        h = int(hashlib.md5(feat.encode("utf-8", errors="ignore")).hexdigest(), 16)
        for b in range(64):
            vec[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(64):
        if vec[b] > 0:
            out |= 1 << b
    return out


def _hamming64(a: int, b: int) -> int:
    x = a ^ b
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _pair_record(
    *,
    group_key: str,
    group_size: int,
    has_pairwise_contrast: bool,
    best_model: Optional[str],
    best_quality: Optional[float],
    best_version: Optional[str],
    my_model: Optional[str],
    my_quality: Optional[float],
    my_version: Optional[str],
    peer_models: List[str],
    match_basis: str,
) -> dict:
    delta = None
    if my_quality is not None and best_quality is not None:
        delta = float(best_quality) - float(my_quality)
    spread = None
    if has_pairwise_contrast and my_quality is not None and best_quality is not None:
        spread = abs(float(best_quality) - float(my_quality))
    return {
        "group_key": group_key,
        "group_size": int(group_size),
        "has_pairwise_contrast": bool(has_pairwise_contrast),
        "best_model": best_model,
        "best_version": best_version,
        "best_quality": best_quality,
        "my_model": my_model,
        "my_version": my_version,
        "my_quality": my_quality,
        "peer_models": peer_models,
        "delta_to_best": delta,
        "quality_spread_in_group": spread,
        "match_basis": match_basis,
    }


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentCrossModelPairMapper(Mapper):
    """Annotate rows that share the same cohort key.

    Set :attr:`REQUIRES_FULL_DATASET_PASS` so Ray (and any executor) runs a
    full-dataset pass instead of per-batch ``process``.

    Default: exact ``pair_key_meta`` (e.g. ``agent_lineage_sample_id``). Optional
    **similar cohorts** when lineage ids differ:

    - ``normalized_query``: group by lowercased / whitespace-collapsed ``query``.
    - ``simhash_lsh``: SimHash over ``query`` (+ optional extra text) with LSH bands
      and ``simhash_max_hamming`` for near-duplicate **query** (proxy for similar
      ``<query, env>`` when env is folded into ``extra_group_text_key``).

    Cross-version regression uses the same groups: compare ``version_meta`` within
    a cohort found by any of the above modes.
    """

    #: If True, RayData runs ``take_all`` → :meth:`apply_full_dataset_annotations` →
    # ``from_items`` instead of ``map_batches`` (which would skip the custom
    # ``run`` and leave pairs empty).
    REQUIRES_FULL_DATASET_PASS = True

    def __init__(
        self,
        pair_key_meta: str = "agent_lineage_sample_id",
        model_meta: str = "agent_lineage_tag_model",
        version_meta: str = "agent_lineage_tag_version",
        score_meta: str = "agent_lineage_tag_quality",
        fallback_score_stat: str = "llm_quality_score",
        min_group_size: int = 2,
        group_key_mode: str = "exact",
        query_key: str = "query",
        extra_group_text_key: str = "",
        simhash_ngram: int = 3,
        num_lsh_bands: int = 8,
        bits_per_band: int = 8,
        simhash_max_hamming: int = 12,
        max_lsh_bucket_size: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pair_key_meta = str(pair_key_meta or "").strip() or "agent_lineage_sample_id"
        self.model_meta = str(model_meta or "").strip() or "agent_lineage_tag_model"
        self.version_meta = str(version_meta or "").strip() or "agent_lineage_tag_version"
        self.score_meta = str(score_meta or "").strip() or "agent_lineage_tag_quality"
        self.fallback_score_stat = str(fallback_score_stat or "").strip()
        self.min_group_size = max(2, int(min_group_size))
        mode = (group_key_mode or "exact").strip().lower()
        if mode not in ("exact", "normalized_query", "simhash_lsh"):
            mode = "exact"
        self.group_key_mode = mode
        self.query_key = str(query_key or "query").strip() or "query"
        self.extra_group_text_key = str(extra_group_text_key or "").strip()
        self.simhash_ngram = max(2, int(simhash_ngram))
        self.num_lsh_bands = max(1, min(16, int(num_lsh_bands)))
        self.bits_per_band = max(2, min(16, int(bits_per_band)))
        if self.num_lsh_bands * self.bits_per_band > 64:
            self.num_lsh_bands = max(1, 64 // self.bits_per_band)
        self.simhash_max_hamming = max(0, int(simhash_max_hamming))
        self.max_lsh_bucket_size = max(32, int(max_lsh_bucket_size))

    def process_single(self, sample):
        return sample

    def apply_full_dataset_annotations(self, rows: List[dict]) -> None:
        """Compute cohorts and set ``meta.agent_cross_model_pair`` on each row in place.

        Callers (HF ``run`` or Ray) must pass a list of **mutable** per-row dicts
        (typically a ``deepcopy`` of the table). Empty input is a no-op.
        """
        if not rows:
            return

        if self.group_key_mode == "normalized_query":
            groups, basis = self._groups_normalized_query(rows)
        elif self.group_key_mode == "simhash_lsh":
            groups, basis = self._groups_simhash_lsh(rows)
        else:
            groups, basis = self._groups_exact(rows)

        for gk, members in groups.items():
            scores: List[Optional[float]] = [self._score_for_row(r) for _, r in members]
            models = [self._meta_get(r, self.model_meta) for _, r in members]

            best_idx = 0
            best_sc: Optional[float] = None
            for j, sc in enumerate(scores):
                if sc is None:
                    continue
                if best_sc is None or sc > best_sc:
                    best_sc = sc
                    best_idx = j
            if best_sc is None and members:
                best_idx = 0

            _best_i, best_row = members[best_idx]
            best_model = self._meta_get(best_row, self.model_meta)
            best_version = self._meta_get(best_row, self.version_meta)
            best_quality = self._score_for_row(best_row)

            peer_models = sorted(
                {str(m) for m in models if m is not None and str(m).strip()},
            )
            has_contrast = len(members) >= self.min_group_size and len(peer_models) >= 2

            for _i, row in members:
                m = self._meta_dict(row)
                my_model = self._meta_get(row, self.model_meta)
                my_version = self._meta_get(row, self.version_meta)
                my_quality = self._score_for_row(row)
                m[MetaKeys.agent_cross_model_pair] = _pair_record(
                    group_key=str(gk),
                    group_size=len(members),
                    has_pairwise_contrast=has_contrast,
                    best_model=str(best_model) if best_model is not None else None,
                    best_quality=best_quality,
                    best_version=str(best_version) if best_version is not None else None,
                    my_model=str(my_model) if my_model is not None else None,
                    my_quality=my_quality,
                    my_version=str(my_version) if my_version is not None else None,
                    peer_models=peer_models,
                    match_basis=basis,
                )

        for row in rows:
            m = self._meta_dict(row)
            if MetaKeys.agent_cross_model_pair not in m:
                m[MetaKeys.agent_cross_model_pair] = _pair_record(
                    group_key="",
                    group_size=0,
                    has_pairwise_contrast=False,
                    best_model=None,
                    best_quality=None,
                    best_version=None,
                    my_model=None,
                    my_quality=None,
                    my_version=None,
                    peer_models=[],
                    match_basis=basis if groups else "exact_pair_key",
                )

    def _meta_dict(self, row: dict) -> dict:
        """Ensure ``Fields.meta`` on the row is a mutable dict (HF may stringify)."""
        k = Fields.meta
        m = row.get(k)
        if isinstance(m, dict):
            return m
        if isinstance(m, str) and m.strip():
            try:
                parsed = json.loads(m)
                if isinstance(parsed, dict):
                    row[k] = parsed
                    return parsed
            except json.JSONDecodeError:
                pass
        row[k] = {}
        return row[k]

    def _meta_get(self, row: dict, key: str) -> Any:
        return self._meta_dict(row).get(key)

    def _score_for_row(self, row: dict) -> Optional[float]:
        v = _as_float(self._meta_get(row, self.score_meta))
        if v is not None:
            return v
        if self.fallback_score_stat:
            st = row.get(Fields.stats) or {}
            if isinstance(st, dict):
                return _as_float(st.get(self.fallback_score_stat))
        return None

    def _grouping_text(self, row: dict) -> str:
        q = _normalize_query_text(row.get(self.query_key))
        if not self.extra_group_text_key:
            return q
        extra = row.get(self.extra_group_text_key)
        if extra is None:
            extra = self._meta_get(row, self.extra_group_text_key)
        ex = _normalize_query_text(extra) if not isinstance(extra, (dict, list)) else ""
        if isinstance(extra, (dict, list)):
            try:
                ex = _normalize_query_text(json.dumps(extra, sort_keys=True, ensure_ascii=False)[:4000])
            except (TypeError, ValueError):
                ex = ""
        if q and ex:
            return q + "\n" + ex
        return q or ex

    def _groups_exact(self, rows: List[dict]) -> Tuple[Dict[str, List[Tuple[int, dict]]], str]:
        groups: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
        for i, row in enumerate(rows):
            gk = self._meta_get(row, self.pair_key_meta)
            if gk is None or str(gk).strip() == "":
                continue
            groups[str(gk)].append((i, row))
        return groups, "exact_pair_key"

    def _groups_normalized_query(self, rows: List[dict]) -> Tuple[Dict[str, List[Tuple[int, dict]]], str]:
        groups: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
        for i, row in enumerate(rows):
            gk = self._grouping_text(row)
            if not gk:
                continue
            groups[gk].append((i, row))
        return groups, "normalized_query"

    def _groups_simhash_lsh(self, rows: List[dict]) -> Tuple[Dict[str, List[Tuple[int, dict]]], str]:
        n = len(rows)
        sims: List[Optional[int]] = []
        for row in rows:
            blob = self._grouping_text(row)
            sims.append(_simhash_64(blob, self.simhash_ngram) if blob else None)

        uf = _UnionFind(n)
        mask = (1 << self.bits_per_band) - 1
        for band in range(self.num_lsh_bands):
            shift = band * self.bits_per_band
            if shift >= 64:
                break
            buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for i, sh in enumerate(sims):
                if sh is None:
                    continue
                sub = (sh >> shift) & mask
                buckets[(band, sub)].append(i)
            for lst in buckets.values():
                if len(lst) > self.max_lsh_bucket_size:
                    continue
                for a in range(len(lst)):
                    for b in range(a + 1, len(lst)):
                        ia, ib = lst[a], lst[b]
                        ha, hb = sims[ia], sims[ib]
                        if ha is None or hb is None:
                            continue
                        if _hamming64(ha, hb) <= self.simhash_max_hamming:
                            uf.union(ia, ib)

        clusters: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
        for i, row in enumerate(rows):
            root = uf.find(i)
            clusters[root].append((i, row))

        groups: Dict[str, List[Tuple[int, dict]]] = {}
        for root, members in clusters.items():
            if len(members) == 1 and sims[members[0][0]] is None:
                groups[f"__empty_query__:{root}"] = members
                continue
            sh0 = sims[root]
            if sh0 is not None:
                gk = f"simhash_cluster:{root}:{sh0:x}"
            else:
                gk = f"simhash_cluster:{root}"
            groups[gk] = members
        return groups, "simhash_lsh"

    def run(self, dataset, *, exporter=None, tracer=None):
        from data_juicer.core.data import NestedDataset, add_same_content_to_new_column

        if not isinstance(dataset, NestedDataset):
            dataset = NestedDataset(dataset)

        if Fields.meta not in dataset.features:
            dataset = dataset.map(
                add_same_content_to_new_column,
                fn_kwargs={"new_column_name": Fields.meta, "initial_value": {}},
                num_proc=self.runtime_np(),
                batch_size=self.batch_size,
                desc=f"{self._name}_add_meta",
            )

        rows: List[dict] = copy.deepcopy(dataset.to_list())
        self.apply_full_dataset_annotations(rows)
        new_ds = NestedDataset.from_list(rows)
        free_models()
        return new_ds


# Export helper for recipes / tests (same naming as normalize)
lineage_meta_key = _lineage_meta_key
