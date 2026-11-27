"""Model-independent checks against the original pairwise cosine calculation."""

from unittest.mock import patch

import numpy as np
import pickle
import unittest
import torch
from datasets import Dataset

from data_juicer.ops.filter.text_embd_similarity_filter import TextEmbdSimilarityFilter
from data_juicer.utils.constant import Fields, StatsKeys


class TextEmbdSimilarityMeanTest(unittest.TestCase):
    def setUp(self):
        with patch("data_juicer.ops.filter.text_embd_similarity_filter.prepare_model", return_value="unused"):
            self.op = TextEmbdSimilarityFilter(min_score=0.2, max_score=0.8)

    def prepare(self, references):
        with patch.object(self.op, "_get_embd", return_value=np.asarray(references, dtype=np.float64)):
            self.op.prepare_valid_feature(Dataset.from_dict({"text": ["reference"] * len(references)}))

    def score(self, query):
        sample = {"text": "candidate", Fields.stats: {}}
        with patch.object(self.op, "_get_embd_single", return_value=query):
            return self.op.compute_stats_single(sample)

    def test_matches_pairwise_cosines(self):
        rng = np.random.default_rng(17)
        references = rng.normal(size=(37, 64))
        references[0] = 0
        references[1] *= 1e-12
        references[2] *= 1e6
        self.prepare(references)
        for dtype in (np.float32, np.float64):
            for scale in (0.0, 1e-12, 1e-9, 1e-8, 1.0, 1e6):
                with self.subTest(dtype=dtype, scale=scale):
                    query = (rng.normal(size=64) * scale).astype(dtype)
                    expected = (
                        torch.nn.functional.cosine_similarity(torch.tensor(query).view(1, -1), torch.tensor(references))
                        .mean()
                        .item()
                    )
                    sample = self.score(query)
                    self.assertAlmostEqual(sample[Fields.stats][StatsKeys.text_embd_similarity], expected, delta=1e-12)
                    self.assertEqual(
                        self.op.process_single(sample),
                        self.op.get_keep_boolean(expected, self.op.min_score, self.op.max_score),
                    )

    def test_mean_is_not_renormalized_and_refreshes(self):
        self.prepare([[1, 0], [-1, 0], [0, 1]])
        self.assertAlmostEqual(self.score([0.0, 1.0])[Fields.stats][StatsKeys.text_embd_similarity], 1 / 3)
        self.prepare([[0, -1]])
        self.assertEqual(self.score([0.0, 1.0])[Fields.stats][StatsKeys.text_embd_similarity], -1)
        self.prepare([])
        self.assertNotIn("mean_normalized_embedding", self.op.valid_feature)

    def test_unusual_query_shapes_keep_original_behavior(self):
        references = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.prepare(references)
        for query in ([1.0], [1.0, 2.0, 3.0], None, [1, 0], [1 + 1j, 0j]):
            with self.subTest(query=query):
                try:
                    expected = (
                        torch.nn.functional.cosine_similarity(torch.tensor(query).view(1, -1), torch.tensor(references))
                        .mean()
                        .item()
                    )
                except (TypeError, RuntimeError):
                    expected = None
                self.assertEqual(self.score(query)[Fields.stats][StatsKeys.text_embd_similarity], expected)

    def test_direct_features_and_cached_score(self):
        self.op.valid_feature = {"embeddings": np.array([[1.0, 0.0]])}
        self.assertEqual(self.score([1.0, 0.0])[Fields.stats][StatsKeys.text_embd_similarity], 1)
        sample = {Fields.stats: {StatsKeys.text_embd_similarity: 0.5}}
        with patch.object(self.op, "_get_embd_single", side_effect=AssertionError("must not embed cached sample")):
            self.assertIs(self.op.compute_stats_single(sample), sample)

    def test_nonfinite_references_remain_nonfinite(self):
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                self.prepare([[invalid, 1.0], [1.0, 0.0]])
                self.assertTrue(np.isnan(self.score([1.0, 0.0])[Fields.stats][StatsKeys.text_embd_similarity]))

    def test_threshold_decisions(self):
        self.prepare([[1.0, 0.0], [0.0, 1.0]])
        for similarity in (0.2 - 1e-8, 0.2, 0.2 + 1e-8):
            # Add two cancelling references to control the mean's magnitude.
            references = np.array([[1.0, 0.0], [-1.0, 0.0], [similarity * 3, (1 - (similarity * 3) ** 2) ** 0.5]])
            self.prepare(references)
            expected = (
                torch.nn.functional.cosine_similarity(torch.tensor([[1.0, 0.0]]), torch.tensor(references))
                .mean()
                .item()
            )
            sample = self.score([1.0, 0.0])
            self.assertEqual(self.op.process_single(sample), self.op.get_keep_boolean(expected, 0.2, 0.8))

    def test_prepared_features_survive_serialization(self):
        self.prepare([[1.0, 0.0], [0.0, 1.0]])
        self.op.valid_feature = pickle.loads(pickle.dumps(self.op.valid_feature))
        self.assertEqual(self.score([1.0, 0.0])[Fields.stats][StatsKeys.text_embd_similarity], 0.5)
