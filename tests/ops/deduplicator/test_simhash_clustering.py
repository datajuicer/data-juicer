"""Clustering tests independent of the optional native SimHash extension."""

import random
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator import document_simhash_deduplicator as module
from data_juicer.utils.constant import HashKeys


def graph_oracle(rows, edges, show_num):
    graph = defaultdict(set)
    for x, y in edges:
        graph[x].add(y)
        graph[y].add(x)
    remaining = set(int(row[HashKeys.simhash]) for row in rows)
    components = {}
    traces = {}
    cluster = 0
    while remaining:
        value = remaining.pop()
        if value not in graph or value in components:
            continue
        pending = [value]
        components[value] = cluster
        while pending:
            for neighbor in graph[pending.pop()]:
                if neighbor not in components:
                    components[neighbor] = cluster
                    pending.append(neighbor)
        if len(traces) < show_num:
            traces[cluster] = []
        cluster += 1
    seen = set()
    kept = []
    for row in rows:
        value = int(row[HashKeys.simhash])
        key = ("cluster", components[value]) if value in components else ("hash", value)
        if key[0] == "cluster" and key[1] in traces and len(traces[key[1]]) < 2:
            traces[key[1]].append(row)
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept, traces


class SimhashClusteringTest(unittest.TestCase):
    def check_graph(self, hashes, edges):
        rows = [{"id": i, "text": f"payload-{i}", HashKeys.simhash: str(h)} for i, h in enumerate(hashes)]
        dataset = Dataset.from_list(rows)
        for show_num in (0, 1, 3, 100):
            with self.subTest(show_num=show_num):
                backend = SimpleNamespace(find_all=lambda *args: edges)
                with patch.object(module, "simhash", backend):
                    result, traces = module.DocumentSimhashDeduplicator().process(dataset, show_num=show_num)
                expected_rows, expected_traces = graph_oracle(rows, edges, show_num)
                self.assertEqual(result.to_list(), expected_rows)
                self.assertEqual(traces, expected_traces)
        self.assertEqual(dataset.to_list(), rows)

    def test_transitive_matches_duplicates_and_isolates(self):
        large = 2**64 - 1
        self.check_graph([3, 1, 9, 2, 1, large, large, 8, 7], [(1, 2), (2, 3), (7, 8), (8, 7), (9, 9)])

    def test_hamming_matches(self):
        hashes = [0, 1, 3, 7, 15, 2**63, 2**64 - 1, 1]
        unique = list(dict.fromkeys(hashes))
        edges = [(x, y) for i, x in enumerate(unique) for y in unique[i + 1 :] if (x ^ y).bit_count() <= 1]
        self.check_graph(hashes, edges)

    def test_random_graphs_and_edge_order(self):
        rng = random.Random(29)
        for case in range(12):
            hashes = [rng.getrandbits(64) for _ in range(40)]
            rows = rng.choices(hashes, k=70) + hashes
            edges = [tuple(rng.sample(hashes, 2)) for _ in range(case * 5)]
            with self.subTest(case=case):
                self.check_graph(rows, edges)
                self.check_graph(rows, [(y, x) for x, y in reversed(edges)])

    def test_long_chain(self):
        self.check_graph(list(range(2000, 0, -1)), [(i, i + 1) for i in range(1, 2000)])

    def test_empty_and_singleton_do_not_query(self):
        for rows in ([], [{"text": "only", HashKeys.simhash: "1"}]):
            dataset = Dataset.from_list(rows)
            with patch.object(module, "simhash", None):
                result, traces = module.DocumentSimhashDeduplicator().process(dataset, show_num=1)
            self.assertEqual(result.to_list(), rows)
            self.assertEqual(traces, {})
