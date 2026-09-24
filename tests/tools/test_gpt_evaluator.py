import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

ROOT = Path(__file__).resolve().parents[2]

OK_A = "review\n8,6"
OK_B = "review\n4,10"
OK_C = "review\n6,6"
ZERO = "review\n0,0"


class FakeWriter:
    def __init__(self):
        self.rows = []
        self.closed = False

    def write(self, row):
        self.rows.append(row)

    def close(self):
        self.closed = True


class GPTEvaluatorTest(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        spec = importlib.util.spec_from_file_location(
            "gpt_evaluator", ROOT / "tools/evaluator/gpt_eval/gpt_evaluator.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.module)
        except ImportError as e:
            self.skipTest(f"gpt_evaluator dependencies are not installed: {e}")

    def run_evaluator(self, reviews):
        """Runs GPTEvaluator.run over len(reviews) // 2 questions with the judge replies given in ``reviews``."""
        n = len(reviews) // 2
        evaluator = object.__new__(self.module.GPTEvaluator)
        evaluator.questions = [{"question_id": i, "category": "general", "text": f"q{i}"} for i in range(n)]
        evaluator.answers = [{"model_id": "target", "text": f"a{i}"} for i in range(n)]
        evaluator.baseline = [{"model_id": "baseline", "text": f"b{i}"} for i in range(n)]
        evaluator.prompt_templates = {
            "general": {"system_prompt": "s", "prompt_template": "{question} {answer_1} {answer_2}", "defaults": {}}
        }
        evaluator.reviewers = {"general": {"metadata": {"temperature": 0, "max_tokens": 8, "model": "m"}}}
        evaluator.worker_num = 1
        evaluator.max_retry = 1
        evaluator.debug = False
        evaluator.result_writer = FakeWriter()

        class FakePool:
            def __init__(self, processes):
                pass

            def map(self, func, requests):
                assert len(requests) == len(reviews)
                return list(reviews)

        out = io.StringIO()
        with patch.object(self.module, "Pool", FakePool), contextlib.redirect_stdout(out):
            evaluator.run()
        rows = evaluator.result_writer.rows
        per_question = [r for r in rows if "question_id" in r]
        summary = [r for r in rows if "question_id" not in r]
        return per_question, summary, out.getvalue()

    def test_scored_comparisons_are_averaged(self):
        _, summary, _ = self.run_evaluator([OK_A, OK_B, OK_C, OK_C])
        self.assertEqual(summary, [{"target": 7.5, "baseline": 5.5}])

    def test_failed_comparison_count_is_printed(self):
        _, _, out = self.run_evaluator([OK_A, OK_B, OK_C, OK_C])
        self.assertIn("Failed comparisons (left out of the averages): 0/2", out)

    def test_genuine_zero_scores_are_still_counted(self):
        _, summary, _ = self.run_evaluator([OK_A, OK_B, ZERO, ZERO])
        self.assertEqual(summary, [{"target": 4.5, "baseline": 2.5}])

    def test_failed_judge_call_is_left_out_and_counted(self):
        _, summary, out = self.run_evaluator([OK_A, OK_B, "error", OK_C])
        self.assertEqual(summary, [{"target": 9.0, "baseline": 5.0}])
        self.assertIn("1/2", out)

    def test_unparseable_reply_is_left_out_and_counted(self):
        _, summary, out = self.run_evaluator([OK_A, OK_B, "no scores here", OK_C])
        self.assertEqual(summary, [{"target": 9.0, "baseline": 5.0}])
        self.assertIn("1/2", out)

    def test_failed_comparison_keeps_a_null_score_in_the_rows(self):
        rows, _, _ = self.run_evaluator([OK_A, OK_B, "error", OK_C])
        self.assertEqual(rows[0]["score1"], [8.0, 6.0])
        self.assertIsNone(rows[1]["score1"])
        self.assertEqual(rows[1]["score2"], [6.0, 6.0])

    def test_all_comparisons_failed_reports_no_averages(self):
        rows, summary, out = self.run_evaluator(["error", "error", "error", "error"])
        self.assertEqual(summary, [])
        self.assertEqual(len(rows), 2)
        self.assertIn("2/2", out)


if __name__ == "__main__":
    unittest.main()
