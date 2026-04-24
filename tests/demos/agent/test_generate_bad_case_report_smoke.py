"""Smoke test for generate_bad_case_report (no charts, no LLM)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "demos" / "agent" / "scripts" / "generate_bad_case_report.py"

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestGenerateBadCaseReportSmoke(DataJuicerTestCaseBase):
    def test_report_minimal_jsonl_no_charts(self) -> None:
        row = {
            "query": "hello",
            "response": "world",
            "__dj__meta__": {
                "agent_bad_case_tier": "none",
                "agent_request_model": "test-model",
                "agent_pt": "20250101",
            },
            "__dj__stats__": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            inp = tdir / "one.jsonl"
            out = tdir / "out.html"
            line = json.dumps(row, ensure_ascii=False) + "\n"
            inp.write_text(line, encoding="utf-8")
            cmd = [
                sys.executable,
                str(_SCRIPT),
                "--input",
                str(inp),
                "--output",
                str(out),
                "--report-pii-variants",
                "safe",
                "--no-charts",
                "--sample-headlines",
                "0",
                "--drilldown-limit",
                "0",
                "--no-drilldown-export",
            ]
            env = dict(**os.environ)
            env.pop("BAD_CASE_REPORT_LLM", None)
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            html = out.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("sec-charts", html)
            self.assertIn("sec-dialog-metrics", html)

    def test_multi_input_jsonl_concatenates_rows(self) -> None:
        row_a = {
            "query": "a",
            "response": "ra",
            "__dj__meta__": {
                "agent_bad_case_tier": "none",
                "agent_request_model": "m",
                "agent_pt": "20250101",
            },
            "__dj__stats__": {},
        }
        row_b = {
            "query": "b",
            "response": "rb",
            "__dj__meta__": {
                "agent_bad_case_tier": "high_precision",
                "agent_request_model": "m",
                "agent_pt": "20250101",
            },
            "__dj__stats__": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            inp1 = tdir / "part1.jsonl"
            inp2 = tdir / "part2.jsonl"
            out = tdir / "out.html"
            inp1.write_text(json.dumps(row_a, ensure_ascii=False) + "\n", encoding="utf-8")
            inp2.write_text(json.dumps(row_b, ensure_ascii=False) + "\n", encoding="utf-8")
            cmd = [
                sys.executable,
                str(_SCRIPT),
                "--input",
                str(inp1),
                "--input",
                str(inp2),
                "--output",
                str(out),
                "--report-pii-variants",
                "safe",
                "--no-charts",
                "--sample-headlines",
                "0",
                "--drilldown-limit",
                "0",
                "--no-drilldown-export",
            ]
            env = dict(**os.environ)
            env.pop("BAD_CASE_REPORT_LLM", None)
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            html = out.read_text(encoding="utf-8")
            self.assertIn("sec-data-provenance", html)
            self.assertIn("2</strong> 个 jsonl", html)
            self.assertIn(str(inp1.resolve()), html)
            self.assertIn(str(inp2.resolve()), html)

    def test_safe_report_omits_pii_headlines_audit_report_keeps(self) -> None:
        rows = [
            {
                "query": "q1",
                "response": "r1",
                "__dj__meta__": {
                    "agent_bad_case_tier": "high_precision",
                    "agent_request_id": "rid-1",
                    "agent_insight_llm": {"headline": "VISIBLE_HEADLINE"},
                },
                "__dj__stats__": {},
            },
            {
                "query": "q2",
                "response": "r2",
                "__dj__meta__": {
                    "agent_bad_case_tier": "high_precision",
                    "agent_request_id": "rid-2",
                    "agent_insight_llm": {"headline": "HIDDEN_HEADLINE"},
                    "pii_llm_suspect": {
                        "suspected": [{"field": "query", "category": "t", "evidence": "x"}],
                    },
                },
                "__dj__stats__": {},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            inp = tdir / "two.jsonl"
            out = tdir / "out.html"
            inp.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                str(_SCRIPT),
                "--input",
                str(inp),
                "--output",
                str(out),
                "--report-pii-variants",
                "both",
                "--no-charts",
                "--sample-headlines",
                "10",
                "--drilldown-limit",
                "0",
                "--no-drilldown-export",
                "--no-insight-semantic-cluster",
            ]
            env = dict(**os.environ)
            env.pop("BAD_CASE_REPORT_LLM", None)
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            html_safe = out.read_text(encoding="utf-8")
            audit_path = out.with_name(out.stem + "_pii_audit" + out.suffix)
            self.assertTrue(audit_path.is_file(), proc.stdout)
            html_audit = audit_path.read_text(encoding="utf-8")
            self.assertIn("VISIBLE_HEADLINE", html_safe)
            self.assertNotIn("HIDDEN_HEADLINE", html_safe)
            self.assertIn("report-variant-banner", html_safe)
            self.assertIn("HIDDEN_HEADLINE", html_audit)
            self.assertIn("report-variant-banner", html_audit)

    def test_skill_insight_macro_splits_cn_punctuation_without_reprocess(self) -> None:
        row = {
            "query": "q",
            "response": "r",
            "__dj__meta__": {
                "agent_bad_case_tier": "none",
                "agent_request_model": "m",
                "agent_skill_insights": ["alpha_label，beta_label"],
            },
            "__dj__stats__": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            inp = tdir / "si.jsonl"
            out = tdir / "out.html"
            inp.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            cmd = [
                sys.executable,
                str(_SCRIPT),
                "--input",
                str(inp),
                "--output",
                str(out),
                "--report-pii-variants",
                "safe",
                "--no-charts",
                "--sample-headlines",
                "0",
                "--drilldown-limit",
                "0",
                "--no-drilldown-export",
            ]
            env = dict(**os.environ)
            env.pop("BAD_CASE_REPORT_LLM", None)
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            html = out.read_text(encoding="utf-8")
            self.assertIn("alpha_label", html)
            self.assertIn("beta_label", html)


if __name__ == "__main__":
    unittest.main()
