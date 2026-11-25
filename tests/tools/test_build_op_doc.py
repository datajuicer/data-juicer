import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

ROOT = Path(__file__).resolve().parents[2]


class BuildOpDocTest(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        spec = importlib.util.spec_from_file_location("build_op_doc", ROOT / ".pre-commit-hooks/build_op_doc.py")
        self.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.builder)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ops = self.root / "ops/pipeline"
        self.docs = self.root / "docs/operators/pipeline"
        self.ops.mkdir(parents=True)
        self.docs.mkdir(parents=True)
        (self.root / "format").mkdir()
        # Redirect filesystem inputs, while using the real scanner and renderer.
        paths = patch.multiple(
            self.builder,
            ROOT=self.root,
            DOC_PATH=str(ROOT / "docs/Operators.md"),
            DOC_OP_PATH=self.root / "docs/operators",
            OP_CODE_PREFIX=str(self.root / "ops"),
            FORMATTER_CODE_PREFIX=str(self.root / "format"),
            OP_TEST_PREFIX=str(self.root / "tests/ops"),
        )
        paths.start()
        self.addCleanup(paths.stop)

    def test_recipe_names_are_independent_of_source_filenames(self):
        cases = [
            ("literal_pipeline", "", '@OPERATORS.register_module("recipe_literal")', "recipe_literal"),
            (
                "constant_pipeline",
                'OP_NAME = "recipe_constant"',
                "@OPERATORS.register_module(OP_NAME)",
                "recipe_constant",
            ),
            (
                "named_pipeline",
                'NAME: str = "recipe_named"',
                "@OPERATORS.register_module(module_name=NAME)",
                "recipe_named",
            ),
            ("default_pipeline", "", "@OPERATORS.register_module()", "PublicPipeline"),
        ]
        for stem, assignment, decorator, expected in cases:
            (self.ops / f"{stem}.py").write_text(
                'raise RuntimeError("Metadata collection must never import this module")\n'
                f"{assignment}\n{decorator}\n"
                'class PublicPipeline:\n    """Generate response."""\n',
                encoding="utf-8",
            )
            (self.docs / f"{stem}.md").write_text("# Details\n", encoding="utf-8")

        records, _ = self.builder.get_op_list_from_code()
        actual = {record.name: record.info for record in records}
        self.assertEqual(
            actual,
            {expected: f"[info](operators/pipeline/{stem}.md)" for stem, _, _, expected in cases},
        )

    def test_vllm_recipe_names_links_and_translation_survive_generation(self):
        for kind in ("llm", "vlm"):
            stem = f"{kind}_inference_with_ray_vllm_pipeline"
            (self.ops / f"{stem}.py").write_bytes((ROOT / f"data_juicer/ops/pipeline/{stem}.py").read_bytes())
            (self.docs / f"{stem}.md").write_text("# Details\n", encoding="utf-8")

        old_records, _ = self.builder.parse_op_record_from_current_doc()
        old_by_name = {record.name: record for record in old_records}
        records, _ = self.builder.get_op_list_from_code()
        self.assertEqual(
            {record.name for record in records}, {"llm_ray_vllm_engine_pipeline", "vlm_ray_vllm_engine_pipeline"}
        )
        rendered = self.builder.generate_op_table_section("pipeline", records, old_by_name)
        for kind in ("llm", "vlm"):
            name = f"{kind}_ray_vllm_engine_pipeline"
            self.assertIn(f"| {name} |", rendered)
            self.assertIn(f"[info](operators/pipeline/{kind}_inference_with_ray_vllm_pipeline.md)", rendered)
            self.assertIn(old_by_name[name].desc_zh, rendered)
            self.assertNotIn(f"| {kind}_inference_with_ray_vllm_pipeline |", rendered)

    def test_registration_mentioned_in_docstrings_is_not_an_operator(self):
        (self.ops / "helper.py").write_text(
            'class Helper:\n    """Use OPERATORS.register_module to register an OP."""\n', encoding="utf-8"
        )
        records, counts = self.builder.get_op_list_from_code()
        self.assertEqual(records, [])
        self.assertEqual(counts, {})

    def test_dynamic_names_fail_instead_of_silently_using_filename(self):
        (self.ops / "dynamic_pipeline.py").write_text(
            "OP_NAME = make_name()\n@OPERATORS.register_module(OP_NAME)\n"
            'class PublicPipeline:\n    """Generate response."""\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Cannot statically resolve registered OP name"):
            self.builder.get_op_list_from_code()


if __name__ == "__main__":
    unittest.main()
