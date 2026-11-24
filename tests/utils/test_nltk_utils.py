import io
import os
import pickle
import tempfile
import unittest

import nltk

import data_juicer.utils.nltk_utils as nltk_utils
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class NLTKUtilsTest(DataJuicerTestCaseBase):
    def _with_nltk_path(self, paths, body):
        old_path = list(nltk.data.path)
        try:
            nltk.data.path[:] = list(paths)
            return body(nltk)
        finally:
            nltk.data.path[:] = old_path

    def test_ensure_nltk_resource_maps_problem_path_to_existing_file(self):
        with tempfile.TemporaryDirectory() as nltk_data_dir:
            punkt = os.path.join(nltk_data_dir, "tokenizers", "punkt", "english.pickle")
            os.makedirs(os.path.dirname(punkt), exist_ok=True)
            with open(punkt, "wb") as f:
                pickle.dump({"tokenizer": "ok"}, f)

            def body(_nltk):
                self.assertTrue(nltk_utils.ensure_nltk_resource("tokenizers/punkt_tab/english"))
                self.assertFalse(nltk_utils.ensure_nltk_resource("tokenizers/missing"))

            self._with_nltk_path([nltk_data_dir], body)

    def test_clean_nltk_cache_handles_full_clean_reset_and_empty_paths(self):
        with tempfile.TemporaryDirectory() as nltk_data_dir:
            for subdir in ["tokenizers", "taggers", "chunkers", "corpora", "stemmers"]:
                subdir_path = os.path.join(nltk_data_dir, subdir)
                os.makedirs(subdir_path)
                with open(os.path.join(subdir_path, "cached.bin"), "w") as f:
                    f.write("old")

            def clean_all(_nltk):
                nltk_utils.clean_nltk_cache()

            self._with_nltk_path([nltk_data_dir, os.path.join(nltk_data_dir, "missing")], clean_all)

            for subdir in ["tokenizers", "taggers", "chunkers", "corpora", "stemmers"]:
                subdir_path = os.path.join(nltk_data_dir, subdir)
                self.assertTrue(os.path.isdir(subdir_path))
                self.assertFalse(os.path.exists(os.path.join(subdir_path, "cached.bin")))

            keep_dir = os.path.join(nltk_data_dir, "tokenizers")
            keep_file = os.path.join(keep_dir, "keep.bin")
            with open(keep_file, "w") as f:
                f.write("keep")

            def clean_selected(_nltk):
                nltk_utils.clean_nltk_cache(packages=["punkt"])

            self._with_nltk_path([nltk_data_dir], clean_selected)
            self.assertTrue(os.path.exists(keep_file))

            def reset(_nltk):
                nltk_utils.clean_nltk_cache(complete_reset=True)

            self._with_nltk_path([nltk_data_dir], reset)
            self.assertTrue(os.path.isdir(nltk_data_dir))
            self.assertFalse(os.path.exists(keep_file))

            def empty(_nltk):
                nltk_utils.clean_nltk_cache()

            self._with_nltk_path([], empty)

    def test_patch_nltk_pickle_security_extends_allowlist(self):
        old_loader_marker = object()
        old_allowed_marker = object()
        old_loader = getattr(nltk.data, "restricted_pickle_load", old_loader_marker)
        old_allowed = getattr(nltk.data, "ALLOWED_PICKLE_CLASSES", old_allowed_marker)
        try:
            nltk.data.restricted_pickle_load = lambda payload: "restricted"
            nltk.data.ALLOWED_PICKLE_CLASSES = set()

            self.assertTrue(nltk_utils.patch_nltk_pickle_security())

            # The patch must extend NLTK's allowlist with the model classes
            # Data-Juicer relies on.
            self.assertIn(
                "nltk.tokenize.punkt.PunktSentenceTokenizer",
                nltk.data.ALLOWED_PICKLE_CLASSES,
            )

            # The patched loader accepts both bytes and file-like objects for a
            # benign payload. NOTE: we intentionally do NOT assert on the
            # loader's security boundary (i.e. whether arbitrary payloads are
            # accepted) so that this test does not turn the current loading
            # behavior into a protected contract.
            payload = pickle.dumps({"ok": True})
            self.assertEqual(nltk.data.restricted_pickle_load(io.BytesIO(payload)), {"ok": True})
            self.assertEqual(nltk.data.restricted_pickle_load(payload), {"ok": True})
        finally:
            if old_loader is old_loader_marker:
                delattr(nltk.data, "restricted_pickle_load")
            else:
                nltk.data.restricted_pickle_load = old_loader
            if old_allowed is old_allowed_marker:
                delattr(nltk.data, "ALLOWED_PICKLE_CLASSES")
            else:
                nltk.data.ALLOWED_PICKLE_CLASSES = old_allowed

    def test_create_physical_resource_alias_success_and_missing_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_source = os.path.join(tmpdir, "missing.bin")
            self.assertFalse(
                nltk_utils.create_physical_resource_alias(
                    missing_source,
                    os.path.join(tmpdir, "alias", "missing.bin"),
                )
            )

            source = os.path.join(tmpdir, "source.bin")
            with open(source, "w") as f:
                f.write("model")

            alias = os.path.join(tmpdir, "alias", "source.bin")
            self.assertTrue(nltk_utils.create_physical_resource_alias(source, alias))
            with open(alias) as f:
                self.assertEqual(f.read(), "model")

    def test_create_physical_resource_alias_replaces_existing_file_and_rejects_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "source.bin")
            with open(source, "w") as f:
                f.write("new")

            alias = os.path.join(tmpdir, "alias.bin")
            with open(alias, "w") as f:
                f.write("old")

            self.assertTrue(nltk_utils.create_physical_resource_alias(source, alias))
            with open(alias) as f:
                self.assertEqual(f.read(), "new")

            blocked_alias = os.path.join(tmpdir, "blocked")
            os.makedirs(blocked_alias)
            self.assertFalse(nltk_utils.create_physical_resource_alias(source, blocked_alias))

    def test_setup_resource_aliases_no_source_is_successful_noop(self):
        with tempfile.TemporaryDirectory() as nltk_data_dir:
            def body(_nltk):
                self.assertTrue(nltk_utils.setup_resource_aliases())

            self._with_nltk_path([nltk_data_dir], body)


if __name__ == "__main__":
    unittest.main()
