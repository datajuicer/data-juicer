import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


@unittest.skipUnless(importlib.util.find_spec("wandb"), "requires the optional W&B dependency")
class HpoEntryPointTest(DataJuicerTestCaseBase):
    def test_import_does_not_start_a_sweep(self):
        import wandb

        with patch.object(wandb, "sweep") as sweep, patch.object(wandb, "agent") as agent:
            with patch.object(sys, "argv", ["sphinx-build"]):
                module = importlib.import_module("data_juicer.tools.hpo.execute_hpo_wandb")
                importlib.reload(module)
                sweep.assert_not_called()
                agent.assert_not_called()
                with self.assertRaisesRegex(ValueError, "--hpo_config"):
                    module.main()
                sweep.assert_not_called()
                agent.assert_not_called()

    def test_cli_passes_search_space_and_trial_limit_to_wandb(self):
        import wandb

        from data_juicer.tools.hpo.execute_hpo_wandb import main

        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "sweep.yaml"
            config.write_text("sweep_name: docs-test\nmetric:\n  name: quality_score\nsweep_max_count: 2\n")
            with patch.object(sys, "argv", ["execute_hpo_wandb.py", "--hpo_config", str(config)]):
                with patch.object(wandb, "sweep", return_value="sweep-id") as sweep:
                    with patch.object(wandb, "agent") as agent:
                        main()
            sweep.assert_called_once_with(
                sweep={"sweep_name": "docs-test", "metric": {"name": "quality_score"}, "sweep_max_count": 2},
                project="docs-test",
            )
            self.assertEqual(agent.call_count, 1)
            self.assertEqual(agent.call_args.args, ("sweep-id",))
            self.assertEqual(agent.call_args.kwargs["count"], 2)
            self.assertTrue(callable(agent.call_args.kwargs["function"]))


if __name__ == "__main__":
    unittest.main()
