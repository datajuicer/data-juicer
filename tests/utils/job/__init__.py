"""Fixtures shared by the job utility tests."""

import unittest
from unittest.mock import patch


def isolate_job_logging():
    """Keep JobUtils construction from replacing the runner's global log sinks."""
    for method in ("remove", "add"):
        patcher = patch(f"data_juicer.utils.job.common.logger.{method}")
        unittest.addModuleCleanup(patcher.stop)
        patcher.start()
