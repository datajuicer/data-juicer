import os
import sys
import unittest
import unittest.mock
import jsonlines
import regex as re
from loguru import logger

import data_juicer.utils.logger_utils
from data_juicer.utils.logger_utils import setup_logger, get_log_file_path, make_log_summarization

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

@unittest.skip('This case could break the logger.')
class LoggerUtilsTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.temp_output_path = 'tmp/test_logger_utils/'
        data_juicer.utils.logger_utils.LOGGER_SETUP = False

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')
        super().tearDown()

    def get_log_messages(self, content):
        lines = content.strip().split('\n')
        messages = []
        for line in lines:
            line = line.strip()
            if line:
                if ' - ' in line:
                    messages.append(' - '.join(line.strip().split(' - ')[1:]))
                else:
                    messages.append(line)
        return messages

    def test_logger_utils(self):
        setup_logger(self.temp_output_path)
        logger.info('info test')
        logger.warning('warning test')
        logger.error('error test')
        logger.debug('debug test')
        print('extra normal info')
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'log.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'log_ERROR.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'log_WARNING.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'log_DEBUG.txt')))
        with open(os.path.join(self.temp_output_path, 'log.txt'), 'r') as f:
            content = f.read()
            messages = self.get_log_messages(content)
            self.assertEqual(len(messages), 5)
            self.assertEqual(messages, ['info test', 'warning test', 'error test', 'debug test', 'extra normal info'])

        with jsonlines.open(os.path.join(self.temp_output_path, 'log_ERROR.txt'), 'r') as reader:
            messages = [line for line in reader]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]['record']['message'], 'error test')
        with jsonlines.open(os.path.join(self.temp_output_path, 'log_WARNING.txt'), 'r') as reader:
            messages = [line for line in reader]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]['record']['message'], 'warning test')
        with jsonlines.open(os.path.join(self.temp_output_path, 'log_DEBUG.txt'), 'r') as reader:
            messages = [line for line in reader]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]['record']['message'], 'debug test')

        self.assertEqual(get_log_file_path(), os.path.abspath(os.path.join(self.temp_output_path, 'log.txt')))

        # setup again
        setup_logger(os.path.join(self.temp_output_path, 'second_setup'))
        logger.info('info test')
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'log.txt')))
        self.assertFalse(os.path.exists(os.path.join(self.temp_output_path, 'second_setup', 'log.txt')))

    def test_make_log_summarization(self):
        setup_logger(self.temp_output_path)
        logger.info('normal log 1')
        logger.error(f'An error occurred in fake_op_1 when processing sample '
                     f'"fake_sample_1" -- {ModuleNotFoundError}: err msg 1 -- '
                     f'detailed error msg 1')
        logger.info('normal log 2')
        logger.warning('warning message')
        logger.info('normal log 3')
        logger.error(f'An error occurred in fake_op_2 when processing sample '
                     f'"fake_sample_1" -- {ValueError}: err msg 1 -- detailed '
                     f'error msg 1')
        logger.info('normal log 4')
        logger.error(f'An error occurred in fake_op_3 when processing sample '
                     f'"fake_sample_3" -- {ModuleNotFoundError}: err msg 3 -- '
                     f'detailed error msg 3')
        logger.info('normal log 5')

        make_log_summarization()
        with open(os.path.join(self.temp_output_path, 'log.txt')) as f:
            content = f.read()
            # find start words
            self.assertIn('Processing finished with:', content)
            # find number of warnings and errors
            warn_num = re.findall(r'Warnings: (\d+)', content)
            self.assertEqual(len(warn_num), 1)
            self.assertEqual(int(warn_num[0]), 1)
            err_num = re.findall(r'Errors: (\d+)', content)
            self.assertEqual(len(err_num), 1)
            self.assertEqual(int(err_num[0]), 3)
            # find table head content
            self.assertIn('OP/Method', content)
            self.assertIn('Error Type', content)
            self.assertIn('Error Message', content)
            self.assertIn('Error Count', content)
            # find end words
            log_fn = re.findall(r'Error/Warning details can be found in the log file \[(.+)\] and its related log files\.', content)
            self.assertEqual(len(log_fn), 1)
            self.assertEqual(log_fn[0], os.path.abspath(os.path.join(self.temp_output_path, 'log.txt')))
            self.assertTrue(os.path.exists(log_fn[0]))


class IsNotebookTest(DataJuicerTestCaseBase):
    """Tests for is_notebook() without breaking the global logger."""

    def test_is_notebook_returns_false_outside_notebook(self):
        """In a normal test environment, is_notebook() should return False."""
        from data_juicer.utils.logger_utils import is_notebook
        self.assertFalse(is_notebook())

    def test_is_notebook_returns_false_when_get_ipython_is_none(self):
        """When get_ipython is None, is_notebook returns False."""
        from data_juicer.utils import logger_utils
        from data_juicer.utils.logger_utils import is_notebook

        original = logger_utils.get_ipython
        try:
            logger_utils.get_ipython = None
            self.assertFalse(is_notebook())
        finally:
            logger_utils.get_ipython = original

    def test_is_notebook_exception_returns_false(self):
        """When get_ipython() raises, is_notebook returns False."""
        from data_juicer.utils import logger_utils
        from data_juicer.utils.logger_utils import is_notebook

        original = logger_utils.get_ipython

        def exploding():
            raise RuntimeError("boom")

        try:
            logger_utils.get_ipython = exploding
            self.assertFalse(is_notebook())
        finally:
            logger_utils.get_ipython = original


class StreamToLoguruTest(DataJuicerTestCaseBase):
    """Tests for StreamToLoguru without modifying global sys.stdout/stderr."""

    def test_write_non_caller_module(self):
        """Writing from a non-caller module goes through raw info path."""
        from data_juicer.utils.logger_utils import StreamToLoguru
        stream = StreamToLoguru(level="INFO", caller_names=("nonexistent_module",))
        # Should not raise
        stream.write("hello from test\n")

    def test_getvalue_returns_buffer_content(self):
        """getvalue() returns accumulated buffer content."""
        from data_juicer.utils.logger_utils import StreamToLoguru
        stream = StreamToLoguru()
        stream.write("first ")
        stream.write("second")
        value = stream.getvalue()
        self.assertIn("first ", value)
        self.assertIn("second", value)

    def test_flush_does_not_raise(self):
        """flush() should not raise."""
        from data_juicer.utils.logger_utils import StreamToLoguru
        stream = StreamToLoguru()
        stream.flush()  # Should not raise

    def test_isatty_returns_false(self):
        """isatty() should return False."""
        from data_juicer.utils.logger_utils import StreamToLoguru
        stream = StreamToLoguru()
        self.assertFalse(stream.isatty())

    def test_buffer_truncate_on_write(self):
        """Buffer content is truncated to BUFFER_SIZE after write."""
        from data_juicer.utils.logger_utils import StreamToLoguru
        stream = StreamToLoguru()
        stream.BUFFER_SIZE = 10  # Small buffer for test
        stream.write("a" * 100)
        # truncate(10) keeps only the first 10 characters of content
        self.assertEqual(len(stream.buffer.getvalue()), 10)


class HiddenPrintsTest(DataJuicerTestCaseBase):
    """Tests for the HiddenPrints context manager."""

    def test_hidden_prints_suppresses_stdout(self):
        """HiddenPrints redirects stdout to devnull."""
        from data_juicer.utils.logger_utils import HiddenPrints
        import io

        original_stdout = sys.stdout
        with HiddenPrints():
            # stdout should be redirected to devnull
            self.assertNotEqual(sys.stdout, original_stdout)
            print("this should be suppressed")
        # After exiting, stdout should be restored
        self.assertIs(sys.stdout, original_stdout)

    def test_hidden_prints_restores_stdout_on_exception(self):
        """HiddenPrints restores stdout even if exception occurs inside."""
        from data_juicer.utils.logger_utils import HiddenPrints

        original_stdout = sys.stdout
        try:
            with HiddenPrints():
                raise ValueError("test error")
        except ValueError:
            pass
        self.assertIs(sys.stdout, original_stdout)


class GetCallerNameTest(DataJuicerTestCaseBase):
    """Tests for get_caller_name()."""

    def test_get_caller_name_depth_zero(self):
        """get_caller_name(0) returns the caller's module name."""
        from data_juicer.utils.logger_utils import get_caller_name
        name = get_caller_name(depth=0)
        # Should be this test module's name
        self.assertIn("test_logger_utils", name)


class RedirectSysOutputTest(DataJuicerTestCaseBase):
    """Tests for redirect_sys_output() — verifies notebook guard."""

    def test_redirect_noop_in_notebook(self):
        """redirect_sys_output does nothing when is_notebook() returns True."""
        from data_juicer.utils import logger_utils
        from data_juicer.utils.logger_utils import redirect_sys_output

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with unittest.mock.patch.object(
            logger_utils, "is_notebook", return_value=True
        ):
            redirect_sys_output("INFO")

        # stdout/stderr should be unchanged
        self.assertIs(sys.stdout, original_stdout)
        self.assertIs(sys.stderr, original_stderr)


class MakeLogSummarizationTest(DataJuicerTestCaseBase):
    """Tests for make_log_summarization edge cases."""

    def test_make_log_summarization_no_log_file(self):
        """make_log_summarization returns early when no log file exists."""
        with unittest.mock.patch(
            "data_juicer.utils.logger_utils.get_log_file_path",
            return_value=None,
        ):
            # Should return None (early exit) without error
            result = make_log_summarization()
            self.assertIsNone(result)


class SetupLoggerIsolatedTest(DataJuicerTestCaseBase):
    """Tests for setup_logger branches that don't break the global logger.

    These tests reset LOGGER_SETUP before and after each test to avoid
    contaminating other tests.
    """

    def setUp(self):
        super().setUp()
        self.tmp_dir = os.path.join('tmp', 'test_setup_logger_isolated')
        os.makedirs(self.tmp_dir, exist_ok=True)
        # Save original state
        self._orig_setup = data_juicer.utils.logger_utils.LOGGER_SETUP
        self._orig_handlers = dict(logger._core.handlers)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

    def tearDown(self):
        # Restore global state
        data_juicer.utils.logger_utils.LOGGER_SETUP = self._orig_setup
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        # Remove any handlers we added
        current_ids = set(logger._core.handlers.keys())
        orig_ids = set(self._orig_handlers.keys())
        for handler_id in current_ids - orig_ids:
            try:
                logger.remove(handler_id)
            except ValueError:
                pass
        if os.path.exists(self.tmp_dir):
            os.system(f'rm -rf {self.tmp_dir}')
        super().tearDown()

    def test_setup_logger_override_mode_removes_existing_file(self):
        """setup_logger in override mode removes an existing log file."""
        log_file = os.path.join(self.tmp_dir, 'log.txt')
        with open(log_file, 'w') as f:
            f.write("old content")
        self.assertTrue(os.path.exists(log_file))

        data_juicer.utils.logger_utils.LOGGER_SETUP = False
        setup_logger(
            self.tmp_dir, filename='log.txt', mode='o', redirect=False,
        )
        # The old file should have been removed (a new one may be created
        # by the file sink, but the old content should be gone)
        if os.path.exists(log_file):
            with open(log_file) as f:
                self.assertNotIn("old content", f.read())

    def test_setup_logger_redirect_auto(self):
        """setup_logger with redirect='auto' resolves based on is_notebook."""
        data_juicer.utils.logger_utils.LOGGER_SETUP = False
        # In non-notebook env, redirect='auto' should redirect
        setup_logger(
            self.tmp_dir, filename='log_auto.txt', redirect='auto',
        )
        # LOGGER_SETUP should be True after setup
        self.assertTrue(data_juicer.utils.logger_utils.LOGGER_SETUP)

    def test_setup_logger_skips_when_already_setup(self):
        """setup_logger is a no-op when LOGGER_SETUP is already True."""
        data_juicer.utils.logger_utils.LOGGER_SETUP = True
        handlers_before = set(logger._core.handlers.keys())
        setup_logger(self.tmp_dir, filename='log_skip.txt', redirect=False)
        handlers_after = set(logger._core.handlers.keys())
        # No new handlers should be added
        self.assertEqual(handlers_before, handlers_after)


if __name__ == '__main__':
    unittest.main()
