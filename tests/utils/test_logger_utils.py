import os
import unittest
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


if __name__ == '__main__':
    unittest.main()
