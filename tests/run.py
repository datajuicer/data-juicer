# The code is from adapted from
# https://github.com/alibaba/FederatedScope/blob/master/tests/run.py

# Data-Juicer adopts Apache 2.0 license, the original license of this file
# is as follows:
# --------------------------------------------------------
# Copyright (c) Alibaba, Inc. and its affiliates

import argparse
import os
import sys
import unittest
import coverage

# start the coverage immediately
cov = coverage.Coverage(include='data_juicer/**')
cov.start()

from loguru import logger

from data_juicer.utils.unittest_utils import set_clear_model_flag, get_partial_test_cases, set_from_fork_flag

file_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(file_dir)

parser = argparse.ArgumentParser('test runner')
parser.add_argument('--tag', choices=["standalone", "ray"],
                    default="standalone",
                    help="the tag of tests being run")
parser.add_argument('--pattern', default='test_*.py', help='test file pattern')
parser.add_argument('--mode', default='partial',
                    help='test mode. Should be one of the ["partial", '
                         '"regression"]. "partial" means only test on the '
                         'unit tests of the changed files. "regression" means '
                         'test on all unit tests.')
parser.add_argument('--from-fork', default=False, type=bool,
                    help='Whether this unittest is conducted on a forked repo '
                         'instead of a branch in the Data-Juicer repo.')
parser.add_argument('--test_dir',
                    default='tests',
                    help='directory to be tested')
parser.add_argument('--clear_model',
                    default=False,
                    type=bool,
                    help='whether to clear the downloaded models for tests. '
                         'It\'s False in default.')
args = parser.parse_args()

set_clear_model_flag(args.clear_model)
set_from_fork_flag(args.from_fork)

class TaggedTestLoader(unittest.TestLoader):
    def __init__(self, tag="standalone", included_test_files=None):
        super().__init__()
        self.tag = tag
        if isinstance(included_test_files, str):
            included_test_files = [included_test_files]
        self.included_test_files = included_test_files
    
    def loadTestsFromTestCase(self, testCaseClass):
        # set tag to testcase class
        setattr(testCaseClass, 'current_tag', self.tag)
        test_names = self.getTestCaseNames(testCaseClass)
        loaded_suite = self.suiteClass()
        for test_name in test_names:
            test_case = testCaseClass(test_name)
            test_method = getattr(test_case, test_name)
            if self.tag in getattr(test_method, '__test_tags__', ["standalone"]):
                loaded_suite.addTest(test_case)
        return loaded_suite

    def _match_path(self, path, full_path, pattern):
        # override this method to use alternative matching strategy
        match = super()._match_path(path, full_path, pattern)
        if self.included_test_files is not None:
            for included_test_file in self.included_test_files:
                if included_test_file in full_path:
                    return match
            return False
        else:
            return match

def gather_test_cases(test_dir, pattern, tag, mode='partial'):
    test_to_run = unittest.TestSuite()
    partial_test_files = get_partial_test_cases() if mode == 'partial' else None
    test_loader = TaggedTestLoader(tag, included_test_files=partial_test_files)
    discover = test_loader.discover(test_dir, pattern=pattern, top_level_dir=None)
    for suite_discovered in discover:
        print('suite_discovered', suite_discovered)
        for test_suite in suite_discovered:
            print('test_suite', test_suite)
            if isinstance(test_suite, unittest.loader._FailedTest):
                raise test_suite._exception
            for test_case in test_suite:
                logger.info(f'Add test case [{test_case._testMethodName}]'
                            f' from {test_case.__class__.__name__}')
                test_to_run.addTest(test_case)
    return test_to_run


def main():
    global cov
    runner = unittest.TextTestRunner()
    test_suite = gather_test_cases(os.path.abspath(args.test_dir),
                                   args.pattern, args.tag, args.mode)
    logger.info(f'There are {len(test_suite._tests)} test cases to run.')
    res = runner.run(test_suite)

    cov.stop()
    cov.save()
    cov.combine()

    if not res.wasSuccessful():
        exit(1)

    try:
        cov.report(ignore_errors=True)
    except Exception as e:
        logger.error(f'Failed to print coverage report: {e}')

    try:
        cov.html_report(directory=f'coverage_report_{args.tag}', ignore_errors=True)
    except Exception as e:
        logger.error(f'Failed to generate coverage report in html: {e}')


if __name__ == '__main__':
    main()
