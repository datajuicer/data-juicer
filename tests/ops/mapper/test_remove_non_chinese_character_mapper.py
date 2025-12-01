import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.remove_non_chinese_character_mapper import \
    RemoveNonChineseCharacterlMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RemoveNonChineseCharacterlMapperrTest(DataJuicerTestCaseBase):

    def setUp(self, keep_alphabet=True, keep_number=True, keep_punc=True):
        super().setUp()
        self.op = RemoveNonChineseCharacterlMapper(keep_alphabet, keep_number,
                                                   keep_punc)

    def _run_remove_non_chinese_character(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_remove_non_chinese_character(self):

        samples = [{
            'text': '特殊的康熙部首或者扩展部首会被去除，⼏几⺇',
            'target': '特殊的康熙部首或者扩展部首会被去除几'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁'
        }, {
            'text': 'ftp://exam匹配ple汉字ma-niè包括rdas繁體字h@hqbchd.ckdhnfes.cds',
            'target': '匹配汉字包括繁體字'
        }, {
            'text': '👊    所有的非汉字a44sh都12@46h会被*&……*qb^4525去掉',
            'target': '所有的非汉字都会被去掉'
        }]
        self.setUp(False, False, False)
        self._run_remove_non_chinese_character(samples)

    def test_remove_non_chinese_character2(self):

        samples = [{
            'text': '特殊的康熙部首或者扩展部首会被去除，⼏几⺇',
            'target': '特殊的康熙部首或者扩展部首会被去除几'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁dasoidhaofgom'
        }, {
            'text': 'ftp://exam匹配ple汉字ma-niè包括rdas繁體字h@hqbchd.ckdhnfes.cds',
            'target': 'ftpexam匹配ple汉字mani包括rdas繁體字hhqbchdckdhnfescds'
        }, {
            'text': '👊    所有的非汉字a44sh都12@46h会被*&……*qb^4525去掉',
            'target': '所有的非汉字ash都h会被qb去掉'
        }]
        self.setUp(True, False, False)
        self._run_remove_non_chinese_character(samples)

    def test_remove_non_chinese_character3(self):

        samples = [{
            'text': '特殊的康熙部首或者扩展部首会被去除，⼏几⺇',
            'target': '特殊的康熙部首或者扩展部首会被去除几'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁126445'
        }, {
            'text': 'f://exam匹配ple汉12字ma-niè包括rdas繁88體字h@hqbchd.ds1',
            'target': '匹配汉12字包括繁88體字1'
        }, {
            'text': '👊    所有的非汉字a44sh都12@46h会被*&……*qb^4525去掉',
            'target': '所有的非汉字44都1246会被4525去掉'
        }]
        self.setUp(False, True, False)
        self._run_remove_non_chinese_character(samples)

    def test_remove_non_chinese_character4(self):

        samples = [{
            'text': '特殊的康熙部首或者扩展部首会被去除，⼏几⺇',
            'target': '特殊的康熙部首或者扩展部首会被去除，几'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁.'
        }, {
            'text': 'f://exam匹配ple汉12字ma-niè包括rdas繁88體字h@hqbchd.ds1',
            'target': '//匹配汉字-包括繁體字.'
        }, {
            'text': '👊    所有的非汉字a44sh都12@46h会被*&……*qb^4525去掉',
            'target': '    所有的非汉字都会被*&*去掉'
        }]
        self.setUp(False, False, True)
        self._run_remove_non_chinese_character(samples)

    def test_remove_non_chinese_character5(self):

        samples = [{
            'text': '特殊的康熙部首或者扩展部首会被去除，⼏几⺇',
            'target': '特殊的康熙部首或者扩展部首会被去除几'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁dasoidhao1264fg45om'
        }, {
            'text': 'f://exam匹配ple汉12字ma-niè包括rdas繁88體字h@hqbchd.ds1',
            'target': 'fexam匹配ple汉12字mani包括rdas繁88體字hhqbchdds1'
        }, {
            'text': '👊    所有的非汉字a44sh都12@46h会被*&……*qb^4525去掉',
            'target': '所有的非汉字a44sh都1246h会被qb4525去掉'
        }]
        self.setUp(True, True, False)
        self._run_remove_non_chinese_character(samples)

    def test_remove_non_chinese_character6(self):

        samples = [{
            'text': '特殊的康熙部首或者扩展部首会被去除，⼏几⺇',
            'target': '特殊的康熙部首或者扩展部首会被去除，几'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁dasoidhaofg.om'
        }, {
            'text': 'f://exam匹配ple汉12字ma-niè包括rdas繁88體字h@hqbchd.ds1',
            'target': 'f//exam匹配ple汉字ma-ni包括rdas繁體字hhqbchd.ds'
        }, {
            'text': '👊    所有的非汉字a44sh都12@46h会被*&……*qb^4525去掉',
            'target': '    所有的非汉字ash都h会被*&*qb去掉'
        }]
        self.setUp(True, False, True)
        self._run_remove_non_chinese_character(samples)

    def test_remove_non_chinese_character7(self):

        samples = [{
            'text': '特殊的康熙部首或者扩展部首会被去除，⼏几⺇',
            'target': '特殊的康熙部首或者扩展部首会被去除，几'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁1264.45'
        }, {
            'text': 'f://exam匹配ple汉12字ma-niè包括rdas繁88體字h@hqbchd.ds1',
            'target': '//匹配汉12字-包括繁88體字.1'
        }, {
            'text': '👊    所有的非汉字a44sh都12@46h会被*&……*qb^4525去掉',
            'target': '    所有的非汉字44都1246会被*&*4525去掉'
        }]
        self.setUp(False, True, True)
        self._run_remove_non_chinese_character(samples)

    def test_remove_non_chinese_character8(self):

        samples = [{
            'text': '特殊的康熙部首或者扩展部首会被去除，⼏几⺇',
            'target': '特殊的康熙部首或者扩展部首会被去除，几'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁dasoidhao1264fg.45om'
        }, {
            'text': 'f://exam匹配ple汉12字ma-niè包括rdas繁88體字h@hqbchd.ds1',
            'target': 'f//exam匹配ple汉12字ma-ni包括rdas繁88體字hhqbchd.ds1'
        }, {
            'text': '👊    所有的非汉字a44sh都12@46h会被*&……*qb^4525去掉',
            'target': '    所有的非汉字a44sh都1246h会被*&*qb4525去掉'
        }]
        self.setUp(True, True, True)
        self._run_remove_non_chinese_character(samples)


if __name__ == '__main__':
    unittest.main()
