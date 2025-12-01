import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.chinese_convert_mapper import ChineseConvertMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ChineseConvertMapperTest(DataJuicerTestCaseBase):

    def setUp(self, mode='s2t'):
        super().setUp()

        self.op = ChineseConvertMapper(mode)

    def _run_chinese_convert(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_s2t(self):

        samples = [{
            'text': '这是几个简体字，会被转换为繁体字',
            'target': '這是幾個簡體字，會被轉換爲繁體字'
        }, {
            'text': '如果本身都是繁體字就不會被轉換',
            'target': '如果本身都是繁體字就不會被轉換'
        }, {
            'text': '试试繁体afadf字$#@#和简体字，以及各123213*&dasd種不同字符数字的组合轉換效果',
            'target': '試試繁體afadf字$#@#和簡體字，以及各123213*&dasd種不同字符數字的組合轉換效果'
        }]
        self.setUp('s2t')
        self._run_chinese_convert(samples)

    def test_t2s(self):

        samples = [{
            'text': '這是幾個繁體字，會被轉換爲簡體字',
            'target': '这是几个繁体字，会被转换为简体字'
        }, {
            'text': '如果本身都是简体字，就不会被转换',
            'target': '如果本身都是简体字，就不会被转换'
        }, {
            'text': '试试繁体afadf字$#@#和简体字，以及各123213*&dasd種不同字符数字的组合轉換效果',
            'target': '试试繁体afadf字$#@#和简体字，以及各123213*&dasd种不同字符数字的组合转换效果'
        }]
        self.setUp('t2s')
        self._run_chinese_convert(samples)

    def test_s2tw(self):

        samples = [{
            'text': '群贤毕至，少长咸集',
            'target': '群賢畢至，少長鹹集'
        }, {
            'text': '为你我用了半年的积蓄，漂洋过海来看你',
            'target': '為你我用了半年的積蓄，漂洋過海來看你'
        }, {
            'text': '米线面粉里面启发吊钩',
            'target': '米線麵粉裡面啟發吊鉤'
        }]
        self.setUp('s2tw')
        self._run_chinese_convert(samples)

    def test_tw2s(self):

        samples = [{
            'text': '群賢畢至，少長鹹集',
            'target': '群贤毕至，少长咸集'
        }, {
            'text': '為你我用了半年的積蓄，漂洋過海來看你',
            'target': '为你我用了半年的积蓄，漂洋过海来看你'
        }, {
            'text': '米線麵粉裡面啟發吊鉤',
            'target': '米线面粉里面启发吊钩'
        }]
        self.setUp('tw2s')
        self._run_chinese_convert(samples)

    def test_s2hk(self):

        samples = [{
            'text': '群贤毕至，少长咸集',
            'target': '羣賢畢至，少長鹹集'
        }, {
            'text': '为你我用了半年的积蓄，漂洋过海来看你',
            'target': '為你我用了半年的積蓄，漂洋過海來看你'
        }, {
            'text': '米线面粉里面启发吊钩',
            'target': '米線麪粉裏面啓發吊鈎'
        }]
        self.setUp('s2hk')
        self._run_chinese_convert(samples)

    def test_hk2s(self):

        samples = [{
            'text': '羣賢畢至，少長鹹集',
            'target': '群贤毕至，少长咸集'
        }, {
            'text': '為你我用了半年的積蓄，漂洋過海來看你',
            'target': '为你我用了半年的积蓄，漂洋过海来看你'
        }, {
            'text': '米線麪粉裏面啓發吊鈎',
            'target': '米线面粉里面启发吊钩'
        }]
        self.setUp('hk2s')
        self._run_chinese_convert(samples)

    def test_s2twp(self):

        samples = [{
            'text': '网络连接异常，请检查信息安全',
            'target': '網路連線異常，請檢查資訊安全'
        }, {
            'text': '今年想去新西兰和马尔代夫旅游',
            'target': '今年想去紐西蘭和馬爾地夫旅遊'
        }, {
            'text': '我打个出租车打到了一辆奔驰，准备在车上吃冰棍和奶酪',
            'target': '我打個計程車打到了一輛賓士，準備在車上吃冰棒和乳酪'
        }]
        self.setUp('s2twp')
        self._run_chinese_convert(samples)

    def test_tw2sp(self):

        samples = [{
            'text': '網路連線異常，請檢查資訊保安',
            'target': '网络连接异常，请检查信息保安'
        }, {
            'text': '今年想去紐西蘭和馬爾地夫旅遊',
            'target': '今年想去新西兰和马尔代夫旅游'
        }, {
            'text': '我打個計程車打到了一輛賓士，準備在車上吃冰棒和乳酪',
            'target': '我打个出租车打到了一辆奔驰，准备在车上吃冰棍和奶酪'
        }]
        self.setUp('tw2sp')
        self._run_chinese_convert(samples)

    def test_t2tw(self):

        samples = [{
            'text': '羣賢畢至，少長鹹集',
            'target': '群賢畢至，少長鹹集'
        }, {
            'text': '爲你我用了半年的積蓄，漂洋過海來看你',
            'target': '為你我用了半年的積蓄，漂洋過海來看你'
        }, {
            'text': '米線麪粉裏面啓發吊鉤',
            'target': '米線麵粉裡面啟發吊鉤'
        }]
        self.setUp('t2tw')
        self._run_chinese_convert(samples)

    def test_tw2t(self):

        samples = [{
            'text': '群賢畢至，少長鹹集',
            'target': '羣賢畢至，少長鹹集'
        }, {
            'text': '為你我用了半年的積蓄，漂洋過海來看你',
            'target': '爲你我用了半年的積蓄，漂洋過海來看你'
        }, {
            'text': '米線麵粉裡面啟發吊鈎',
            'target': '米線麪粉裏面啓發吊鈎'
        }]
        self.setUp('tw2t')
        self._run_chinese_convert(samples)

    def test_t2hk(self):

        samples = [{
            'text': '說他癡人說夢,他深感不悅',
            'target': '説他痴人説夢,他深感不悦'
        }, {
            'text': '爲你我用了半年的積蓄，漂洋過海來看你',
            'target': '為你我用了半年的積蓄，漂洋過海來看你'
        }, {
            'text': '一隻憂鬱的臺灣烏龜',
            'target': '一隻憂鬱的台灣烏龜'
        }]
        self.setUp('t2hk')
        self._run_chinese_convert(samples)

    def test_hk2t(self):

        samples = [{
            'text': '説他痴人説夢,他深感不悦',
            'target': '說他癡人說夢,他深感不悅'
        }, {
            'text': '為你我用了半年的積蓄，漂洋過海來看你',
            'target': '爲你我用了半年的積蓄，漂洋過海來看你'
        }, {
            'text': '一隻憂鬱的台灣烏龜',
            'target': '一隻憂鬱的臺灣烏龜'
        }]
        self.setUp('hk2t')
        self._run_chinese_convert(samples)

    def test_t2jp(self):

        samples = [{
            'text': '他需要修復心臟瓣膜',
            'target': '他需要修復心臓弁膜'
        }, {
            'text': '舊字體歷史假名遣 新字體現代假名遣',
            'target': '旧字体歴史仮名遣 新字体現代仮名遣'
        }, {
            'text': '藝術 缺航 飲料罐',
            'target': '芸術 欠航 飲料缶'
        }]
        self.setUp('t2jp')
        self._run_chinese_convert(samples)

    def test_jp2t(self):

        samples = [{
            'text': '他需要修復心臓弁膜',
            'target': '他需要修復心臟瓣膜'
        }, {
            'text': '旧字体歴史仮名遣 新字体現代仮名遣',
            'target': '舊字體歷史假名遣 新字體現代假名遣'
        }, {
            'text': '芸術 欠航 飲料缶',
            'target': '藝術 缺航 飲料罐'
        }]
        self.setUp('jp2t')
        self._run_chinese_convert(samples)


if __name__ == '__main__':
    unittest.main()
