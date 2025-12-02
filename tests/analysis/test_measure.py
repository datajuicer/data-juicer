import os
import torch
import unittest

import numpy as np
import torch.distributions as td

from data_juicer.analysis.measure import Measure, KLDivMeasure, EntropyMeasure, CrossEntropyMeasure, JSDivMeasure, RelatedTTestMeasure

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class MeasureTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()

        self.temp_output_path = 'tmp/test_measure/'
        if not os.path.exists(self.temp_output_path):
            os.makedirs(self.temp_output_path)

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')

        super().tearDown()

    def test_convert_to_tensor(self):
        measure = Measure()

        # tensor to tensor
        x = torch.rand((16, 3, 224, 224), dtype=torch.float32)
        self.assertTrue(measure._convert_to_tensor(x).equal(x))
        # categorical to tensor
        cat = td.Categorical(torch.rand((16, 4096)))
        self.assertTrue(measure._convert_to_tensor(cat).equal(cat.probs))
        # file to tensor
        torch.save(x, os.path.join(self.temp_output_path, 'temp_tensor.pt'))
        self.assertTrue(measure._convert_to_tensor(os.path.join(self.temp_output_path, 'temp_tensor.pt')).equal(x))
        # scalar
        self.assertTrue(measure._convert_to_tensor(1.0).equal(torch.tensor(1.0)))
        # list/tuple
        self.assertTrue(measure._convert_to_tensor([1, 2]).equal(torch.tensor([1, 2])))

    def test_convert_to_categorical(self):
        measure = Measure()

        # tensor to categorical
        x = torch.rand((16, 4096), dtype=torch.float32)
        self.assertIsInstance(measure._convert_to_categorical(x), td.Categorical)
        # categorical to categorical
        cat = td.Categorical(torch.rand((16, 4096)))
        self.assertIsInstance(measure._convert_to_categorical(cat), td.Categorical)
        # file to tensor
        torch.save(x, os.path.join(self.temp_output_path, 'temp_tensor.pt'))
        self.assertIsInstance(measure._convert_to_categorical(os.path.join(self.temp_output_path, 'temp_tensor.pt')), td.Categorical)
        # scalar: ValueError
        with self.assertRaises(ValueError):
            _ = measure._convert_to_categorical(1.0)
        # list/tuple
        self.assertIsInstance(measure._convert_to_categorical([1, 2]), td.Categorical)

    def test_convert_to_ndarray(self):
        measure = Measure()

        # tensor to tensor
        x = torch.rand((16, 3, 224, 224), dtype=torch.float32)
        self.assertIsInstance(measure._convert_to_ndarray(x), np.ndarray)
        # categorical to tensor
        cat = td.Categorical(torch.rand((16, 4096)))
        self.assertIsInstance(measure._convert_to_ndarray(cat), np.ndarray)
        # file to tensor
        torch.save(x, os.path.join(self.temp_output_path, 'temp_tensor.pt'))
        self.assertIsInstance(measure._convert_to_ndarray(os.path.join(self.temp_output_path, 'temp_tensor.pt')), np.ndarray)
        # scalar
        self.assertIsInstance(measure._convert_to_ndarray(1.0), np.ndarray)
        # list/tuple
        self.assertIsInstance(measure._convert_to_ndarray([1, 2]), np.ndarray)


class KLDivMeasureTest(DataJuicerTestCaseBase):

    def test_measure(self):
        dis1 = [0.1, 0.2, 0.3, 0.4]
        dis2 = [0.1, 0.2, 0.3, 0.4]
        dis3 = [0.4, 0.25, 0.3, 0.05]

        measure = KLDivMeasure()
        self.assertEqual(measure.measure(dis1, dis2).item(), 0.0)
        self.assertEqual(measure.measure(dis2, dis1).item(), 0.0)
        self.assertAlmostEqual(measure.measure(dis1, dis3).item(), 0.6485, delta=1e-4)
        self.assertAlmostEqual(measure.measure(dis3, dis1).item(), 0.5063, delta=1e-4)

class JSDivMeasureTest(DataJuicerTestCaseBase):

    def test_measure(self):
        dis1 = [0.1, 0.2, 0.3, 0.4]
        dis2 = [0.1, 0.2, 0.3, 0.4]
        dis3 = [0.4, 0.25, 0.3, 0.05]

        measure = JSDivMeasure()
        self.assertEqual(measure.measure(dis1, dis2).item(), 0.0)
        self.assertEqual(measure.measure(dis2, dis1).item(), 0.0)
        self.assertAlmostEqual(measure.measure(dis1, dis3).item(), 0.1270, delta=1e-4)
        self.assertAlmostEqual(measure.measure(dis3, dis1).item(), 0.1270, delta=1e-4)

class CrossEntropyMeasureTest(DataJuicerTestCaseBase):

    def test_measure(self):
        dis1 = [0.1, 0.2, 0.3, 0.4]
        dis2 = [0.1, 0.2, 0.3, 0.4]
        dis3 = [0.4, 0.25, 0.3, 0.05]

        measure = CrossEntropyMeasure()
        self.assertAlmostEqual(measure.measure(dis1, dis2).item(), 1.2799, delta=1e-4)
        self.assertAlmostEqual(measure.measure(dis2, dis1).item(), 1.2799, delta=1e-4)
        self.assertAlmostEqual(measure.measure(dis1, dis3).item(), 1.9284, delta=1e-4)
        self.assertAlmostEqual(measure.measure(dis3, dis1).item(), 1.7304, delta=1e-4)

class EntropyMeasureTest(DataJuicerTestCaseBase):

    def test_measure(self):
        dis1 = [0.1, 0.2, 0.3, 0.4]
        dis2 = [0.1, 0.2, 0.3, 0.4]
        dis3 = [0.4, 0.25, 0.3, 0.05]

        measure = EntropyMeasure()
        self.assertAlmostEqual(measure.measure(dis1).item(), 1.2799, delta=1e-4)
        self.assertAlmostEqual(measure.measure(dis2).item(), 1.2799, delta=1e-4)
        self.assertAlmostEqual(measure.measure(dis3).item(), 1.2241, delta=1e-4)

class RelatedTTestMeasureTest(DataJuicerTestCaseBase):

    def test_measure_discrete(self):
        feat1 = [[['a'], ['b', 'c']], 'e', 'g', 'e', 'h']
        feat2 = ['a', 'b', 'd', 'a', ['f', ['b', 'a']]]
        measure = RelatedTTestMeasure()
        res = measure.measure(feat1, feat2)
        self.assertEqual(res.statistic, 0.0)
        self.assertEqual(res.pvalue, 1.0)
        self.assertEqual(res.df, 7)

    def test_measure_continuous(self):
        feat1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        feat2 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        measure = RelatedTTestMeasure()
        res = measure.measure(feat1, feat2)
        self.assertEqual(res.statistic, 0.0)
        self.assertEqual(res.pvalue, 1.0)
        self.assertEqual(res.df, 9)


if __name__ == '__main__':
    unittest.main()
