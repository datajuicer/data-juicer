import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.image_segment_mapper import ImageSegmentMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields, MetaKeys


class ImageSegmentMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    def _run_op(self, op, source_list, num_proc=1):
        dataset = Dataset.from_list(source_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res_list = dataset.to_list()

        bbox_nums = [[5], [14, 6]]
        for sample, sample_bbn in zip(res_list, bbox_nums):
            for bb, bbn in zip(sample[Fields.meta][MetaKeys.bbox_tag],
                               sample_bbn):
                self.assertEqual(len(bb), bbn)

    def test_segment_mapper(self):
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }]
        # fix params for reproducibility
        op = ImageSegmentMapper(
            imgsz=1024, conf=0.9, iou=0.5, model_path='FastSAM-x.pt')
        self._run_op(op, ds_list)

    def test_cpu(self):
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }]
        # fix params for reproducibility
        op = ImageSegmentMapper(imgsz=1024,
                                conf=0.9,
                                iou=0.5,
                                model_path='FastSAM-x.pt',
                                accelerator='cpu')
        self._run_op(op, ds_list)

    def test_multi_process(self):
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }]
        # fix params for reproducibility
        op = ImageSegmentMapper(
            imgsz=1024, conf=0.9, iou=0.5, model_path='FastSAM-x.pt')
        self._run_op(op, ds_list, num_proc=2)


if __name__ == '__main__':
    unittest.main()
