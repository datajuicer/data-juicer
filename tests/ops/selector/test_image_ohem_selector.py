import tempfile
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.selector.image_ohem_selector import ImageOHEMSelector
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageOHEMSelectorTest(DataJuicerTestCaseBase):
    @staticmethod
    def _dataset():
        return Dataset.from_list([{"id": i, "images": []} for i in range(5)])

    def test_select_top_ratio_and_attach_loss(self):
        calls = []

        def score_fn(model, samples, images, device, **kwargs):
            calls.append(len(samples))
            return [sample["id"] for sample in samples]

        result = ImageOHEMSelector(
            score_fn=score_fn, top_ratio=0.4, batch_size=2, device="cpu"
        ).process(self._dataset())

        self.assertEqual(calls, [2, 2, 1])
        self.assertEqual([sample["id"] for sample in result], [4, 3])
        self.assertEqual(
            [sample["__dj__stats__"]["image_ohem_loss"] for sample in result],
            [4.0, 3.0],
        )

    def test_topk_limits_ratio(self):
        result = ImageOHEMSelector(
            score_fn=lambda model, samples, images, device: [sample["id"] for sample in samples],
            top_ratio=1.0,
            topk=2,
            device="cpu",
        ).process(self._dataset())
        self.assertEqual([sample["id"] for sample in result], [4, 3])

    def test_load_functions_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as score_file:
            score_file.write(
                "def model_factory(offset=0):\n"
                "    return {'offset': offset}\n\n"
                "def score_fn(model, samples, images, device, multiplier=1):\n"
                "    return [(sample['id'] + model['offset']) * multiplier "
                "for sample in samples]\n"
            )
            score_file.flush()
            result = ImageOHEMSelector(
                score_file=score_file.name,
                model_kwargs={"offset": 1},
                score_kwargs={"multiplier": 2},
                topk=1,
                device="cpu",
            ).process(self._dataset())
        self.assertEqual(result[0]["id"], 4)
        self.assertEqual(result[0]["__dj__stats__"]["image_ohem_loss"], 10.0)

    def test_reject_wrong_number_of_losses(self):
        selector = ImageOHEMSelector(
            score_fn=lambda model, samples, images, device: [1.0],
            topk=1,
            batch_size=2,
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "one loss per sample"):
            selector.process(self._dataset())

    def test_validate_parameters(self):
        with self.assertRaises(ValueError):
            ImageOHEMSelector(score_fn=lambda *args: [], top_ratio=1.1)
        with self.assertRaises(ValueError):
            ImageOHEMSelector(score_fn=lambda *args: [], topk=-1)
        with self.assertRaises(ValueError):
            ImageOHEMSelector(score_fn=lambda *args: [])


if __name__ == "__main__":
    unittest.main()
