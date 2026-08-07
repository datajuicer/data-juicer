import io
import sys
import tempfile
import unittest

from PIL import Image

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

        result = ImageOHEMSelector(score_fn=score_fn, top_ratio=0.4, batch_size=2, device="cpu").process(
            self._dataset()
        )

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

    def test_load_dataclass_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as score_file:
            score_file.write(
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n\n"
                "@dataclass\n"
                "class ScoreConfig:\n"
                "    offset: float = 1.0\n\n"
                "def score_fn(model, samples, images, device):\n"
                "    config = ScoreConfig()\n"
                "    return [sample['id'] + config.offset for sample in samples]\n"
            )
            score_file.flush()
            selector = ImageOHEMSelector(score_file=score_file.name, topk=1, device="cpu")
            result = selector.process(self._dataset())

        self.assertIn(selector._module_name, sys.modules)
        self.assertEqual(result[0]["id"], 4)
        sys.modules.pop(selector._module_name, None)

    def test_score_modules_have_unique_names(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as score_file:
            score_file.write("def score_fn(model, samples, images, device):\n    return [0] * len(samples)\n")
            score_file.flush()
            first_selector = ImageOHEMSelector(score_file=score_file.name, topk=1)
            second_selector = ImageOHEMSelector(score_file=score_file.name, topk=1)

        self.assertNotEqual(first_selector._module_name, second_selector._module_name)
        sys.modules.pop(first_selector._module_name, None)
        sys.modules.pop(second_selector._module_name, None)

    def test_failed_module_is_removed(self):
        module_names_before = set(sys.modules)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+") as score_file:
            score_file.write("raise RuntimeError('module load failed')\n")
            score_file.flush()
            with self.assertRaisesRegex(RuntimeError, "module load failed"):
                ImageOHEMSelector(score_file=score_file.name, topk=1)

        new_module_names = set(sys.modules) - module_names_before
        self.assertFalse(any(name.startswith("data_juicer_image_ohem_") for name in new_module_names))

    def test_load_images_from_bytes_only(self):
        first_image = Image.new("RGB", (3, 2), color="red")
        first_image_buffer = io.BytesIO()
        first_image.save(first_image_buffer, format="PNG")
        selector = ImageOHEMSelector(score_fn=lambda *args: [], topk=1)
        images = selector._images(
            {
                "images": ["/path/does/not/exist.png"],
                "image_bytes": [first_image_buffer.getvalue()],
            }
        )
        self.assertEqual(images[0].size, (3, 2))

    def test_load_images_with_partial_bytes(self):
        first_image = Image.new("RGB", (3, 2), color="red")
        first_image_buffer = io.BytesIO()
        first_image.save(first_image_buffer, format="PNG")

        with tempfile.NamedTemporaryFile(suffix=".png") as second_image_file:
            Image.new("RGB", (5, 4), color="blue").save(second_image_file.name)
            selector = ImageOHEMSelector(score_fn=lambda *args: [], topk=1)
            images = selector._images(
                {
                    "images": ["/path/does/not/exist.png", second_image_file.name],
                    "image_bytes": [first_image_buffer.getvalue(), None],
                }
            )

        self.assertEqual([image.size for image in images], [(3, 2), (5, 4)])

    def test_release_model_after_success_and_error(self):
        class Model:
            def __init__(self):
                self.devices = []

            def to(self, device):
                self.devices.append(device)
                return self

            def eval(self):
                return self

        success_model = Model()
        success_selector = ImageOHEMSelector(
            model_factory=lambda: success_model,
            score_fn=lambda model, samples, images, device: [sample["id"] for sample in samples],
            topk=1,
            device="cuda",
        )
        success_selector.process(self._dataset())
        self.assertIsNone(success_selector._model)
        self.assertEqual(success_model.devices, ["cuda", "cpu"])

        failed_model = Model()

        def failed_score_fn(model, samples, images, device):
            raise RuntimeError("scoring failed")

        failed_selector = ImageOHEMSelector(
            model_factory=lambda: failed_model,
            score_fn=failed_score_fn,
            topk=1,
            device="cuda",
        )
        with self.assertRaisesRegex(RuntimeError, "scoring failed"):
            failed_selector.process(self._dataset())
        self.assertIsNone(failed_selector._model)
        self.assertEqual(failed_model.devices, ["cuda", "cpu"])

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
