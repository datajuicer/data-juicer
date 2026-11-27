import unittest

import torch

from data_juicer.ops.common.ldeq_face_keypoints_func import Normalize


class LDEQNormalizeTest(unittest.TestCase):
    def test_softmax_heatmaps_are_finite_and_normalized(self):
        layer = Normalize(2, "softargmax")
        logits = torch.tensor([[[[0.0, 1000.0], [-1000.0, 3.0]]] * 2])
        output = layer(logits)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.sum(dim=(2, 3)), torch.ones(1, 2))


if __name__ == "__main__":
    unittest.main()
