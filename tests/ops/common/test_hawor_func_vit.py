import unittest
from unittest import mock

import torch

from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase

from .test_hawor_imports import load_test_modules

hw, hv, dw = load_test_modules()


class HaworVitTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_resize_absolute_positions(self):
        pos = torch.arange(5 * 3, dtype=torch.float32).reshape(1, 5, 3)
        self.assertEqual(hv.get_abs_pos(pos, 2, 2, 2, 2).shape, (1, 5, 3))
        resized = hv.get_abs_pos(pos, 1, 2, 2, 2)
        self.assertEqual(resized.shape, (1, 3, 3))
        no_cls = hv.get_abs_pos(pos[:, 1:], 1, 2, 2, 2, has_cls_token=False)
        self.assertEqual(no_cls.shape, (1, 2, 3))

    @TEST_TAG("standalone")
    def test_drop_path_forwards_probability_and_training_mode(self):
        drop = hv.DropPath(0.25)
        inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        expected = object()
        for training in (True, False):
            with (
                self.subTest(training=training),
                mock.patch.object(hv.timm.layers, "drop_path", return_value=expected) as drop_path,
            ):
                drop.train(training)
                self.assertIs(drop(inputs), expected)
                drop_path.assert_called_once_with(inputs, 0.25, training)
        self.assertEqual(drop.extra_repr(), "p=0.25")

    @TEST_TAG("standalone")
    def test_small_network_blocks(self):
        self.assertEqual(hv.Mlp(4, 6, 5)(torch.randn(2, 3, 4)).shape, (2, 3, 5))
        self.assertEqual(hv.Attention(6, num_heads=2, attn_head_dim=2)(torch.randn(2, 3, 6)).shape, (2, 3, 6))
        self.assertEqual(hv.Block(6, 2, mlp_ratio=2)(torch.randn(2, 3, 6)).shape, (2, 3, 6))

    @TEST_TAG("standalone")
    def test_patch_embedding(self):
        patch = hv.PatchEmbed(img_size=(8, 8), patch_size=4, embed_dim=6)
        tokens, shape = patch(torch.randn(2, 3, 8, 8))
        self.assertEqual((tokens.shape, shape), (torch.Size([2, 9, 6]), (3, 3)))

    @TEST_TAG("standalone")
    def test_hybrid_embedding(self):
        class ListBackbone(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3, stride=2, padding=1)

            def forward(self, value):
                return [self.conv(value)]

        hybrid = hv.HybridEmbed(ListBackbone(), img_size=8, embed_dim=4)
        self.assertEqual(hybrid(torch.randn(2, 3, 8, 8)).shape, (2, 16, 4))

    @TEST_TAG("standalone")
    def test_small_vit_forward_and_freezing(self):
        torch.manual_seed(2)
        model = hv.ViT(
            img_size=(8, 8),
            patch_size=8,
            embed_dim=8,
            depth=2,
            num_heads=2,
            mlp_ratio=2,
            qkv_bias=True,
            drop_path_rate=0.1,
            frozen_stages=0,
            freeze_attn=True,
            freeze_ffn=True,
        )
        model.init_weights()
        self.assertEqual(model.get_num_layers(), 2)
        self.assertEqual(model.no_weight_decay(), {"pos_embed", "cls_token"})
        output = model(torch.randn(1, 3, 8, 8))
        self.assertEqual(output.shape, (1, 8, 1, 1))
        self.assertTrue(all((not p.requires_grad for p in model.patch_embed.parameters())))
        self.assertTrue(all((not p.requires_grad for p in model.blocks[0].attn.parameters())))
        model.train()
        self.assertFalse(model.patch_embed.training)

    @TEST_TAG("standalone")
    def test_vit_uses_checkpoint(self):
        checkpointed = hv.ViT(img_size=8, patch_size=8, embed_dim=8, depth=1, num_heads=2, use_checkpoint=True)
        with mock.patch.object(torch.utils.checkpoint, "checkpoint", side_effect=lambda fn, x: fn(x)) as checkpoint:
            self.assertEqual(checkpointed(torch.randn(1, 3, 8, 8)).shape, (1, 8, 1, 1))
            checkpoint.assert_called_once()


if __name__ == "__main__":
    unittest.main()
