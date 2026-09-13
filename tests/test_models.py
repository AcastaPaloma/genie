"""Functional checks use explicit tiny overrides; production defaults stay full-size."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import TOKENIZER_TRAINING, DYNAMICS_TRAINING, build_optimizer
from cookbook import Cookbook
from dataloader import ClipDataset, build_loader, prepare_video
from dynamics_model import DynamicsModel
from lam import LAM
from transformer import SelfAttention, SpatioTemporalTransformerBlock
from vae import VAE


def tiny_vae():
    return VAE(img=(16, 32), enc_width=16, dec_width=24, enc_layers=1,
               dec_layers=1, st_enc_heads=2, st_dec_heads=3,
               enc_head_dim=8, dec_head_dim=8, num_codes=16, code_width=4, T=3)


def tiny_lam():
    return LAM(img=(16, 32), d_width=16, st_blocks=1, st_heads=2,
               head_dim=8, code_width=4, T=3)


def tiny_dynamics():
    # Deliberately use hidden width != heads * head_dim.
    return DynamicsModel(img=(16, 32), d_width=20, st_blocks=2,
                         st_heads=3, head_dim=8, num_codes=16, action_width=4, video_width=4, T=3)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_full_default_architectures_without_allocating_weights(self):
        with torch.device("meta"):
            vae, lam, dyn = VAE(), LAM(), DynamicsModel()
            logits = dyn(torch.zeros(1, 16, 960, dtype=torch.long), torch.zeros(1, 15, 32))
        self.assertEqual(logits.shape, (1, 16, 960, 1024))
        for blocks, width, count, heads, head_dim in (
            (vae.enc_blocks, 512, 12, 8, 64),
            (vae.dec_blocks, 1024, 20, 16, 64),
            (lam.encoder.blocks, 1024, 20, 16, 64),
            (lam.decoder.blocks, 1024, 20, 16, 64),
            (dyn.blocks, 5120, 48, 36, 128),
        ):
            self.assertEqual(len(blocks), count)
            for block in blocks:
                self.assertEqual(block.embedding_dim, width)
                for attention in (block.spatial_attention, block.temporal_attention):
                    self.assertEqual(attention.num_heads, heads)
                    self.assertEqual(attention.head_dim, head_dim)
                    self.assertEqual(attention.q_proj.out_features, heads * head_dim)
        self.assertEqual(vae.cb.cookbook.weight.shape, (1024, 32))
        self.assertEqual(lam.cookbook.cookbook.weight.shape, (8, 32))
        self.assertEqual(lam.encoder.patch, 16)
        self.assertEqual(lam.encoder.n_patches, 60)
        self.assertIsInstance(dyn.blocks[0].spatial_attention.q_norm, nn.LayerNorm)
        # Literal table dimensions + independent attention + 4x FFNs exceed
        # the reported 10.1B. Keep that unresolved discrepancy visible.
        self.assertGreater(sum(p.numel() for p in dyn.parameters()), 19_000_000_000)

    def test_end_to_end_training_interfaces(self):
        video = torch.rand(2, 3, 16, 32, 3)
        vae, lam, dyn = tiny_vae(), tiny_lam(), tiny_dynamics()
        result = vae(video, return_details=True)
        self.assertEqual(result["reconstruction"].shape, video.shape)
        self.assertTrue(torch.equal(vae.unpatchify(vae.patchify(video)), video))
        result["loss"].backward()
        self.assertGreater(vae.patch_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(vae.cb.cookbook.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(vae.patch_proj.weight.grad).all())
        with torch.no_grad():
            vectors, ids, _ = vae.encode(video)
            torch.testing.assert_close(vae.decode(vectors), vae.decode_tokens(ids))
        lam.eval()
        self.assertFalse(lam.encoder.training)
        lam.train()
        self.assertTrue(lam.decoder.training)
        pred, action_ids, loss, _, _ = lam(video)
        self.assertEqual(pred.shape, video[:, 1:].shape)
        self.assertTrue(((pred >= 0) & (pred <= 1)).all())
        self.assertEqual(action_ids.shape, (2, 2))
        loss.backward()
        self.assertTrue(torch.isfinite(lam.encoder.patch_proj.weight.grad).all())
        actions, _, _ = lam.encode(video)
        actions.retain_grad()
        masked = torch.zeros_like(ids, dtype=torch.bool)
        masked[:, 1:] = True
        logits = dyn(ids, actions, mask=masked)
        self.assertEqual(logits.shape, (2, 3, 32, 16))
        F.cross_entropy(logits[masked], ids[masked]).backward()
        self.assertIsNone(actions.grad)
        self.assertGreater(dyn.action_proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(dyn.blocks[0].spatial_attention.q_proj.weight.grad).all())

    def test_dynamics_masking_causality_and_action_signal(self):
        dyn = tiny_dynamics().eval()
        ids = torch.randint(0, 16, (1, 3, 32))
        actions = torch.randn(1, 2, 4)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, 2] = True
        with torch.no_grad():
            baseline = dyn(ids, actions, mask=mask)
            changed_ids = ids.clone()
            changed_ids[:, 2] = (changed_ids[:, 2] + 1) % 16
            torch.testing.assert_close(baseline, dyn(changed_ids, actions, mask=mask))
            altered_actions = actions.clone()
            altered_actions[:, 1] += 5
            changed = dyn(changed_ids, altered_actions, mask=mask)
            torch.testing.assert_close(baseline[:, :2], changed[:, :2])
            self.assertGreater((baseline[:, 2] - changed[:, 2]).abs().max().item(), 1e-4)
        with self.assertRaises(ValueError):
            dyn(ids, torch.zeros(1, 3, 4))

    def test_quantized_video_input_path(self):
        dyn = tiny_dynamics().eval()
        video = torch.randn(1, 3, 32, 4, requires_grad=True)
        actions = torch.randn(1, 2, 4, requires_grad=True)
        mask = torch.zeros(1, 3, 32, dtype=torch.bool)
        mask[:, -1] = True
        logits = dyn(video, actions, mask=mask)
        self.assertEqual(logits.shape, (1, 3, 32, 16))
        altered = video.detach().clone()
        altered[:, -1] += 100
        torch.testing.assert_close(logits, dyn(altered, actions, mask=mask))
        logits.square().mean().backward()
        self.assertIsNone(video.grad)
        self.assertIsNone(actions.grad)
        self.assertGreater(dyn.video_proj.weight.grad.abs().sum().item(), 0)

    def test_fused_attention_matches_inspection_path(self):
        attention = SelfAttention(20, 3, head_dim=8, qk_norm=True).eval()
        x = torch.randn(2, 3, 5, 20)
        boolean_mask = torch.zeros(5, 5, dtype=torch.bool)
        boolean_mask[:, -1] = True
        additive_mask = torch.zeros(5, 5).masked_fill(boolean_mask, float("-inf"))
        for mask in (None, boolean_mask, additive_mask):
            for causal in (False, True):
                with self.subTest(mask_type=None if mask is None else mask.dtype, causal=causal):
                    fast = attention(x, attention_mask=mask, is_causal=causal)
                    slow, weights = attention(x, attention_mask=mask, is_causal=causal, return_attention=True)
                    torch.testing.assert_close(fast, slow, atol=1e-6, rtol=1e-5)
                    if causal:
                        self.assertTrue((weights.triu(1) == 0).all())

    def test_st_block_cannot_see_future_frames(self):
        block = SpatioTemporalTransformerBlock(16, 2, head_dim=8).eval()
        x = torch.randn(1, 3, 4, 16)
        changed = x.clone()
        changed[:, 2] = torch.randn_like(changed[:, 2]) * 10
        torch.testing.assert_close(block(x)[:, :2], block(changed)[:, :2])

    def test_chunked_quantizer_matches_brute_force(self):
        cb = Cookbook(K=7, code_width=4, search_chunk_size=3)
        x = torch.randn(2, 3, 4, requires_grad=True)
        expected = (x.detach().unsqueeze(-2) - cb.cookbook.weight.detach()).square().sum(-1).argmin(-1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            quantized, ids, loss = cb(x)
        torch.testing.assert_close(ids, expected)
        torch.testing.assert_close(quantized, cb.cookbook(ids))
        (quantized.sum() + loss).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(cb.cookbook.weight.grad).all())

    def test_rgb_layout_fps_and_episode_boundaries(self):
        frames = np.zeros((20, 4, 8, 3), dtype=np.uint8)
        for i in range(20):
            frames[i] = i
        ds = ClipDataset(frames, np.array([[0, 10], [10, 10]]), T=3,
                         source_fps=35, content_size=(4, 8), image_size=(16, 16))
        self.assertEqual(len(ds), 6)  # span=8; three windows in each episode
        torch.testing.assert_close(ds[2][:, 0, 0, 0], torch.tensor([2., 5., 9.]) / 255)
        torch.testing.assert_close(ds[3][:, 0, 0, 0], torch.tensor([10., 13., 17.]) / 255)
        prepared = prepare_video(frames[:1])
        self.assertEqual(prepared.shape, (1, 96, 160, 3))
        torch.testing.assert_close(prepared[:, 90:], prepared[:, 89:90].expand(-1, 6, -1, -1))
        gray = np.full((2, 1, 4, 8), 255, dtype=np.uint8)
        self.assertTrue((prepare_video(gray, layout="TCHW") == 1).all())
        with tempfile.TemporaryDirectory() as directory:
            fp, ep = Path(directory) / "frames.npy", Path(directory) / "episodes.npy"
            np.save(fp, frames)
            np.save(ep, np.array([[0, 20]]))
            _, loader = build_loader(batch_size=1, num_workers=0, frames_path=fp,
                                     episodes_path=ep, drop_last=False)
            self.assertEqual(next(iter(loader)).shape, (1, 16, 96, 160, 3))

    def test_optimizer_configuration_and_schedule_endpoints(self):
        for config in (TOKENIZER_TRAINING, DYNAMICS_TRAINING):
            optimizer, scheduler = build_optimizer(nn.Linear(2, 2), config)
            self.assertEqual(optimizer.defaults["betas"], (0.9, 0.9))
            self.assertEqual(optimizer.defaults["weight_decay"], 1e-4)
            schedule = scheduler.lr_lambdas[0]
            self.assertEqual(schedule(0), 1 / config.warmup_steps)
            self.assertAlmostEqual(schedule(config.warmup_steps), 1)
            self.assertAlmostEqual(schedule(config.steps), config.min_lr / config.max_lr)


if __name__ == "__main__":
    unittest.main()
