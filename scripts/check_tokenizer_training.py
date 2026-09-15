"""CPU checks for DDP sample weighting and mid-epoch checkpoint/resume."""
import contextlib
import io
import os
from pathlib import Path
import pickle
import shutil
import socket
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.multiprocessing as mp
from array_record.python.array_record_module import ArrayRecordWriter
from train_tokenizer import EpochBatches, train
from vae import VAE
from cookbook import Cookbook


def tiny_model():
    torch.manual_seed(77)
    return VAE(img=16, enc_width=8, dec_width=8, enc_layers=1, dec_layers=1,
               st_enc_heads=1, st_dec_heads=1, enc_head_dim=8, dec_head_dim=8,
               code_width=4, num_codes=8)


def make_data(root):
    path = Path(root) / "train"
    path.mkdir()
    writer = ArrayRecordWriter(str(path / "test.array_record"), "group_size:1")
    try:
        for i in range(5):
            frames = np.full((16, 60, 80, 3), i * 40, dtype=np.uint8)
            writer.write(pickle.dumps(dict(sequence_length=16, raw_video=frames.tobytes())))
    finally:
        writer.close()
    return str(path)


def run(root, filename, batch_size=2, **kwargs):
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        train(data_dir=str(Path(root) / "train"), device="cpu", num_workers=0,
              epochs=2, batch_size=batch_size, checkpoint=str(Path(root) / filename),
              checkpoint_every=2, model=tiny_model(), validation_every=0, **kwargs)


def distributed_worker(rank, port, root):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    run(root, "distributed.pt")


class TrainingChecks(unittest.TestCase):
    def test_dynamics_loader_preserves_legacy_and_stabilized_math(self):
        from dynamics_model import load_tokenizer
        with tempfile.TemporaryDirectory() as root:
            video = torch.rand(1, 16, 16, 16, 3)
            for stabilize in (False, True):
                config = {**tiny_model().model_config, "stabilize": stabilize}
                original = VAE(**config).eval()
                saved_config = dict(config)
                if not stabilize:
                    saved_config.pop("stabilize")
                path = Path(root) / f"tokenizer-{stabilize}.pt"
                torch.save({"model_config": saved_config,
                            "model_state_dict": original.state_dict()}, path)
                loaded = load_tokenizer(path, torch.device("cpu"))
                self.assertEqual(loaded.stabilize, stabilize)
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                with torch.no_grad():
                    torch.testing.assert_close(loaded(video), original(video), rtol=0, atol=0)

    def test_straight_through_preserves_tiny_codes_and_encoder_gradient(self):
        for dtype in (torch.float32, torch.bfloat16):
            cb = Cookbook(K=8, code_width=4)
            x = torch.full((2,3,4),10000.0,dtype=dtype,requires_grad=True)
            quantized, indices, _ = cb(x)
            expected = cb.cookbook(indices).to(dtype)
            torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
            quantized.float().sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
            self.assertIsNone(cb.cookbook.weight.grad)

    def test_large_decoder_residuals_keep_reconstruction_gradients(self):
        model = tiny_model()
        model.dec_blocks[-1].register_forward_hook(lambda module, args, output: output * 10000)
        video = torch.rand(1,16,16,16,3)
        result = model(video, return_details=True)
        result['reconstruction_loss'].backward()
        self.assertTrue(torch.isfinite(model.proj_dec_to_pp_width.weight.grad).all())
        self.assertGreater(model.proj_dec_to_pp_width.weight.grad.abs().max().item(), 0)
        self.assertGreater(result['reconstruction'].min().item(), 0.01)
        self.assertLess(result['reconstruction'].max().item(), 0.99)

    def test_validation_stops_and_checkpoints_identical_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            make_data(root)
            shutil.copytree(Path(root)/'train', Path(root)/'val')
            model = tiny_model()
            model.decode = lambda z: torch.zeros(z.shape[0], z.shape[1],16,16,3) + z.sum()*0
            checkpoint = Path(root)/'stopped.pt'
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'collapsed'):
                train(data_dir=str(Path(root)/'train'), device='cpu', num_workers=0,
                      epochs=1, batch_size=2, model=model, checkpoint=checkpoint,
                      validation_every=1, health_after=0)
            saved = torch.load(checkpoint, weights_only=True)
            self.assertTrue(saved['health_failure'])
            self.assertEqual(saved['global_step'], 1)
            self.assertIn('different validation clips produce identical outputs',
                          saved['validation']['collapse_reasons'])
            self.assertTrue((checkpoint.parent/'validation/step_0000001.png').exists())

    def test_partition_has_exact_coverage_even_with_empty_ranks(self):
        for size in range(1, 14):
            for world in (1, 2, 3):
                for start in range(size):
                    actual = []
                    for rank in range(world):
                        sampler = EpochBatches(size, 2, rank, world, 42, 1, start)
                        for step, batch in enumerate(sampler):
                            real_count, _ = sampler.counts(step)
                            actual.extend(batch[:real_count])
                    expected = torch.randperm(size, generator=torch.Generator().manual_seed(43)).tolist()[start:]
                    self.assertEqual(sorted(actual), sorted(expected))

    def test_resume_preserves_weights_optimizer_and_sample_cursor(self):
        with tempfile.TemporaryDirectory() as root:
            make_data(root)
            run(root, "full.pt")
            run(root, "partial.pt", max_steps=2)
            partial = torch.load(Path(root) / "partial.pt", weights_only=True)
            self.assertEqual(partial["next_sample"], 4)
            self.assertFalse(partial["epoch_complete"])
            run(root, "resumed.pt", resume=str(Path(root) / "partial.pt"))
            full = torch.load(Path(root) / "full.pt", weights_only=True)
            resumed = torch.load(Path(root) / "resumed.pt", weights_only=True)
            self.assertEqual(full["global_step"], resumed["global_step"])
            self.assertEqual([x["sequences"] for x in resumed["history"]], [5, 5])
            for name, tensor in full["model_state_dict"].items():
                torch.testing.assert_close(tensor, resumed["model_state_dict"][name], rtol=0, atol=0)
            for idx, state in full["optimizer_state_dict"]["state"].items():
                for key, tensor in state.items():
                    torch.testing.assert_close(tensor, resumed["optimizer_state_dict"]["state"][idx][key], rtol=0, atol=0)

    def test_ddp_matches_global_batch_with_uneven_tail(self):
        with tempfile.TemporaryDirectory() as root:
            make_data(root)
            run(root, "single.pt", batch_size=4)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            mp.spawn(distributed_worker, args=(port, root), nprocs=2, join=True)
            single = torch.load(Path(root) / "single.pt", weights_only=True)
            distributed = torch.load(Path(root) / "distributed.pt", weights_only=True)
            self.assertEqual(single["global_step"], distributed["global_step"])
            self.assertEqual([x["sequences"] for x in distributed["history"]], [5, 5])
            for name, tensor in single["model_state_dict"].items():
                torch.testing.assert_close(tensor, distributed["model_state_dict"][name], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
