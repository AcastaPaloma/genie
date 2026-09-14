"""Genie 1 video tokenizer: Table 7 defaults, channels-last RGB video."""

import torch
from torch import nn

from config import (TOKENIZER_ENCODER as ENC, TOKENIZER_DECODER as DEC,
                    VIDEO_PATCH, IMAGE_SIZE, CHANNELS, SEQUENCE_LENGTH,
                    VIDEO_CODES, CODE_WIDTH)
from transformer import SpatioTemporalTransformerBlock
from cookbook import Cookbook
from patches import VideoPatches, position_parameter



class VAE(VideoPatches):
    """Quantized video autoencoder; input/output (B,T,H,W,C) in [0,1].

    Defaults: encoder 512/12/8, decoder 1024/20/16 (width/blocks/heads).
    Images default to 96x160, padded from 90x160 by the data pipeline.
    Both integer square sizes and (height,width) sizes are accepted.
    """

    def __init__(self, patch=VIDEO_PATCH, img=IMAGE_SIZE, channels=CHANNELS,
                 enc_width=ENC.width, T=SEQUENCE_LENGTH, enc_layers=ENC.layers,
                 st_enc_heads=ENC.heads, dec_width=DEC.width,
                 dec_layers=DEC.layers, st_dec_heads=DEC.heads, *,
                 enc_head_dim=ENC.head_dim, dec_head_dim=DEC.head_dim,
                 code_width=CODE_WIDTH, num_codes=VIDEO_CODES,
                 ffn_expansion=ENC.ffn_expansion):
        super().__init__(img, patch, channels)
        if min(enc_width, dec_width, T, enc_layers, dec_layers) < 1:
            raise ValueError("Widths, context length and layer counts must be positive")
        self.model_config = dict(patch=patch, img=img, channels=channels,
            enc_width=enc_width, T=T, enc_layers=enc_layers, st_enc_heads=st_enc_heads,
            dec_width=dec_width, dec_layers=dec_layers, st_dec_heads=st_dec_heads,
            enc_head_dim=enc_head_dim, dec_head_dim=dec_head_dim,
            code_width=code_width, num_codes=num_codes, ffn_expansion=ffn_expansion)
        self.enc_width, self.dec_width = enc_width, dec_width
        self.cb = Cookbook(K=num_codes, code_width=code_width)
        self.patch_proj = nn.Linear(self.patch_dim, enc_width)
        self.spatial_pos = position_parameter(self.n_patches, enc_width)
        self.temporal_pos = position_parameter(T, enc_width)
        self.dec_spatial_pos = position_parameter(self.n_patches, dec_width)
        self.dec_temporal_pos = position_parameter(T, dec_width)
        self.enc_blocks = nn.ModuleList([
            SpatioTemporalTransformerBlock(enc_width, st_enc_heads,
                head_dim=enc_head_dim, ffn_expansion_factor=ffn_expansion)
            for _ in range(enc_layers)
        ])
        self.proj_enc_to_codebook_width = nn.Linear(enc_width, code_width)
        self.proj_codebook_to_dec_width = nn.Linear(code_width, dec_width)
        self.dec_blocks = nn.ModuleList([
            SpatioTemporalTransformerBlock(dec_width, st_dec_heads,
                head_dim=dec_head_dim, ffn_expansion_factor=ffn_expansion)
            for _ in range(dec_layers)
        ])
        self.proj_dec_to_pp_width = nn.Linear(dec_width, self.patch_dim)

    def encode(self, video):
        """Return quantized vectors (B,T,N,32), IDs (B,T,N), and VQ loss."""
        x = self.patchify(video)
        t = video.shape[1]
        if not 1 <= t <= self.temporal_pos.shape[0]:
            raise ValueError("Frame count exceeds configured tokenizer context")
        x = self.patch_proj(x)
        x = x + self.spatial_pos[None, None] + self.temporal_pos[None, :t, None]
        for block in self.enc_blocks:
            x = block(x)
        return self.cb(self.proj_enc_to_codebook_width(x))

    def decode(self, z_q):
        if z_q.ndim != 4 or tuple(z_q.shape[2:]) != (self.n_patches, self.cb.code_width):
            raise ValueError("Expected codebook vectors (B,T,n_patches,code_width)")
        t = z_q.shape[1]
        if not 1 <= t <= self.dec_temporal_pos.shape[0]:
            raise ValueError("Frame count exceeds configured decoder context")
        x = self.proj_codebook_to_dec_width(z_q)
        x = x + self.dec_spatial_pos[None, None] + self.dec_temporal_pos[None, :t, None]
        for block in self.dec_blocks:
            x = block(x)
        # Bounded RGB reconstruction is an implementation choice for [0,1] data.
        return self.unpatchify(self.proj_dec_to_pp_width(x).sigmoid())

    def decode_tokens(self, indices):
        return self.decode(self.cb.cookbook(indices))

    def forward(self, video, *, return_details=False):
        z_q, indices, vq_loss = self.encode(video)
        reconstruction = self.decode(z_q)
        if return_details:
            reconstruction_loss = nn.functional.mse_loss(reconstruction, video)
            return dict(reconstruction=reconstruction, indices=indices,
                        vq_loss=vq_loss, reconstruction_loss=reconstruction_loss,
                        loss=reconstruction_loss + vq_loss)
        return reconstruction


def train_tokenizer(*, data_dir=None, epochs=1, batch_size=1, device=None,
                    learning_rate=None, warmup_steps=0, num_workers=0, seed=42,
                    checkpoint=None, log_every=100, model=None):
    """Train the tokenizer for full epochs over all local training chunks.

    batch_size counts 16-frame sequences. No fixed batch limit and no dropped
    last batch. Model size stays at full defaults unless a model is explicitly
    supplied (e.g. for a unit test). This is a single-device training entrypoint.
    """
    from dataclasses import replace
    from pathlib import Path
    import random
    import numpy as np
    from torch.utils.data import DataLoader
    from config import TOKENIZER_TRAINING, CONTENT_SIZE, build_optimizer
    from dataloader import ArrayRecordClipDataset

    if min(epochs, batch_size, log_every) < 1 or num_workers < 0 or warmup_steps < 0:
        raise ValueError("epochs, batch_size, log_every must be positive; workers/warmup nonnegative")
    root = Path(__file__).resolve().parent
    data_dir = Path(data_dir) if data_dir is not None else root / "data/train"
    checkpoint = Path(checkpoint) if checkpoint is not None else root / "data/checkpoints/vae_latest.pt"
    learning_rate = TOKENIZER_TRAINING.max_lr if learning_rate is None else learning_rate
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    image_size = model.image_size if model is not None else IMAGE_SIZE
    content_size = tuple(min(content, size) for content, size in zip(CONTENT_SIZE, image_size))
    print(f"Indexing training clips in {data_dir} ...", flush=True)
    dataset = ArrayRecordClipDataset(data_dir, T=16, content_size=content_size, image_size=image_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
        # Readers are opened in each spawned worker, never inherited from a fork.
        **({"multiprocessing_context": "spawn", "persistent_workers": True} if num_workers else {}),
    )
    print(f"{dataset.record_count:,} records -> {len(dataset):,} sequences of 16 frames; "
          f"{len(loader):,} batches/epoch; device={device}", flush=True)
    model = (VAE() if model is None else model).to(device)
    if model.temporal_pos.shape[0] < 16 or model.channels != 3:
        raise ValueError("Tokenizer must accept 16-frame RGB sequences")
    training_config = replace(TOKENIZER_TRAINING, max_lr=learning_rate,
                              min_lr=learning_rate, warmup_steps=warmup_steps,
                              steps=epochs * len(loader), global_batch_size=batch_size)
    optimizer, scheduler = build_optimizer(model, training_config)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history, global_step = [], 0

    for epoch in range(1, epochs + 1):
        model.train()  # Inherited nn.Module.train(), not a custom training-loop method.
        totals = dict(loss=0.0, reconstruction_loss=0.0, vq_loss=0.0)
        sequences_seen = 0
        for batch_number, video in enumerate(loader, 1):
            # Shuffling changes only sequence order, never frame order inside it.
            video = video.to(device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                result = model(video, return_details=True)
                loss = result["loss"]
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {batch_number}")
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            global_step += 1
            sequences_seen += video.shape[0]
            for key in totals:
                totals[key] += result[key].detach().item() * video.shape[0]
            if batch_number == 1 or batch_number % log_every == 0 or batch_number == len(loader):
                print(f"epoch {epoch}/{epochs} batch {batch_number}/{len(loader)} "
                      f"loss={loss.detach().item():.6f} "
                      f"recon={result['reconstruction_loss'].detach().item():.6f} "
                      f"vq={result['vq_loss'].detach().item():.6f}", flush=True)

        metrics = {key: value / sequences_seen for key, value in totals.items()}
        metrics.update(epoch=epoch, sequences=sequences_seen, batches=len(loader))
        history.append(metrics)
        # Save after every complete epoch. Atomic replacement avoids a half-written
        # latest checkpoint if saving is interrupted.
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        torch.save({
            "model_config": model.model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch, "global_step": global_step, "history": history,
            "training_config": vars(training_config), "source_files": dataset.paths,
            "sequence_length": 16, "seed": seed,
        }, temporary)
        temporary.replace(checkpoint)
        print(f"Finished epoch {epoch}: {sequences_seen:,} sequences; saved {checkpoint}", flush=True)

    model.eval()
    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the video tokenizer on all DOOM training clips.")
    parser.add_argument("--data-dir", default=None, help="ArrayRecord split directory (default: data/train)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="16-frame sequences per optimizer batch")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:1, mps, cpu")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Default: constant LR; use 10000 for the paper's tokenizer warmup")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None, help="Default: data/checkpoints/vae_latest.pt")
    parser.add_argument("--log-every", type=int, default=100)
    train_tokenizer(**vars(parser.parse_args()))
