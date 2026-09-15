"""Genie 1 latent action model using the tokenizer's channels-last RGB layout."""
import torch
from torch import nn
from torch.nn import functional as F

from config import (ACTION_MODEL as ARCH, ACTION_PATCH, IMAGE_SIZE, CHANNELS,
                    CODE_WIDTH, ACTION_CODES, SEQUENCE_LENGTH)
from transformer import SpatioTemporalTransformerBlock
from cookbook import Cookbook
from patches import VideoPatches, position_parameter


class LAMEncoder(VideoPatches):
    def __init__(self, patch=ACTION_PATCH, img=IMAGE_SIZE, channels=CHANNELS,
                 d_width=ARCH.width, code_width=CODE_WIDTH, T=SEQUENCE_LENGTH,
                 st_blocks=ARCH.layers, st_heads=ARCH.heads, *,
                 head_dim=ARCH.head_dim, ffn_expansion=ARCH.ffn_expansion):
        super().__init__(img, patch, channels)
        if min(T, st_blocks, d_width) < 1:
            raise ValueError("Context, blocks and width must be positive")
        self.d_width = d_width
        self.patch_proj = nn.Linear(self.patch_dim, d_width)
        self.spatial_pos = position_parameter(self.n_patches, d_width)
        self.temporal_pos = position_parameter(T, d_width)
        self.blocks = nn.ModuleList([
            SpatioTemporalTransformerBlock(d_width, st_heads, head_dim=head_dim,
                ffn_expansion_factor=ffn_expansion)
            for _ in range(st_blocks)
        ])
        self.to_code = nn.Linear(d_width, code_width)

    def forward(self, video):
        # Same (B,T,H,W,C) pixels as VAE; only the patch size is different.
        x = self.patchify(video)
        t = video.shape[1]
        if not 1 <= t <= self.temporal_pos.shape[0]:
            raise ValueError("Frame count exceeds configured LAM context")
        x = self.patch_proj(x)
        x = x + self.spatial_pos[None, None] + self.temporal_pos[None, :t, None]
        for block in self.blocks:
            x = block(x)
        return self.to_code(x.mean(dim=2))  # (B,T,code_width)


class LAMDecoder(VideoPatches):
    def __init__(self, patch=ACTION_PATCH, img=IMAGE_SIZE, channels=CHANNELS,
                 d_width=ARCH.width, code_width=CODE_WIDTH, T=SEQUENCE_LENGTH,
                 st_heads=ARCH.heads, st_blocks=ARCH.layers, *,
                 head_dim=ARCH.head_dim, ffn_expansion=ARCH.ffn_expansion):
        super().__init__(img, patch, channels)
        if min(T, st_blocks, d_width) < 1:
            raise ValueError("Context, blocks and width must be positive")
        self.d_width = d_width
        self.patch_proj = nn.Linear(self.patch_dim, d_width)
        self.spatial_pos = position_parameter(self.n_patches, d_width)
        self.temporal_pos = position_parameter(T, d_width)
        self.blocks = nn.ModuleList([
            SpatioTemporalTransformerBlock(d_width, st_heads, head_dim=head_dim,
                ffn_expansion_factor=ffn_expansion)
            for _ in range(st_blocks)
        ])
        self.action_proj = nn.Linear(code_width, d_width)
        self.to_pixels = nn.Linear(d_width, self.patch_dim)

    def forward(self, video, z_q):
        x = self.patchify(video)
        b, t = video.shape[:2]
        if not 1 <= t <= self.temporal_pos.shape[0]:
            raise ValueError("Frame count exceeds configured LAM context")
        if tuple(z_q.shape) != (b, t, self.action_proj.in_features):
            raise ValueError("Expected one codebook action vector per input frame")
        x = self.patch_proj(x)
        x = x + self.spatial_pos[None, None] + self.temporal_pos[None, :t, None]
        x = x + self.action_proj(z_q).unsqueeze(2)
        for block in self.blocks:
            x = block(x)
        return self.unpatchify(self.to_pixels(x).sigmoid())


class LAM(nn.Module):
    def __init__(self, K=ACTION_CODES, code_width=CODE_WIDTH, d_width=ARCH.width,
                 T=SEQUENCE_LENGTH, channels=CHANNELS, *, img=IMAGE_SIZE,
                 patch=ACTION_PATCH, st_blocks=ARCH.layers, st_heads=ARCH.heads,
                 head_dim=ARCH.head_dim, ffn_expansion=ARCH.ffn_expansion):
        super().__init__()
        self.K = K
        kwargs = dict(patch=patch, img=img, channels=channels, d_width=d_width,
                      code_width=code_width, T=T, st_blocks=st_blocks,
                      st_heads=st_heads, head_dim=head_dim, ffn_expansion=ffn_expansion)
        self.encoder = LAMEncoder(**kwargs)
        self.cookbook = Cookbook(K=K, code_width=code_width)
        self.decoder = LAMDecoder(**kwargs)
        # Saved in checkpoints so LAM(**model_config) rebuilds a matching model.
        self.model_config = dict(K=K, **kwargs)

    # The trainer reads these off the model the same way it reads the VAE,
    # which inherits them from VideoPatches. Forward them to the encoder.
    @property
    def image_size(self):
        return self.encoder.image_size

    @property
    def channels(self):
        return self.encoder.channels

    @property
    def temporal_pos(self):
        return self.encoder.temporal_pos

    def encode(self, video):
        """Return T-1 action vectors/IDs for transitions frame t -> frame t+1."""
        if video.ndim != 5 or video.shape[1] < 2:
            raise ValueError("LAM needs at least two video frames")
        z = self.encoder(video)
        return self.cookbook(z[:, 1:])

    def forward(self, video, return_details=False):
        z_q, idx, vq_loss = self.encode(video)
        pred = self.decoder(video[:, :-1], z_q)
        if not return_details:
            return pred
        counts, perplexity = self.cookbook.usage(idx) 
        recon = F.mse_loss(pred, video[:, 1:])
        return dict(prediction=pred, actions=z_q, indices=idx, counts=counts,
                    perplexity=perplexity, vq_loss=vq_loss,
                    reconstruction_loss=recon, loss=recon + vq_loss)



def train_lam(*, data_dir=None, epochs=1, batch_size=1, device=None,
                    learning_rate=None, warmup_steps=0, num_workers=0, seed=42,
                    checkpoint=None, log_every=100, model=None):
    """Train the latent action model for full epochs over all local training chunks.

    batch_size counts 16-frame sequences. No fixed batch limit and no dropped
    last batch. Model size stays at full defaults unless a model is explicitly
    supplied (e.g. for a unit test). This is a single-device training entrypoint.
    """
    from dataclasses import replace
    from pathlib import Path
    import random
    import numpy as np
    from torch.utils.data import DataLoader
    from config import LAM_TRAINING, TOKENIZER_TRAINING, CONTENT_SIZE, build_optimizer
    from dataloader import ArrayRecordClipDataset

    if min(epochs, batch_size, log_every) < 1 or num_workers < 0 or warmup_steps < 0:
        raise ValueError("epochs, batch_size, log_every must be positive; workers/warmup nonnegative")
    root = Path(__file__).resolve().parent
    data_dir = Path(data_dir) if data_dir is not None else root / "data/train"
    checkpoint = Path(checkpoint) if checkpoint is not None else root / "data/checkpoints/lam_latest.pt"
    # LAM_TRAINING is the paper's co-training schedule (Table 9). Standalone LAM
    # pretraining is our staging choice; it borrows the tokenizer rate (Table 8)
    # because a lone LAM is trained like the tokenizer, on reconstruction + VQ loss.
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
    model = (LAM() if model is None else model).to(device)
    if model.temporal_pos.shape[0] < 16 or model.channels != 3:
        raise ValueError("LAM must accept 16-frame RGB sequences")
    training_config = replace(LAM_TRAINING, max_lr=learning_rate,
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
        totals = dict(loss=0.0, reconstruction_loss=0.0, vq_loss=0.0, perplexity=0.0)
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
                      f"vq={result['vq_loss'].detach().item():.6f} "
                      f"ppl={result['perplexity'].item():.2f}", flush=True)

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
    # Example: python lam.py --epochs 1 --batch-size 8 --learning-rate 3e-4 --warmup-steps 1000 --device cuda
    import argparse

    parser = argparse.ArgumentParser(description="Train the latent action model on all DOOM training clips.")
    parser.add_argument("--data-dir", default=None, help="ArrayRecord split directory (default: data/train)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="16-frame sequences per optimizer batch")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:1, mps, cpu")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Default: tokenizer rate 3e-4 for standalone LAM pretraining")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Default: constant LR; the paper's co-training warmup is 5000")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None, help="Default: data/checkpoints/lam_latest.pt")
    parser.add_argument("--log-every", type=int, default=100)
    train_lam(**vars(parser.parse_args()))
