"""Genie 1 latent action model using the tokenizer's channels-last RGB layout."""

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

    def encode(self, video):
        """Return T-1 action vectors/IDs for transitions frame t -> frame t+1."""
        if video.ndim != 5 or video.shape[1] < 2:
            raise ValueError("LAM needs at least two video frames")
        z = self.encoder(video)
        return self.cookbook(z[:, 1:])

    def forward(self, video):
        z_q, idx, vq_loss = self.encode(video)
        pred = self.decoder(video[:, :-1], z_q)
        recon = F.mse_loss(pred, video[:, 1:])
        return pred, idx, recon + vq_loss, recon, vq_loss
