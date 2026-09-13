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


if __name__ == "__main__":
    import argparse
    import numpy as np
    from PIL import Image
    from dataloader import prepare_video

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    with Image.open(args.image) as image:
        video = prepare_video(np.asarray(image.convert("RGB"))[None]).unsqueeze(0)
    with torch.inference_mode():
        print(VAE().eval()(video).shape)
