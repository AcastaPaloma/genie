import torch
from torch import nn

from transformer import SpatioTemporalTransformerBlock
from cookbook import Cookbook

class VAE(nn.Module):
    """Reconstruct channels-last video through a quantized ST autoencoder.

    Input must be floating point with shape (B, T, img, img, channels).
    Defaults follow Genie v1 Table 7: encoder 512/12/8, decoder 1024/20/16
    (width/blocks/heads), and a 1024-entry codebook of width 32.
    T in the constructor sets the maximum supported number of frames.
    Returns reconstructed pixels with shape (B, T, img, img, channels).
    """

    def __init__(self, patch=4, img=84, channels=3, enc_width=512, T=16,
                 enc_layers=12, st_enc_heads=8, dec_width=1024,
                 dec_layers=20, st_dec_heads=16):
        super().__init__()

        self.cb = Cookbook(K=1024, code_width=32)
        if min(patch, img, channels, enc_width, dec_width, T) <= 0:
            raise ValueError("All dimensions must be positive")
        if img % patch != 0:
            raise ValueError("img must be divisible by patch")
        if any(not isinstance(n, int) or n < 1 for n in (enc_layers, dec_layers)):
            raise ValueError("enc_layers and dec_layers must be positive integers")

        self.patch, self.img, self.channels = patch, img, channels
        self.enc_width, self.dec_width = enc_width, dec_width
        self.n_patches = (img // patch) ** 2
        patch_dim = patch * patch * channels
        self.patch_proj = nn.Linear(patch_dim, enc_width)
        self.spatial_pos = nn.Parameter(torch.empty(self.n_patches, enc_width))
        self.temporal_pos = nn.Parameter(torch.empty(T, enc_width))
        nn.init.normal_(self.spatial_pos, std=0.02)
        nn.init.normal_(self.temporal_pos, std=0.02)
        # Implementation choice: Genie v1 does not specify decoder position encoding.
        self.dec_spatial_pos = nn.Parameter(torch.empty(self.n_patches, dec_width))
        self.dec_temporal_pos = nn.Parameter(torch.empty(T, dec_width))
        nn.init.normal_(self.dec_spatial_pos, std=0.02)
        nn.init.normal_(self.dec_temporal_pos, std=0.02)

        self.enc_blocks = nn.ModuleList([
            SpatioTemporalTransformerBlock(
                embedding_dim=enc_width, num_heads=st_enc_heads
            )
            for _ in range(enc_layers)
        ])

        self.proj_enc_to_codebook_width = nn.Linear(enc_width, self.cb.code_width)
        self.proj_codebook_to_dec_width = nn.Linear(self.cb.code_width, dec_width)

        self.dec_blocks = nn.ModuleList([
            SpatioTemporalTransformerBlock(
                embedding_dim=dec_width, num_heads=st_dec_heads
            )
            for _ in range(dec_layers)
        ])

        self.proj_dec_to_pp_width = nn.Linear(dec_width, patch_dim)

    def patchify(self, video):
        """Return row-major patches: (B, T, 84, 84, 3) -> (B, T, 441, 48)."""
        if video.ndim != 5 or tuple(video.shape[2:]) != (
            self.img, self.img, self.channels
        ):
            raise ValueError(
                f"Expected video shape (B, T, {self.img}, {self.img}, "
                f"{self.channels}), got {tuple(video.shape)}"
            )

        B, T, H, W, C = video.shape
        p = self.patch
        x = video.reshape(B, T, H // p, p, W // p, p, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        return x.reshape(B, T, self.n_patches, p * p * C)

    def unpatchify(self, patches):
        """Invert patchify: (B, T, 441, 48) -> (B, T, 84, 84, 3)."""
        p, C = self.patch, self.channels
        if patches.ndim != 4 or tuple(patches.shape[2:]) != (
            self.n_patches, p * p * C
        ):
            raise ValueError(
                f"Expected patches shape (B, T, {self.n_patches}, "
                f"{p * p * C}), got {tuple(patches.shape)}"
            )
        B, T = patches.shape[:2]
        grid = self.img // p
        x = patches.reshape(B, T, grid, grid, p, p, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        return x.reshape(B, T, self.img, self.img, C)

    def encode(self, video):
        """Return quantized vectors, integer token IDs, and differentiable VQ loss."""
        x = self.patchify(video)                 # (B, T, 441, 48)
        if not video.is_floating_point():
            raise TypeError("video must contain floating-point pixels")
        T = video.shape[1]
        if not 1 <= T <= self.temporal_pos.shape[0]:
            raise ValueError(
                f"Expected 1 to {self.temporal_pos.shape[0]} frames, got {T}"
            )
        x = self.patch_proj(x)                  # (B, T, 441, D)
        x = x + self.spatial_pos[None, None, :, :]
        x = x + self.temporal_pos[None, :T, None, :]

        for block in self.enc_blocks:
            x = block(x)

        z = self.proj_enc_to_codebook_width(x)  # (B, T, 441, 32)
        return self.cb(z)

    def decode(self, z_q):
        """Reconstruct video from (B, T, n_patches, 32) codebook vectors."""
        if z_q.ndim != 4 or tuple(z_q.shape[2:]) != (self.n_patches, self.cb.code_width):
            raise ValueError("Expected codebook vectors with shape (B, T, n_patches, 32)")
        T = z_q.shape[1]
        if not 1 <= T <= self.dec_temporal_pos.shape[0]:
            raise ValueError("Number of frames exceeds the decoder's supported range")
        x = self.proj_codebook_to_dec_width(z_q)
        x = x + self.dec_spatial_pos[None, None, :, :]
        x = x + self.dec_temporal_pos[None, :T, None, :]

        for block in self.dec_blocks:
            x = block(x)

        patch_pixels = self.proj_dec_to_pp_width(x)  # (B, T, 441, 48)
        return self.unpatchify(patch_pixels)         # (B, T, 84, 84, 3)

    def decode_tokens(self, indices):
        """Decode integer codebook IDs with shape (B, T, n_patches)."""
        return self.decode(self.cb.cookbook(indices))

    def forward(self, video, *, return_details=False):
        z_q, indices, vq_loss = self.encode(video)
        reconstruction = self.decode(z_q)
        if return_details:
            reconstruction_loss = nn.functional.mse_loss(reconstruction, video)
            return {
                "reconstruction": reconstruction,
                "indices": indices,
                "vq_loss": vq_loss,
                "reconstruction_loss": reconstruction_loss,
                "loss": reconstruction_loss + vq_loss,
            }
        return reconstruction

if __name__ == "__main__":
    import argparse

    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an image to pass through the VAE")
    args = parser.parse_args()

    with Image.open(args.image) as image:
        image = image.convert("RGB").resize((84, 84))
        video = torch.from_numpy(np.array(image)).float() / 255.0
    video = video.unsqueeze(0).unsqueeze(0)  # (1, 1, 84, 84, 3)

    v = VAE()
    with torch.inference_mode():
        print(v(video).shape)
