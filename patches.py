"""Shared channels-last video layout for the tokenizer and latent action model."""

import torch
from torch import nn


class VideoPatches(nn.Module):
    def __init__(self, img, patch, channels):
        super().__init__()
        self.image_size = (img, img) if isinstance(img, int) else tuple(img)
        if len(self.image_size) != 2 or min(*self.image_size, patch, channels) <= 0:
            raise ValueError("Image, patch, and channel dimensions must be positive")
        if any(size % patch for size in self.image_size):
            raise ValueError("Both image dimensions must be divisible by patch size")
        self.img, self.patch, self.channels = img, patch, channels
        self.grid = tuple(size // patch for size in self.image_size)
        self.n_patches = self.grid[0] * self.grid[1]
        self.patch_dim = patch * patch * channels

    def patchify(self, video):
        expected = (*self.image_size, self.channels)
        if video.ndim != 5 or tuple(video.shape[2:]) != expected:
            raise ValueError(f"Expected (B, T, {expected}), got {tuple(video.shape)}")
        if not video.is_floating_point():
            raise TypeError("Video must contain floating-point pixels in [0, 1]")
        b, t = video.shape[:2]
        h, w = self.grid
        p, c = self.patch, self.channels
        return video.reshape(b, t, h, p, w, p, c).permute(
            0, 1, 2, 4, 3, 5, 6
        ).reshape(b, t, self.n_patches, self.patch_dim)

    def unpatchify(self, patches):
        if patches.ndim != 4 or tuple(patches.shape[2:]) != (self.n_patches, self.patch_dim):
            raise ValueError("Patch tensor does not match the configured video layout")
        b, t = patches.shape[:2]
        h, w = self.grid
        p, c = self.patch, self.channels
        return patches.reshape(b, t, h, w, p, p, c).permute(
            0, 1, 2, 4, 3, 5, 6
        ).reshape(b, t, *self.image_size, c)


def position_parameter(*shape):
    # Learned absolute positions and this initialization are implementation choices.
    result = nn.Parameter(torch.empty(*shape))
    nn.init.normal_(result, std=0.02)
    return result
