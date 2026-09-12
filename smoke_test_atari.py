"""Run one real Atari Breakout sequence through the factorized transformer."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F

from dataloader import load_breakout_sequence
from transformer import SpatioTemporalTransformerBlock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()

    frames, actions = load_breakout_sequence(sequence_length=args.frames)
    video = torch.from_numpy(frames).float().div_(255).unsqueeze(0)
    _, _, height, width, channels = video.shape
    if height % args.patch_size or width % args.patch_size:
        raise ValueError("patch-size must divide the 84x84 Atari frame dimensions")

    patches = F.avg_pool2d(video.permute(0, 1, 4, 2, 3).flatten(0, 1), args.patch_size, args.patch_size)
    patch_height, patch_width = patches.shape[-2:]
    patches = patches.reshape(1, args.frames, channels, patch_height, patch_width).permute(0, 1, 3, 4, 2)

    embed = nn.Linear(channels, args.embedding_dim)
    model = SpatioTemporalTransformerBlock(args.embedding_dim, args.heads).eval()
    with torch.inference_mode():
        output = model(embed(patches))

    expected_shape = (1, args.frames, patch_height, patch_width, args.embedding_dim)
    assert output.shape == expected_shape, (output.shape, expected_shape)
    assert torch.isfinite(output).all(), "transformer produced NaN or infinity"
    action_info = "no actions" if actions is None else f"actions={actions.tolist()}"
    print(f"PASS: {tuple(frames.shape)} -> {tuple(output.shape)}; {action_info}")


if __name__ == "__main__":
    main()
