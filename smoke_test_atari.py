"""Run one real Atari Breakout sequence through the factorized transformer."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F

from dataloader import load_breakout_sequence
from transformer import SpatioTemporalTransformerBlock


def show(name: str, tensor: torch.Tensor) -> None:
    print(f"{name:32} shape={tuple(tensor.shape)!s:20} dtype={tensor.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()

    frames, actions = load_breakout_sequence(sequence_length=args.frames)
    print("\nInput from the Atari dataset")
    print(f"frames (NumPy)                   shape={frames.shape} dtype={frames.dtype}")
    if actions is None:
        print("actions                            absent")
    else:
        print(f"actions (NumPy labels)           shape={actions.shape} dtype={actions.dtype} values={actions.tolist()}")

    # ArrayRecord bytes are read-only; copy before converting to a PyTorch tensor.
    video = torch.from_numpy(frames.copy()).float().div_(255).unsqueeze(0)
    print("\nVideo-to-token preparation")
    show("normalized video (B,T,H,W,C)", video)
    _, _, height, width, channels = video.shape
    if height % args.patch_size or width % args.patch_size:
        raise ValueError("patch-size must divide the 84x84 Atari frame dimensions")

    image_batch = video.permute(0, 1, 4, 2, 3).flatten(0, 1)
    show("channels-first frames (B*T,C,H,W)", image_batch)
    patches = F.avg_pool2d(image_batch, args.patch_size, args.patch_size)
    show("pooled patches (B*T,C,h,w)", patches)
    patch_height, patch_width = patches.shape[-2:]
    patches = patches.reshape(1, args.frames, channels, patch_height, patch_width).permute(0, 1, 3, 4, 2)
    show("patch grid (B,T,h,w,C)", patches)

    embed = nn.Linear(channels, args.embedding_dim)
    model = SpatioTemporalTransformerBlock(args.embedding_dim, args.heads).eval()
    embedded_patches = embed(patches)
    show("embedded tokens (B,T,h,w,D)", embedded_patches)
    print("\nInside the factorized transformer")
    print(f"spatial sequence (B,T,N,D)       shape={(1, args.frames, patch_height * patch_width, args.embedding_dim)}")
    print(f"spatial attention (B,T,heads,N,N) shape={(1, args.frames, args.heads, patch_height * patch_width, patch_height * patch_width)}")
    print(f"temporal sequence (B,N,T,D)       shape={(1, patch_height * patch_width, args.frames, args.embedding_dim)}")
    print(f"temporal attention (B,N,heads,T,T) shape={(1, patch_height * patch_width, args.heads, args.frames, args.frames)}")
    with torch.inference_mode():
        output = model(embedded_patches)
    show("transformer output (B,T,h,w,D)", output)

    expected_shape = (1, args.frames, patch_height, patch_width, args.embedding_dim)
    assert output.shape == expected_shape, (output.shape, expected_shape)
    assert torch.isfinite(output).all(), "transformer produced NaN or infinity"
    print("\nPASS: the real frame sequence completed the transformer forward pass.")
    print("Actions are dataset labels only in this smoke test; the transformer has no action input or prediction head yet.")


if __name__ == "__main__":
    main()
