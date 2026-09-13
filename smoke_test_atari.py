"""Legacy filename: inspect one DOOM clip through one tokenizer-width ST block.

Uses the shared production dimensions; this is not a training entrypoint.
"""

import argparse
import torch
from torch import nn

from config import TOKENIZER_ENCODER as ARCH, IMAGE_SIZE, VIDEO_PATCH, SEQUENCE_LENGTH, FPS
from dataloader import build_loader, FRAMES_NPY, EPS_NPY
from patches import VideoPatches
from transformer import SpatioTemporalTransformerBlock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", default=FRAMES_NPY)
    parser.add_argument("--episodes-path", default=EPS_NPY)
    parser.add_argument("--source-fps", type=float, default=FPS)
    args = parser.parse_args()
    _, loader = build_loader(batch_size=1, num_workers=0, drop_last=False,
                             frames_path=args.frames_path, episodes_path=args.episodes_path,
                             source_fps=args.source_fps, T=SEQUENCE_LENGTH)
    video = next(iter(loader))
    patches = VideoPatches(IMAGE_SIZE, VIDEO_PATCH, 3).patchify(video)
    embed = nn.Linear(VIDEO_PATCH ** 2 * 3, ARCH.width)
    block = SpatioTemporalTransformerBlock(ARCH.width, ARCH.heads, head_dim=ARCH.head_dim).eval()
    with torch.inference_mode():
        output = block(embed(patches))
    assert output.shape == (*patches.shape[:-1], ARCH.width)
    assert torch.isfinite(output).all()
    print(f"PASS: video {tuple(video.shape)} -> ST features {tuple(output.shape)}")


if __name__ == "__main__":
    main()
