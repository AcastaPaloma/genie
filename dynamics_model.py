"""Genie 1 dynamics: full Table 12 width, depth and attention configuration.

The paper leaves the exact input embedding modules unspecified. Support the
user's codebook-vector path as well as discrete video token IDs. Both produce
model-width embeddings before action addition. Training/sampling is external.
"""

import torch
from torch import nn
from torch.nn import functional as F

from config import (DYNAMICS as ARCH, VIDEO_CODES, CODE_WIDTH, IMAGE_SIZE,
                    VIDEO_PATCH, SEQUENCE_LENGTH)
from patches import VideoPatches, position_parameter
from transformer import SpatioTemporalTransformerBlock


class DynamicsModel(nn.Module):
    def __init__(self, d_width=ARCH.width, st_blocks=ARCH.layers,
                 st_heads=ARCH.heads, head_dim=ARCH.head_dim, *,
                 img=IMAGE_SIZE, patch=VIDEO_PATCH, T=SEQUENCE_LENGTH,
                 num_codes=VIDEO_CODES, action_width=CODE_WIDTH,
                 video_width=CODE_WIDTH, qk_norm=ARCH.qk_norm,
                 ffn_expansion=ARCH.ffn_expansion):
        super().__init__()
        if min(d_width, st_blocks, T, num_codes, action_width, video_width) < 1:
            raise ValueError("Widths, blocks, context and vocabulary must be positive")
        self.n_patches = VideoPatches(img, patch, 3).n_patches
        self.d_width, self.num_codes, self.mask_id = d_width, num_codes, num_codes
        self.action_width, self.video_width = action_width, video_width
        self.video_embedding = nn.Embedding(num_codes + 1, d_width)
        self.video_proj = nn.Linear(video_width, d_width)
        self.action_proj = nn.Linear(action_width, d_width)
        self.spatial_pos = position_parameter(self.n_patches, d_width)
        self.temporal_pos = position_parameter(T, d_width)
        self.trans_blocks = nn.ModuleList([
            SpatioTemporalTransformerBlock(
                d_width, st_heads, head_dim=head_dim, qk_norm=qk_norm,
                ffn_expansion_factor=ffn_expansion, temporal_causal=True,
            ) for _ in range(st_blocks)
        ])
        self.output_norm = nn.LayerNorm(d_width)
        self.to_logits = nn.Linear(d_width, num_codes)

    @property
    def blocks(self):
        return self.trans_blocks

    def forward(self, video_tokens, action_tokens, *, mask=None):
        """Return token logits (B,T,N,K), not a shape or decoded video.

        video_tokens: integer IDs (B,T,N), or quantized vectors (B,T,N,32).
        action_tokens: LAM codebook vectors (B,T-1,32), for transitions t -> t+1.
        mask: boolean (B,T,N), True at positions to replace with learned MASK.

        The selected action is added to every patch of its destination frame;
        the first frame has no incoming action. Use one video representation
        consistently during training and generation; its input layers are distinct.
        The training loop masks targets and computes token cross-entropy.
        """
        is_ids = video_tokens.ndim == 3 and video_tokens.dtype in (torch.int32, torch.int64)
        is_vectors = (video_tokens.ndim == 4 and video_tokens.is_floating_point()
                      and video_tokens.shape[-1] == self.video_width)
        if not (is_ids or is_vectors) or video_tokens.shape[2] != self.n_patches:
            raise ValueError("Expected integer IDs (B,T,N) or quantized vectors (B,T,N,video_width)")
        b, t, n = video_tokens.shape[:3]
        if not 1 <= t <= self.temporal_pos.shape[0]:
            raise ValueError("Frame count exceeds configured dynamics context")
        if tuple(action_tokens.shape) != (b, t - 1, self.action_width):
            raise ValueError("Expected T-1 transition action vectors (B,T-1,action_width)")
        if not action_tokens.is_floating_point():
            raise TypeError("Actions must be floating-point LAM codebook vectors")
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (b, t, n)):
            raise ValueError("mask must be boolean with shape (B,T,N)")
        if is_ids:
            if mask is not None:
                video_tokens = video_tokens.masked_fill(mask, self.mask_id)
            x = self.video_embedding(video_tokens)
        else:
            # Tokenizer is pretrained/frozen during dynamics training.
            x = self.video_proj(video_tokens.detach())
            if mask is not None:
                x = torch.where(mask.unsqueeze(-1), self.video_embedding.weight[self.mask_id], x)
        actions = self.action_proj(action_tokens.detach())
        # Pad AFTER projection: first frame gets zero even when projection has bias.
        actions = F.pad(actions, (0, 0, 1, 0)).unsqueeze(2)
        x = x + actions + self.spatial_pos[None, None] + self.temporal_pos[None, :t, None]
        for block in self.trans_blocks:
            x = block(x)
        return self.to_logits(self.output_norm(x))


if __name__ == "__main__":
    # Inspect full-size settings without an import-time allocation of billions of weights.
    with torch.device("meta"):
        model = DynamicsModel()
    print(f"{len(model.trans_blocks)} ST blocks, width {model.d_width}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
