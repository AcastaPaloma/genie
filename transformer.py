"""Reusable spatiotemporal-transformer building blocks.

The classes here operate on token embeddings, rather than pixels or discrete
code IDs. Component-specific modules (a video tokenizer, a latent-action
model, or a dynamics model) are responsible for producing those embeddings
and for adding positional or action conditioning.
"""

from __future__ import annotations

from typing import Literal, overload

import torch
from torch import Tensor, nn


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over locations within each video frame.

    Each frame attends over its own spatial locations only. The time axis is
    kept independent, which makes this module usable as the spatial part of a
    factorized spatiotemporal transformer.

    Accepted inputs are either a token sequence ``(B, T, N, D)`` or a spatial
    grid ``(B, T, H, W, D)``. The output has exactly the same shape as the
    input. This module intentionally does not add positional embeddings,
    normalization, residual connections, or a feed-forward network; those are
    responsibilities of the surrounding transformer block.

    Args:
        embedding_dim: Token width ``D``.
        num_heads: Number of independent attention heads.
        head_dim: Query/key/value width per head. Defaults to
            ``embedding_dim // num_heads``.
        bias: Whether linear projections include biases.
        attention_dropout: Dropout probability applied to normalized attention
            weights.
        projection_dropout: Dropout probability applied after the output
            projection.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if head_dim is None:
            if embedding_dim % num_heads:
                raise ValueError(
                    "embedding_dim must be divisible by num_heads when head_dim is omitted"
                )
            head_dim = embedding_dim // num_heads
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0.0 <= projection_dropout < 1.0:
            raise ValueError("projection_dropout must be in [0, 1)")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention_dim = num_heads * head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(embedding_dim, self.attention_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, self.attention_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, self.attention_dim, bias=bias)
        self.out_proj = nn.Linear(self.attention_dim, embedding_dim, bias=bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        return_attention: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        return_attention: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply spatial self-attention.

        Args:
            x: Float token embeddings of shape ``(B, T, N, D)`` or
                ``(B, T, H, W, D)``.
            return_attention: Return the pre-dropout attention weights with
                shape ``(B, T, num_heads, N, N)`` for inspection.
        """
        sequence, original_layout = self._to_sequence(x)
        batch, frames, locations, _ = sequence.shape

        q = self._split_heads(self.q_proj(sequence))
        k = self._split_heads(self.k_proj(sequence))
        v = self._split_heads(self.v_proj(sequence))

        # (B, T, heads, N, head_dim) @ (B, T, heads, head_dim, N)
        # gives one N-by-N spatial attention map per frame and per head.
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        attention_for_return = attention
        attention = self.attention_dropout(attention)

        attended = torch.matmul(attention, v)
        attended = attended.transpose(2, 3).reshape(
            batch, frames, locations, self.attention_dim
        )
        output = self.projection_dropout(self.out_proj(attended))
        output = self._restore_layout(output, original_layout)

        if return_attention:
            return output, attention_for_return
        return output

    def _split_heads(self, x: Tensor) -> Tensor:
        """Convert ``(B, T, N, H*d_h)`` to ``(B, T, H, N, d_h)``."""
        batch, frames, locations, _ = x.shape
        return x.reshape(
            batch, frames, locations, self.num_heads, self.head_dim
        ).transpose(2, 3)

    def _to_sequence(self, x: Tensor) -> tuple[Tensor, tuple[int, int] | None]:
        if not x.is_floating_point():
            raise TypeError("x must contain floating-point token embeddings")
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {x.shape[-1]}"
            )

        if x.ndim == 4:
            return x, None
        if x.ndim == 5:
            batch, frames, height, width, embedding = x.shape
            return x.reshape(batch, frames, height * width, embedding), (height, width)
        raise ValueError(
            "Expected x with shape (B, T, N, D) or (B, T, H, W, D), "
            f"got {tuple(x.shape)}"
        )

    @staticmethod
    def _restore_layout(x: Tensor, layout: tuple[int, int] | None) -> Tensor:
        if layout is None:
            return x
        batch, frames, _, embedding = x.shape
        height, width = layout
        return x.reshape(batch, frames, height, width, embedding)
