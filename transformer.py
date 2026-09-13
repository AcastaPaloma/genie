"""Reusable factorized spatiotemporal-transformer building blocks.

``SelfAttention`` is agnostic to the meaning of axes: it attends over the
penultimate (sequence) axis of ``(*, sequence_length, embedding_dim)``.
``SpatioTemporalTransformerBlock`` uses two independent instances of that
primitive: one over locations inside each frame and one over time at each
location.
"""

from __future__ import annotations
from typing import Literal, overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """Configurable multi-head self-attention over a sequence axis.

    The input may contain any number of leading batch-like dimensions. For
    example, spatial attention can receive ``(B, T, N, D)`` and attend over
    ``N``, while temporal attention can receive ``(B, N, T, D)`` and attend
    over ``T``.

    This primitive does not add positional embeddings, normalization,
    residual connections, or an FFN. Its caller owns those choices.

    Args:
        embedding_dim: Token width ``D``.
        num_heads: Number of independent attention heads.
        head_dim: Query, key, and value width per head. Defaults to
            ``embedding_dim // num_heads``.
        bias: Whether linear projections include biases.
        attention_dropout: Dropout probability applied to attention weights.
        projection_dropout: Dropout probability after the output projection.
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
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        _validate_attention_config(
            embedding_dim,
            num_heads,
            head_dim,
            attention_dropout,
            projection_dropout,
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or embedding_dim // num_heads
        self.attention_dim = num_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embedding_dim, self.attention_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, self.attention_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, self.attention_dim, bias=bias)
        self.out_proj = nn.Linear(self.attention_dim, embedding_dim, bias=bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)
        # Genie reports QK normalization, but not its exact implementation.
        # Use per-head LayerNorm on Q/K, retaining scaled dot-product attention.
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
        is_causal: bool = False,
        return_attention: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
        is_causal: bool = False,
        return_attention: Literal[True],
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
        is_causal: bool = False,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply self-attention over ``x``'s penultimate axis.

        Args:
            x: Floating-point token embeddings with shape ``(*, L, D)``.
            attention_mask: Optional mask broadcastable to
                ``(*, num_heads, L, L)``. A boolean ``True`` blocks an edge;
                a floating-point mask is added to the attention logits.
            is_causal: Block edges from a query position to future keys.
            return_attention: Return pre-dropout attention weights with shape
                ``(*, num_heads, L, L)``.
        """
        self._validate_input(x)
        sequence_length = x.shape[-2]

        q = self.q_norm(self._split_heads(self.q_proj(x)))
        k = self.k_norm(self._split_heads(self.k_proj(x)))
        v = self._split_heads(self.v_proj(x))

        if not return_attention:
            # Flatten leading batch axes so CUDA SDPA can select fused kernels.
            # This preserves model capacity while avoiding materialized NxN maps.
            leading = q.shape[:-3]
            q, k, v = (a.reshape(-1, self.num_heads, sequence_length, self.head_dim)
                       for a in (q, k, v))
            mask = None
            causal = is_causal
            if attention_mask is not None:
                if attention_mask.ndim < 2 or attention_mask.shape[-2:] != (sequence_length, sequence_length):
                    raise ValueError("attention_mask must end with (L, L)")
                if attention_mask.dtype != torch.bool and not attention_mask.is_floating_point():
                    raise TypeError("attention_mask must be boolean or floating-point")
                mask = attention_mask.to(device=x.device)
                if mask.dtype == torch.bool:
                    # Public interface True=blocked; SDPA True=allowed.
                    mask = ~mask
                    if is_causal:
                        mask = mask & ~self._causal_mask(sequence_length, x.device)
                else:
                    mask = mask.to(dtype=q.dtype)
                    if is_causal:
                        mask = mask.masked_fill(self._causal_mask(sequence_length, x.device), float("-inf"))
                mask = torch.broadcast_to(mask, (*leading, self.num_heads, sequence_length, sequence_length))
                mask = mask.reshape(-1, self.num_heads, sequence_length, sequence_length)
                causal = False
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=causal,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
            )
            attended = attended.reshape(*leading, self.num_heads, sequence_length, self.head_dim)
            attended = attended.transpose(-3, -2).reshape(*x.shape[:-2], sequence_length, self.attention_dim)
            return self.projection_dropout(self.out_proj(attended))

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = self._apply_masks(logits, attention_mask, is_causal)
        attention = torch.softmax(logits.float(), dim=-1).to(v.dtype)
        attention = torch.nan_to_num(attention)  # All-blocked rows have no update.
        attention_for_return = attention
        attention = self.attention_dropout(attention)

        attended = torch.matmul(attention, v)
        attended = attended.transpose(-3, -2).reshape(
            *x.shape[:-2], sequence_length, self.attention_dim
        )
        output = self.projection_dropout(self.out_proj(attended))

        if return_attention:
            return output, attention_for_return
        return output

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim < 3:
            raise ValueError(
                "Expected x with shape (*, sequence_length, embedding_dim), "
                f"got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError("x must contain floating-point token embeddings")
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {x.shape[-1]}"
            )

    def _split_heads(self, x: Tensor) -> Tensor:
        """Convert ``(*, L, H*d_h)`` to ``(*, H, L, d_h)``."""
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim).transpose(-3, -2)

    @staticmethod
    def _causal_mask(sequence_length: int, device: torch.device) -> Tensor:
        return torch.ones(
            sequence_length, sequence_length, dtype=torch.bool, device=device
        ).triu(diagonal=1)

    def _apply_masks(
        self,
        logits: Tensor,
        attention_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        if is_causal:
            logits = logits.masked_fill(
                self._causal_mask(logits.shape[-1], logits.device),
                float("-inf"),
            )

        if attention_mask is None:
            return logits
        if attention_mask.ndim < 2 or attention_mask.shape[-2:] != logits.shape[-2:]:
            raise ValueError(
                "attention_mask must end with (sequence_length, sequence_length); "
                f"got {tuple(attention_mask.shape)} for logits {tuple(logits.shape)}"
            )
        if attention_mask.dtype == torch.bool:
            return logits.masked_fill(attention_mask.to(device=logits.device), float("-inf"))
        if not attention_mask.is_floating_point():
            raise TypeError("attention_mask must be boolean or floating-point")
        return logits + attention_mask.to(device=logits.device, dtype=logits.dtype)


class FeedForward(nn.Module):
    """Position-wise transformer MLP that preserves the token width."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        expansion_factor: float = 4.0,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        hidden_dim = int(embedding_dim * expansion_factor)
        if hidden_dim <= 0:
            raise ValueError("embedding_dim * expansion_factor must be positive")

        self.in_proj = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.activation = nn.GELU()
        self.hidden_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim, bias=bias)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_dropout(self.out_proj(self.hidden_dropout(self.activation(self.in_proj(x)))))


class SpatioTemporalTransformerBlock(nn.Module):
    """One factorized spatial-attention, temporal-attention, and FFN block.

    Spatial attention is applied independently within each frame. Temporal
    attention then operates independently at each spatial location; by default
    it is causal, so frame ``t`` cannot read frame ``t + 1`` or later.

    Separate spatial and temporal attention instances have the same interface
    but distinct parameters. This is the factorized ST-transformer pattern used
    by Genie-style architectures.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        ffn_expansion_factor: float = 4.0,
        temporal_causal: bool = True,
        bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        attention_kwargs = {
            "head_dim": head_dim,
            "bias": bias,
            "attention_dropout": attention_dropout,
            "projection_dropout": projection_dropout,
            "qk_norm": qk_norm,
        }

        self.embedding_dim = embedding_dim
        self.temporal_causal = temporal_causal
        self.spatial_norm = nn.LayerNorm(embedding_dim)
        self.spatial_attention = SelfAttention(
            embedding_dim, num_heads, **attention_kwargs
        )
        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.temporal_attention = SelfAttention(
            embedding_dim, num_heads, **attention_kwargs
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForward(
            embedding_dim,
            expansion_factor=ffn_expansion_factor,
            bias=bias,
            dropout=ffn_dropout,
        )

    def forward(
        self,
        x: Tensor,
        *,
        temporal_attention_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Apply one ST block to ``(B, T, N, D)`` or ``(B, T, H, W, D)``.

        ``temporal_attention_mask`` has the same convention as
        :class:`SelfAttention` and is combined with the optional causal mask.
        """
        sequence, grid_shape = self._to_spatial_sequence(x)

        if return_attention:
            spatial_update, spatial_weights = self.spatial_attention(
                self.spatial_norm(sequence), return_attention=True
            )
        else:
            spatial_update = self.spatial_attention(self.spatial_norm(sequence))
        spatial_state = sequence + spatial_update

        temporal_input = self.temporal_norm(spatial_state).transpose(1, 2)
        if return_attention:
            temporal_update, temporal_weights = self.temporal_attention(
                temporal_input,
                attention_mask=temporal_attention_mask,
                is_causal=self.temporal_causal,
                return_attention=True,
            )
        else:
            temporal_update = self.temporal_attention(
                temporal_input,
                attention_mask=temporal_attention_mask,
                is_causal=self.temporal_causal,
            )
        temporal_state = spatial_state + temporal_update.transpose(1, 2)

        output = temporal_state + self.ffn(self.ffn_norm(temporal_state))
        output = self._restore_spatial_layout(output, grid_shape)

        if return_attention:
            return output, {"spatial": spatial_weights, "temporal": temporal_weights}
        return output

    def _to_spatial_sequence(self, x: Tensor) -> tuple[Tensor, tuple[int, int] | None]:
        if x.ndim not in (4, 5):
            raise ValueError(
                "Expected x with shape (B, T, N, D) or (B, T, H, W, D), "
                f"got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {x.shape[-1]}"
            )
        if x.ndim == 4:
            return x, None

        batch, frames, height, width, embedding = x.shape
        return x.reshape(batch, frames, height * width, embedding), (height, width)

    @staticmethod
    def _restore_spatial_layout(x: Tensor, grid_shape: tuple[int, int] | None) -> Tensor:
        if grid_shape is None:
            return x
        batch, frames, _, embedding = x.shape
        height, width = grid_shape
        return x.reshape(batch, frames, height, width, embedding)


def _validate_attention_config(
    embedding_dim: int,
    num_heads: int,
    head_dim: int | None,
    attention_dropout: float,
    projection_dropout: float,
) -> None:
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if head_dim is None and embedding_dim % num_heads:
        raise ValueError(
            "embedding_dim must be divisible by num_heads when head_dim is omitted"
        )
    if head_dim is not None and head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if not 0.0 <= attention_dropout < 1.0:
        raise ValueError("attention_dropout must be in [0, 1)")
    if not 0.0 <= projection_dropout < 1.0:
        raise ValueError("projection_dropout must be in [0, 1)")
