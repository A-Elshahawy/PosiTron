from __future__ import annotations

import math
from typing import Final, TypeVar

import torch
from PositionEncondings.alibi import ALiBi
from PositionEncondings.relativePE import RelativePositionEncoding as Rel_PE
from PositionEncondings.rotaryPE import RotaryPositionEncoding
from torch import Tensor, nn

from .utils import clone

T = TypeVar("T", bound=nn.Module)


class MultiHeadedAttention(nn.Module):
    """
    Base class for multi-headed attention mechanisms.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads: Final[int] = n_heads
        self.d_model: Final[int] = d_model
        self.d_k: Final[int] = d_model // n_heads
        self.dropout = nn.Dropout(p=dropout)
        self.linears = clone(nn.Linear(d_model, d_model), 4)

    def _project_qkv(
        self, query: Tensor, key: Tensor, value: Tensor, batch_size: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project and reshape query, key, and value tensors."""
        return [
            lin(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

    def _attention_forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Compute scaled dot-product attention."""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = scores.softmax(dim=-1)
        p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def _combine_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """Combine attention heads and apply final linear transformation."""
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)


class AbsoluteMultiHeadedAttention(MultiHeadedAttention):
    """
    Multi-headed attention with absolute position encoding.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__(n_heads, d_model, dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        batch_size = query.size(0)

        query, key, value = self._project_qkv(query, key, value, batch_size)

        x, self.attn = self._attention_forward(query, key, value, mask)
        return self._combine_heads(x, batch_size)


class RelativeMultiHeadAttention(MultiHeadedAttention):
    """
    Multi-headed attention with relative position encoding.
    """

    def __init__(
        self, d_model: int, n_heads: int, max_length: int, dropout: float
    ) -> None:
        super().__init__(n_heads, d_model, dropout)
        self.relative_pos_embedding = Rel_PE(self.d_k, max_length)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        batch_size, query_len, _ = query.size()
        batch_size, key_len, _ = key.size()

        # Project and split into heads
        Q = (
            self.linears[0](query)
            .view(batch_size, query_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.linears[1](key)
            .view(batch_size, key_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.linears[2](value)
            .view(batch_size, key_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (self.d_k**0.5)

        # Add relative positional scores
        rel_pos_embeddings = self.relative_pos_embedding(
            max(query_len, key_len)
        ).permute(2, 0, 1)
        rel_scores = torch.einsum(
            "bhqd,dqk->bhqk", Q, rel_pos_embeddings[:, :query_len, :key_len]
        )
        scores += rel_scores

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention probabilities and output
        attn_weights = scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.einsum("bhqk,bhvd->bhqd", attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        return output


class RotaryMultiHeadAttention(MultiHeadedAttention):
    """
    Multi-headed attention with rotary position encoding.
    """

    def __init__(
        self, n_heads: int, d_model: int, dropout: float = 0.1, max_length: int = 5000
    ) -> None:
        super().__init__(n_heads, d_model, dropout)
        self.rope = RotaryPositionEncoding(self.d_k, max_length=max_length)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key, value = (
            lin(x).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        )

        # Apply RoPE to query and key
        query = self.rope(query)
        key = self.rope(key)

        # Compute attention
        x, self.attn = self._attention_forward(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_heads * self.d_k)

        return self.linears[-1](x)


class AliBiMultiHeadAttention(MultiHeadedAttention):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(n_heads, d_model, dropout)
        self.alibi = ALiBi(self.n_heads)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        batch_size, query_len, _ = query.size()

        query, key, value = self._project_qkv(query, key, value, batch_size)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add ALiBi bias
        alibi_bias = self.alibi(query_len, query.device)  # Get ALiBi bias
        scores = scores + alibi_bias

        # Compute attention probabilities and output
        x, self.attn = self._attention_forward(query, key, value, mask)
        return self._combine_heads(x, batch_size)
