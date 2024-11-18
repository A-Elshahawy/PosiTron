from __future__ import annotations

import math
from typing import Final, TypeVar

import torch
from torch import Tensor, nn

from .utils import clone

T = TypeVar("T", bound=nn.Module)


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention mechanism as described in 'Attention is All You Need'.

    Allows the model to jointly attend to information from different representation
    subspaces at different positions.

    Attributes:
        h: Number of attention heads
        d_model: Model dimension
        d_k: Dimension of keys/queries (d_model // h)
        attn: Attention weights from the last forward pass
        linears: Linear transformations for Q, K, V, and output
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1) -> None:
        """
        Initialize the Multi-headed attention module.

        Args:
            h: Number of attention heads
            d_model: Model dimension (must be divisible by h)
            dropout: Dropout probability

        Raises:
            AssertionError: If d_model is not divisible by h
        """
        super().__init__()
        assert d_model % h == 0, f"d_model ({d_model}) must be divisible by h ({h})"

        # Save dimensions
        self.h: Final[int] = h
        self.d_model: Final[int] = d_model
        self.d_k: Final[int] = d_model // h

        # Initialize layers
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.attn: Tensor | None = None

        # Create four identical linear transformations
        self.linears: nn.ModuleList = clone(nn.Linear(self.d_model, self.d_model), 4)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """
        Compute multi-headed attention.

        Args:
            query: Query tensor of shape [batch_size, query_len, d_model]
            key: Key tensor of shape [batch_size, key_len, d_model]
            value: Value tensor of shape [batch_size, value_len, d_model]
            mask: Optional mask tensor of shape [batch_size, 1, query_len, key_len]

        Returns:
            Output tensor of shape [batch_size, query_len, d_model]
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size: int = query.size(0)

        # Linear projections and reshape
        query, key, value = self._project_qkv(query, key, value, batch_size)

        # Apply attention
        x, self.attn = self._attention_forward(query, key, value, mask, self.dropout)

        # Combine heads and apply final linear transformation
        return self._combine_heads(x, batch_size)

    def _project_qkv(
        self, query: Tensor, key: Tensor, value: Tensor, batch_size: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project and reshape query, key, and value tensors."""
        return [
            lin(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

    def _attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
        dropout: nn.Dropout,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query: Transformed query tensor
            key: Transformed key tensor
            value: Transformed value tensor
            mask: Optional attention mask
            dropout: Dropout module

        Returns:
            Tuple of (attended values, attention weights)
        """
        d_k: Final[int] = query.size(-1)
        scores: Tensor = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn: Tensor = scores.softmax(dim=-1)
        p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def _combine_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """Combine attention heads and apply final linear transformation."""
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


"""
from typing import TypeVar, TypeAlias, Tuple
import torch
import torch.nn as nn
from .utils import clone


T = TypeVar("T", bound=nn.Module)
Tensor : TypeAlias = torch.Tensor

class MultiHeadedAttention(nn.Module):


    def __init__(self, h: int, d_model: int, dropout: float=0.1, *args, **kwargs):
        super(MultiHeadedAttention, self).__init__(*args, **kwargs)
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # * assume d_v always equals d_k
        self.d_k = d_model // h
        self.attn = None
        self.linears = clone(nn.Linear(self.d_model, self.d_model), 4)


    def forward(self, query:Tensor, key: Tensor, value: Tensor, mask:Tensor=None):
        if mask is not None:
            # * same mask applied for all heads
            mask = mask.unsqueeze()
        nbatches = query.size(0)

        # ?  1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # ? 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # ? 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


    def attention(self, query: Tensor,
                    key: Tensor,
                    value: Tensor,
                    mask: Tensor=None,
                    dropout: float=None
                    )-> Tuple[Tensor| Tensor]:

        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e-9)
        p_attn = scores.softmax(dim=-1)

        if dropout is not None:
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn , value), p_attn
    """

"""
from typing import Optional, Tuple
import torch
from torch import nn
from torch import Tensor
from .utils import clone

class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert d_model % h == 0, "d_model must be divisible by h"
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        self.dropout = nn.Dropout(p=dropout)
        self.attn: Optional[Tensor] = None
        self.linears = clone(nn.Linear(self.d_model, self.d_model), 4)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # * 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # * 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # * 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        dropout: Optional[nn.Dropout] = None
    ) -> Tuple[Tensor, Tensor]:
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = scores.softmax(dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
        """
