import copy
from typing import TypeVar

import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)


def clone(module: T, N: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class LayerNorm(nn.Module):
    """
    A custom normalization layer that normalizes input tensor along the last dimension.

    Attributes:
        a_2 (nn.Parameter): Learnable scaling parameter initialized to ones.
        b_2 (nn.Parameter): Learnable shifting parameter initialized to zeros.
        eps (float): A small constant added to the standard deviation for numerical stability.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies normalization to the input tensor `x` using learnable parameters.
    """

    def __init__(self, features: tuple, eps: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float) -> None:
        super(SubLayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer) -> torch.Tensor:
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_ff: int, d_model: int, dropout=0.1, *args, **kwargs):
        super(PositionWiseFeedForward, self).__init__(*args, **kwargs)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
