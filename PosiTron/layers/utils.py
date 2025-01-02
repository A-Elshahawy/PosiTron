import copy
import math
from typing import TypeVar

import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)
Tensor = torch.Tensor


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
        forward(x: Tensor) -> Tensor:
            Applies normalization to the input tensor `x` using learnable parameters.
    """

    def __init__(self, features: tuple, eps: float = 1e-6) -> None:
        super().__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer) -> Tensor:
        "Apply residual connection to any sublayer with the same size."

        return x + self.dropout(sublayer(self.norm(x)))


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_ff: int, d_model: int, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    """
    Implements word embeddings with scaled output for transformer models.

    Embeds input tokens into continuous vector space and scales by sqrt(d_model).
    The scaling helps maintain variance of activations through the network.
    """

    def __init__(self, d_model: int, vocab: int, *args, **kwargs) -> None:
        """
        Initialize the embedding layer.

        Args:
            d_model: Dimension of the embeddings
            vocab: Size of the vocabulary
        """
        super().__init__(*args, **kwargs)
        self.lut: nn.Embedding = nn.Embedding(vocab, d_model)
        self.d_model: int = d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        Embed and scale input tokens.

        Args:
            x: Input tensor of token indices [batch_size, seq_len]

        Returns:
            Scaled embedding tensor [batch_size, seq_len, d_model]
        """
        return self.lut(x) * math.sqrt(self.d_model)
