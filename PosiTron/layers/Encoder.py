from typing import TypeVar

import torch.nn as nn
from torch import Tensor

from .utils import LayerNorm, SubLayerConnection, clone

T = TypeVar("T", bound=nn.Module)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer: T, N: int) -> None:
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder layer consisting of self-attention and feed-forward sub-layers.

    This layer applies self-attention followed by a feed-forward network to the input tensor.
    Each sub-layer is wrapped with a residual connection and layer normalization. The mask
    is used in the self-attention mechanism to prevent attending to certain positions.

    Args:
        size (int): The size of the input and output dimensions.
        self_Attn (nn.Module): Self-attention mechanism.
        feed_forward (nn.Module): Feed-forward network.
        dropout (float): Dropout rate.

    Attributes:
        size (int): The size of the input and output dimensions.
        self_Atten (nn.Module): Self-attention mechanism.
        feed_forward (nn.Module): Feed-forward network.
        dropout (nn.Dropout): Dropout layer.
        sub_layer (nn.ModuleList): List containing two SubLayerConnection modules.

    Methods:
        forward(x, mask): Performs a forward pass through the encoder layer.
    """

    def __init__(
        self, size: int, self_attn: T, feed_forward: T, dropout: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.size = size
        self.self_Atten = self_attn
        self.feed_forward = feed_forward
        self.sub_layer = clone(SubLayerConnection(self.size, dropout), 2)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.sub_layer[0](x, lambda x: self.self_Atten(x, x, x, mask))
        return self.sub_layer[1](x, self.feed_forward)
