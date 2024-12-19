from typing import TypeVar

import torch
import torch.nn as nn

from .utils import LayerNorm, SubLayerConnection, clone

T = TypeVar("T", bound=nn.Module)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer: T, N: int) -> None:
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through decoder layers.

        Args:
            x: Target sequence tensor
            memory: Encoder output tensor
            src_mask: Source padding mask
            tgt_mask: Target sequence mask

        Returns:
            Processed tensor after passing through all layers
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    A decoder layer consisting of self-attention, cross-attention, and feed-forward sublayers.

    Args:
        size: The size of the input and output tensors
        self_attn: The self-attention mechanism
        src_attn: The cross-attention mechanism
        dropout: The dropout rate to apply after each sublayer
        feed_forward: The feed-forward network
    """

    def __init__(
        self, size: int, self_attn: T, src_attn: T, feed_forward: T, dropout: float
    ) -> None:
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clone(SubLayerConnection(size, dropout), 3)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            x: Input tensor
            memory: Encoder memory
            src_mask: Source mask
            tgt_mask: Target mask

        Returns:
            Processed tensor after passing through sublayers
        """
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayers[2](x, self.feed_forward)
