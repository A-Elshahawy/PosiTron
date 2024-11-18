"""
This module implements various position encoding and embedding mechanisms for transformer models using PyTorch.

Position encodings are crucial for providing positional information to transformer models, which inherently lack
a sense of order due to their attention mechanism. This module includes implementations for absolute, relative,
and rotary position encodings, as well as word embeddings with scaled output.

Classes:
    - Absolute_Position_Encoding: Implements absolute positional encoding using fixed sinusoidal functions.
    - Relative_Position_Encoding: Implements relative positional encoding with learnable embeddings based on
        relative distances between sequence positions.
    - Rotary_Position_Encoding: Placeholder for rotary position encoding, which is not yet implemented.
    - Embeddings: Implements word embeddings with scaled output to maintain variance through the network.

Usage:
    These classes are typically used in transformer models to enhance the model's ability to understand the order
    and relationships between tokens in a sequence. The positional encodings are added to the input embeddings
    before being fed into the transformer layers, while the Embeddings class is used to convert token indices
    into dense vectors.

Dependencies:
    - math: For mathematical operations.
    - typing: For type annotations, including TypeVar, TypeAlias, and Tuple.
    - torch: PyTorch library for building neural network components.
    - torch.nn: PyTorch's neural network module for defining layers and models.

Note:
    The Rotary_Position_Encoding class is currently a placeholder and does not contain any implementation. It can
    be extended in the future to include rotary position encoding functionality.
"""

import math
from typing import TypeAlias, TypeVar

import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)
Tensor: TypeAlias = torch.Tensor


class AbsolutePositionEncoding(nn.Module):
    "Implement the PE function of the base transformer."

    def __init__(
        self, d_model: int, dropout: float, max_lenght: int = 5000, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_lenght, d_model)
        position = torch.arange(0, max_lenght).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("AbsPE", pe)

    def forward(self, x: Tensor):
        x = x + self.AbsPE[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ? ==================================================================================================
# ? ==================================================================================================


class RelativePositionEncoding(nn.Module):
    """
    Implements `Transformer-XL` relative positional encodings for transformer models.

    Generates learnable embeddings based on relative distances between sequence positions.
    The embeddings are learned parameters rather than fixed sinusoidal functions.

    Attributes:
        max_length: Maximum sequence length supported
        d_model: Dimension of the position embeddings
        relative_embedding: Embedding layer for relative positions
    """

    def __init__(self, d_model: int, max_length: int, *args, **kwargs) -> None:
        """
        Initialize relative position encoding.

        Args:
            d_model: Dimension of the position embeddings
            max_length: Maximum sequence length supported
        """
        super().__init__(*args, **kwargs)
        self.max_length: int = max_length
        self.d_model: int = d_model
        vocab_size: int = 2 * max_length - 1
        self.relative_embedding: nn.Embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: int) -> Tensor:
        """
        Generate relative position embeddings.

        Args:
            x: Length of input sequence

        Returns:
            Tensor of shape (x, x, d_model) containing relative position embeddings
        """
        # Generate position indices
        range_vec: Tensor = torch.arange(
            x, device=self.relative_embedding.weight.device
        )

        # Calculate relative positions between all pairs
        range_mat: Tensor = range_vec[None, :] - range_vec[:, None]

        # Clip relative positions to max_length
        clipping_mat: Tensor = torch.clamp(
            range_mat, -self.max_length + 1, self.max_length - 1
        )

        # Shift to positive indices for embedding lookup
        relative_positions: Tensor = clipping_mat + self.max_length - 1

        return self.relative_embedding(relative_positions)


class RotaryPositionEncoding(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, sinusoidal_pos, query_layer, key_layer, value_layer=None):
        """
        Apply rotary position embeddings to the input tensors.

        Args:
            sinusoidal_pos: Tensor containing sinusoidal positional embeddings
            query_layer: Query tensor
            key_layer: Key tensor
            value_layer: Optional value tensor

        Returns:
            Tuple of tensors with rotary embeddings applied
        """
        # Split sinusoidal position embeddings into sin and cos components
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)

        # Create expanded sin/cos position embeddings
        # [θ0,θ1,θ2......θd/2-1] -> [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

        # Apply rotary embeddings to query
        rotate_half_query_layer = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
        ).reshape_as(query_layer)
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos

        # Apply rotary embeddings to key
        rotate_half_key_layer = torch.stack(
            [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
        ).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos

        # Optionally apply rotary embeddings to value
        if value_layer is not None:
            rotate_half_value_layer = torch.stack(
                [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
            ).reshape_as(value_layer)
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer

        return query_layer, key_layer


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
