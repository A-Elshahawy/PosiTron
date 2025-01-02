import math

import torch
import torch.nn as nn

Tensor: type = torch.Tensor


class RotaryPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_length: int = 5000):
        """
        Args:
            dim: Head dimension (must be even to split into sin and cos components).
            max_length: Maximum sequence length for positional encoding.
        """
        super().__init__()
        assert dim % 2 == 0, "Head dimension must be even for RoPE."

        self.dim = dim
        self.max_length = max_length

        # Precompute sine and cosine terms
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        self.register_buffer("sin", torch.sin(position * div_term), persistent=False)
        self.register_buffer("cos", torch.cos(position * div_term), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, num_heads, seq_len, head_dim]

        Returns:
            Rotated tensor of the same shape as input.
        """
        *_, seq_len, head_dim = x.size()

        # Ensure RoPE is compatible with the input's head_dim
        sin = self.sin[:seq_len, : head_dim // 2].unsqueeze(0).unsqueeze(0)
        cos = self.cos[:seq_len, : head_dim // 2].unsqueeze(0).unsqueeze(0)

        # Apply rotations
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split even and odd dimensions
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated
