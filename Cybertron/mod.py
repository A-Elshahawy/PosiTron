import math

import torch
import torch.nn as nn


class RotaryMultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, max_length: int, dropout: float = 0.1
    ):
        """
        Multi-Head Attention with Rotary Positional Embeddings.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            max_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.num_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Linear projections for query, key, and value
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Rotary positional embedding
        self.rotary_emb = RotaryPositionEncoding(d_model, max_length)

        # Dropout and softmax
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the input tensor into multiple heads.

        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor of shape [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the multiple heads back into a single tensor.

        Args:
            x: Tensor of shape [batch_size, num_heads, seq_len, d_k]

        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Forward pass for Multi-Head Attention with Rotary Embeddings.

        Args:
            query: Query tensor of shape [batch_size, seq_len, d_model]
            key: Key tensor of shape [batch_size, seq_len, d_model]
            value: Value tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor of shape [batch_size, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.size()

        # Project to query, key, and value
        Q = self.split_heads(
            self.query_proj(query)
        )  # [batch_size, num_heads, seq_len, d_k]
        K = self.split_heads(
            self.key_proj(key)
        )  # [batch_size, num_heads, seq_len, d_k]
        V = self.split_heads(
            self.value_proj(value)
        )  # [batch_size, num_heads, seq_len, d_k]

        # Apply rotary embeddings to Q and K
        Q, K = self.rotary_emb.apply_rotary_embeddings(Q, K)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # [batch_size, num_heads, seq_len, seq_len]

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match the attention scores dimensions
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention probabilities
        attention_probs = self.softmax(
            scores
        )  # [batch_size, num_heads, seq_len, seq_len]
        attention_probs = self.dropout(attention_probs)

        # Compute attention output
        context = torch.matmul(
            attention_probs, V
        )  # [batch_size, num_heads, seq_len, d_k]

        # Combine heads and project to output
        output = self.out_proj(
            self.combine_heads(context)
        )  # [batch_size, seq_len, d_model]

        return output


class RotaryPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        """
        Rotary Positional Embedding implementation.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model

        # Generate base frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cache for position embeddings
        position = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", position, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        self.register_buffer("pos_emb", emb)

    def _rotate_half(self, x):
        """
        Rotates half of the dimensions for rotary embeddings.

        Args:
            x: Input tensor [batch_size, num_heads, seq_len, head_dim]

        Returns:
            Rotated tensor [batch_size, num_heads, seq_len, head_dim]
        """
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_embeddings(self, query_layer, key_layer, value_layer=None):
        """
        Apply rotary position embeddings to query, key, and optional value layers.

        Args:
            query_layer: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key_layer: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value_layer: Optional value tensor [batch_size, num_heads, seq_len, head_dim]

        Returns:
            Transformed query, key, and optional value layers.
        """
        batch_size, num_heads, seq_len, head_dim = query_layer.size()

        # Extract sinusoidal positional embeddings
        sin, cos = self.pos_emb[:seq_len].chunk(2, dim=-1)
        sin = sin.view(1, 1, seq_len, -1)  # Shape: [1, 1, seq_len, d_model // 2]
        cos = cos.view(1, 1, seq_len, -1)  # Shape: [1, 1, seq_len, d_model // 2]

        # Match head_dim for query, key, and value layers
        sin = sin[:, :, :, :head_dim]
        cos = cos[:, :, :, :head_dim]

        # Apply rotary transformation
        def apply_to_layer(layer):
            rotated = self._rotate_half(layer)
            return layer * cos + rotated * sin

        query_layer = apply_to_layer(query_layer)
        key_layer = apply_to_layer(key_layer)

        if value_layer is not None:
            value_layer = apply_to_layer(value_layer)
            return query_layer, key_layer, value_layer

        return query_layer, key_layer


# Example usage
# if __name__ == "__main__":
#     # Example parameters
#     d_model = 512
#     max_len = 5000
#     batch_size = 2
#     num_heads = 8
#     seq_len = 10
#     head_dim = d_model // num_heads

#     # Initialize rotary positional encoding
#     rotary_encoding = RotaryPositionEncoding(d_model, max_len)

#     # Mock input layers
#     query_layer = torch.randn(batch_size, num_heads, seq_len, head_dim)
#     key_layer = torch.randn(batch_size, num_heads, seq_len, head_dim)
#     value_layer = torch.randn(batch_size, num_heads, seq_len, head_dim)

#     # Apply rotary embeddings
#     query_layer, key_layer, value_layer = rotary_encoding.apply_rotary_embeddings(
#         query_layer, key_layer, value_layer
#     )

#     print(f"Query layer shape: {query_layer.shape}")
#     print(f"Key layer shape: {key_layer.shape}")
#     print(f"Value layer shape: {value_layer.shape}")

# if __name__ == "__main__":
#     # Example parameters
#     d_model:int = 512
#     n_heads:int = 2
#     max_length:int = 5000
#     seq_len:int = 500
#     batch_size:int = 2

#     # Initialize multi-head attention with rotary embeddings
#     mha = RotaryMultiHeadAttention(d_model, n_heads, max_length)

#     # Mock input tensors
#     query = torch.randn(batch_size, seq_len, d_model)
#     key = torch.randn(batch_size, seq_len, d_model)
#     value = torch.randn(batch_size, seq_len, d_model)
#     mask = torch.ones(batch_size, seq_len, seq_len)  # Optional mask

#     # Compute attention output
#     output = mha(query, key, value, mask)
#     print(f"Output shape: {output.shape}")  # Expected: [batch_size, seq_len, d_model]
