import torch
import torch.nn as nn

Tensor = torch.Tensor


class RelativePositionEncoding(nn.Module):
    """
    Implements `Transformer-XL` relative positional encodings for transformer models.

    Generates learnable embeddings based on relative distances between sequence positions.
    Attributes:
        max_length: Maximum sequence length supported.
        d_k: Dimension of the position embeddings per attention head.
        relative_embedding: Embedding layer for relative positions.
    """

    def __init__(self, d_k: int, max_length: int) -> None:
        """
        Initialize relative position encoding.

        Args:
            d_k: Dimension of the position embeddings per head.
            max_length: Maximum sequence length supported.
        """
        super().__init__()
        self.d_k: int = d_k
        self.max_length: int = max_length
        self.relative_embedding = nn.Embedding(2 * max_length - 1, d_k)

    def forward(self, seq_len: int) -> Tensor:
        """
        Generate relative position embeddings for a given sequence length.

        Args:
            seq_len: Input sequence length.

        Returns:
            Tensor of shape (seq_len, seq_len, d_k).
        """
        # Generate relative position indices
        range_vec = torch.arange(seq_len, device=self.relative_embedding.weight.device)
        range_mat = range_vec[None, :] - range_vec[:, None]
        clipped_mat = torch.clamp(range_mat, -self.max_length + 1, self.max_length - 1)
        relative_positions = clipped_mat + self.max_length - 1

        return self.relative_embedding(relative_positions)
