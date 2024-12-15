from typing import TypeVar

import torch
import torch.nn as nn
from layers.utils import subsequent_mask

T = TypeVar("T", bound=nn.Module)
type Tensor = torch.Tensor


class GreedyDecoder:
    """Perform greedy decoding for sequence generation."""

    def __init__(self, model: T, max_len: int, start_symbol: int) -> None:
        """
        Initialize decoder with model and decoding parameters.

        Args:
            model: The model to use for decoding
            max_len: Maximum length of generated sequence
            start_symbol: Token to start sequence generation
        """
        self.model = model
        self.max_len = max_len
        self.start_symbol = start_symbol

    def decode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Perform greedy decoding on input sequence.

        Args:
            src: Source sequence tensor of shape (batch_size, seq_len)
            src_mask: Source sequence mask tensor of shape (batch_size, 1, seq_len)

        Returns:
            Generated sequence tensor of shape (batch_size, max_len)
        """
        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(self.start_symbol).type_as(src.data)

        for _ in range(self.max_len - 1):
            out = self.model.decode(
                memory, src_mask, subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )

        return ys
