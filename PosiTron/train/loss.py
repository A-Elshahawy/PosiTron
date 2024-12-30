from typing import TypeVar

import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)
Tensor = torch.Tensor


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0) -> None:
        """
        Initialize the LabelSmoothing module.

        Args:
            size (int): The number of classes in the classification task.
            padding_idx (int): The index of the padding token.
            smoothing (float, optional): The label smoothing value. Defaults to 0.0.
        """
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LabelSmoothing module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, vocab_size].
            target (torch.Tensor): Target tensor of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Tensor of shape [batch_size, seq_len, vocab_size] with smoothed labels.
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class LossCompute:
    def __init__(self, generator: T, criterion: T) -> None:
        """
        Initialize the LossCompute module.

        Args:
            generator (nn.Module): The generator module.
            criterion (nn.Module): The loss function module.
        """
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: Tensor, y: Tensor, norm: float) -> tuple[float, torch.Tensor]:
        """
        Compute the loss of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, vocab_size].
            y (torch.Tensor): Target tensor of shape [batch_size, seq_len].
            norm (float): The normalization factor.

        Returns:
            Tuple[float, torch.Tensor]: A tuple containing the loss value and the loss tensor.
        """
        x = self.generator(x)
        loss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return loss.data * norm, loss
