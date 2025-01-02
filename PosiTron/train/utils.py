import torch
from layers.utils import subsequent_mask

Tensor = torch.Tensor


class Batch:
    """Manage batch data with masking during training."""

    def __init__(self, src: Tensor, tgt: Tensor | None = None, pad: int = 2) -> None:
        """Manage batch data with masking during training.

        Args:
            src (Tensor): Source tensor [batch_size, seq_len]
            tgt (Tensor | None): Target tensor [batch_size, seq_len] (default is None)
            pad (int): Padding value (default is 2)
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.n_tokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt: Tensor, pad: int) -> Tensor:
        """
        Create standard target mask for transformer encoder.

        Args:
            tgt (Tensor): Target tensor [batch_size, seq_len]
            pad (int): Padding value

        Returns:
            tgt_mask (Tensor): Mask for target tensor [batch_size, seq_len, seq_len]
        """
        # Create a mask where the padding character is False, and the rest is True
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # Add a mask for the subsequent elements in the sequence
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(1)).type_as(tgt_mask.data)
        return tgt_mask


def lr_scheduler(step: int, model_size: int, factor: float, warmup: int) -> float:
    """Create a learning rate scheduler with warmup.

    Args:
        step (int): The step number.
        model_size (int): The model size.
        factor (float): The factor to scale the learning rate.
        warmup (int): The number of steps to warm up the learning rate.

    Returns:
        float: The learning rate.
    """
    step = max(1, step)  # Avoid zero raising to negative power
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
