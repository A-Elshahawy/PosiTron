import time
from typing import Any, Callable, Optional, TypeVar

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

T = TypeVar("T", bound=nn.Module)


class TrainingState:
    """Track training progress and statistics."""

    def __init__(self):
        self.steps = 0
        self.accumulated_steps = 0
        self.samples = 0
        self.n_tokens = 0


class Trainer:
    """Base class for training models."""

    @staticmethod
    def run_epoch(
        data_loader: DataLoader,
        model: T,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
        ],
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        mode: str = "train",
        accumulate_iter: int = 1,
        train_state: Optional[TrainingState] = None,
        batch_process_fn: Optional[
            Callable[
                [Any],
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    int,
                    torch.Tensor,
                ],
            ]
        ] = None,
        log_interval: int = 40,
    ) -> tuple[float, TrainingState]:
        """
        Runs one epoch of training or evaluation.

        Args:
            data_loader: DataLoader providing the data batches.
            model: The PyTorch model to be trained or evaluated.
            loss_fn: A callable that computes the loss. It should accept model outputs, target labels, and n_tokens.
            optimizer: The PyTorch optimizer.
            scheduler: (Optional) Learning rate scheduler.
            mode: "train" for training, "eval" for evaluation. Use "train+log" for detailed training.
            accumulate_iter: Number of batches to accumulate gradients before updating the model.
            train_state: (Optional) Training state object to track the process.
            batch_process_fn: (Optional) A function to pre process the batch before giving it to the forward method
            log_interval: Log details every log_interval batches.

        Returns:
             A tuple containing the average loss per token and the training state object.
        """
        train_state = train_state or TrainingState()
        start = time.time()
        total_tokens = total_loss = tokens = n_accumulated = 0

        pbar = tqdm(data_loader, desc=f"Epoch {train_state.steps}")
        for i, batch in enumerate(pbar):
            if batch_process_fn:
                src, tgt, src_mask, tgt_mask, n_tokens, tgt_y = batch_process_fn(batch)
            else:
                src, tgt, src_mask, tgt_mask, n_tokens, tgt_y = (
                    batch.src,
                    batch.tgt,
                    batch.src_mask,
                    batch.tgt_mask,
                    batch.n_tokens,
                    batch.tgt_y,
                )

            output = model.forward(src, tgt, src_mask, tgt_mask)

            loss, loss_node = loss_fn(output, tgt_y, n_tokens)

            if mode in ["train", "train+log"]:
                loss_node.backward()
                train_state.steps += 1
                train_state.samples += src.size(0)
                train_state.n_tokens = n_tokens

                if (i + 1) % accumulate_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accumulated += 1
                    train_state.accumulated_steps += 1

                if scheduler:
                    scheduler.step()

            total_loss += loss_node.item()
            total_tokens += n_tokens
            tokens += n_tokens

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / (i + 1):.4f}",
                    "tokens/sec": f"{tokens / (time.time() - start):.2f}",
                }
            )

            if (i + 1) % log_interval == 0 and mode in ["train", "train+log"]:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(
                    f"Epoch Step: {i:6d} | Accumulation Step: {n_accumulated:3d} | "
                    f"Loss: {loss / n_tokens:6.2f} | "
                    f"Tokens / Sec: {tokens / elapsed:7.1f} | "
                    f"Learning Rate: {lr:6.1e}"
                )
                start = time.time()
                tokens = 0

        return total_loss / total_tokens, train_state
