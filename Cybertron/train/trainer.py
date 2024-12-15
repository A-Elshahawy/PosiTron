import time
from typing import TypeVar

import torch.nn as nn

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
        data_iter,
        model,
        loss_fn,
        optimizer,
        scheduler=None,
        mode="train",
        accumulate_iter=1,
        train_state=None,
    ):
        train_state = train_state or TrainingState()
        start = time.time()
        total_tokens = total_loss = tokens = n_accumulated = 0

        for i, batch in enumerate(data_iter):
            output = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

            loss, loss_node = loss_fn(output, batch.tgt_y, batch.n_tokens)

            if mode in ["train", "train+log"]:
                loss_node.backward()
                train_state.steps += 1
                train_state.samples += batch.src.size(0)
                train_state.n_tokens = batch.n_tokens

                if i % accumulate_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_accumulated += 1
                    train_state.accumulated_steps += 1

                if scheduler:
                    scheduler.step()

            total_loss += loss_node.item()
            total_tokens += batch.n_tokens
            tokens += batch.n_tokens

            if i % 40 == 0 and mode in ["train", "train+log"]:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(
                    f"Epoch Step: {i:6d} | Accumulation Step: {n_accumulated:3d} | "
                    f"Loss: {loss / batch.n_tokens:6.2f} | "
                    f"Tokens / Sec: {tokens / elapsed:7.1f} | "
                    f"Learning Rate: {lr:6.1e}"
                )
                start = time.time()
                tokens = 0

        return total_loss / total_tokens, train_state
