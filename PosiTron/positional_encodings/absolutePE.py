import math

import torch
import torch.nn as nn

Tensor = torch.Tensor


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
