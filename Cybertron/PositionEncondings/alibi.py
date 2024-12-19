import math

import torch

type Tensor = torch.Tensor


class AliBi(torch.nn.Module):
    def __init__(self, n_heads: int = 8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.slope = self._get_slopes(self, n_heads)
        self.register_buffer(
            "slopes_tensor", torch.tensor(self.slope).view(1, n_heads, 1, 1)
        )

    def _get_slopes(self, n):
        """
        Generates a list of slopes for ALiBi attention biases.

        Args:
            n (int): The number of attention heads.

        Returns:
            list: A list of floats representing the slopes for ALiBi.
        """

        def get_slopes_power_of_2(n):
            """Helper function to generate slopes when n is a power of 2."""
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def forward(self, seq_len: int) -> Tensor:
        positions = torch.arange(0, seq_len, dtype=torch.float32)
        alibi_bias = (
            (positions[:, None] + positions[None, :]).abs().unsqueeze(0).unsqueeze(0)
        )

        alibi = -alibi_bias * self.slopes_tensor

        return alibi
