import torch
import torch.nn as nn
import numpy as np


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True) -> None:
        super(TimeEncoder, self).__init__()
        if time_dim % 2 != 0:
            raise ValueError("Time dimension must be an even number.")

        self.time_dim = time_dim
        num_frequencies = self.time_dim // 2
        self.w = nn.Linear(1, num_frequencies)

        self.w.weight = nn.Parameter(
            (
                torch.from_numpy(
                    1 / 10 ** np.linspace(0, 9,
                                          num_frequencies, dtype=np.float32)
                )
            ).reshape(num_frequencies, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(num_frequencies))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # (batch_size, seq_len, num_frequencies)
        output_w = self.w(timestamps)

        output_sin = torch.sin(output_w)
        output_cos = torch.cos(output_w)

        output = torch.cat([output_sin, output_cos], dim=-1)

        return output
