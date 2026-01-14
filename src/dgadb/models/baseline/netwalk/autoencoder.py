import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int,  # num_nodes
        hidden_channels: int
    ) -> None:
        super().__init__()
        self.encoder_layer = nn.Linear(in_channels, hidden_channels, bias=True)
        self.decoder_layer = nn.Linear(hidden_channels, in_channels, bias=True)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.encoder_layer.weight, gain=1.0)
        nn.init.xavier_uniform_(self.decoder_layer.weight, gain=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # (batch_size, num_nodes)
        return torch.sigmoid(self.encoder_layer(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder_layer(h))

    def forward(self, x: torch.Tensor, corrupt_prob: float) -> tuple[torch.Tensor, torch.Tensor]:
        x_corrupted = (
            (x + torch.rand_like(x) * 0.1) * corrupt_prob +
            x * (1 - corrupt_prob)
            if self.training
            else x
        )
        h = self.encode(x_corrupted)
        x_hat = self.decode(h)
        return x_hat, h
