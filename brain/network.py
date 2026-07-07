from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    """Simple fully connected neural network built with pure PyTorch."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: Iterable[int],
        output_size: int,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_layers = list(hidden_layers)
        self.output_size = output_size

        layer_sizes = [input_size, *self.hidden_layers, output_size]
        layers: list[nn.Module] = []

        for index in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[index], layer_sizes[index + 1]))
            if index < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "input_size": self.input_size,
                "hidden_layers": self.hidden_layers,
                "output_size": self.output_size,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device | None = None) -> "NeuralNetwork":
        checkpoint = torch.load(path, map_location=map_location)
        network = cls(
            input_size=checkpoint["input_size"],
            hidden_layers=checkpoint["hidden_layers"],
            output_size=checkpoint["output_size"],
        )
        network.load_state_dict(checkpoint["state_dict"])
        return network
