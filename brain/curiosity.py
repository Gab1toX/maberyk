from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from brain.network import NeuralNetwork


class CuriosityModule(nn.Module):
    """Intrinsic curiosity based on next-observation prediction error."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_layers: Iterable[int] = (128, 128),
    ) -> None:
        super().__init__()

        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_layers = list(hidden_layers)

        predictor_input_size = observation_size + action_size
        error_input_size = observation_size * 2

        self.next_observation_predictor = NeuralNetwork(
            input_size=predictor_input_size,
            hidden_layers=self.hidden_layers,
            output_size=observation_size,
        )
        self.prediction_error_network = NeuralNetwork(
            input_size=error_input_size,
            hidden_layers=self.hidden_layers,
            output_size=1,
        )
        self.error_loss = nn.MSELoss()

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        predicted_next_observation = self.predict_next_observation(observation, action)
        result = {"predicted_next_observation": predicted_next_observation}

        if next_observation is not None:
            prediction_error = self.prediction_error(predicted_next_observation, next_observation)
            predicted_error = self.predict_error(predicted_next_observation, next_observation)
            result["prediction_error"] = prediction_error
            result["intrinsic_reward"] = prediction_error.detach()
            result["predicted_error"] = predicted_error

        return result

    def predict_next_observation(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        predictor_input = torch.cat((observation, action), dim=-1)
        return self.next_observation_predictor(predictor_input)

    def prediction_error(
        self,
        predicted_next_observation: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> torch.Tensor:
        squared_error = (predicted_next_observation - next_observation).pow(2)
        return squared_error.mean(dim=-1, keepdim=True)

    def predict_error(
        self,
        predicted_next_observation: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> torch.Tensor:
        error_input = torch.cat((predicted_next_observation.detach(), next_observation), dim=-1)
        return self.prediction_error_network(error_input)

    def reward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            predicted_next_observation = self.predict_next_observation(observation, action)
            return self.prediction_error(predicted_next_observation, next_observation)

    def loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> torch.Tensor:
        predicted_next_observation = self.predict_next_observation(observation, action)
        actual_error = self.prediction_error(predicted_next_observation, next_observation)
        predicted_error = self.predict_error(predicted_next_observation, next_observation)

        prediction_loss = actual_error.mean()
        error_prediction_loss = self.error_loss(predicted_error, actual_error.detach())
        return prediction_loss + error_prediction_loss

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "hidden_layers": self.hidden_layers,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "CuriosityModule":
        checkpoint = torch.load(path, map_location=map_location)
        module = cls(
            observation_size=checkpoint["observation_size"],
            action_size=checkpoint["action_size"],
            hidden_layers=checkpoint["hidden_layers"],
        )
        module.load_state_dict(checkpoint["state_dict"])
        return module
