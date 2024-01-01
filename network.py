import os
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING, Any, Iterable

import torch
from torch import nn


# prevent circular import
if TYPE_CHECKING:
    from environment import State, Environment, Action
    from dqn.dqn import ExperienceBatch, TdTargetBatch
else:
    Experience = object
    State = object
    Action = object
    Environment = object
    ExperienceBatch = object
    TdTargetBatch = object


class NeuralNetwork(nn.Module):
    @staticmethod
    def device() -> torch.device:
        """Utility function to determine whether we can run on GPU"""
        device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        return torch.device(device)

    @staticmethod
    def tensorify(array: Iterable) -> torch.Tensor:
        """Create a PyTorch tensor, and make sure it's on the GPU if possible"""
        return torch.tensor(array, device=NeuralNetwork.device())

    def __init__(self, inputs: int, outputs: int, neurons: int = 128):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inputs, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, outputs),
        )

        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-4, amsgrad=True)

        # move to gpu if possible
        self.to(NeuralNetwork.device())

    def create_copy(self):
        raise NotImplementedError  # need to subclass this

    def copy_from(self, other: "NeuralNetwork"):
        self.load_state_dict(other.state_dict())

    # do not call directly, call get_q_values() instead
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """PyTorch internal function to perform forward pass.
        Do not call directly, use get_q_values() instead.

        Args:
            state (torch.Tensor): a tensor of length 6 (the state has 6 variables)

        Returns:
            torch.Tensor: a tensor of length 3 (one q-value for each action)
        """

        return self.linear_relu_stack(state)

    def backprop(self, expected: torch.Tensor, actual: torch.Tensor):
        criterion = torch.nn.HuberLoss()
        loss = criterion(expected, actual)

        self.optim.zero_grad()
        loss.backward()

        # clip gradients
        nn.utils.clip_grad.clip_grad_value_(self.parameters(), 100.0)

        self.optim.step()  # gradient descent
