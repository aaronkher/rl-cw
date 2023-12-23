import os
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from environment import State, Environment, Action


@dataclass
class NeuralNetworkResult:
    tensor: torch.Tensor

    def best_action(self) -> Action:
        argmax: torch.Tensor = self.tensor.argmax()  # this is a tensor with one item
        best_action = argmax.item()
        return cast(Action, best_action)

    def best_action_q_value(self) -> float:
        return self.tensor[self.best_action()].item()

    def q_value_for_action(self, action: Action) -> float:
        return self.tensor[action].item()


class NeuralNetwork(nn.Module):
    @staticmethod
    def device() -> str:
        """Utility function to determine whether we can run on GPU"""
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return device

    def __init__(self, env: Environment):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(env.observation_space_length, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_count),
        )

        self.optim = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)

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

    # need to return the q value for an action AND
    # return the corresponding action so DQN class knows what to use
    def get_q_values(self, state: State) -> NeuralNetworkResult:
        """For a given state, pass it through the neural network and return
        the q-values for each action in that state.

        Args:
            state (State): The state to get the q-values for.

        Returns:
            NeuralNetworkResult: An object that wraps the raw tensor, with
                utility methods such as q_value_for_action() to make our lives
                easier.
        """

        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device())
        neural_network_output = self(state_tensor)
        return NeuralNetworkResult(neural_network_output)

    def get_best_action(self, state: State) -> Action:
        """Get the best action in a given state according to the neural network.

        Args:
            state (State): The state to get the best action for.

        Returns:
            Action: The best action in the given state.
        """

        neural_network_result = self.get_q_values(state)
        return neural_network_result.best_action()

    def backprop(self, nn_result: NeuralNetworkResult, td_target: float):
        # "prediction" is y_t and "label" is y_hat.

        # How should this function work?
        # should it take floats as parameters, or tensors?
        # It should use MSE and gradient descent. right?
        # gradient descent is for updating the whole list of q-values. here
        # we are only trying to update one q-value. how does this work?

        # raise "TODO"

        self.optim.zero_grad()

        y_hat = nn_result.tensor
        y_t = nn_result.tensor.clone()
        best_action = nn_result.best_action()
        y_t[best_action] = td_target

        criterion = torch.nn.MSELoss()
        loss = criterion(y_hat, y_t)

        loss.backward()

        self.optim.step()  # gradient descent

        # criterion = torch.nn.MSELoss()
        # predictions = self(state)
        # loss = criterion(predictions, label)

        # optim = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        # optim.step()  # gradient descent

        # loss.backward()  # backward pass
