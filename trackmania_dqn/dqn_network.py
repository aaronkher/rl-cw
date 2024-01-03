import os
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING, Any, Iterable

import torch
from torch import nn


from network import NeuralNetwork

# prevent circular import
if TYPE_CHECKING:
    from environment import State, Environment, Action
    from trackmania_dqn.dqn import TransitionBatch, TdTargetBatch
else:
    Experience = object
    State = object
    Action = object
    Environment = object
    TransitionBatch = object
    TdTargetBatch = object


@dataclass
class DqnNetworkResult:
    tensor: torch.Tensor

    def best_action(self) -> Action:
        argmax: torch.Tensor = self.tensor.argmax()  # this is a tensor with one item
        best_action = argmax.item()
        return cast(Action, best_action)

class DqnNetworkResultBatch:
    def __init__(self, tensor: torch.Tensor):
        # Tensor[[QValue * 3], [QValue * 3], ...]
        self.tensor = tensor

    def for_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # actions = Tensor[[Action], [Action], ...]
        # where Action is int

        # Tensor[[QValue], [QValue], ...]
        return self.tensor.gather(1, actions)

    def best_actions(self) -> torch.Tensor:
        return self.tensor.argmax(1)

    def __getitem__(self, index: int) -> DqnNetworkResult:
        """Override index operator e.g. batch[0] -> NeuralNetworkResult"""
        return DqnNetworkResult(self.tensor[index])

    def __mul__(self, other: float) -> "DqnNetworkResultBatch":
        """Override * operator e.g. batch * 0.9"""
        return DqnNetworkResultBatch(self.tensor * other)


class TrackManiaDqnNetwork(NeuralNetwork):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        inputs = observation_space.shape[0]
        outputs = action_space.shape[0]

        super(TrackManiaDqnNetwork, self).__init__(inputs, outputs)

    def create_copy(self) -> "TrackManiaDqnNetwork":
        copy = TrackManiaDqnNetwork(self.observation_space, self.action_space)
        copy.copy_from(self)
        return copy

    def get_q_values(self, state: State) -> DqnNetworkResult:
        """For a given state, pass it through the neural network and return
        the q-values for each action in that state.

        Args:
            state (State): The state to get the q-values for.

        Returns:
            NeuralNetworkResult: An object that wraps the raw tensor, with
                utility methods such as q_value_for_action() to make our lives
                easier.
        """

        neural_network_output = self(state.tensor)
        return DqnNetworkResult(neural_network_output)

    def get_q_values_batch(self, states: torch.Tensor) -> DqnNetworkResultBatch:
        # states = Tensor[State, State, ...]
        # where State is Tensor[position, velocity]

        batch_output = self(states)
        # batch_output = Tensor[[QValue * 3], [QValue * 3], ...]
        # where QValue is float

        return DqnNetworkResultBatch(batch_output)

    def get_best_action(self, state: State) -> Action:
        """Get the best action in a given state according to the neural network.

        Args:
            state (State): The state to get the best action for.

        Returns:
            Action: The best action in the given state.
        """

        neural_network_result = self.get_q_values(state)
        return neural_network_result.best_action()

    def train(self, prev_states, actions_chosen, td_targets: TdTargetBatch):
        # Tensor[[QValue * 3], [QValue * 3], ...]
        # where QValue is float
        q_values = self(prev_states)

        # Tensor[[QValue], [QValue], ...]
        actions_chosen_q_values = q_values.gather(1, actions_chosen)
        # y_hat = predicted (policy network)

        # Tensor[[TDTarget], [TDTarget], ...]
        # where TDTarget is QValue
        td_targets_tensor = td_targets.tensor.unsqueeze(1)
        # y = actual (target network)

        self.gradient_descent(actions_chosen_q_values, td_targets_tensor)
