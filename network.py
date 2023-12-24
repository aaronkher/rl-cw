import os
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING, Any, Iterable

import torch
from torch import nn
from torch.nn.functional import relu


# prevent circular import
if TYPE_CHECKING:
    from environment import State, Environment, Action
    from dqn import Experience
else:
    Experience = object
    State = object
    Action = object
    Environment = object


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
    def device() -> torch.device:
        """Utility function to determine whether we can run on GPU"""
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return torch.device(device)

    @staticmethod
    def tensorify(array: Iterable) -> torch.Tensor:
        """Create a PyTorch tensor, and make sure it's on the GPU if possible"""
        return torch.tensor(array, device=NeuralNetwork.device())

    def __init__(self, env: Environment):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(env.observation_space_length, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, env.action_count),
        # )
        # self.layers = [
        #     nn.Linear(env.observation_space_length, 128),
        #     nn.Linear(128, 128),
        #     nn.Linear(128, env.action_count),
        # ]
        self.layer1 = nn.Linear(env.observation_space_length, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, env.action_count)

        # self.optim = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-2, amsgrad=True)

    # do not call directly, call get_q_values() instead
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """PyTorch internal function to perform forward pass.
        Do not call directly, use get_q_values() instead.

        Args:
            state (torch.Tensor): a tensor of length 6 (the state has 6 variables)

        Returns:
            torch.Tensor: a tensor of length 3 (one q-value for each action)
        """

        # return self.linear_relu_stack(state)
        x = relu(self.layer1(state))
        x = relu(self.layer2(x))
        return self.layer3(x)

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

        neural_network_output = self(state)
        return NeuralNetworkResult(neural_network_output)

    def get_q_values_batch(self, states: list[State]) -> list[NeuralNetworkResult]:
        batch_tensor = torch.tensor(states, dtype=torch.float32).to(self.device())
        batch_output = self(batch_tensor)
        return [NeuralNetworkResult(x) for x in batch_output]

    def get_best_action(self, state: State) -> Action:
        """Get the best action in a given state according to the neural network.

        Args:
            state (State): The state to get the best action for.

        Returns:
            Action: The best action in the given state.
        """

        neural_network_result = self.get_q_values(state)
        return neural_network_result.best_action()

    def backprop(self, experiences: list[Experience], td_targets: list[float]):
        self.optim.zero_grad()

        # Tensor[State, State, ...]
        # where State is Tensor[position, velocity]
        experience_states = torch.stack([exp.old_state for exp in experiences])

        # Tensor[[QValue * 3], [QValue * 3], ...]
        # where QValue is float
        q_values = self(experience_states)

        # Tensor[[Action], [Action], ...]
        # where Action is int
        actions_chosen = self.tensorify([[exp.action] for exp in experiences])

        # Tensor[[QValue], [QValue], ...]
        actions_chosen_q_values = q_values.gather(1, actions_chosen)

        # Tensor[[TDTarget], [TDTarget], ...]
        # where TDTarget is QValue
        td_targets_tensor = self.tensorify([[td_target] for td_target in td_targets])

        criterion = torch.nn.MSELoss()
        loss = criterion(actions_chosen_q_values, td_targets_tensor)
        loss.backward()

        self.optim.step()  # gradient descent
