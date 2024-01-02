import os
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING, Any, Iterable

import torch
from torch import nn


# prevent circular import
if TYPE_CHECKING:
    from environment import State, Environment, Action
    from dqn import ExperienceBatch, TdTargetBatch
else:
    Experience = object
    State = object
    Action = object
    Environment = object
    ExperienceBatch = object
    TdTargetBatch = object


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


class NeuralNetworkResultBatch:
    def __init__(self, tensor: torch.Tensor):
        # Tensor[[QValue * 3], [QValue * 3], ...]
        self.tensor = tensor

    def __getitem__(self, index: int) -> NeuralNetworkResult:
        """Override index operator e.g. batch[0] -> NeuralNetworkResult"""
        return NeuralNetworkResult(self.tensor[index])

    def __mul__(self, other: float) -> "NeuralNetworkResultBatch":
        """Override * operator e.g. batch * 0.9"""
        return NeuralNetworkResultBatch(self.tensor * other)


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
        
        # ACTION SPACE DETAILS
        self.ACTIONS_COUNT = 5 # TODO: DON'T HARDCODE

        # STATE SPACE DETAILS
        self.IMAGE_ROWS = 5 # TODO: DON'T HARDCODE
        self.IMAGE_COLUMNS = 5 # TODO: DON'T HARDCODE
        self.IMAGE_CHANNELS = 1 # 1 for greyscale # TODO: DON'T HARDCODE

        # CNN - for input image processing
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.IMAGE_CHANNELS, out_channels=16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            input_size = torch.zeros(1, self.IMAGE_CHANNELS, self.IMAGE_ROWS, self.IMAGE_COLUMNS)
            # flatten for passing to FNN
            self.flattened_size = self.conv_layers(input_size).shape[1]

        # Feed Forward Neural Network (FNN) - for DQN
        ffn_input_neurons = self.flattened_size
        ffn_neurons = 128
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(ffn_input_neurons, ffn_neurons),
            nn.ReLU(),
            nn.Linear(ffn_neurons, self.ACTIONS_COUNT)
        )

        self.optim = torch.optim.AdamW(self.parameters(), lr=1e-4, amsgrad=True)

    # do not call directly, call get_q_values() instead
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """PyTorch internal function to perform forward pass.
        Do not call directly, use get_q_values() instead.

        Args:
            state (torch.Tensor): a tensor of length 6 (the state has 6 variables)

        Returns:
            torch.Tensor: a tensor of length 3 (one q-value for each action)
        """
        x = self.conv_layers(state)
        return self.linear_relu_stack(x)

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
        batch_size = 1
        state_tensor = state.tensor.view(batch_size, self.IMAGE_CHANNELS, self.IMAGE_ROWS, self.IMAGE_COLUMNS)  
        neural_network_output = self(state_tensor)
        return NeuralNetworkResult(neural_network_output)

    def get_q_values_batch(self, states: torch.Tensor) -> NeuralNetworkResultBatch:
        # states = Tensor[State, State, ...]
        # where State is Tensor[position, velocity]

        batch_output = self(states)
        # batch_output = Tensor[[QValue * 3], [QValue * 3], ...]
        # where QValue is float

        return NeuralNetworkResultBatch(batch_output)

    def get_best_action(self, state: State) -> Action:
        """Get the best action in a given state according to the neural network.

        Args:
            state (State): The state to get the best action for.

        Returns:
            Action: The best action in the given state.
        """

        neural_network_result = self.get_q_values(state)
        return neural_network_result.best_action()

    def backprop(self, experiences: ExperienceBatch, td_targets: TdTargetBatch):
        # Tensor[State, State, ...]
        # where State is Tensor[position, velocity]
        experience_states = experiences.old_states

        # Tensor[[QValue * 3], [QValue * 3], ...]
        # where QValue is float
        q_values = self(experience_states)

        # Tensor[[Action], [Action], ...]
        # where Action is int
        actions_chosen = experiences.actions

        # Tensor[[QValue], [QValue], ...]
        actions_chosen_q_values = q_values.gather(1, actions_chosen)
        # y_hat = predicted (policy network)

        # Tensor[[TDTarget], [TDTarget], ...]
        # where TDTarget is QValue
        td_targets_tensor = td_targets.tensor.unsqueeze(1)
        # y = actual (target network)

        criterion = torch.nn.HuberLoss()
        loss = criterion(actions_chosen_q_values, td_targets_tensor)

        self.optim.zero_grad()
        loss.backward()

        # clip gradients
        nn.utils.clip_grad.clip_grad_value_(self.parameters(), 100.0)

        self.optim.step()  # gradient descent