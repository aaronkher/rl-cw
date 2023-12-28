import os
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING, Any, Iterable

import torch
from torch import nn
from torch.nn.functional import relu
from torch.nn.utils.clip_grad import clip_grad_value_


# prevent circular import
if TYPE_CHECKING:
    from environment import State, Environment, Action
    from dqn import TransitionBatch, TdTargetBatch, DQN, Transition
else:
    Experience = object
    State = object
    Action = object
    Environment = object
    TransitionBatch = object
    TdTargetBatch = object
    DQN = object

    import collections
    Transition = collections.namedtuple(
        "Transition", ("state", "action", "next_state", "reward")
    )


@dataclass
class NeuralNetworkResult:
    tensor: torch.Tensor

    def best_action(self) -> Action:
        return self.tensor.max(1).indices.view(1, 1)

    def best_action_q_value(self) -> float:
        return self.tensor[self.best_action()].item()

    def q_value_for_action(self, action: Action) -> float:
        return self.tensor[action].item()


class NeuralNetworkResultBatch:
    def __init__(self, batch_output: torch.Tensor):
        # Tensor[[QValue * 3], [QValue * 3], ...]
        self.batch_output = batch_output

    def __getitem__(self, index: int) -> NeuralNetworkResult:
        """Override index operator e.g. batch[0] -> NeuralNetworkResult"""
        return NeuralNetworkResult(self.batch_output[index])

    def __mul__(self, other: float) -> "NeuralNetworkResultBatch":
        """Override * operator e.g. batch * 0.9"""
        return NeuralNetworkResultBatch(self.batch_output * other)


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

        self.layer1 = nn.Linear(env.observation_space_length, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, env.action_count)

        # self.optim = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)
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

        x = torch.nn.functional.relu(self.layer1(state))
        x = torch.nn.functional.relu(self.layer2(x))
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

    def backprop(self, transitions: TransitionBatch, dqn: DQN):
        # state_action_values = self(experiences.old_states).gather(
        #     1, experiences.actions
        # )

        # # # Tensor[[TDTarget], [TDTarget], ...]
        # # # where TDTarget is QValue
        # td_targets_tensor = td_targets.tensor  # y = actual (target network)

        # criterion = torch.nn.SmoothL1Loss()
        # loss = criterion(state_action_values, td_targets_tensor)

        # self.optim.zero_grad()
        # loss.backward()

        # clip_grad_value_(self.parameters(), 100)
        # self.optim.step()  # gradient descent

        device = self.device()
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(dqn.buffer_batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                dqn.target_network(non_final_next_states).max(1).values
            )

        expected_state_action_values = (next_state_values * dqn.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optim.zero_grad()
        loss.backward()

        clip_grad_value_(self.parameters(), 100)
        self.optim.step()
