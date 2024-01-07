from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim.optimizer import Optimizer as Optimizer
from dqn.dqn_network import DqnNetwork, DqnNetworkResultBatch
from environment.environment import Action, Environment, State
from replay_buffer import TransitionBatch
from network import NeuralNetwork

# prevent circular import
if TYPE_CHECKING:
    from ddpg.ddpg import TdTargetBatch
else:
    TdTargetBatch = object


class CriticNetworkResults:
    def __init__(self, tensor: torch.Tensor):
        # Tensor[[QValue], [QValue], ...]
        self.tensor = tensor


class CriticNetwork(DqnNetwork):
    def __init__(self, env: Environment):
        inputs = env.action_count + env.observation_space_length
        outputs = 1
        super(DqnNetwork, self).__init__(inputs, outputs)

        self.environment = env

        self.reset_output_weights()

    def create_copy(self) -> "CriticNetwork":
        copy = CriticNetwork(self.environment)
        copy.copy_from(self)
        return copy

    def create_stack(self) -> nn.Sequential:
        n = 256
        return nn.Sequential(
            nn.Linear(self.inputs, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, self.outputs),
        )

    def create_optim(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        input = torch.cat([state, action], dim=1)
        output = self(input)
        return output

    def train(self, experiences: TransitionBatch, td_targets: TdTargetBatch) -> float:
        # Tensor[State, State, ...]
        # where State is Tensor[position, velocity]
        experience_states = experiences.old_states

        # Tensor[Action, Action, ...]
        # where Action is int
        experience_actions = experiences.actions

        # Tensor[[QValue], [QValue], ...]
        # where QValue is float
        q_values = self.get_q_values(experience_states, experience_actions)

        # Tensor[[TDTarget], [TDTarget], ...]
        # where TDTarget is QValue
        td_targets_tensor = td_targets.tensor
        # y = actual (target network)

        return self.gradient_descent(td_targets_tensor, q_values)

    def gradient_descent(self, td_targets: torch.Tensor, q_values: torch.Tensor) -> float:
        # loss = torch.nn.functional.mse_loss(td_targets, q_values)
        criterion = torch.nn.HuberLoss()
        loss = criterion(td_targets, q_values)

        self.optim.zero_grad()
        loss.backward()

        self.optim.step()

        return loss.item()
