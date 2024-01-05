from typing import TYPE_CHECKING

import torch
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
        outputs = env.action_count
        super(DqnNetwork, self).__init__(inputs, outputs)

        self.environment = env

    def create_copy(self) -> "CriticNetwork":
        copy = CriticNetwork(self.environment)
        copy.copy_from(self)
        return copy

    def get_q_values(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        input = torch.cat([state, action], dim=1)
        output = self(input)
        return output

    def train(self, experiences: TransitionBatch, td_targets: TdTargetBatch):
        # Tensor[State, State, ...]
        # where State is Tensor[position, velocity]
        experience_states = experiences.old_states

        # Tensor[Action, Action, ...]
        # where Action is int
        experience_actions = experiences.actions.unsqueeze(1)

        # Tensor[[QValue], [QValue], ...]
        # where QValue is float
        q_values = self.get_q_values(experience_states, experience_actions)

        # Tensor[[TDTarget], [TDTarget], ...]
        # where TDTarget is QValue
        td_targets_tensor = td_targets.tensor
        # y = actual (target network)

        self.gradient_descent(q_values, td_targets_tensor)

    def gradient_descent(self, q_values: torch.Tensor, td_targets: torch.Tensor):
        self.optim.zero_grad()

        loss = torch.nn.functional.mse_loss(q_values, td_targets)
        loss.backward()

        self.optim.step()
