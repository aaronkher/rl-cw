from ddpg.critic_network import CriticNetwork
from environment.environment import Action, ContinuousAction, Environment, State
from network import NeuralNetwork
from replay_buffer import TransitionBatch


import torch
from torch import nn
from torch.optim.optimizer import Optimizer as Optimizer


class ActorNetwork(NeuralNetwork):
    def __init__(self, env: Environment, critic_network: CriticNetwork):
        inputs = env.observation_space_length
        outputs = env.action_count
        super(ActorNetwork, self).__init__(inputs, outputs)

        self.environment = env
        self.critic_network = critic_network

        self.reset_output_weights()

    def create_stack(self) -> nn.Sequential:
        n = 256
        return nn.Sequential(
            nn.Linear(self.inputs, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, self.outputs),
            nn.Sigmoid(),
        )

    def create_optim(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def create_copy(self) -> "ActorNetwork":
        copy = ActorNetwork(self.environment, self.critic_network)
        copy.copy_from(self)
        return copy

    def get_action(self, state: State) -> ContinuousAction:
        input = state.tensor
        output: torch.Tensor = self(input)
        action = ContinuousAction(output)
        return action

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        output = self(states)
        return output

    def train(self, experiences: TransitionBatch) -> float:
        # run the states through the network to figure out what we
        # would have done
        states = experiences.old_states
        actions = self.get_actions(states)

        # ask the critic network to criticize these actions we chose
        q_values = self.critic_network.get_q_values(states, actions)

        return self.gradient_ascent(q_values)

    def gradient_ascent(self, q_values: torch.Tensor) -> float:
        # take the mean
        mean_qvalue = -q_values.sum()

        # backprop
        self.optim.zero_grad()
        mean_qvalue.backward()

        nn.utils.clip_grad.clip_grad_value_(self.parameters(), 1000.0)

        self.optim.step()

        return mean_qvalue.item()
