import torch
from environment import Environment
from network import NeuralNetwork
from replay_buffer import TransitionBatch


class ActorNetwork(NeuralNetwork):
    def __init__(self, env: Environment):
        self.environment = env

        inputs = env.observation_space_length
        outputs = env.action_count
        super(ActorNetwork, self).__init__(inputs, outputs)

    def create_copy(self) -> "ActorNetwork":
        copy = ActorNetwork(self.environment)
        copy.copy_from(self)
        return copy

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        output = self(states)
        return output

    def train(self, experiences: TransitionBatch):
        pass
