import torch
from environment import Environment, State, Action
from network import NeuralNetwork
from replay_buffer import TransitionBatch
from critic_network import CriticNetwork


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
    
    def get_action(self, state: State) -> Action:
        output = self(torch.Tensor(state))
        return output

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        output = self(states)
        return output

    def train(self, experiences: TransitionBatch, critic_network: CriticNetwork):
        # Tensor[QValue, QValue, ...]
        q_values = critic_network.get_q_values(
            experiences.old_states,
            self.get_action_batch(experiences.old_states)
            )

        self.gradient_ascent(q_values)

    def gradient_ascent(self, q_values: torch.Tensor):
        self.optim.zero_grad()

        loss = -q_values.mean()
        loss.backward()

        self.optim.step()

    def update_target_weights(self, network_to_copy: NeuralNetwork, target_network_learning_rate: float):
        for target_param, param in zip(self.linear_relu_stack.parameters(), network_to_copy.parameters()): # TODO rename linear_relu_stack to "network" or eq.
            target_param.data.copy_(param.data * target_network_learning_rate + target_param.data * (1.0 - target_network_learning_rate))
