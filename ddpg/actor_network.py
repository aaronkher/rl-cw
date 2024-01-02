import torch
from ddpg.critic_network import CriticNetwork
from environment import Action, Environment, State
from network import NeuralNetwork
from replay_buffer import TransitionBatch


class ActorNetwork(NeuralNetwork):
    def __init__(self, env: Environment, critic_network: CriticNetwork):
        inputs = env.observation_space_length
        outputs = env.action_count
        super(ActorNetwork, self).__init__(inputs, outputs)

        self.environment = env
        self.critic_network = critic_network

    def create_copy(self) -> "ActorNetwork":
        copy = ActorNetwork(self.environment, self.critic_network)
        copy.copy_from(self)
        return copy

    def get_action(self, state: State) -> Action:
        input = state.tensor
        output = self(input)
        action = output.detach().cpu().item()  # single item tensor -> float
        return action

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        output = self(states)
        return output

    def train(self, experiences: TransitionBatch):
        # run the states through the network to figure out what we
        # would have done
        states = experiences.old_states
        actions = self.get_actions(states)

        # ask the critic network to criticize these actions we chose
        q_values = self.critic_network.get_q_values(states, actions)

        self.gradient_ascent(q_values)

    def gradient_ascent(self, q_values: torch.Tensor):
        q_values = -q_values

        # take the mean
        mean_qvalue = q_values.mean()

        # backprop
        self.optim.zero_grad()
        mean_qvalue.backward()
        self.optim.step()
