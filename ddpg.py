import torch
import numpy as np
import collections
import math
import random

from dataclasses import dataclass
from dqn import TdTargetBatch, Experience

from environment import Environment, Action, State, ActionResult
from dqn_network import NeuralNetwork, NeuralNetworkResult, NeuralNetworkResultBatch
from data_helper import plot_episode_data, EpisodeData

from ddpg_critic_network import DDPGCriticNetwork, DDPGCriticNetworkResult
from ddpg_actor_network import DDPGActorNetwork

class ExperienceBatch:
    def __init__(self, experiences: list[Experience]):
        self.size = len(experiences)

        # Tensor[[0], [2], [1], ...]
        self.actions = NeuralNetwork.tensorify([exp.action for exp in experiences])

        # Tensor[-0.99, -0.99, ...]
        self.rewards = NeuralNetwork.tensorify([exp.reward for exp in experiences])

        # Tensor[State, State, ...]
        # states are already torch tensors, so we can just use torch.stack
        self.old_states = torch.stack([exp.old_state for exp in experiences])

        # Tensor[False, False, True, ...]
        self.terminal = NeuralNetwork.tensorify([exp.terminal for exp in experiences])

class ReplayBuffer:
    def __init__(self, max_len=10000):
        self.buffer = collections.deque(maxlen=max_len)

    def add_experience(self, experience: Experience):
        self.buffer.append(experience)

    def get_batch(self, batch_size: int) -> ExperienceBatch:
        experiences = random.sample(self.buffer, batch_size)
        return ExperienceBatch(experiences)

    def size(self) -> int:
        return len(self.buffer)

class DDPG:
    def __init__(
        self,
        episode_count: int,
        timestep_count: int,
        gamma: float,
        # epsilon_start: float = 0.9,
        # epsilon_min: float = 0.01,
        # epsilon_decay: float = 0.05,
        # C: int = 50,
        beta = 0.9,
        buffer_batch_size: int = 100,
        target_network_learning_rate = 0.9,
    ):
        self.episode_count = episode_count
        self.timestep_count = timestep_count

        # self.epsilon_start = epsilon_start
        # self.epsilon_min = epsilon_min
        # self.decay_rate = epsilon_decay
        # self.epsilon = epsilon_start

        self.gamma = gamma
        self.beta = beta
        # self.C = C
        self.buffer_batch_size = buffer_batch_size # replay memory D capacity

        self.target_network_learning_rate = target_network_learning_rate

        self.environment = Environment("MountainCarContinuous-v0")

        # initialise replay memory
        self.replay_buffer = ReplayBuffer()

        # initialise critic network
        self.critic_network = DDPGCriticNetwork(self.environment).to(NeuralNetwork.device())
        
        # initialise actor network
        self.actor_network = DDPGActorNetwork(self.environment).to(NeuralNetwork.device())

        # initialise target critic network
        self.target_critic_network = DDPGCriticNetwork(self.environment).to(NeuralNetwork.device())
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())

        # initialise target actor network
        self.target_actor_network = DDPGActorNetwork(self.environment).to(NeuralNetwork.device())
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())

# old methods from dqn.py
        # # initialise q1
        # self.policy_network = NeuralNetwork(self.environment).to(NeuralNetwork.device())
        # # initialise q2
        # self.target_network = NeuralNetwork(self.environment).to(NeuralNetwork.device())
        # # copy q2 to q1
        # self.policy_network.load_state_dict(self.target_network.state_dict())

    def get_action_according_to_policy(self, state: State) -> Action:
        return self.actor_network.get_action(state)

    def get_action(self, state: State):
        # randomly generate noise and add it to the action

        # TODO come up with a not stupid way of doing this
        # noise = np.random.uniform(-1, 1)

        noise = np.random.normal(0, 0.1) # mean 0, std 0.1

        action = self.get_action_according_to_policy(state)
        action += noise

        # clamp action between -1 and 1
        action = min(max(action, -1.0), 1.0)
        return action

    def execute_action(self, action: Action) -> ActionResult:
        return self.environment.take_action(action)

    # using policy
    def get_q_value(self, state: State) -> DDPGCriticNetworkResult:
        return self.target_critic_network.get_q_value(state)

    def compute_td_targets_batch(self, experiences: ExperienceBatch) -> TdTargetBatch:
        # td target is:
        # reward + discounted qvalue  (if not terminal)
        # reward + 0                  (if terminal)

        # Tensor[-0.99, -0.99, ...]
        rewards = experiences.rewards

        # Tensor[[QValue * 3], [QValue * 3], ...]
        qvalues = self.target_critic_network.get_q_value_batch(
            experiences.old_states,
            experiences.actions
        )
        qvalues_tensor = qvalues.batch_output

        # Tensor[QValue, QValue, ...]
        discounted_qvalues_tensor = qvalues_tensor * self.gamma
        discounted_qvalues_tensor[experiences.terminal] = 0

        # # reformat rewards tensor to same shape as discounted_qvalues_tensor
        # # Tensor[[-0.99], [-0.99], ...]
        # rewards = rewards.unsqueeze(1) # don't think i need this anymore so commeting

        # Tensor[[TDTarget], [TDTarget], ...]
        td_targets = rewards + discounted_qvalues_tensor
        return TdTargetBatch(td_targets)

    # def decay_epsilon(self, episode):
    #     # epsilon = epsilon_min + (epsilon_start - epsilon_min) x epsilon^-decay_rate * episode
    #     self.epsilon = self.epsilon_min + (
    #         self.epsilon_start - self.epsilon_min
    #     ) * math.exp(-self.decay_rate * episode)
    #     print(f"Epsilon decayed to {self.epsilon}")

    def update_target_networks(self):
        # update target critic network
        torch.add(
            torch.mul(self.critic_network.parameter(), self.beta),
            torch.mul(self.target_critic_network.parameter(), 1-self.beta),
            out=self.target_critic_network.parameter()
        )

        # update target actor network
        torch.add(
            torch.mul(self.actor_network.parameter(), self.beta),
            torch.mul(self.target_actor_network.parameter(), 1-self.beta),
            out=self.target_actor_network.parameter()
        )

    def backprop_critic(self, experiences: ExperienceBatch, td_targets: TdTargetBatch):
        self.critic_network.backprop(experiences, td_targets)

    def backprop_actor(self, experiences: ExperienceBatch, critic_network: DDPGCriticNetwork):
        self.actor_network.backprop(experiences, critic_network)

    def train(self):
        episodes = []

        try:
            # timestep_C_count = 0
            for episode in range(self.episode_count):
                print(f"Episode: {episode}")
                self.environment.reset()

                timestep = 0
                reward_sum = 0
                won = False

                for timestep in range(self.timestep_count):
                    state = self.environment.current_state  # S_t

                    action = self.get_action(state)  # A_t
                    action_result = self.execute_action(action)

                    reward_sum += action_result.reward

                    # print(
                    #     f"Episode {episode} Timestep {timestep} | Action {action}, Reward {action_result.reward:.0f}, Total Reward {reward_sum:.0f}"
                    # )

                    experience = Experience(
                        action_result.old_state,
                        action_result.new_state,
                        action,
                        action_result.reward,
                        action_result.terminal and action_result.won,
                    )
                    self.replay_buffer.add_experience(experience)

                    if self.replay_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.replay_buffer.get_batch(
                            self.buffer_batch_size
                        )
                        # TODO td_target calculation isn't the same as DQN because we're using a continuous action space
                        td_targets = self.compute_td_targets_batch(replay_batch) # y (target network)

                        # Pass back target networks values to update
                        # Gradient descent on critic
                        self.backprop_critic(replay_batch, td_targets)

                        # Gradient ascent on actor
                        self.backprop_actor(replay_batch, self.critic_network)

                        self.update_target_networks()

                    # timestep_C_count += 1
                    # if timestep_C_count == self.C:
                    #     self.update_target_network()
                    #     timestep_C_count = 0

                    # process termination
                    if action_result.terminal:
                        won = action_result.won
                        won_str = "won" if won else "lost"
                        print(
                            f"Episode {episode+1} ended ({won_str}) after {timestep+1} timestaps"
                            f" with total reward {reward_sum:.2f}"
                        )
                        break

                episodes.append(EpisodeData(episode, reward_sum, timestep, won))
                # self.decay_epsilon(episode)
                # print(f"Episode {episode} finished with total reward {reward_sum}")

        except KeyboardInterrupt:
            pass

        plot_episode_data(episodes)
