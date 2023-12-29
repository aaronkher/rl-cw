import random
from dataclasses import dataclass
import math
import time
import collections

import torch
import numpy as np

from environment import Environment, Action, State, ActionResult
from network import NeuralNetwork, NeuralNetworkResult, NeuralNetworkResultBatch
from data_helper import plot_episode_data, EpisodeData, create_figure


@dataclass
class Experience:
    old_state: State
    new_state: State
    action: Action
    reward: float
    terminal: bool
    td_error: float


@dataclass
class TdTargetBatch:
    # Tensor[[TDTarget], [TDTarget], ...]
    tensor: torch.Tensor


class ExperienceBatch:
    def __init__(self, experiences: list[Experience]):
        self.size = len(experiences)

        # Tensor[[0], [2], [1], ...]
        self.actions = NeuralNetwork.tensorify([[exp.action] for exp in experiences])

        # Tensor[-0.99, -0.99, ...]
        self.rewards = NeuralNetwork.tensorify([exp.reward for exp in experiences])

        # Tensor[State, State, ...]
        # states are already torch tensors, so we can just use torch.stack
        self.old_states = torch.stack([exp.old_state for exp in experiences])

        self.new_states = torch.stack([exp.new_state for exp in experiences])

        # Tensor[False, False, True, ...]
        self.terminal = NeuralNetwork.tensorify([exp.terminal for exp in experiences])


class ReplayBuffer:
    def __init__(self, replay_buffer_size: int, omega: float = 0.5):
        self.buffer = collections.deque(maxlen=replay_buffer_size)
        self.omega = omega

    def add_experience(self, experience: Experience):
        self.buffer.append(experience)

    def get_batch(self, batch_size: int) -> ExperienceBatch:
        experiences = random.sample(self.buffer, batch_size)
        return ExperienceBatch(experiences)

    """
    Samples experiences proportional to the magnitude of td-error.
    Large TD error, greater probability of being sampled.
    """

    def prioritized_replay_sampling(self, batch_size: int):
        c = 0.01  # small constant

        # prioritiy of experience is proportional to the magnitude of the td-err
        priorities = (
            np.array(
                [abs(exp.td_error) + c for exp in list(self.buffer)], dtype=np.float32
            )
            ** self.omega
        )

        # w = omega
        # k = buffer size
        # P(i)= p_i^w / ∑_k p_k^w
        probabilities = priorities / priorities.sum()

        indicies = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indicies]

        return ExperienceBatch(experiences)

    def get_buffer(self):
        return ExperienceBatch(list(self.buffer))

    def size(self) -> int:
        return len(self.buffer)


class DQN:
    def __init__(
        self,
        episode_count: int,
        timestep_count: int,
        gamma: float,
        epsilon_start: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.02,
        C: int = 50,
        buffer_batch_size: int = 100,
        replay_buffer_size: int = 10000,
    ):
        self.episode_count = episode_count
        self.timestep_count = timestep_count

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = epsilon_decay
        self.epsilon = epsilon_start

        self.gamma = gamma
        self.C = C
        self.buffer_batch_size = buffer_batch_size

        self.environment = Environment()

        self.replay_buffer = ReplayBuffer(replay_buffer_size=replay_buffer_size)
        self.policy_network = NeuralNetwork(self.environment)  # q1
        self.target_network = NeuralNetwork(self.environment)  # q2
        # copy q2 to q1
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def get_best_action(self, state: State) -> Action:
        return self.policy_network.get_best_action(state)

    def get_action_using_epsilon_greedy(self, state: State):
        if np.random.uniform(0, 1) < self.epsilon:
            # pick random action
            action = random.choice(self.environment.action_list)
        else:
            # pick best action
            action = self.get_best_action(state)
        return action

    def execute_action(self, action: Action) -> ActionResult:
        return self.environment.take_action(action)

    # using policy
    def get_q_values(self, state: State) -> NeuralNetworkResult:
        return self.policy_network.get_q_values(state)

    # using target network here to estimate q values
    def get_q_value_for_action(self, state: State, action: Action) -> float:
        neural_network_result = self.target_network.get_q_values(state)
        return neural_network_result.q_value_for_action(action)

    """
    Double DQN: Yt ≡ Rt+1+γQ(St+1, argmax aQ(St+1, a; θt), θ−t)
    from: https://arxiv.org/pdf/1509.06461.pdf
    θt = policy network (in our case)
    θ−t = target network (in our case)
    """

    def compute_td_target(self, reward: float, new_state: State) -> float:
        # TD Target is the last reward + the expected reward of the
        # best action in the next state, discounted.

        # the reward and state after the last action was taken:
        last_reward = reward  # R_t
        current_state = new_state  # S_t+1

        if self.environment.is_terminated:
            td_target = last_reward
        else:
            # policy network (θt) used here to get best action
            action = self.get_best_action(current_state)

            # target network (θ−t) used here to calculate q value for action
            td_target = last_reward + self.gamma * self.get_q_value_for_action(
                current_state, action
            )

        return td_target

    """
    Double DQN: Yt ≡ Rt+1+γQ(St+1, argmax aQ(St+1, a; θt), θ−t)
    from: https://arxiv.org/pdf/1509.06461.pdf
    θt = policy network (in our case)
    θ−t = target network (in our case)
    """

    def compute_td_targets_batch(self, experiences: ExperienceBatch) -> TdTargetBatch:
        # Tensor[[Reward], [Reward], ...]
        rewards = experiences.rewards.unsqueeze(1)  # R_t+1

        # get best actions according to policy network
        # Tensor[[QValue * 3], [QValue * 3], ...]
        policy_network_q_values = self.policy_network.get_q_values_batch(
            experiences.new_states
        )
        # Tensor[Action, Action, ...]
        best_actions = policy_network_q_values.batch_output.argmax(dim=1)
        # Tensor[[Action], [Action], ...]
        best_actions = best_actions.unsqueeze(1)

        # calculate q values using target network
        q_values = self.target_network.get_q_values_batch(experiences.new_states)

        # get the target network's q value of the policy network's chosen action
        max_q_values = q_values.batch_output.gather(1, best_actions)
        # discounted reward is 0 for terminal states
        max_q_values[experiences.terminal] = 0.0

        td_targets = rewards + (self.gamma * max_q_values)

        return TdTargetBatch(td_targets)

    def decay_epsilon(self, episode):
        # epsilon = epsilon_min + (epsilon_start - epsilon_min) x epsilon^-decay_rate * episode
        self.epsilon = self.epsilon_min + (
            self.epsilon_start - self.epsilon_min
        ) * math.exp(-self.decay_rate * episode)
        print(f"Epsilon decayed to {self.epsilon}")

    def update_target_network(self):
        target_net_state = self.target_network.state_dict()
        policy_net_state = self.policy_network.state_dict()
        tau = 0.005

        for key in policy_net_state:
            target_net_state[key] = (
                tau * policy_net_state[key] + (1 - tau) * target_net_state[key]
            )

        self.target_network.load_state_dict(target_net_state)

    def backprop(self, experiences: ExperienceBatch, td_targets: TdTargetBatch):
        self.policy_network.backprop(experiences, td_targets)

    def train(self):
        episodes = []
        create_figure()
        try:
            timestep_C_count = 0
            for episode in range(self.episode_count):
                print(f"Episode: {episode}")
                self.environment.reset()

                timestep = 0
                reward_sum = 0
                won = False

                for timestep in range(self.timestep_count):
                    state = self.environment.current_state  # S_t

                    action = self.get_action_using_epsilon_greedy(state)  # A_t
                    action_result = self.execute_action(action)

                    reward_sum += action_result.reward

                    # print(
                    #     f"Episode {episode} Timestep {timestep} | Action {action}, Reward {action_result.reward:.0f}, Total Reward {reward_sum:.0f}"
                    # )

                    td_target = self.compute_td_target(
                        action_result.reward, action_result.new_state
                    )

                    experience = Experience(
                        action_result.old_state,  
                        action_result.new_state,
                        action,
                        action_result.reward,
                        action_result.terminal and not action_result.won,
                        td_target,
                    )

                    self.replay_buffer.add_experience(experience)

                    if self.replay_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.replay_buffer.prioritized_replay_sampling(
                            self.buffer_batch_size
                        )

                        # compute td targets for the whole of the replay batch
                        td_targets = self.compute_td_targets_batch(replay_batch)

                        self.backprop(replay_batch, td_targets)

                    timestep_C_count += 1
                    if timestep_C_count == self.C:
                        self.update_target_network()
                        timestep_C_count = 0

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
                self.decay_epsilon(episode)

                print(f"Episode {episode} finished with total reward {reward_sum}")
                if episode % 10 == 0:
                    plot_episode_data(episodes) # comment out if you don't want live plot updates
    
            time.sleep(10*1000)

        except KeyboardInterrupt:
            pass

        # plot_episode_data(episodes) # uncomment if you want to plot after training
