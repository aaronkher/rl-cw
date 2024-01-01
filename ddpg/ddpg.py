import collections
from dataclasses import dataclass

import torch
from data_helper import LivePlot
from environment import Action, MountainCarContinuousEnv, State
from replay_buffer import ExperienceBatch, ReplayBuffer


@dataclass
class TdTargetBatch:
    # Tensor[TDTarget, TDTarget, ...]
    tensor: torch.Tensor


class DDPG:
    def __init__(
        self,
        episode_count: int,
        gamma: float,
        buffer_batch_size: int,
        target_network_learning_rate: float,
    ):
        self.episode_count = episode_count
        self.gamma = gamma
        self.buffer_batch_size = buffer_batch_size
        self.target_network_learning_rate = target_network_learning_rate

        self.environment = MountainCarContinuousEnv()
        self.replay_buffer = ReplayBuffer()

        # {} used as placeholders
        self.actor_network = {}
        self.target_actor_network = {}
        # copy weights

        self.critic_network = {}
        self.target_critic_network = {}
        # copy weights

    def get_action(self, state: State) -> Action:
        raise "TODO"

    def compute_td_targets(self, experiences: ExperienceBatch) -> TdTargetBatch:
        raise "TODO"

    def train_critic_network(self, experiences: ExperienceBatch, td_targets: TdTargetBatch):
        raise "TODO"

    def train_actor_network(self, experiences: ExperienceBatch):
        raise "TODO"

    def update_target_networks(self):
        raise "TODO"

    def decay_epsilon(self, episode: int):  # todo rename
        raise "TODO"

    def train(self):
        plot = LivePlot()
        plot.create_figure()

        try:
            recent_rewards = collections.deque(maxlen=30)
            for episode in range(self.episode_count):
                self.environment.reset()

                reward_sum = 0
                transition = None
                timestep = 0

                for timestep in range(10 * 1000):
                    state = self.environment.current_state  # S_t
                    action = self.get_action(state)  # A_t

                    transition = self.environment.take_action(action)
                    reward_sum += transition.reward
                    self.replay_buffer.add_experience(transition)

                    if self.replay_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.replay_buffer.get_batch(self.buffer_batch_size)
                        td_targets = self.compute_td_targets(replay_batch)

                        self.train_critic_network(replay_batch, td_targets)
                        self.train_actor_network(replay_batch)

                    self.update_target_networks()

                    # process termination
                    if transition.end_of_episode():
                        break

                # episode ended
                recent_rewards.append(reward_sum)

                # print episode result
                assert transition is not None
                won = transition.truncated
                won_str = "(won) " if won else "(lost)"
                running_avg = sum(recent_rewards) / len(recent_rewards)
                print(
                    f"Episode {episode+1: <3} | {timestep+1: >3} timesteps {won_str}"
                    f" | reward {reward_sum: <6.2f} | avg {running_avg: <6.2f} (last {len(recent_rewards): <2})"
                )

                self.decay_epsilon(episode)

                # episodes.append(EpisodeData(episode, reward_sum, timestep, won))
                plot.add_episode(reward_sum, won, running_avg)
                if episode % 5 == 0:
                    plot.draw()

        except KeyboardInterrupt:  # ctrl-c received while training
            pass  # stop training

        try:
            plot.draw()  # draw final results
        except KeyboardInterrupt:  # ctrl-c used to close final results
            pass  # we try/except/pass to not print KeyboardInterrupt error
