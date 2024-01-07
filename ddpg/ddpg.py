import collections
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import time
import numpy as np
from data_helper import LivePlot
from ddpg.actor_network import ActorNetwork
from ddpg.critic_network import CriticNetwork
from environment.environment import Action, ContinuousAction, ContinuousActionEnv, Environment, State
from environment.trackmania import TrackmaniaEnv
from replay_buffer import TransitionBatch, TransitionBuffer


@dataclass
class TdTargetBatch:
    # Tensor[TDTarget, TDTarget, ...]
    tensor: torch.Tensor


class DDPG:
    def __init__(
        self,
        environment: ContinuousActionEnv,
        episode_count: int,
        gamma: float,
        buffer_batch_size: int,
        target_network_learning_rate: float,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        checkpoint_id: str | None = None,
    ):
        self.episode_count = episode_count
        self.gamma = gamma
        self.buffer_batch_size = buffer_batch_size
        self.target_network_learning_rate = target_network_learning_rate

        self.environment = environment
        self.transition_buffer = TransitionBuffer(omega=0.5)

        self.critic_network = CriticNetwork(self.environment)
        self.target_critic_network = self.critic_network.create_copy()

        self.actor_network = ActorNetwork(self.environment, self.critic_network)
        self.target_actor_network = self.actor_network.create_copy()

        self.latest_checkpoint = None
        self.checkpoint_id = checkpoint_id
        if checkpoint_id is not None:
            self.critic_network.load_checkpoint(checkpoint_id, filename="critic_weights")
            self.target_critic_network.load_checkpoint(checkpoint_id, filename="target_critic_weights")

            self.actor_network.load_checkpoint(checkpoint_id, filename="actor_weights")
            self.target_actor_network.load_checkpoint(checkpoint_id, filename="target_actor_weights")

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.previous_noise = mu

    def get_action(self, state: State) -> ContinuousAction:
        perfect_action = self.actor_network.get_action(state)
        assert isinstance(perfect_action, ContinuousAction)

        noise = self.compute_OU_noise()
        action = perfect_action + noise
        
        # tmp
        # gas_b4, steer_b4 = action.action

        action.clamp(-1, 1)

        # tmp
        # gas, steer = action.action
        # print(f"gas={gas_b4: <6.2f} -> {gas: <6.2f}, steer = {steer_b4: <6.2f} -> {steer: <6.2f}")

        return action

    def compute_OU_noise(self) -> np.ndarray:
        # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        r = np.random.randn(self.environment.action_count).astype(np.float32)
        delta_noise = self.theta * (self.mu - self.previous_noise) + self.sigma * r
        self.previous_noise += delta_noise
        return self.previous_noise

    def compute_td_targets(self, experiences: TransitionBatch) -> TdTargetBatch:
        # td target is:
        # reward + discounted qvalue  (if not terminal)
        # reward + 0                  (if terminal)

        new_states = experiences.new_states

        # ask the actor network what actions it would choose
        actions = self.target_actor_network.get_actions(new_states)

        # ask the critic network to criticize these actions
        # Tensor[[QValue], [QValue], ...]
        discounted_qvalues = self.target_critic_network.get_q_values(new_states, actions)
        discounted_qvalues[experiences.terminal] = 0
        discounted_qvalues *= self.gamma

        # Tensor[[Reward], [Reward], ...]
        rewards = experiences.rewards.unsqueeze(1)

        # Tensor[TDTarget, TDTarget, ...]
        td_targets = rewards + discounted_qvalues
        return TdTargetBatch(td_targets)

    def train_critic_network(self, experiences: TransitionBatch, td_targets: TdTargetBatch) -> float:
        return self.critic_network.train(experiences, td_targets)

    def train_actor_network(self, experiences: TransitionBatch) -> float:
        return self.actor_network.train(experiences)

    def update_target_networks(self):
        self.target_critic_network.polyak_update(self.critic_network, self.target_network_learning_rate)
        self.target_actor_network.polyak_update(self.actor_network, self.target_network_learning_rate)

    def decay_noise(self):  # todo rename
        self.sigma = max(0.01, self.sigma * 0.99)

    def train(self, seed: bool = True):
        plot = LivePlot()

        self.high_score = float("-inf")
        high_score_episode = 0

        start = datetime.now()

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
                    self.transition_buffer.add(transition)

                    if self.transition_buffer.size() > self.buffer_batch_size:
                        replay_batch = self.transition_buffer.get_batch(self.buffer_batch_size)
                        td_targets = self.compute_td_targets(replay_batch)

                        critic_loss = self.train_critic_network(replay_batch, td_targets)
                        actor_loss = self.train_actor_network(replay_batch)
                        plot.add_losses(actor_loss, critic_loss, can_redraw=False)

                        self.update_target_networks()

                    # process termination
                    if transition.end_of_episode():
                        break

                # episode ended
                recent_rewards.append(reward_sum)

                if isinstance(self.environment, TrackmaniaEnv):
                    time_taken = time.time() - self.environment.last_reset
                else:
                    time_taken = -1.0

                # print episode result
                assert transition is not None
                won = self.environment.won(transition)
                won_str = "(won) " if won else "(lost)"
                running_avg = sum(recent_rewards) / len(recent_rewards)
                print(
                    f"Episode {episode+1: <3} | {timestep+1: >3} timesteps {won_str}"
                    f" | reward {reward_sum: <7.2f} | avg {running_avg: <6.2f} (last {len(recent_rewards)}) "
                    f" | sigma {self.sigma: <5.2f} | time_taken {time_taken:.2f}"
                )

                now = datetime.now()
                running_for = now - start

                suffix = None  # if should create checkpoint, will be a str
                if episode == self.episode_count - 1:  # create checkpoint after 10 episodes for startup
                    suffix = " (startup checkpoint)"
                if episode % 100 == 0 and episode != 0:  # create checkpoint every 100 episodes
                    suffix = f" (ep {episode})"
                if reward_sum > self.high_score + 0.01:
                    self.high_score = reward_sum
                    if (
                        episode > high_score_episode + 15
                    ):  # create checkpoint if high score has been beaten for 15 episodes
                        high_score_episode = episode
                        suffix = f" (hs {self.high_score:.1f})"

                should_save_checkpoint = suffix is not None
                if should_save_checkpoint:
                    checkpoint_info = {
                        "episode_number": episode,
                        "reward": reward_sum,
                        "won": won,
                        "running_since": start,
                        "running_for": running_for,
                        "start_checkpoint": self.checkpoint_id,
                        "previous_checkpoint": self.latest_checkpoint,
                        "suffix": suffix,
                    }

                    self.critic_network.save_checkpoint(**checkpoint_info, filename="critic_weights")
                    self.target_critic_network.save_checkpoint(
                        **checkpoint_info, filename="target_critic_weights"
                    )
                    self.actor_network.save_checkpoint(**checkpoint_info, filename="actor_weights")
                    checkpoint_folder, self.latest_checkpoint = self.target_actor_network.save_checkpoint(
                        **checkpoint_info, filename="target_actor_weights"
                    )

                    plot.save_img(checkpoint_folder / "plot.png")
                    plot.save_csv(checkpoint_folder / "data.csv")

                self.decay_noise()

                if not won:
                    time_taken = -1.0
                plot.add_episode(reward_sum, won, running_avg, time_taken)
                if isinstance(self.environment, TrackmaniaEnv):
                    self.environment.save_replay()

        except KeyboardInterrupt as e:  # ctrl-c received while training
            if seed:
                raise e
            pass  # stop training

        try:
            plot.draw()  # draw final results
        except KeyboardInterrupt:  # ctrl-c used to close final results
            pass  # we try/except/pass to not print KeyboardInterrupt error

        plot.close()
