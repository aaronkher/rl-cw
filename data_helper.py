from dataclasses import dataclass

import matplotlib.pyplot as plt

from environment import Transition


# @dataclass
# class EpisodeData:
#     episode_number: int
#     timesteps: int
#     won: bool
#     reward_sum: float
#     reward_running_avg: float


class LivePlot:
    def __init__(self):
        # self.episodes: list[EpisodeData] = []

        # self.episode_numbers: list[int] = []
        self.rewards: list[float] = []
        self.running_avgs: list[float] = []

        self.won_episodes_numbers: list[int] = []
        self.won_episode_rewards: list[float] = []

        self.lost_episodes_numbers: list[int] = []
        self.lost_episode_rewards: list[float] = []

    def create_figure(self):
        plt.ion()

        self.figure = plt.figure()

        plt.title("Total reward per episode")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")

    def add_episode(self, reward: float, won: bool, running_avg: float):
        episode_number = len(self.rewards)
        self.rewards.append(reward)
        self.running_avgs.append(running_avg)

        if won:
            self.won_episodes_numbers.append(episode_number)
            self.won_episode_rewards.append(reward)
        else:
            self.lost_episodes_numbers.append(episode_number)
            self.lost_episode_rewards.append(reward)

        self.figure.canvas.flush_events()

    def draw(self):
        # plot total reward per episode and change colour based on if the episode was won or lost
        figure = plt.gcf()
        axes = plt.gca()

        # plot rewards line
        plt.plot(self.rewards, color="b")

        # plot running average line
        plt.plot(self.running_avgs, color="g")

        axes.scatter(
            self.lost_episodes_numbers,
            self.lost_episode_rewards,
            color="r",
        )
        axes.scatter(
            self.won_episodes_numbers,
            self.won_episode_rewards,
            color="g",
        )

        figure.canvas.draw()
        figure.canvas.flush_events()
