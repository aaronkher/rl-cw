from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class EpisodeData:
    episode_number: int
    reward_sum: float
    timestep_count: int
    won: bool

# create function to plot total reward per episode
def plot_episode_data(Episodes: list[EpisodeData]):
    print(Episodes)
    plt.figure(1)
    colours = np.where([episode.won for episode in Episodes], 'g', 'r')
    # plot total reward per episode and change colour based on if the episode was won or lost
    plt.plot([episode.episode_number for episode in Episodes], [episode.reward_sum for episode in Episodes])
    plt.scatter([episode.episode_number for episode in Episodes if not episode.won], [episode.reward_sum for episode in Episodes if not episode.won], color='r')
    plt.scatter([episode.episode_number for episode in Episodes if episode.won], [episode.reward_sum for episode in Episodes if episode.won], color='g')

    plt.title("Total reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.show()
