from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class EpisodeData:
    episode_number: int
    reward_sum: float
    timestep_count: int
    won: bool


# create function to plot total reward per episode
def plot_episode_data(Episodes: list[EpisodeData]):
    # plot total reward per episode and change colour based on if the episode was won or lost
    figure = plt.gcf()
    axes = plt.gca()
    
    # comment out the plt.plot if you don't want the lines connecting the points
    plt.plot(
        [episode.episode_number for episode in Episodes],
        [episode.reward_sum for episode in Episodes],
    )

    axes.scatter(
        [episode.episode_number for episode in Episodes if not episode.won],
        [episode.reward_sum for episode in Episodes if not episode.won],
        color="r",
    )
    axes.scatter(
        [episode.episode_number for episode in Episodes if episode.won],
        [episode.reward_sum for episode in Episodes if episode.won],
        color="g",
    )

    figure.canvas.draw()
    figure.canvas.flush_events()

def create_figure():
    plt.ion()

    figure = plt.figure()

    plt.title("Total reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")