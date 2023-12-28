from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import os


@dataclass
class EpisodeData:
    episode_number: int
    reward_sum: float
    timestep_count: int
    won: bool


# create function to plot total reward per episode
def plot_episode_data(
    Episodes: list[EpisodeData],
    hyperparameters: dict,
    save_folder: str = "results",
):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # print(Episodes)
    plt.clf()
    plt.figure(1)
    colours = np.where([episode.won for episode in Episodes], "g", "r")
    # plot total reward per episode and change color based on if the episode was won or lost
    # plt.plot(
    #     [episode.episode_number for episode in Episodes],
    #     [episode.reward_sum for episode in Episodes],
    # )
    plt.scatter(
        [episode.episode_number for episode in Episodes if not episode.won],
        [episode.reward_sum for episode in Episodes if not episode.won],
        color="r",
    )
    plt.scatter(
        [episode.episode_number for episode in Episodes if episode.won],
        [episode.reward_sum for episode in Episodes if episode.won],
        color="g",
    )

    # Include hyperparameter values in the title
    title = (
        f"Total Reward per Episode\n"
        f"Gamma: {hyperparameters['gamma']}, "
        f"Epsilon Start: {hyperparameters['epsilon_start']}, "
        f"Epsilon Min: {hyperparameters['epsilon_min']}, "
        f"Epsilon Decay: {hyperparameters['epsilon_decay']}, "
        f"C: {hyperparameters['C']}, "
        f"Buffer Batch Size: {hyperparameters['buffer_batch_size']}"
    )

    save_filename = (
        f"g:{hyperparameters['gamma']},"
        f"eS:{hyperparameters['epsilon_start']},"
        f"eM:{hyperparameters['epsilon_min']},"
        f"eD:{hyperparameters['epsilon_decay']},"
        f"C:{hyperparameters['C']},"
        f"BBS:{hyperparameters['buffer_batch_size']},"
        f".png"
    )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")

    save_path = os.path.join(save_folder, save_filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show(block=False)