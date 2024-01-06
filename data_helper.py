import time


import matplotlib
import matplotlib.pyplot as plt

from environment.environment import Transition


class LivePlot:
    def __init__(self):
        self.rewards = RewardsGraph("Total reward per episode")
        self.actor_loss = PlotGraph("Actor loss")
        self.critic_loss = PlotGraph("Critic loss", data_color="r", avg_color="y")

        self.last_draw = 0
        self.last_draw_duration = 0.001

        self.create_figure()

    def create_figure(self):
        plt.ion()

        self.figure = plt.figure(figsize=(14, 4))

    def add_episode(self, reward: float, won: bool, running_avg: float):
        self.rewards.add_data_point(reward, running_avg, won)
        self.tick()

    def add_losses(self, actor_loss: float, critic_loss: float | None = None):
        self.actor_loss.add_data_point(actor_loss)
        if critic_loss is not None:
            self.critic_loss.add_data_point(critic_loss)

        self.tick()

    def tick(self):
        self.figure.canvas.flush_events()

        # we want to spend less than 5% time drawing the figure
        time_since_last_draw = time.time() - self.last_draw
        if self.last_draw_duration < 0.05 * time_since_last_draw:
            start = time.time()
            self.last_draw = start

            self.draw()
            self.last_draw_duration = time.time() - start

        # print(
        #     f"self.last_draw = {self.last_draw:.2f}, self.last_draw_time = {self.last_draw_duration:.2f}, "
        #     f"time_since_last_draw = {time_since_last_draw:.2f}, * 0.01 = {0.01 * time_since_last_draw:.2f}, "
        #     f"% time spent drawing = {100 * self.last_draw_duration / time_since_last_draw:.2f}"
        # )

    def draw(self):
        figure = plt.gcf()

        self.rewards.subplot(131)
        self.actor_loss.subplot(132)
        self.critic_loss.subplot(133)

        figure.canvas.draw()
        figure.canvas.flush_events()


class PlotGraph:
    def __init__(
        self,
        name: str | None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        data_color: str = "b",
        avg_color: str = "g",
    ):
        self.name = name
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data: list[float] = []
        self.running_avgs: list[float] = []
        self.data_color = data_color
        self.avg_color = avg_color

    def add_data_point(self, data_point: float, running_avg: float | None = None):
        self.data.append(data_point)
        self.running_avgs.append(running_avg or self.calculate_running_avg())

    def subplot(self, loc: int):
        axes = plt.subplot(loc)

        if self.name is not None:
            plt.title(self.name)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)

        plt.plot(self.data, color=self.data_color)
        plt.plot(self.running_avgs, color=self.avg_color)
        return axes

    def calculate_running_avg(self, last: int = 30):
        data_to_avg = self.data[-last:]
        return sum(data_to_avg) / len(data_to_avg)


class RewardsGraph(PlotGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(xlabel="Episode", ylabel="Total reward", *args, **kwargs)

        self.won_episodes_numbers: list[int] = []
        self.won_episode_rewards: list[float] = []

        self.lost_episodes_numbers: list[int] = []
        self.lost_episode_rewards: list[float] = []

    def add_data_point(self, data_point: float, running_avg: float, won: bool):
        episode_number = len(self.data)
        super().add_data_point(data_point, running_avg)

        if won:
            self.won_episodes_numbers.append(episode_number)
            self.won_episode_rewards.append(data_point)
        else:
            self.lost_episodes_numbers.append(episode_number)
            self.lost_episode_rewards.append(data_point)

    def subplot(self, loc: int):
        axes = super().subplot(loc)

        # plot blue or red dot depending on if the episode was won or lost
        axes.scatter(self.lost_episodes_numbers, self.lost_episode_rewards, color="r")
        axes.scatter(self.won_episodes_numbers, self.won_episode_rewards, color="g")

        return axes
