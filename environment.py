from dataclasses import dataclass

import torch

import gymnasium

from network import NeuralNetwork

Action = int
State = torch.Tensor


@dataclass
class ActionResult:
    action: Action
    old_state: State
    new_state: State
    reward: float
    terminal: bool
    won: bool


class Environment:
    def __init__(self):
        # self.env = gymnasium.make("MountainCar-v0", render_mode="human")
        self.env = gymnasium.make("MountainCar-v0")

        self.reset()
        self.last_action_taken: ActionResult

    @property
    def action_list(self) -> list[Action]:
        # [0, 1, 2] for acrobot
        return list(range(self.env.action_space.start, self.env.action_space.n))  # type: ignore

    @property
    def action_count(self) -> int:
        # 3 for acrobot
        return len(self.action_list)

    @property
    def observation_space_length(self) -> int:
        # 6 for acrobot
        return sum(self.env.observation_space.shape)  # type: ignore

    def take_action(self, action: Action) -> ActionResult:
        old_state = self.current_state
        (new_state, _reward, terminated, truncated, info) = self.env.step(action)
        new_state = NeuralNetwork.tensorify(new_state)
        reward = float(_reward)

        (x_axis_position, velocity) = new_state
        reward += abs(float(velocity)) / 0.07 / 2

        # clamp reward between -1 and 1
        # reward = min(max(reward, -1.0), 1.0)

        self.last_action_taken = ActionResult(
            action,
            old_state,
            new_state,
            reward,
            terminated or truncated,
            terminated,
        )
        return self.last_action_taken

    def reset(self):
        (current_state, _) = self.env.reset()
        current_state = NeuralNetwork.tensorify(current_state)

        self.last_action_taken = ActionResult(
            action=None,  # type: ignore
            old_state=None,  # type: ignore
            new_state=current_state,
            reward=0.0,
            terminal=False,
            won=False,
        )

    @property
    def current_state(self) -> State:
        return self.last_action_taken.new_state

    @property
    def is_terminated(self) -> bool:
        return self.last_action_taken.terminal

    @property
    def last_reward(self) -> float:
        return self.last_action_taken.reward

    def render(self):
        self.env.render()
