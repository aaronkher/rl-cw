from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import gymnasium

Action = int
State = npt.NDArray[np.float64]


@dataclass
class ActionTaken:
    action: Action
    old_state: State
    new_state: State
    reward: float
    terminated: bool  # win
    truncated: bool  # loss (too many episodes)


class Environment:
    def __init__(self):
        self.env = gymnasium.make("Acrobot-v1", render_mode="human")
        # self.env = gymnasium.make("Acrobot-v1")

        self.reset()
        self.last_action_taken: ActionTaken

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

    def take_action(self, action: Action) -> ActionTaken:
        old_state = self.current_state
        (new_state, reward, terminated, truncated, info) = self.env.step(action)

        # tmp: add more reward the higher the tip
        cos_angle = new_state[0]
        sin_angle = new_state[1]
        angle = np.arctan2(sin_angle, cos_angle)
        cos_angle2 = new_state[3]
        sin_angle2 = new_state[4]
        angle2 = np.arctan2(sin_angle2, cos_angle2)
        reward += abs(angle) - abs(angle2) - 5
        print(reward)

        self.last_action_taken = ActionTaken(
            action, old_state, new_state, float(reward), terminated, truncated
        )
        return self.last_action_taken

    def reset(self):
        (current_state, _) = self.env.reset()
        self.last_action_taken = ActionTaken(
            action=None,  # type: ignore
            old_state=None,  # type: ignore
            new_state=current_state,
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    @property
    def current_state(self) -> State:
        return self.last_action_taken.new_state

    @property
    def is_terminated(self) -> bool:
        return self.last_action_taken.terminated

    @property
    def last_reward(self) -> float:
        return self.last_action_taken.reward

    def render(self):
        self.env.render()
