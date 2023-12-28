from dataclasses import dataclass

import torch

import gymnasium

from network import NeuralNetwork

Action = torch.Tensor
State = torch.Tensor


@dataclass
class ActionResult:
    action: Action
    old_state: State
    new_state: State | None
    reward: torch.Tensor
    terminal: bool
    won: bool


class Environment:
    def __init__(self):
        # self.env = gymnasium.make("MountainCar-v0", render_mode="human")
        self.env = gymnasium.make("CartPole-v1")

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
        assert old_state is not None, "env needs resetting"

        (observation, reward, terminated, truncated, info) = self.env.step(
            action.item()
        )
        # new_state = torch.from_numpy(new_state).to(NeuralNetwork.device())
        reward = torch.tensor([reward], device=NeuralNetwork.device())

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=NeuralNetwork.device()
            ).unsqueeze(0)

        self.last_action_taken = ActionResult(
            action,
            old_state,
            next_state,
            reward,
            terminated or truncated,
            truncated,
        )
        return self.last_action_taken

    def reset(self):
        (current_state, _) = self.env.reset()
        current_state = torch.tensor(
            current_state, dtype=torch.float32, device=NeuralNetwork.device()
        ).unsqueeze(0)

        self.last_action_taken = ActionResult(
            action=None,  # type: ignore
            old_state=None,  # type: ignore
            new_state=current_state,
            reward=None,  # type: ignore
            terminal=False,
            won=False,
        )

    @property
    def current_state(self) -> State | None:
        return self.last_action_taken.new_state

    @property
    def is_terminated(self) -> bool:
        return self.last_action_taken.terminal

    @property
    def last_reward(self) -> float:
        return self.last_action_taken.reward.item()

    def render(self):
        self.env.render()
