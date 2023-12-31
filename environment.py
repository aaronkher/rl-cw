from dataclasses import dataclass

import torch
import gymnasium
from network import NeuralNetwork



Action = int


@dataclass
class State:
    tensor: torch.Tensor  # 1D array
    terminal: bool


@dataclass
class Transition:
    """The result of taking A_t in S_t, obtaining R_t and transitionning
    to S_t+1."""

    action: Action  # A_t
    old_state: State  # S_t
    new_state: State  # S_t+1
    reward: float  # R_t
    truncated: bool  # out of timesteps

    def end_of_episode(self) -> bool:
        return self.new_state.terminal or self.truncated


class Environment:
    def __init__(self):
        self.env = gymnasium.make("highway-v0")

        self.reset()
        self.current_state: State
        self.last_action_taken: Transition | None

    @property
    def action_list(self) -> list[Action]:
        # [0, 1] for cartpole
        return list(range(self.env.action_space.start, self.env.action_space.n))  # type: ignore

    @property
    def action_count(self) -> int:
        return len(self.action_list)

    @property
    def observation_space_length(self) -> int:
        return sum(self.env.observation_space.shape)  # type: ignore

    def take_action(self, action: Action) -> Transition:
        old_state = self.current_state
        (new_state_ndarray, _reward, terminated, truncated, _) = self.env.step(action)

        device = NeuralNetwork.device()
        new_state_tensor = torch.from_numpy(new_state_ndarray).flatten().to(device)
        new_state = State(new_state_tensor, terminated)
        reward = float(_reward)

        self.current_state = new_state
        self.last_action_taken = Transition(
            action,
            old_state,
            new_state,
            reward,
            truncated,
        )
        return self.last_action_taken

    def reset(self):
        (current_state, _) = self.env.reset()
        current_state = NeuralNetwork.tensorify(current_state).flatten()
        current_state = State(current_state, False)

        self.current_state = current_state
        self.last_action_taken = None

    @property
    def needs_reset(self) -> bool:
        return self.last_action_taken is None or self.last_action_taken.end_of_episode()

    @property
    def last_reward(self) -> float:
        assert self.last_action_taken is not None
        return self.last_action_taken.reward