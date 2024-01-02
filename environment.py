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
        if self.env is None:
            raise Exception("Env error")
        if self.env.observation_space is None or self.env.observation_space.shape is None:
            raise Exception("observation space error")
        
        self.reset()

        print(f"Observation Space Dimensions: {self.env.observation_space.shape}")
        print(f"Action Space Dimensions: {self.env.action_space.shape}")
        print(f"Number of Actions: {self.env.action_space.n}")  # type: ignore
        print("")
        self.current_state: State
        self.last_action_taken: Transition | None

    @property
    def action_list(self) -> list[Action]:
        return list(range(self.env.action_space.n)) # type: ignore

    @property
    def action_count(self) -> int:
        return self.env.action_space.n # type: ignore

    @property
    def observation_space_length(self) -> int:
        if self.env.observation_space.shape:
            return self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        else:
            raise Exception("Invalid observation space shape.")

    def take_action(self, action: Action) -> Transition:
        old_state = self.current_state
        new_state_ndarray, reward, terminated, truncated, _ = self.env.step(action)

        device = NeuralNetwork.device()
        new_state_tensor = torch.from_numpy(new_state_ndarray).to(device)
        new_state = State(new_state_tensor, terminated)
        reward = float(reward)

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
        initial_state_ndarray, _ = self.env.reset()
        device = NeuralNetwork.device()
        initial_state_tensor = torch.from_numpy(initial_state_ndarray).to(device)
        initial_state = State(initial_state_tensor, False)
        self.current_state = initial_state
        self.last_action_taken = None

    @property
    def needs_reset(self) -> bool:
        return self.last_action_taken is None or self.last_action_taken.end_of_episode()
