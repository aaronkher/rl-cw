from dataclasses import dataclass
from abc import ABC, abstractmethod, abstractproperty

import torch
import gymnasium

from network import NeuralNetwork

ContinuousAction = torch.Tensor
DiscreteAction = int
Action = ContinuousAction | DiscreteAction


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


class Environment(ABC):
    @abstractmethod
    def won(self, transition: Transition) -> bool:
        ...

    @abstractproperty
    def observation_space_length(self) -> int:
        ...

    @abstractproperty
    def action_count(self) -> int:
        ...

    @abstractmethod
    def random_action(self) -> Action:
        ...

    @abstractmethod
    def take_action(self, action: Action) -> Transition:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractproperty
    def current_state(self) -> State:
        ...

    @abstractproperty
    def needs_reset(self) -> bool:
        ...

    @abstractproperty
    def last_reward(self) -> float:
        ...


class DiscreteActionEnv(Environment):
    @abstractmethod
    def action_list(self) -> list[Action]:
        ...

    def action_count(self) -> int:
        return len(self.action_list())


class ContinuousActionEnv(Environment):
    @property
    def action_count(self) -> int:
        ...
