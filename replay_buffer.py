import collections
from dataclasses import dataclass
from typing import Deque, Iterable
import typing
import numpy as np
from numpy import ndarray
import torch
from environment.environment import Transition
from network import NeuralNetwork


# === PARENT CLASS ===

T = typing.TypeVar("T")


class Buffer:
    def get_buffer(self) -> Deque[T]:
        raise NotImplementedError

    def get_priorities(self) -> np.ndarray | None:
        return None

    def add(self, item: T):
        self.get_buffer().append(item)

    def get_batch(self, batch_size: int) -> list[T]:
        buffer = self.get_buffer()
        priorities = self.get_priorities()

        indices = np.random.choice(len(buffer), batch_size, p=priorities)
        experiences = [buffer[idx] for idx in indices]

        return experiences

    def size(self):
        buffer = self.get_buffer()
        return len(buffer)


# === TRANSITION/TIMESTEP BATCH ===


@dataclass
class Experience:
    transition: Transition
    td_error: float


class TransitionBatch:
    def __init__(self, experiences: list[Experience]):
        self.experiences = experiences
        self.size = len(experiences)

        # Tensor[[0], [2], [1], ...]
        self.actions = NeuralNetwork.tensorify([exp.transition.action.tensor() for exp in experiences])

        # Tensor[-0.99, -0.99, ...]
        self.rewards = NeuralNetwork.tensorify([exp.transition.reward for exp in experiences])

        # Tensor[State, State, ...]
        # states are already torch tensors, so we can just use torch.stack
        self.old_states = torch.stack([exp.transition.old_state.tensor for exp in experiences])
        self.new_states = torch.stack([exp.transition.new_state.tensor for exp in experiences])

        # Tensor[False, False, True, ...]
        self.terminal = NeuralNetwork.tensorify([exp.transition.new_state.terminal for exp in experiences])

    def update_td_errors(self, td_errors: Iterable[float]):
        for exp, td_error in zip(self.experiences, td_errors):
            exp.td_error = td_error


class TransitionBuffer(Buffer):
    def __init__(self, max_len: int = 10000, omega: float = 0.5):
        self.buffer: Deque[Experience] = collections.deque(maxlen=max_len)
        self.omega = omega

    def get_buffer(self) -> Deque[Experience]:
        return self.buffer

    def add(self, transition: Transition):
        experience = Experience(transition, 9.0)
        self.buffer.append(experience)

    def get_priorities(self) -> ndarray | None:
        priorities = np.array([exp.td_error for exp in self.buffer])
        priorities /= priorities.sum()
        return priorities

    def get_batch(self, batch_size: int) -> TransitionBatch:
        batch = super().get_batch(batch_size)
        return TransitionBatch(batch)


# === TRAJECTORY/EPISODE BUFFER ===


Trajectory = TransitionBuffer


@dataclass
class TrajectoryBatch:
    trajectories: list[Trajectory]


class TrajectoryBuffer(Buffer):
    def __init__(self, max_len: int):
        self.buffer: Deque[Trajectory] = collections.deque(maxlen=max_len)

    def get_buffer(self) -> Deque[Trajectory]:
        return self.buffer

    def get_batch(self, batch_size: int) -> TrajectoryBatch:
        batch = super().get_batch(batch_size)
        return TrajectoryBatch(batch)
