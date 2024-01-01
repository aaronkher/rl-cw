import collections
from dataclasses import dataclass
from typing import Deque, Iterable
import numpy as np
import torch
from environment import Transition
from network import NeuralNetwork


@dataclass
class Experience:
    transition: Transition
    td_error: float


class ExperienceBatch:
    def __init__(self, replay_buffer: "ReplayBuffer", experiences: list[Experience]):
        self.replay_buffer = replay_buffer
        self.experiences = experiences
        self.size = len(experiences)

        # Tensor[[0], [2], [1], ...]
        self.actions = NeuralNetwork.tensorify([[exp.transition.action] for exp in experiences])

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


class ReplayBuffer:
    def __init__(self, max_len=10000, omega=0.5):
        self.buffer: Deque[Experience] = collections.deque(maxlen=max_len)
        self.omega = omega

    def add_experience(self, transition: Transition):
        experience = Experience(transition, 9.0)
        self.buffer.append(experience)

    def get_batch(self, batch_size: int) -> ExperienceBatch:
        priorities = np.array([exp.td_error for exp in self.buffer])
        priorities /= priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=priorities)
        experiences = [self.buffer[idx] for idx in indices]

        return ExperienceBatch(self, experiences)

    def size(self) -> int:
        return len(self.buffer)
