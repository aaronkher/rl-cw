import collections
from dataclasses import dataclass
from typing import Deque, Iterable
import typing
import numpy as np
from numpy import ndarray
import torch
from environment import Transition
from network import NeuralNetwork


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


class TransitionBuffer:
    def __init__(self, max_len: int = 10000, omega: float = 0.5):
        self.buffer: Deque[Experience] = collections.deque(maxlen=max_len)
        self.new_transitions: list[Transition] = []
        self.omega = omega

    def get_buffer(self) -> Deque[Experience]:
        return self.buffer

    def add(self, transition: Transition):
        self.new_transitions.append(transition)
    
    def get_priorities(self) -> ndarray | None:
        priorities = np.array([exp.td_error for exp in self.buffer])
        priorities /= priorities.sum()
        return priorities

    def get_batch(self, batch_size: int) -> TransitionBatch:
        batch_size -= len(self.new_transitions)
        buffer = self.get_buffer()
        priorities = self.get_priorities()
        
        indices = np.random.choice(len(buffer), batch_size, p=priorities)
        experiences = [buffer[idx] for idx in indices]

        new_transitions = self.new_transitions[:batch_size]
        new_experiences = [Experience(transition, -1) for transition in new_transitions]
        del self.new_transitions[:batch_size]

        experiences.extend(new_experiences)

        return TransitionBatch(experiences)
    
    def size(self):
        buffer = self.get_buffer()
        return len(buffer)
