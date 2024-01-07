from abc import abstractmethod
import math

import numpy as np
import gymnasium
import torch
from environment.environment import (
    Action,
    ContinuousAction,
    ContinuousActionEnv,
    DiscreteAction,
    DiscreteActionEnv,
    Environment,
    State,
    Transition,
)
from network import NeuralNetwork


class GymnasiumEnv(Environment):
    def __init__(self, env_name: str, render: bool = False):
        render_mode = "human" if render else None
        self.env = gymnasium.make(env_name, render_mode=render_mode)

        self.reset()
        self._current_state: State
        self.last_action_taken: Transition | None

    @abstractmethod
    def won(self, transition: Transition) -> bool:
        ...

    @property
    def observation_space_length(self) -> int:
        # 4 for cartpole
        return sum(self.env.observation_space.shape)  # type: ignore

    def take_action(self, action: Action) -> Transition:
        old_state = self._current_state
        (new_state_ndarray, _reward, terminated, truncated, _) = self.env.step(action.gymnasium())

        device = NeuralNetwork.device()
        new_state_tensor = torch.from_numpy(new_state_ndarray).to(device)
        new_state = State(new_state_tensor, terminated)
        reward = float(_reward)

        self._current_state = new_state
        self.last_action_taken = Transition(
            action,
            old_state,
            new_state,
            reward,
            truncated,
        )

        self.last_action_taken.reward = self.reward_engineering(self.last_action_taken)

        return self.last_action_taken

    def reward_engineering(self, transition: Transition) -> float:
        return transition.reward  # override this to do reward engineering

    def reset(self):
        (current_state, _) = self.env.reset()
        current_state = NeuralNetwork.tensorify(current_state)
        current_state = State(current_state, False)

        self._current_state = current_state
        self.last_action_taken = None

    @property
    def current_state(self) -> State:
        return self._current_state

    @property
    def needs_reset(self) -> bool:
        return self.last_action_taken is None or self.last_action_taken.end_of_episode()

    @property
    def last_reward(self) -> float:
        assert self.last_action_taken is not None
        return self.last_action_taken.reward


class DiscreteGymnasiumEnv(GymnasiumEnv, DiscreteActionEnv):
    @property
    def action_list(self) -> list[DiscreteAction]:
        # [0, 1] for cartpole
        actions = range(self.env.action_space.start, self.env.action_space.n)  # type: ignore
        return [DiscreteAction(action) for action in actions]

    @property
    def action_count(self) -> int:
        # 2 for cartpole
        return len(self.action_list)


class CartpoleEnv(DiscreteGymnasiumEnv):
    def __init__(self, render: bool = False):
        super().__init__("CartPole-v1", render)

    def won(self, transition: Transition) -> bool:
        # truncated means we didn't survive till the end
        return not transition.truncated


class MountainCarEnv(DiscreteGymnasiumEnv):
    def __init__(self, render: bool = False):
        super().__init__("MountainCar-v0", render)

    def reward_engineering(self, transition: Transition) -> float:
        old_state = transition.old_state
        _, old_velocity = old_state.tensor 
        new_state = transition.new_state
        _, new_velocity = new_state.tensor

        acceleration = (abs(float(new_velocity)) - abs(float(old_velocity))) * 100

        return transition.reward + acceleration

    def won(self, transition: Transition) -> bool:
        # terminated means we reached the finish line
        return transition.new_state.terminal
        


class ContinuousGymnasiumEnv(GymnasiumEnv, ContinuousActionEnv):
    def __init__(self, env_name: str, render: bool = False):
        super().__init__(env_name, render)

        device = NeuralNetwork.device()
        self.low = torch.tensor(self.env.action_space.low, device=device)  # type: ignore
        self.high = torch.tensor(self.env.action_space.high, device=device)  # type: ignore

    @property
    def action_count(self) -> int:
        assert self.env.action_space.shape is not None
        return math.prod(self.env.action_space.shape)

    def interpolate_action(self, action: torch.Tensor) -> ContinuousAction:
        # assume the input is in the range [0, 1]
        # interpolate it to [low, high]
        action *= self.high - self.low  # [0, high - low]
        action += self.low  # [low, high]

        assert self.low <= action <= self.high
        return ContinuousAction(action)

    def take_action(self, action: ContinuousAction):
        # clip action between -1 and 1
        action_tensor = action.action
        action_tensor = action_tensor.clamp(-1, 1)

        # bring action from [-1, 1] -> [low, high]
        action_tensor *= 0.5  # [-0.5, 0.5]
        action_tensor += 0.5  # [0, 1]
        action = self.interpolate_action(action_tensor)

        return super().take_action(action)

    def random_action(self) -> ContinuousAction:
        # create random float between low and high
        action = torch.rand(1, device=NeuralNetwork.device())  # [0, 1]
        return self.interpolate_action(action)


class PendulumEnv(ContinuousGymnasiumEnv):
    def __init__(self, render: bool = False):
        super().__init__("Pendulum-v1", render)

    def won(self, transition: Transition) -> bool:
        # there is no winning in this one
        return True

    def random_action(self) -> ContinuousAction:
        # random float between -2 and 2
        action = torch.rand(1) * 4 - 2
        return ContinuousAction(action)


class MountainCarContinuousEnv(ContinuousGymnasiumEnv):
    def __init__(self, render: bool = False):
        super().__init__("MountainCarContinuous-v0", render)

    def reward_engineering(self, transition: Transition) -> float:
        old_state = transition.old_state
        _, old_velocity = old_state.tensor 
        new_state = transition.new_state
        _, new_velocity = new_state.tensor

        acceleration = (abs(float(new_velocity)) - abs(float(old_velocity))) * 100

        return transition.reward + acceleration

    def won(self, transition: Transition) -> bool:
        # terminated means we reached the finish line
        return transition.new_state.terminal