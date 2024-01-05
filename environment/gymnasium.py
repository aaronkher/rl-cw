from abc import abstractmethod
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
        return self.last_action_taken

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


class PendulumEnv(GymnasiumEnv, ContinuousActionEnv):
    def __init__(self, render: bool = False):
        super().__init__("Pendulum-v1", render)

    def won(self, transition: Transition) -> bool:
        # there is no winning in this one
        return True

    @property
    def action_count(self) -> int:
        # todo do not hardcode
        return 1

    def take_action(self, action: ContinuousAction):
        return super().take_action(action)  # type: ignore

    def random_action(self) -> ContinuousAction:
        # random float between -2 and 2
        action = torch.rand(1) * 4 - 2
        return ContinuousAction(action)
