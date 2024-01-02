from dataclasses import dataclass

import torch
import gymnasium

from network import NeuralNetwork


Action = int | float


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
    def __init__(self, env_name: str, render: bool = False):
        # self.env = gymnasium.make("CartPole-v1")
        render_mode = "human" if render else None
        self.env = gymnasium.make(env_name, render_mode=render_mode)

        self.reset()
        self.current_state: State
        self.last_action_taken: Transition | None

    def won(self, transition: Transition) -> bool:
        raise NotImplementedError  # needs to be subclassed

    @property
    def action_list(self) -> list[Action]:
        # [0, 1] for cartpole
        return list(range(self.env.action_space.start, self.env.action_space.n))  # type: ignore

    @property
    def action_count(self) -> int:
        # 2 for cartpole
        return len(self.action_list)

    @property
    def observation_space_length(self) -> int:
        # 4 for cartpole
        return sum(self.env.observation_space.shape)  # type: ignore

    def take_action(self, action: Action) -> Transition:
        old_state = self.current_state
        (new_state_ndarray, _reward, terminated, truncated, _) = self.env.step(action)

        device = NeuralNetwork.device()
        new_state_tensor = torch.from_numpy(new_state_ndarray).to(device)
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
        current_state = NeuralNetwork.tensorify(current_state)
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


class CartpoleEnv(Environment):
    def __init__(self, render: bool = False):
        super().__init__("CartPole-v1", render)

    def won(self, transition: Transition) -> bool:
        # truncated means we didn't survive till the end
        return not transition.truncated


class PendulumEnv(Environment):
    def __init__(self, render: bool = False):
        super().__init__("Pendulum-v1", render)

    def won(self, transition: Transition) -> bool:
        # there is no winning in this one
        return True 

    @property
    def action_count(self) -> int:
        # todo do not hardcode
        return 1

    def take_action(self, action: Action):
        return super().take_action([action])  # type: ignore
