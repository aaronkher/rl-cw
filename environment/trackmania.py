import math
from typing import Iterable

import torch
import numpy as np
import tmrl

from network import NeuralNetwork
from environment.environment import Action, DiscreteAction, DiscreteActionEnv, State, Transition


class TrackmaniaEnv(DiscreteActionEnv):
    def __init__(self):
        self.env = tmrl.get_environment()

        self.reset()
        self._current_state: State
        self.last_action_taken: Transition | None

        self.timestep_penalty = 0.

        # hardcoded for track RL01 straight
        self.track_length = 20.86
        self.track_length_done = 0

    def won(self, transition: Transition) -> bool:
        return transition.reward == 100 - self.timestep_penalty

    def action_list(self) -> list[Action]:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8]

    @property
    def observation_space_length(self) -> int:
        inputs = 0
        for input in self.env.observation_space:
            inputs += math.prod(input.shape)
        return inputs

    def tensorify_state(self, state: Iterable[np.ndarray], terminated: bool) -> State:
        state_cat = np.concatenate([input.flatten() for input in state], dtype=np.float32)

        device = NeuralNetwork.device()
        state_tensor = torch.from_numpy(state_cat).to(device)
        return State(state_tensor, terminated)

    def format_action(self, nn_action: Action) -> np.ndarray:
        assert isinstance(nn_action, DiscreteAction)

        # 0-2: gas 3-5: nothing 6-8: brake
        # 0/3/6: left 1/4/7: straight 2/5/8: right

        accel = nn_action // 3
        direction = nn_action % 3

        gas = 1 if accel == 0 else 0
        brake = 1 if accel == 2 else 0
        steer = direction - 1  # -1 is left, 0 is straight, 1 is right

        return np.array([gas, brake, steer])

    def take_action(self, action: Action) -> Transition:
        old_state = self.current_state
        formatted_action = self.format_action(action)
        (new_state_ndarray, _reward, terminated, truncated, _) = self.env.step(formatted_action)

        new_state = self.tensorify_state(new_state_ndarray, terminated)
        reward = float(_reward)

        # reward engineering
        reward -= self.timestep_penalty # adding penalty for each timestep
        # if reward != 100 - self.timestep_penalty: # if we lost
        #     reward -= 50 # add penalty for losing

        if not terminated:
            self.track_length_done += reward
        elif terminated and reward != 100 - self.timestep_penalty:
            # we lost, punish it for the distance it didn't do
            reward -= self.track_length - self.track_length_done
            self.track_length_done = 0


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
        current_state = self.tensorify_state(current_state, False)

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
