import math
import os
from typing import Iterable
from pathlib import Path
import json
from dataclasses import dataclass
import time

import torch
import numpy as np
import tmrl
from tmrl.custom.utils import control_keyboard

from network import NeuralNetwork
from environment.environment import (
    Action,
    DiscreteAction,
    DiscreteActionEnv,
    ContinuousAction,
    ContinuousActionEnv,
    Environment,
    State,
    Transition,
)


class TrackmaniaEnv(Environment):
    def __init__(self):
        self.env = tmrl.get_environment()

        self.reset()
        self._current_state: State
        self.last_action_taken: Transition | None

        self.timestep_penalty = 0.00

        # hardcoded for track RL01 straight
        self.track_length = 22.0
        self.track_length_done = 0

    def track_progress(self, state) -> float:
        return float(state[1])

    def won(self, transition: Transition) -> bool:
        # return abs(transition.reward - (100 - self.timestep_penalty)) < 0.001
        # reward_if_won = 100 - self.timestep_penalty
        # return abs(transition.reward - reward_if_won) < 0.001

        return self.track_progress(transition.new_state.tensor) == 1.0

    @property
    def observation_space_length(self) -> int:
        return 20

    def tensorify_state(self, state: tuple[np.ndarray], terminated: bool) -> State:
        state = self.format_state(state)

        device = NeuralNetwork.device()
        state_tensor = torch.from_numpy(state).to(device)
        return State(state_tensor, terminated)   
    
    def format_state(self, state:tuple[np.ndarray]) -> State:
        speed, progress, lidars, prev_action1, prev_action2 = state
        inputs = np.concatenate([speed, lidars[3].flatten()])
        return inputs

    def take_action(self, raw_action: Action, gas: float, steer: float) -> Transition:
        assert 0 <= gas <= 1
        assert -1 <= steer <= 1
        formatted_action = np.array([gas, 0.0, steer])

        old_state = self.current_state
        (new_state_ndarray, _reward, terminated, truncated, _) = self.env.step(formatted_action)

        new_state = self.tensorify_state(new_state_ndarray, terminated)
        reward = float(_reward)

        # reward engineering
        progress = self.track_progress(new_state_ndarray)
        won = progress > 0.99
        if terminated and not won:
            # we lost, punish it for the distance it didn't do
            reward -= (1 - progress) * 100

        reward -= self.timestep_penalty  # adding penalty for each timestep

        self._current_state = new_state
        self.last_action_taken = Transition(
            raw_action,
            old_state,
            new_state,
            reward,
            truncated,
        )
        return self.last_action_taken

    def reset(self):
        (current_state, _) = self.env.reset()
        self._current_state = self.tensorify_state(current_state, False)

        self.last_action_taken = None

    # keycodes from: https://community.bistudio.com/wiki/DIK_KeyCodes
    def save_replay(self):
        R = 0x13
        UP = 0xC8
        ENTER = 0x1C
        LEFT = 0xCB
        DOWN = 0xD0
        wait_time = 0.05
        control_keyboard.PressKey(R)
        time.sleep(wait_time)
        control_keyboard.ReleaseKey(R)
        time.sleep(wait_time)
        for i in range(20):
            control_keyboard.PressKey(LEFT)
            control_keyboard.ReleaseKey(LEFT)
        control_keyboard.PressKey(DOWN)
        time.sleep(wait_time)
        control_keyboard.ReleaseKey(DOWN)
        time.sleep(wait_time)
        control_keyboard.PressKey(ENTER)
        time.sleep(wait_time)
        control_keyboard.ReleaseKey(ENTER)
        time.sleep(wait_time)
        control_keyboard.PressKey(ENTER)
        time.sleep(wait_time)
        control_keyboard.ReleaseKey(ENTER)
        time.sleep(wait_time)
        control_keyboard.PressKey(ENTER)
        time.sleep(wait_time)
        control_keyboard.ReleaseKey(ENTER)
        time.sleep(wait_time)

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


class KeyboardTrackmania(TrackmaniaEnv, DiscreteActionEnv):
    def __init__(self):
        set_virtual_gamepad(False)
        super().__init__()

    @property
    def action_list(self) -> list[DiscreteAction]:
        actions = [0, 1, 2, 3, 4, 5]
        return [DiscreteAction(action) for action in actions]

    def format_action(self, nn_action: DiscreteAction) -> tuple[float, float]:
        assert isinstance(nn_action, DiscreteAction)
        action = nn_action.action

        # 0-2: gas 3-5: nothing
        # 0/3: left 1/4: straight 2/5: right

        accel = action // 3
        direction = action % 3

        gas = 1 if accel == 0 else 0
        # brake = 1 if accel == 2 else 0
        # brake = 0
        steer = direction - 1  # -1 is left, 0 is straight, 1 is right

        return gas, steer

    def take_action(self, action: DiscreteAction) -> Transition:
        gas, steer = self.format_action(action)
        return super().take_action(action, gas, steer)


class ControllerTrackmania(TrackmaniaEnv, ContinuousActionEnv):
    def __init__(self):
        set_virtual_gamepad(True)
        super().__init__()

    @property
    def action_count(self) -> int:
        return 2  # gas and steer

    def take_action(self, action: ContinuousAction) -> Transition:
        gas, steer = action.action
        gas = gas * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        return super().take_action(action, float(gas), float(steer))

    def random_action(self) -> ContinuousAction:
        # torch.rand(1) returns float in [0, 1]
        gas = torch.rand(1) * 2 - 1
        steer = torch.rand(1) * 2 - 1
        tensor = NeuralNetwork.tensorify([gas, steer])
        return ContinuousAction(tensor)


def set_virtual_gamepad(virtual_gamepad: bool):
    userprofile_path = os.getenv("USERPROFILE")  # C:\Users\...
    assert userprofile_path is not None

    config_path = Path(userprofile_path) / "TmrlData" / "config" / "config.json"
    config_contents = json.loads(config_path.read_text())

    config_contents["VIRTUAL_GAMEPAD"] = virtual_gamepad

    with config_path.open("w") as config_file:
        json.dump(config_contents, config_file, indent=2)
