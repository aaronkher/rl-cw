from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import gymnasium as gymnasium  

Action = int
State = npt.NDArray[np.float64]

@dataclass
class ActionTaken:
    action: Action
    old_state: State
    new_state: State
    reward: float
    terminated: bool

class Environment:
    def __init__(self):
        self.env = gymnasium.make('Acrobot-v1', render_mode="human")
        
        self.reset()
        self.last_action_taken: ActionTaken

    @property
    def action_list(self) -> list[Action]:
        # todo replace with self.env.action_space.something
        return [0, 1, 2]

    @property
    def action_count(self) -> int:
        len(self.action_space())

    @property
    def observation_space_length(self) -> int:
        # todo replace with self.env.observation_space.something
        return 6
    
    def take_action(self, action: Action):
        old_state = self.current_state
        (new_state, reward, terminated, truncated, info) = self.env.step(action)

        self.current_state = new_state
        self.last_action_taken = ActionTaken(action, old_state, new_state, reward, terminated)

    def reset(self):
        (current_state, _) = self.env.reset()
        self.last_action_taken = ActionTaken(
            action=None,
            old_state=None,
            new_state=current_state,
            reward=0.0,
            terminated=False
        )

    @property
    def current_state(self) -> State:
        self.last_action_taken.new_state

    @property
    def is_terminated(self) -> bool:
        self.last_action_taken.terminated
    
    @property
    def last_reward(self) -> float:
        self.last_action_taken.reward

    def render(self):
        self.env.render()

    

