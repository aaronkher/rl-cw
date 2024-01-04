import tmrl

from environment.environment import Action, ContinuousActionEnv, Environment, Transition


class TrackmaniaEnv(ContinuousActionEnv):
    def __init__(self):
        self.env = tmrl.get_environment()

    def won(self, transition: Transition) -> bool:
        return True  # todo

    @property
    def action_count(self) -> int:
        raise NotImplementedError

    @property
    def observation_space_length(self) -> int:
        raise NotImplementedError

    def take_action(self, action: Action) -> Transition:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def needs_reset(self) -> bool:
        raise NotImplementedError

    @property
    def last_reward(self) -> float:
        raise NotImplementedError
