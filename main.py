import sys

from ddpg.ddpg import DDPG
from dqn.dqn import DQN
from environment.gymnasium import (
    CartpoleEnv,
    LunarLanderEnv,
    LunarLanderContinuousEnv,
    MountainCarContinuousEnv,
    MountainCarEnv,
    PendulumEnv,
)
from environment.trackmania import ControllerTrackmania, KeyboardTrackmania
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("checkpoint_id", type=str, help="The checkpoint id to load from", nargs="?")
    parser.add_argument("--epsilon-start", type=float, help="The epsilon start value")
    parser.add_argument("--dqn", action="store_true")
    parser.add_argument("--ddpg", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint_id: str | None = args.checkpoint_id
    epsilon_start: float | None = args.epsilon_start

    episode_count = 10**100
    timestep_count = 10**100

    # env = KeyboardTrackmania()
    # env = LunarLanderEnv()
    dqn_env = MountainCarEnv()

    create_dqn_agent = lambda chk=checkpoint_id, eps=episode_count, **kwargs: DQN(
        environment=dqn_env,
        episode_count=eps,
        timestep_count=timestep_count,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_min=0.01,
        epsilon_decay=0.01,
        buffer_batch_size=256,
        checkpoint_id=chk,
        **kwargs
    )

    for double_dqn in [True, False]:
        for prioritised_replay in [True, False]:
            agent = create_dqn_agent(
                double_dqn=double_dqn,
                prioritised_replay=prioritised_replay,
                eps=400,
            )
            agent.train()

    for polyak in [True, False]:
        agent = create_dqn_agent(
            polyak=polyak,
            double_dqn=False,
            prioritised_replay=False,
            eps=400,
        )
        agent = agent.train()

    for random in [True, False]:
        agent = create_dqn_agent(random=random, eps=400)
        agent = agent.train()

    ddpg_env = MountainCarContinuousEnv()
    create_ddpg_agent = lambda chk=checkpoint_id, eps=episode_count, **kwargs: DDPG(
        environment=ddpg_env,
        episode_count=eps,
        gamma=0.99,
        buffer_batch_size=256,
        target_network_learning_rate=0.005,
        checkpoint_id=chk,
        **kwargs
    )

    for prioritised_replay in [True, False]:
        agent = create_ddpg_agent(prioritised_replay=prioritised_replay)
        agent.train()

    for random in [True, False]:
        agent = create_ddpg_agent(random=random)
        agent.train()
