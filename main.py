import sys

from ddpg.ddpg import DDPG
from dqn.dqn import DQN
from environment.trackmania import TrackmaniaEnv
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "checkpoint_id", type=str, help="The checkpoint id to load from", nargs="?"
    )
    parser.add_argument("--epsilon_start", type=float, help="The epsilon start value")
    return parser.parse_args()


if __name__ == "__main__":
    env = TrackmaniaEnv()
    args = parse_args()
    checkpoint_id: str | None = args.checkpoint_id
    epsilon_start: float | None = args.epsilon_start
    high_score = 0
    best_agent: DQN | None = None

    for i in range(9):
        agent = DQN(
            environment=env,
            episode_count=50,
            timestep_count=10 * 1000,
            gamma=0.99,
            epsilon_start=args.epsilon_start or 0.5,
            epsilon_min=0.01,
            epsilon_decay=0.01,
            buffer_batch_size=256,
            checkpoint_id=checkpoint_id,
        )

        agent.train()
        # ddpg = DDPG(
        #     episode_count=600,
        #     gamma=0.99,
        #     buffer_batch_size=128,
        #     target_network_learning_rate=0.005,
        # )
        # ddpg.train()

        if agent.high_score > high_score:
            best_agent = agent

    assert not isinstance(best_agent, type(None))

    best_agent_continued = DQN(
        environment=env,
        episode_count=10 * 1000,
        timestep_count=10 * 1000,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_min=0.01,
        epsilon_decay=0.01,
        buffer_batch_size=256,
        checkpoint_id=best_agent.latest_checkpoint,
    )
