import sys

from ddpg.ddpg import DDPG
from dqn.dqn import DQN
from environment.trackmania import TrackmaniaEnv
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("checkpoint_id", type=str, help="The checkpoint id to load from")
    parser.add_argument("--epsilon_start", type=float, help="The epsilon start value")
    return parser.parse_args()

if __name__ == "__main__":
    env = TrackmaniaEnv()
    args = parse_args()
    checkpoint_id = args.checkpoint_id
    epsilon_start = args.epsilon_start

    dqn = DQN(
        environment=env,
        episode_count=10 * 1000,
        timestep_count=10 * 1000,
        gamma=0.99,
        epsilon_start=args.epsilon_start or 0.9,
        epsilon_min=0.01,
        epsilon_decay=0.01,
        buffer_batch_size=256,
        checkpoint_id=checkpoint_id,
    )

    dqn.train()
    # ddpg = DDPG(
    #     episode_count=600,
    #     gamma=0.99,
    #     buffer_batch_size=128,
    #     target_network_learning_rate=0.005,
    # )
    # ddpg.train()
