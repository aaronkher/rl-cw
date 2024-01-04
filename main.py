from ddpg.ddpg import DDPG
from dqn.dqn import DQN
from environment.trackmania import TrackmaniaEnv

if __name__ == "__main__":
    env = TrackmaniaEnv()
    dqn = DQN(
        environment=env,
        episode_count=10*1000,
        timestep_count=10*1000,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_min=0.10,
        epsilon_decay=0.001,
        buffer_batch_size=256,
    )
    dqn.train()

    # ddpg = DDPG(
    #     episode_count=600,
    #     gamma=0.99,
    #     buffer_batch_size=128,
    #     target_network_learning_rate=0.005,
    # )
    # ddpg.train()
