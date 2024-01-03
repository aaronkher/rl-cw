from ddpg.ddpg import DDPG
from dqn.dqn import DQN

if __name__ == "__main__":
    dqn = DQN(
        episode_count=600,
        timestep_count=10*1000,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.02,
        buffer_batch_size=128,
    )
    dqn.train()

    # ddpg = DDPG(
    #     episode_count=600,
    #     gamma=0.99,
    #     buffer_batch_size=128,
    #     target_network_learning_rate=0.005,
    # )
    # ddpg.train()
