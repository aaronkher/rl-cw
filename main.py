from dqn import DQN
from ddpg import DDPG

if __name__ == "__main__":
    # dqn = DQN(
    #     episode_count=1000,
    #     timestep_count=100,
    #     gamma=0.9,
    #     epsilon_start=0.9,
    #     epsilon_min=0.01,
    #     epsilon_decay=0.02,
    #     C=50,
    #     buffer_batch_size=100,
    # )
    # dqn.train()

    ddpg = DDPG(
        episode_count=1000,
        timestep_count=1000,
        gamma=0.9,
        beta=0.9,
        buffer_batch_size=100,
    )
    ddpg.train()
