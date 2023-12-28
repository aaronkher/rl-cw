from dqn import DQN

if __name__ == "__main__":
    dqn = DQN(
        episode_count=600,
        timestep_count=100000,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_min=0.05,
        epsilon_decay=1000,
        C=1,
        buffer_batch_size=128,
    )
    dqn.train()
