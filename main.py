from dqn import DQN

if __name__ == "__main__":
    dqn = DQN(
        episode_count=1000,
        timestep_count=100,
        gamma=0.9,
        epsilon_start=0.95,
        epsilon_min=0.001,
        epsilon_decay=0.005,
        C=20,
        buffer_batch_size=100,
    )
    dqn.train()
