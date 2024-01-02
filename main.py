from dqn import DQN

if __name__ == "__main__":
    dqn = DQN(
        episode_count=600,
        timestep_count=10*1000,
        gamma=0.9,
        epsilon_start=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.02,
        C=25,
        buffer_batch_size=1000,
    )
    dqn.train()
