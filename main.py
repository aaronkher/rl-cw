from dqn import DQN

if __name__ == "__main__":
    dqn = DQN(
        episode_count=1000,
        timestep_count=10000000,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.02,
        C=1,
        buffer_batch_size=128,
        replay_buffer_size=10000,
    )
    dqn.train()
