from dqn import DQN

if __name__ == "__main__":
    dqn = DQN(
        episode_count=1000,
        timestep_count=100,
        epsilon=0.9,
        gamma=0.9,
    )
    dqn.train()
