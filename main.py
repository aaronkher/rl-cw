from dqn import DQN

if __name__ == "__main__":
    dqn = DQN(
        episode_count=500,
        timestep_count=1000,
        epsilon=0.2,
        gamma=0.9,
    )
    dqn.train()
