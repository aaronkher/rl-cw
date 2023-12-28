import os
import random

import numpy as np
import torch

from dqn import DQN

RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True)
# https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

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
