import torch
from typing import cast, TYPE_CHECKING, Any, Iterable
from dataclasses import dataclass

from environment import Environment
from dqn_network import NeuralNetwork
from torch import nn

# prevent circular import
if TYPE_CHECKING:
    from environment import State, Environment, Action
    from ddpg import ExperienceBatch, TdTargetBatch
else:
    Experience = object
    State = object
    Action = object
    Environment = object
    ExperienceBatch = object
    TdTargetBatch = object

@dataclass
class DDPGCriticNetworkResult:
    tensor: torch.Tensor
 
class DDPGCriticNetworkResultBatch:
    def __init__(self, batch_output: torch.Tensor):
        # Tensor[[QValue * 3], [QValue * 3], ...]
        self.batch_output = batch_output

    def __getitem__(self, index: int) -> DDPGCriticNetworkResult:
        """Override index operator e.g. batch[0] -> DDPGCriticNetworkResult"""
        return DDPGCriticNetworkResult(self.batch_output[index])

    def __mul__(self, other: float) -> "DDPGCriticNetworkResultBatch":
        """Override * operator e.g. batch * 0.9"""
        return DDPGCriticNetworkResultBatch(self.batch_output * other)

class DDPGCriticNetwork(nn.Module):
    @staticmethod
    def device() -> torch.device:
        """Utility function to determine whether we can run on GPU"""
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return torch.device(device)

    @staticmethod
    def tensorify(array: Iterable) -> torch.Tensor:
        """Create a PyTorch tensor, and make sure it's on the GPU if possible"""
        return torch.tensor(array, device=NeuralNetwork.device())
    
    def __init__(self, env: Environment):
        # call super constructor (and create network actor network)
        super(DDPGCriticNetwork, self).__init__()

        self.env = env

        n=32

        self.critic_network = nn.Sequential(
            nn.Linear(self.env.observation_space_length + 1, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, self.env.action_count),
            nn.Tanh()
        )

        self.optim = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)

        # do not call directly, call get_q_values() instead
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """PyTorch internal function to perform forward pass.
        Do not call directly, use get_q_values() instead.

        Args:
            state (torch.Tensor): a tensor of length 6 (the state has 6 variables)

        Returns:
            torch.Tensor: a tensor of length 3 (one q-value for each action)
        """
        # forward pass actor network to get one action value for each state passed in
        return self.critic_network(state)

    def get_q_value_batch(self, states: torch.Tensor, action: torch.Tensor) -> DDPGCriticNetworkResultBatch:
        # states = Tensor[State, State, ...]
        # where State is Tensor[position, velocity]

        # action = Tensor[Action, Action, ...]

        batch_output = self(torch.cat((states, action), dim=1))
        # batch_output = Tensor[QValue, QValue, ...]
        # where QValue is float

        return DDPGCriticNetworkResultBatch(batch_output)

    def backprop(self, experiences: ExperienceBatch, td_targets: TdTargetBatch):
        self.optim.zero_grad()

        # Tensor[QValue, QValue, ...]
        # where QValue is float
        q_values = self(experiences.old_states)

        # # Tensor[[TDTarget], [TDTarget], ...]
        # # where TDTarget is QValue
        td_targets_tensor = td_targets.tensor # y = actual (target network)

        criterion = torch.nn.MSELoss()
        loss = criterion(q_values, td_targets_tensor)
        loss.backward()
        #clip_grad_norm_(self.parameters(), 1)
 
        self.optim.step()  # gradient descent