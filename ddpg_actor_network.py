import torch
from typing import cast, TYPE_CHECKING, Any, Iterable
from dataclasses import dataclass

from environment import Environment
from dqn_network import NeuralNetwork
from ddpg_critic_network import DDPGCriticNetwork, DDPGCriticNetworkResult, DDPGCriticNetworkResultBatch
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
class DDPGActorNetworkResult:
    tensor: torch.Tensor
 
class DDPGActorNetworkResultBatch:
    def __init__(self, batch_output: torch.Tensor):
        # Tensor[[QValue * 3], [QValue * 3], ...]
        self.batch_output = batch_output

    def __getitem__(self, index: int) -> DDPGActorNetworkResult:
        """Override index operator e.g. batch[0] -> DDPGActorNetworkResult"""
        return DDPGActorNetworkResult(self.batch_output[index])

    def __mul__(self, other: float) -> "DDPGActorNetworkResultBatch":
        """Override * operator e.g. batch * 0.9"""
        return DDPGActorNetworkResultBatch(self.batch_output * other)

class DDPGActorNetwork(nn.Module):
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
        super(DDPGActorNetwork, self).__init__()

        self.env = env

        n=32

        self.actor_network = nn.Sequential(
            nn.Linear(self.env.observation_space_length, n),
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
        return self.actor_network(state)

        # forward pass critic network from state and action to give (singular) q value for that state action pair
        # return self.critic_network(torch.cat((state, action_value_batch), dim=1))

    # need to return the q value for an action AND
    # return the corresponding action so DQN class knows what to use
    def get_action(self, state: State) -> Action:
        """For a given state, pass it through the neural network and return
        the q-values for each action in that state.

        Args:
            state (State): The state to get the action-value for.

        Returns:
            DDPGActorNetworkResult: An object that wraps the raw tensor, with
                utility methods (eh maybe?) to make our lives
                easier.
        """
        neural_network_output = self(state)
        return neural_network_output.item()

    def get_action_batch(self, states: torch.Tensor) -> DDPGActorNetworkResultBatch:
        # states = Tensor[State, State, ...]
        # where State is Tensor[position, velocity]
        batch_output = self(states)
        # batch_output = Tensor[a, a, ...]
        # where a is float [-1,1]

        return DDPGActorNetworkResultBatch(batch_output)

    def backprop(self, experiences: ExperienceBatch, critic_network: DDPGCriticNetwork):
        # perform gradient ascent on the actor network
        self.optim.zero_grad()

        # Tensor[QValue, QValue, ...]
        # where QValue is float
        q_values = critic_network.get_q_value_batch(experiences.old_states, experiences.actions)

        action_values = self(experiences.old_states)

        criterion = torch.nn.MSELoss()
        loss = -criterion(action_values, q_values) # not sure if this is valid syntax
        loss.backward()
        #clip_grad_norm_(self.parameters(), 1)

        self.optim.step()  # gradient descent

        
