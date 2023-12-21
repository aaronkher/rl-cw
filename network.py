import os
import torch
from torch import nn

from environment import State

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self, state_variable_count: int, action_count: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_variable_count, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
        )

    # forward pass
    def forward(self, state: State) -> list[float]:
        state = self.flatten(state)
        logits = self.linear_relu_stack(state)
        return logits
    
    # need to return the q value for an action AND
    # return the corresponding action so DQN class knows what to use
    def get_q_value(self) -> ...:
        
        pass

    def best_action(self, state: State) -> int:
        # action_probabilities = self.forward(state)
        logits = self(state)
        action_probabilities = nn.softmax(logits, dim=1)
        return torch.argmax(action_probabilities)

    def gradient_descent(self, prediction: float, label: float):
        criterion = torch.nn.MSELoss()
        predictions = model(x)
        loss = criterion(predictions, label)

        


        optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        optim.step()  # gradient descent

        loss.backward()  # backward pass


model = NeuralNetwork().to(device)
print(model)
