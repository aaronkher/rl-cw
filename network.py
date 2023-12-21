import os
import torch
from torch import nn

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

    # def forward(self, state):
    #     state = self.flatten(state)
    #     logits = self.linear_relu_stack(state)
    #     return logits
    
    # forward pass
    def forward(self, state) -> list[float]:
        state = self.flatten(state)
        logits = self.linear_relu_stack(state)
        return logits

    def best_action(self, state) -> int:
        # action_probabilities = self.forward(state)
        logits = self(state)
        action_probabilities = nn.softmax(logits, dim=1)
        return torch.argmax(action_probabilities)
    
    # backwards pass
    def update_weights(self):
        # labels = [1, 0, 0]
        # prediction = [0.7, 0.2, 0.1]
        loss = (prediction - labels).sum()
        # loss = (0.7 - 1) + (0.2 - 0) + (0.1 - 0) = 0.3
        loss.backward() # backward pass
        pass

    def gradient_descent(self):
        optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        optim.step() #gradient descent


model = NeuralNetwork().to(device)
print(model)
