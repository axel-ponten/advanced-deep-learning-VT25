import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, inputdim, outputdim=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 2, 4, padding="same")
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(2, 4, 4, padding="same")
        self.fc1 = nn.Linear(int(4 * inputdim/ (4*4)), 128) # twice maxpooling
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, outputdim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x) # don't apply activation function to output, regression
        return x