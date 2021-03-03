import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, input_dim, actions_dim):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 128)
        self.fc_2 = nn.Linear(128, 256)
        self.pi_logits = nn.Linear(256, actions_dim)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        pi = Categorical(logits=self.pi_logits(x))
        return pi


class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 128)
        self.fc_2 = nn.Linear(128, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        value = self.value(x).reshape(-1)
        return value