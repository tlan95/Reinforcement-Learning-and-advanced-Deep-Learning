import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 128)
        self.fc_2 = nn.Linear(128, 256)
        self.pi_logits = nn.Linear(256, n_actions)

    def forward(self, state):
        state = F.relu(self.fc_1(state))
        state = F.relu(self.fc_2(state))
        return Categorical(logits=self.pi_logits(state))


class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 128)
        self.fc_2 = nn.Linear(128, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, state):
        state = F.relu(self.fc_1(state))
        state = F.relu(self.fc_2(state))
        value = self.value(state).reshape(-1)
        return value