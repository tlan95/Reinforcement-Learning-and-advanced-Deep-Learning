import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, input_dim, hidden, output_dim, seed):
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, 450)
        self.fc2 = nn.Linear(450, 350)
        self.fc3 = nn.Linear(350, 350)
        self.fc4 = nn.Linear(350, output_dim)
        self.activation = F.relu #leaky_relu
        self.batch_norm_input = nn.BatchNorm1d(450)
        self.batch_norm_hidden1 = nn.BatchNorm1d(350)
        self.batch_norm_hidden2 = nn.BatchNorm1d(hidden)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        h1 = self.activation(self.batch_norm_input(self.fc1(x)))
        h2 = self.activation(self.batch_norm_hidden1(self.fc2(h1)))
        #h3 = self.activation(self.batch_norm_hidden2(self.fc3(h2)))
        h4 = self.fc4(h2)
        return F.tanh(h4)
        

class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, input_dim, action_dim, hidden, seed, output_dim = 1):
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256+action_dim, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.activation = F.relu #leaky_relu
        self.batch_norm_input = nn.BatchNorm1d(256)
        self.batch_norm_hidden1 = nn.BatchNorm1d(128)
        self.batch_norm_hidden2 = nn.BatchNorm1d(128)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, obs, act):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        h1 = self.activation(self.batch_norm_input(self.fc1(obs)))
        h1 = torch.cat((h1,act), dim=1)
        h2 = self.activation(self.batch_norm_hidden1(self.fc2(h1)))
        #h3 = self.activation(self.batch_norm_hidden2(self.fc3(h2)))
        h4 = self.fc4(h2)
        return h4