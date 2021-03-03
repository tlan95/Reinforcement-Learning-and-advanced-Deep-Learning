import numpy as np
import random
import copy
from collections import namedtuple, deque
from NNmodel import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
RHO = 0.999              # coefficient for soft update of target parameters
LR_ACTOR = 1e-4         # actor learning rate 
LR_CRITIC = 1e-3        # critic learning rate
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20        # update the networks LEARN_NUMBER times after every LEARN_EVERY timesteps
LEARN_NUMBER = 10       # update the networks LEARN_NUMBER times after every LEARN_EVERY timesteps
EPSILON = 1.0           # noise factor
EPSILON_DECAY = 0.999999  # noise factor decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():    
    def __init__(self, state_dim, action_dim, random_seed):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(random_seed)

        # Actor network with its target network
        self.actor_local = Actor(state_dim, action_dim, random_seed).to(device)
        self.actor_target = Actor(state_dim, action_dim, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic network with its target network
        self.critic_local = Critic(state_dim, action_dim, random_seed).to(device)
        self.critic_target = Critic(state_dim, action_dim, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise 
        self.noise = OUNoise(action_dim, random_seed)
        self.epsilon = EPSILON

        # Replay memory
        self.memory = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done, timestamp):
        """Save experience in replay memory, and use random sample from memory to learn."""
        # Save experience
        self.memory.add(state, action, reward, next_state, done)
        # Learn (if there are enough samples in memory)
        if len(self.memory) > BATCH_SIZE and timestamp % LEARN_EVERY == 0:
            for _ in range(LEARN_NUMBER):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Return actions for given state from current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * self.epsilon
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        #   UPDATE CRITIC   #
        actions_next = self.actor_target(next_states.to(device))
        Q_targets_next = self.critic_target(next_states.to(device), actions_next.to(device))
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_local.parameters(), 1)  # Clip the gradient when update critic network
        self.critic_optimizer.step()

        #   UPDATE ACTOR   #
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #   UPDATE TARGET NETWORKS   #
        self.soft_update(self.critic_local, self.critic_target, RHO)
        self.soft_update(self.actor_local, self.actor_target, RHO)      

        #   UPDATE EPSILON AND NOISE   #    
        self.epsilon *= EPSILON_DECAY
        self.noise.reset()           

    def soft_update(self, local_model, target_model, rho):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(rho*target_param.data + (1.0-rho)*local_param.data)

class OUNoise:
    """Noise: Ornstein-Uhlenbeck process."""
    def __init__(self, dim, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(dim)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state to mean value."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update the internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experiences."""
    def __init__(self, action_dim, buffer_size, batch_size, seed):
        self.action_dim = action_dim
        self.memory = deque(maxlen=buffer_size) 
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of memory."""
        return len(self.memory)