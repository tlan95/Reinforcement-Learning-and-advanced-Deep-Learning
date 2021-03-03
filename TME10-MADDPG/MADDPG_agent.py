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
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # update the networks LEARN_NUMBER times after every LEARN_EVERY timesteps
LEARN_NUMBER = 3        # update the networks LEARN_NUMBER times after every LEARN_EVERY timesteps
EPSILON = 1.0           # noise factor
EPSILON_DECAY = 0.99    # noise factor decay
CLIPGRAD = .1           # clipped gradient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, output_dim, input_dim, name, hidden = 256, lr_actor=1.0e-3, lr_critic=1.0e-3, tau = 1.0e-2, seed = 10):
        super(DDPGAgent, self).__init__()
        
        self.seed = seed
        self.actor         = Actor(input_dim, hidden, output_dim, seed).to(device)
        self.critic        = Critic(input_dim = input_dim, action_dim = output_dim, hidden = hidden, seed = seed, output_dim = 1).to(device)
        self.target_actor  = Actor(input_dim, hidden, output_dim, seed).to(device)
        self.target_critic = Critic(input_dim = input_dim, action_dim = output_dim, hidden = hidden, seed = seed, output_dim = 1).to(device)
        self.name = name        
        self.noise = OUNoise(output_dim, seed)
        self.tau = tau
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.clipgrad = CLIPGRAD        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0)
       
    def act(self, state, add_noise=True):
        """Return actions for given state from current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) #.unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().squeeze(0).data.numpy()
        self.actor.train()
        if add_noise:
            action += self.noise.sample() * self.epsilon
        return np.clip(action, -1, 1)
     
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        #   UPDATE CRITIC   #
        actions_next = self.target_actor(next_states.to(device))
        Q_targets_next = self.target_critic(next_states.to(device), actions_next.to(device))
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
    
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.clipgrad)
        self.critic_optimizer.step()

        #   UPDATE ACTOR   #
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #clip_grad_norm_(self.actor.parameters(), self.clipgrad)
        self.actor_optimizer.step()

        #   UPDATE TARGET NETWORKS   #
        self.soft_update(self.critic, self.target_critic )
        self.soft_update(self.actor, self.target_actor)                     
        
        #   UPDATE EPSILON AND NOISE   #   
        self.epsilon *= EPSILON_DECAY
        self.noise.reset()

    def reset(self):
        self.noise.reset()
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
            
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


class MADDPG:
    def __init__(self, state_dim, action_dim, seed = 10):
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(action_dim, state_dim, 1), 
                             DDPGAgent(action_dim, state_dim, 2),
                             DDPGAgent(action_dim, state_dim, 3)]
        
        self.memory = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, seed)

    def act(self, observations):
        """Get actions from all agents in MADDPG"""
        actions = [agent.act(obs) for agent, obs in zip(self.maddpg_agent, observations)]
        return actions

    def step(self, states, actions, rewards, next_states, dones, timestamp):
        """Save experience in replay memory, and use random sample from memory to learn."""
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        #for idx, agent in enumerate(self.maddpg_agent):
            self.memory.add(state, action, reward, next_state, done)
            
        # Learn (if there are enough samples in memory)
        if len(self.memory) > BATCH_SIZE and timestamp % LEARN_EVERY == 0:
            for agent in self.maddpg_agent:
                for _ in range(LEARN_NUMBER):
                    experiences = self.memory.sample()
                    agent.learn(experiences)