import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
from NNmodel import Critic, Actor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCriticAgent(object):
    """Actor-Critic agent"""

    def __init__(self, input_dim, n_actions, gamma, lr, beta, _lambda, _avlambda, target_freq):
        self.gamma = gamma
        self.lr = lr
        self.beta = beta
        self._lambda = _lambda
        self._avlambda = _avlambda
        self.input_dim = input_dim
        self.n_actions = n_actions

        self.actor = Actor(self.input_dim, self.n_actions).to(device)
        self.critic = Critic(self.input_dim).to(device)
        self.critic_target = deepcopy(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.transitions = []

        self.target_freq = target_freq

        self.learn_step = 0
        self.event_count = 0

    def act(self, state):
        action = self.actor(state).sample()
        return action.item()

    def act_greedy(self, state):
        action = self.actor(state).probs.argmax(dim=-1)
        return action.item()

    def store_transition(self, transition):
        if len(self.transitions) == 0:
            self.transitions.append([])
        self.transitions[-1].append(transition)

    @staticmethod
    def _get_transitions(transitions):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for transition in transitions:
            states.append(transition[0])
            actions.append(transition[1])
            rewards.append(transition[2] / 100.0)
            next_states.append(transition[3])
            dones.append(int(transition[4]))

        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.stack(next_states),
            torch.IntTensor(dones).to(device),
        )

    def get_transitions(self, shuffle=True):
        g_states, g_actions, g_rewards, g_next_states, g_dones, g_gaes = [], [], [], [], [], []

        for transitions in self.transitions:
            # get transitions for current trajectory
            states, actions, rewards, next_states, dones = self._get_transitions(transitions)
            # compute GAE for each trajectory
            gaes = self.compute_gae(rewards, states, next_states, 1-dones)
            g_states.append(states)
            g_actions.append(actions)
            g_rewards.append(rewards)
            g_next_states.append(next_states)
            g_dones.append(dones)
            g_gaes.append(gaes)

        self.transitions = []
        if shuffle:
            idx = torch.randperm(torch.cat(g_actions).size(0))
        else:
            idx = torch.arange(torch.cat(g_actions).size(0))
        return (
            torch.cat(g_states)[idx],
            torch.cat(g_actions)[idx],
            torch.cat(g_rewards)[idx],
            torch.cat(g_next_states)[idx],
            torch.cat(g_dones)[idx],
            torch.cat(g_gaes)[idx],
        )

    def compute_gae(self, rewards, states, next_states, masks):
        T = rewards.size(0)
        values = self.critic(states).view(-1)
        next_values = self.critic(next_states).view(-1)
        td_target = rewards + self.gamma * masks * next_values
        td_error = td_target - values

        discount = ((self.gamma * self._avlambda) ** torch.arange(T)).to(device)
        gae = torch.Tensor([(discount[:T - t] * td_error[t:]).sum() for t in range(T)]).to(device)
        return gae.detach()

    def optimize(self):
        self.learn_step += 1
        states, actions, rewards, next_states, dones, gaes = self.get_transitions()

        lambda_returns = self.critic(states).view(-1) + gaes

        # value loss
        value_loss = F.mse_loss(self.critic(states).view(-1), lambda_returns.detach())

        # policy loss
        dist = self.actor(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        policy_loss = - (log_probs * gaes.detach()).mean() - self.beta * entropy

        # backpropagation
        self.critic_optim.zero_grad()
        value_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        clip_grad_norm_(self.actor.parameters(), 0.1)
        return policy_loss, value_loss, gaes, entropy

