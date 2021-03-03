import gym
import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical, kl_divergence
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from NNmodel import Actor, Critic
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(self, env, gamma, _avlambda, beta, delta, init_lr, init_clip_range, n_updates, n_epochs, n_steps, n_mini_batch):
        self.env = env
        self.gamma = gamma
        self._avlambda = _avlambda
        self.beta = beta
        self.delta = delta
        self.init_lr = init_lr
        self.init_clip_range = init_clip_range
        self.lr = self.init_lr
        self.init_clip_range = self.init_clip_range
        self.n_updates = n_updates
        self.epochs = n_epochs
        self.n_steps = n_steps
        self.n_mini_batch = n_mini_batch

        self.n_episodes = 0
        self.episodes_rewards = []

        self.mini_batch_size = self.n_steps // self.n_mini_batch
        assert (self.n_steps % self.n_mini_batch == 0)

        self.current_state = None

        self.input_dim = self.env.observation_space.shape[0]
        self.actions_dim = self.env.action_space.n
        self.actor = Actor(self.input_dim, self.actions_dim).to(device)
        self.critic = Critic(self.input_dim).to(device)

        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.init_lr)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.init_lr)

    def act(self, state):
        action = self.actor(state).sample()
        return action

    def act_greedy(self, state):
        action = self.actor(state).probs.argmax(dim=-1)
        return action

    @staticmethod
    def chunked_mask(mask):
        end_pts = torch.where(mask)[0]
        start = 0
        for end in end_pts:
            yield list(range(start, end + 1))
            start = end + 1

        if end_pts.nelement() == 0:
            yield list(range(len(mask)))

        elif end_pts[-1] != len(mask) - 1:
            yield list(range(end_pts[-1] + 1, len(mask)))

    def sample_trajectories(self):
        """ Sample trajectories with current policy"""

        rewards = np.zeros((self.n_steps,), dtype=np.float32)
        actions = np.zeros((self.n_steps,), dtype=np.int32)
        done = np.zeros((self.n_steps,), dtype=np.bool)
        states = np.zeros((self.n_steps, self.input_dim), dtype=np.float32)
        log_pis = np.zeros((self.n_steps,), dtype=np.float32)
        values = np.zeros((self.n_steps,), dtype=np.float32)

        tmp_rewards = []
        self.current_state = self.env.reset()

        writer = SummaryWriter('runs/lunar/adaptative_kl/updates100') 

        for t in range(self.n_steps):
            with torch.no_grad():
                states[t] = self.current_state 
                state = torch.FloatTensor(self.current_state).to(device)
                pi = self.actor(state)
                values[t] = self.critic(state).cpu().numpy()
                a = self.act(state)
                actions[t] = a.cpu().numpy()
                log_pis[t] = pi.log_prob(a).cpu().numpy()

            new_state, r, end, info = self.env.step(actions[t])
            rewards[t] = r / 100.
            done[t] = False if info.get("TimeLimit.truncated") else end
            tmp_rewards.append(r)
            if end:
                if info.get("TimeLimit.truncated"):
                    print(f"Episode {self.n_episodes} Reward: ", np.sum(tmp_rewards), " TimeLimit.truncated !")
                else:
                    print(f"Episode {self.n_episodes} Reward: ", np.sum(tmp_rewards))
                
                writer.add_scalar("Reward", np.sum(tmp_rewards), self.n_episodes)
                
                self.n_episodes += 1
                self.episodes_rewards.append(np.sum(tmp_rewards))
                if (self.n_episodes + 1) % 100 == 0:
                    print(f"Total Episodes: {self.n_episodes + 1}, "
                          f"Mean Rewards of last 100 episodes: {np.mean(self.episodes_rewards[-100:]):.2f}")
                tmp_rewards = []
                new_state = self.env.reset()

            self.current_state = new_state 

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, device=device)
        values = torch.tensor(values, device=device)
        rewards = torch.tensor(rewards, device=device)
        done = torch.tensor(done, device=device)
        log_pis = torch.tensor(log_pis, device=device)

        advantages = self.compute_gae(rewards, values, done)
        return states, actions, values, log_pis, advantages

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimates"""
        gaes = []
        last_value = self.critic(torch.FloatTensor(self.current_state).to(device))
        next_values = torch.cat([values[1:], last_value.detach()])

        for chunk in self.chunked_mask(dones):
            T = rewards[chunk].size(0)
            td_target = rewards[chunk] + self.gamma * (1 - dones[chunk].int()) * next_values[chunk]
            td_error = td_target - values[chunk]
            discount = ((self.gamma * self._avlambda) ** torch.arange(T)).to(device)
            gae = torch.tensor([(discount[:T - t] * td_error[t:]).sum() for t in range(T)], dtype=torch.float32).to(device)
            gaes.append(gae)
        return torch.cat(gaes)

    def optimize_adaptative_kl(self, states, actions, values, log_pis, advantages, learning_rate, clip_range):
        old_pis = self.actor(states)
        old_probs = old_pis.probs.detach()

        for _ in tqdm(range(self.epochs)):
            # shuffle for each epoch
            indexes = torch.randperm(self.n_steps)
            # for each mini batch
            for start in range(0, self.n_steps, self.mini_batch_size):
                
                # get mini batch
                idx = indexes[start: start + self.mini_batch_size]
                
                # compute loss
                lambda_returns = values[idx] + advantages[idx]
                A_old = (advantages[idx] - advantages[idx].mean()) / (advantages[idx].std() + 1e-8)
                new_pis = self.actor(states[idx])
                new_values = self.critic(states[idx])
                
                # policy loss (with entropy)
                new_log_pis = new_pis.log_prob(actions[idx])
                L = ((new_log_pis - log_pis[idx]).exp() * A_old).mean()
                kl = kl_divergence(Categorical(old_probs[idx]), new_pis).mean()
                entropy = new_pis.entropy().mean()
                policy_loss = -(L - self.beta * kl + 0.01 * entropy)

                # value loss
                clipped_value = values[idx] + (new_values - values[idx]).clamp(min=-clip_range, max=clip_range)
                vf_loss = torch.max((new_values - lambda_returns) ** 2, (clipped_value - lambda_returns) ** 2)
                vf_loss = vf_loss.mean()

                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()

                for pg in self.optim_actor.param_groups:
                    pg['lr'] = learning_rate

                for pg in self.optim_critic.param_groups:
                    pg['lr'] = learning_rate

                policy_loss.backward()
                vf_loss.backward()

                clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optim_actor.step()
                self.optim_critic.step()

        new_pis = self.actor(states)
        kl = kl_divergence(Categorical(old_probs), new_pis).mean()
        if kl >= 1.5 * self.delta:
            self.beta *= 2
        elif kl <= self.delta / 1.5:
            self.beta /= 2

    def optimize_clipped(self, states, actions, values, log_pis, advantages, learning_rate, clip_range):

        for _ in tqdm(range(self.epochs)):
            # shuffle for each epoch
            indexes = torch.randperm(self.n_steps)

            # for each mini batch
            for start in range(0, self.n_steps, self.mini_batch_size):
                
                # get mini batch
                idx = indexes[start: start + self.mini_batch_size]

                # compute loss
                lambda_returns = values[idx] + advantages[idx]
                A_old = (advantages[idx] - advantages[idx].mean()) / (advantages[idx].std() + 1e-8)
                new_pis = self.actor(states[idx])
                new_values = self.critic(states[idx])

                # policy loss (with entropy)
                new_log_pis = new_pis.log_prob(actions[idx])
                ratio = (new_log_pis - log_pis[idx]).exp()
                clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
                Lclip = torch.min(ratio * A_old, clipped_ratio * A_old)
                Lclip = Lclip.mean()
                entropy = new_pis.entropy().mean()
                policy_loss = -(Lclip + 0.01 * entropy)

                # value loss
                clipped_value = values[idx] + (new_values - values[idx]).clamp(min=-clip_range, max=clip_range)
                vf_loss = torch.max((new_values - lambda_returns) ** 2, (clipped_value - lambda_returns) ** 2)
                vf_loss = vf_loss.mean()

                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()

                for pg in self.optim_actor.param_groups:
                    pg['lr'] = learning_rate

                for pg in self.optim_critic.param_groups:
                    pg['lr'] = learning_rate

                policy_loss.backward()
                vf_loss.backward()

                clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optim_actor.step()
                self.optim_critic.step()


    def optimize_without(self, states, actions, values, log_pis, advantages, learning_rate, clip_range):
        old_pis = self.actor(states)
        old_probs = old_pis.probs.detach()

        for _ in tqdm(range(self.epochs)):
            # shuffle for each epoch
            indexes = torch.randperm(self.n_steps)
            # for each mini batch
            for start in range(0, self.n_steps, self.mini_batch_size):
                
                # get mini batch
                idx = indexes[start: start + self.mini_batch_size]
                
                # compute loss
                lambda_returns = values[idx] + advantages[idx]
                A_old = (advantages[idx] - advantages[idx].mean()) / (advantages[idx].std() + 1e-8)
                new_pis = self.actor(states[idx])
                new_values = self.critic(states[idx])
                
                # policy loss (with entropy)
                new_log_pis = new_pis.log_prob(actions[idx])
                L = ((new_log_pis - log_pis[idx]).exp() * A_old).mean()
                entropy = new_pis.entropy().mean()
                policy_loss = -(L + 0.01 * entropy)

                # value loss
                clipped_value = values[idx] + (new_values - values[idx]).clamp(min=-clip_range, max=clip_range)
                vf_loss = torch.max((new_values - lambda_returns) ** 2, (clipped_value - lambda_returns) ** 2)
                vf_loss = vf_loss.mean()

                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()

                for pg in self.optim_actor.param_groups:
                    pg['lr'] = learning_rate

                for pg in self.optim_critic.param_groups:
                    pg['lr'] = learning_rate

                policy_loss.backward()
                vf_loss.backward()

                clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optim_actor.step()
                self.optim_critic.step()

        new_pis = self.actor(states)