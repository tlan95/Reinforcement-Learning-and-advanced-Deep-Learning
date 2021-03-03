import gym
import gridworld
from utils import *
import torch
import itertools
from torch.utils.tensorboard import SummaryWriter
from PPO_agent import PPOAgent
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # config = load_yaml('./configs/config_random_cartpole.yaml')
    config = load_yaml('./configs/config_random_lunar.yaml')
    # config = load_yaml('./configs/config_random_gridworld.yaml')

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    env.seed(config["seed"])

    Gamma = 0.99
    Avlambda = 0.95
    Beta = 1
    Delta = 0.01
    Init_lr = 5e-4      # Initial learning rate
    Init_clip_range = 0.1   # Initial clip range
    N_updates = 100   # number of updates

    # Cartpole
    # N_epochs = 5    # number of epochs (Cartpole) 
    # Lunar
    N_epochs = 10     # number of epochs (Lunar)

    # Cartpole
    # N_steps = 640   # number of steps to run for a single update (Cartpole)
    # Lunar
    N_steps = 2400   # number of steps to run for a single update (Lunar)
    
    N_mini_batch = 10   # number of mini batches


    agent = PPOAgent(env, Gamma, Avlambda, Beta, Delta, Init_lr, Init_clip_range, N_updates, N_epochs, N_steps, N_mini_batch)
    
    mode = "adaptative_kl"
    # mode = "clipped"
    # mode = "without"
    for k in range(agent.n_updates):
        alpha = k / agent.n_updates

        learning_rate = agent.init_lr * (1 - alpha)
        clip_range = agent.init_clip_range * (1 - alpha)

        states, actions, values, log_pis, advantages = agent.sample_trajectories()
        if mode == "adaptative_kl":
            print("Optimize with ppo adaptative kl objective")
            agent.optimize_adaptative_kl(states, actions, values, log_pis, advantages, learning_rate, clip_range)
        elif mode == "clipped":
            print("Optimize with ppo clipped objective")
            agent.optimize_clipped(states, actions, values, log_pis, advantages, learning_rate, clip_range)
        else:
            print("Optimize without adaptative kl nor clipped objective")
            agent.optimize_without(states, actions, values, log_pis, advantages, learning_rate, clip_range)

    agent.env.close()



