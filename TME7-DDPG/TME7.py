import gym
import random
import torch
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from DDPG_agent import DDPGAgent

env = gym.make('Pendulum-v0')
env.seed(5)
agent = DDPGAgent(state_dim=3, action_dim=1, random_seed=5)     # pendulum s=3 a=1  MountainCarContinuous s=2 a=1   LunarLanderContinuous s=8 a=2

N_episodes = 1000     # number of episodes
Max_t = 300       # maximum number of steps in one episode (pendulum=300    MountainCarContinuous=600   LunarLanderContinuous=300)
Print_every = 100     # print the average score for every Print_every episodes

scores_deque = deque(maxlen=Print_every)
time_stamp = 0

writer = SummaryWriter('runs/pendulum/ep1000new') 

for e in range(1, N_episodes+1):
    state = env.reset()
    agent.reset()
    score = 0
    for t in range(Max_t):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done, time_stamp)
        state = next_state
        score += reward
        time_stamp += 1
        if done:
            break 
    scores_deque.append(score)
    writer.add_scalar("Reward", score, e)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)), end="")
    if e % Print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))