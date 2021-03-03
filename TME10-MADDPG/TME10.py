import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import random
import torch
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from MADDPG_agent import MADDPG

"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world



env, scenario, world = make_env('simple_spread')
# state_dim = len(env.observation_space)
# print('state_dim: ', state_dim)
# action_dim = len(env.action_space)
# print('action_dim: ', action_dim)
# print('agents: ', env.n)
state_dim = 14
action_dim = 2
maddpg = MADDPG(state_dim, action_dim)

N_episodes = 1000     # number of episodes
Max_t = 300       # maximum number of steps in one episode 
Print_every = 100     # print the average score for every Print_every episodes
N_agents = env.n    # number of agents

scores_deque = deque(maxlen=Print_every)
time_stamp = 0
episodic_rewards = []

writer = SummaryWriter('runs/simple_spread/ep1000') 

for e in range(1, N_episodes+1):
    states = env.reset()
    scores = np.zeros(N_agents)
    for t in range(Max_t):
        actions = maddpg.act(states)
        next_states, rewards, dones, _ = env.step(actions)
        maddpg.step(states, actions, rewards, next_states, dones, time_stamp)
        states = next_states
        scores += rewards
        time_stamp += 1
        if np.any(dones):
            break 
    max_ = np.max(scores)
    scores_deque.append(max_)
    episodic_rewards.append(max_)
    writer.add_scalar("Reward", max_, e)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)), end="")
    if e % Print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))