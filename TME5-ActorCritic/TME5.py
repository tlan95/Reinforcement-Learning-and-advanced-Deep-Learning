import gym
import gridworld
from utils import *
import torch
import itertools
from torch.utils.tensorboard import SummaryWriter
from AC_agent import ActorCriticAgent
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

    writer = SummaryWriter('runs/lunar/ep1000') 

    GAMMA = 0.99
    LAMBDA = 1.
    AVLAMBDA = 1.
    BETA = 0.01
    TARGET_UPDATE_FREQ = 1
    LR = 0.001
    NB_EPISODES = 1000
    INPUT_DIM = config.featExtractor(env).outSize
    N_ACTIONS = env.action_space.n

    agent = ActorCriticAgent(INPUT_DIM, N_ACTIONS, GAMMA, LR, BETA, LAMBDA, AVLAMBDA, TARGET_UPDATE_FREQ)
    total_rewards = []
    for e in range(NB_EPISODES):
        rsum = 0
        state = env.reset()
        state = torch.FloatTensor(state).to(device)

        for t in itertools.count():
            agent.event_count += 1
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)  
            rsum += reward
            agent.store_transition((state, action, reward, next_state, False if info.get("TimeLimit.truncated") else done))

            if done:
                if info.get("TimeLimit.truncated"):
                    print(f"Episode {e} Reward: ", rsum, " TimeLimit.truncated !")
                total_rewards.append(rsum)
                if e % 100 == 0:
                    print(f"Episode {e} Reward {np.mean(total_rewards[-100:])}")
                if e % agent.target_freq == 0:
                    agent.critic_target = deepcopy(agent.critic)
                break

            state = next_state

        policy_loss, value_loss, gae, entropy = agent.optimize()
        writer.add_scalar("Loss/policy_loss", policy_loss.item(), e)
        writer.add_scalar("Loss/value_loss", value_loss.item(), e)

        writer.add_scalar("Reward/Reward Total", rsum, e)

