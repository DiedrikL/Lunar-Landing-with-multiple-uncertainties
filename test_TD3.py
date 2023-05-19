import torch
import numpy as np
import random
import gymnasium as gym
from TD3_agent import TD3Agent
import csv
import os
from gymnasium.wrappers import RecordVideo


def save_to_csv(filename, episode, score):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if episode == 0:
            writer.writerow(['Episode', 'Score'])
        writer.writerow([episode, score])


def rename(filename_actor, filename_critic1, filename_critic2, scores):
    avg = int(sum(scores) / len(scores))
    if os.path.exists(f'best_checkpoint_actor_{avg}.pth'):
        avg -= 1
    if os.path.exists(f'best_checkpoint_critic1_{avg}.pth'):
        avg -= 1
    os.rename(filename_actor, f'best_checkpoint_actor_{avg}.pth')
    os.rename(filename_critic1, f'best_checkpoint_critic1_{avg}.pth')
    os.rename(filename_critic2, f'best_checkpoint_critic2_{avg}.pth')


# Load saved checkpoints
def load_checkpoints(agent, actor_checkpoint_path, critic1_checkpoint_path, critic2_checkpoint_path):
    agent.actor_local.load_state_dict(torch.load(actor_checkpoint_path))
    agent.critic_local1.load_state_dict(torch.load(critic1_checkpoint_path))
    agent.critic_local2.load_state_dict(torch.load(critic2_checkpoint_path))


# Run trials
def run_trials(agent, env, num_trials, actor, critic1, critic2, re, save):
    scores = []
    max_steps = 1000
    for i in range(num_trials):
        rand_gravity = random.randint(-11, -5)
        rand_wind = random.randint(0, 20)
        env = gym.make("LunarLander-v2", continuous=True, gravity=rand_gravity, enable_wind=True, wind_power=rand_wind,
                       render_mode='rgb_array')
        # env = RecordVideo(env, video_dir)
        state = env.reset()
        agent.reset()
        score = 0
        state = state[0]
        done = False
        steps = 0
        while not done and steps <= max_steps:
            steps += 1
            action = agent.act(state, add_noise=False)
            next_state, reward, done, _, __ = env.step(action.squeeze())
            score += reward
            state = next_state

        if score < 0:
            print("")

        scores.append(score)
        print(f'Trial {i + 1}, Score: {score}')

        if save:
            save_to_csv('TD3_Test.csv', i, score)
    if re:
        rename(actor, critic1, critic2, scores)
    else:
        return scores


def run_all_rename(path):
    renames = True
    for i in range(8):
        actor = path + f"best_checkpoint_actor_{i}.pth"
        critic1 = path + f"best_checkpoint_critic1_{i}.pth"
        critic2 = path + f"best_checkpoint_critic2_{i}.pth"
        load_checkpoints(agent, actor, critic1, critic2)
        run_trials(agent, env, 400, actor, critic1, critic2, renames, False)


def run_average(path):
    renames = False
    avgs = []
    for i in range(3):
        actor = path + f"best_checkpoint_actor.pth"
        critic1 = path + f"best_checkpoint_critic1.pth"
        critic2 = path + f"best_checkpoint_critic2.pth"

        load_checkpoints(agent, actor, critic1, critic2)
        scores = run_trials(agent, env, 400, actor, critic1, critic2, renames, False)
        avgs.append(int(sum(scores) / len(scores)))
    print(avgs)


def run_single(save, rename, path):
    actor = path + f"best_checkpoint_actor.pth"
    critic1 = path + f"best_checkpoint_critic1.pth"
    critic2 = path + f"best_checkpoint_critic2.pth"
    load_checkpoints(agent, actor, critic1, critic2)
    scores = run_trials(agent, env, 400, actor, critic1, critic2, rename, save)
    avg = (int(sum(scores) / len(scores)))
    print("average score:", avg)
    print("top score:", max(scores))


video_dir = "videos_test"
random.seed(8787)

# Create the environment
env = gym.make("LunarLander-v2", continuous=True, render_mode='rgb_array')

# Create the DDPG agent
agent = TD3Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=0,
                 hidden_sizes=(128, 128))

# Define the path to the checkpoints "your/desired/path/here/"
path = input("insert your path here: ")

run_single(True, False, path)
