import torch
import numpy as np
import random
import gymnasium as gym
from DDPG_agent import DDPGAgent
import csv
import os

video_dir = "videos_test"
random.seed(8787)
def save_to_csv(filename, episode, score):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if episode == 0:
            writer.writerow(['Episode', 'Score'])
        writer.writerow([episode, score])

def rename(filename_actor, filename_critic, scores):
    avg = int(sum(scores) / len(scores))
    if os.path.exists(f'best_checkpoint_actor_{avg}.pth'):
        avg -= 1
    if os.path.exists(f'best_checkpoint_critic_{avg}.pth'):
        avg -= 1
    os.rename(filename_actor, f'best_checkpoint_actor_{avg}.pth')
    os.rename(filename_critic, f'best_checkpoint_critic_{avg}.pth')


# Load saved checkpoints
def load_checkpoints(agent, actor_checkpoint_path, critic_checkpoint_path):
    agent.actor_local.load_state_dict(torch.load(actor_checkpoint_path))
    agent.critic_local.load_state_dict(torch.load(critic_checkpoint_path))

# Run trials
def run_trials(agent, env, num_trials, actor, critic):
    scores = []
    max_steps = 1000
    for i in range(num_trials):
        rand_gravity = random.randint(-11, -5)
        rand_wind = random.randint(0, 20)
        env = gym.make("LunarLander-v2", continuous=True, gravity=rand_gravity, enable_wind=True, wind_power=rand_wind,
                       render_mode='rgb_array')
        #env = RecordVideo(env, video_dir)
        state = env.reset()
        agent.reset()
        score = 0
        state = state[0]
        done = False
        steps = 0
        while not done and steps < max_steps:
            steps += 1
            action = agent.act(state, add_noise=False)
            next_state, reward, done, _, __ = env.step(action.squeeze())
            score += reward
            state = next_state
        scores.append(score)
        print(f'Trial {i + 1}, Score: {score}')
        save_to_csv('DDPG_test.csv', i, score)
    return scores
    #rename(actor, critic, scores)




# Create the environment
env = gym.make("LunarLander-v2", continuous=True, render_mode='rgb_array')

# Create the DDPG agent
agent = DDPGAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=0)
def run_all_rename(path):
    for i in range (6):
        actor = path + f"best_checkpoint_actor_{i}.pth"
        critic = path + f"best_checkpoint_critic_{i}.pth"
        load_checkpoints(agent, actor, critic)
        run_trials(agent, env, 400, actor, critic)

def rum_averages(path):
    avgs = []
    for i in range (3):
        actor = path + f"best_checkpoint_actor.pth"
        critic = path + f"best_checkpoint_critic.pth"
        load_checkpoints(agent, actor, critic)
        scores = run_trials(agent, env, 400, actor, critic)
        avgs.append(int(sum(scores) / len(scores)))
    print (avgs)

def run_single_rename(path):
    actor = path + f"best_checkpoint_actor.pth"
    critic = path + f"best_checkpoint_critic.pth"
    load_checkpoints(agent, actor, critic)
    scores = run_trials(agent, env, 400, actor, critic)
    avg = (int(sum(scores) / len(scores)))
    print("average score:", avg)
    print("top score:", max(scores))


# Define the path to the checkpoints "your/desired/path/here/"
path = input("insert your path here: ")

run_single_rename(path)