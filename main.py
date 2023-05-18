import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import deque
import numpy as np
from DDPG_agent import DDPGAgent
from TD3_agent import TD3Agent
import random
import csv

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup video directory and frequency
video_dir = "videos_2"
video_freq = 100

# Function to decide if a video should be recorded
def should_record_video(episode_idx: int) -> bool:
    return episode_idx > 0 and episode_idx % video_freq == 0

def save_training_data(filename, episode, gravity, wind_power, average_score):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if episode == 1:
            writer.writerow(['Episode', 'Gravity', 'Wind Power', 'Average Score'])
        writer.writerow([episode, gravity, wind_power, average_score])

# Function to get the agent depending on user choice
def get_agent(agent_choice, env):
    if agent_choice == 1:
        return DDPGAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0],
                         random_seed=0)
    elif agent_choice == 2:
        return TD3Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=0)
    else:
        raise ValueError("Incorrect input, choose between 1. DDPG and 2. TD3")


# Function to train the agent using DDPG
def train_ddpg(n_episodes=10000, max_t=1000, print_every=10):
    scores_deque = deque(maxlen=print_every)
    scores = []
    max_score = -np.inf
    best_episode = 0
    save = 0
    # Create environment and wrapper for video recording
    env = gym.make("LunarLander-v2", continuous=True, render_mode='rgb_array')
    env.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'render_fps': 50,
    }
    # Get agent
    agent_choice = int(input("What Agent would you like to use?\n1.DDPG agent\n2.TD3 agent\n"))
    agent = get_agent(agent_choice, env)

    for i_episode in range(1, n_episodes + 1):
        rand_gravity = random.randint(-11, -5)
        rand_wind = random.randint(0, 20)
        env = gym.make("LunarLander-v2", continuous=True, gravity=rand_gravity, enable_wind=True, wind_power=rand_wind,
                       render_mode='rgb_array')

        # record video every x episode
        if i_episode % 100 == 0:
            env = RecordVideo(env, video_dir, name_prefix=i_episode)

        state = env.reset()
        agent.reset()
        score = 0
        state = state[0]
        for t in range(max_t):
            action = agent.act(state)[0]
            next_state, reward, done, _, __ = env.step(action)
            agent.step(state, action, reward, next_state, done, episode=i_episode, decrease_sigma_every_n_episodes=50,
                       delta_sigma=0.002)
            state = next_state
            score += reward
            if done:
                break

        env.close()

        scores_deque.append(score)
        scores.append(score)
        if score > max_score:
            max_score = score
            best_episode = i_episode

            if i_episode > 1000:
                if agent_choice == 1:
                    torch.save(agent.actor_local.state_dict(), f'best_checkpoint_actor{save}.pth')
                    torch.save(agent.critic_local.state_dict(), f'best_checkpoint_critic{save}.pth')  # DDPG
                elif agent_choice == 2:
                    torch.save(agent.actor_local.state_dict(), f'best_checkpoint_actor{save}.pth')
                    torch.save(agent.critic_local1.state_dict(), f'best_checkpoint_critic1{save}.pth')  # TD3
                    torch.save(agent.critic_local2.state_dict(), f'best_checkpoint_critic2{save}.pth')
                save += 1
            else:
                if agent_choice == 1:
                    torch.save(agent.actor_local.state_dict(), 'best_checkpoint_actor.pth')
                    torch.save(agent.critic_local.state_dict(), 'best_checkpoint_critic.pth')  # DDPG
                if agent_choice == 2:
                    torch.save(agent.actor_local.state_dict(), 'best_checkpoint_actor.pth')
                    torch.save(agent.critic_local1.state_dict(), 'best_checkpoint_critic1.pth')  # TD3
                    torch.save(agent.critic_local2.state_dict(), 'best_checkpoint_critic2.pth')

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if agent_choice == 1:
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')  # ddpg
            if agent_choice == 2:
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local1.state_dict(), 'checkpoint_critic1.pth')  # TD3
                torch.save(agent.critic_local2.state_dict(), 'checkpoint_critic2.pth')
        save_training_data('Training_data.csv', i_episode, env.gravity, env.wind_power, np.mean(scores_deque))

    return scores

train_ddpg()

