# Lunar-Landing-with-multiple-uncertainties
The Lunar Lander from gymnasium with multiple sources of uncertainties and continuous action and state space. Made in concert with [mikeair8](https://github.com/mikeair8)

Based on work done in https://arxiv.org/abs/2011.11850 with code https://github.com/rogerxcn/lunar_lander_project.

Using the simulator Lunar Lander in Box2D from https://gymnasium.farama.org/environments/box2d/lunar_lander/

Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
To run this code, you will need the following libraries:

torch
gym
gymnasium
numpy
csv
collections
random
You can install these using pip:

bash
Copy code
pip install torch gym gymnasium numpy csv
Also, you'll need the following custom agents:

DDPG_agent
TD3_agent
Usage
The script will interact with the LunarLander-v2 environment from Gym. It will then train an agent using the DDPG (Deep Deterministic Policy Gradient) or TD3 (Twin Delayed DDPG) algorithm.

Files in the repository
DDPG_agent.py: Contains the code for the DDPG agent.
TD3_agent.py: Contains the code for the TD3 agent.
Running the code
You can run the main file in your Python environment. You'll be prompted to choose between the DDPG and TD3 agent.

bash
Copy code
python main.py
During the training process, the script will save the weights of the best performing model to best_checkpoint_actor.pth and best_checkpoint_critic.pth for DDPG or additional best_checkpoint_critic2.pth for TD3.

The script also periodically saves weights to checkpoint_actor.pth and checkpoint_critic.pth every few episodes.

Furthermore, the script saves a CSV file, Training_data.csv, that logs training information for each episode, including episode number, gravity, wind power, and average score.
