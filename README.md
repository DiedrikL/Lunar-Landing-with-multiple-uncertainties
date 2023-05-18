# Lunar-Landing-with-multiple-uncertainties
The Lunar Lander from gymnasium with multiple sources of uncertainties and continuous action and state space. Made in concert with [mikeair8](https://github.com/mikeair8)

Based on work done in https://arxiv.org/abs/2011.11850 with code https://github.com/rogerxcn/lunar_lander_project.

Using the simulator Lunar Lander in Box2D from https://gymnasium.farama.org/environments/box2d/lunar_lander/


# Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
To run this code, you will need 64-bit python >= 3.8 and the following libraries:

```
torch >= 1.12.1
gymnasium[box2d] >= 0.26.1
numpy >= 1.20
moviepy >= 1.0.3
csv
collections
random
```

You can install these using the requirements.txt and pip:
`pip install -r requirements.txt`


## Files in the repository
DDPG_agent.py: Contains the code for the DDPG agent.
TD3_agent.py: Contains the code for the TD3 agent.
lunar_lander.py: Contain a modified version of the Lunar Lander from Gymnasium
main.py: The main run file to train a model
test.py: A file to test models

## Running the code
You can run the main file in your Python environment. You'll be prompted to choose between the DDPG and TD3 agent.

During the training process, the script will save the weights of the best performing model to `best_checkpoint_actor.pth` and `best_checkpoint_critic.pth` for DDPG with an additional `best_checkpoint_critic2.pth` for TD3.

The script also periodically saves weights to `checkpoint_actor.pth` and `checkpoint_critic.pth` every few episodes.

Furthermore, the script saves a CSV file, `Training_data.csv`, that logs training information for each episode, including episode number, gravity, wind power, and average score.

## Testing a model
You can test a model by running test.py and supplying it with a path to a saved model
