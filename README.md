# Lunar-Landing-with-multiple-uncertainties
The Lunar Lander from gymnasium with multiple sources of uncertainties and continuous action and state space. Made in concert with [mikeair8](https://github.com/mikeair8)

Based on work done in https://arxiv.org/abs/2011.11850 with code https://github.com/rogerxcn/lunar_lander_project.

Using the simulator Lunar Lander in Box2D from https://gymnasium.farama.org/environments/box2d/lunar_lander/

We are using the Deep Deterministic Policy Gradient(DDPG) and Twin Delayed Deep Deterministic Policy Gradient(TD3) as the reinforcement learning algorithms.


# Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
To run this code, you will need 64-bit python >= 3.8 and the following libraries:

```
torch >= 1.12.1
swig
gymnasium[box2d] >= 0.26.1
numpy >= 1.20
moviepy >= 1.0.3
```
If you are running on windows you may need to install `Microsoft C++ Build Tools` found here https://visualstudio.microsoft.com/visual-cpp-build-tools/

You can install these using the requirements.txt and pip:
`pip install -r requirements.txt`


## Files in the repository
DDPG_agent.py: Contains the code for the DDPG agent.

TD3_agent.py: Contains the code for the TD3 agent.

lunar_lander.py: Contain a modified version of the Lunar Lander from Gymnasium

main.py: The main run file to train a model

test_DDPG.py: A file to test DDPG models

test_TD3.py: A file to test TD3 models

## Running the code
You can run the main file in your Python environment. You'll be prompted to choose between the DDPG and TD3 agent.

During the training process, the script will save the weights of the best performing model to `best_checkpoint_actor.pth` for both DDPG and TD3. Additinally  `best_checkpoint_critic.pth` for DDPG with `best_checkpoint_critic1.pth` and `best_checkpoint_critic2.pth` for TD3. After running for 1000 episodes it will save the new best checkpoints by adding a `_i` for the i'th better checkpoint. This to ensures that the best model saved is not a result of a lucky combination of uncertanties.

The script also periodically saves weights to `checkpoint_actor.pth` and `checkpoint_critic.pth` every few episodes.

Furthermore, the script saves a CSV file, `Training_data.csv`, that logs training information for each episode, including episode number, gravity, wind power, and average score.

When initializing the agents, whether it be the DDPG or the TD3 agent, they will take 4 paramaters (state_size, action_size, random_seed, hidden_sizes), its important to note that the `hidden sizes` parameter will take hidden sizes in the format (128, 128), where each value is a hidden layer and the value dictates how many nodes that hidden layer will have. the default `hidden sizes` is (128, 128)

## Testing a model
You can test a DDPG model by running `test_DDPG.py` and supplying it with a path to saved DDPG a actor and a critc weights. Or test a TD3 model by running `test_TD3.py` and supplying it with a path to saved a TD3 actor and two critics weights.

If you want to test models that have other structures than the default 128-128 hidden sizes, you will have to change the `hidden sizes` parameter.

The test script will look for files with the name `best_checkpoint_actor.pth` for both DDPG and TD3. Additinally `best_checkpoint_critic.pth` for DDPG with `best_checkpoint_critic1.pth` and `best_checkpoint_critic2.pth` for TD3.
