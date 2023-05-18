import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# Define constants for the buffer size, batch size, discount factor, soft update of target parameters,
# learning rates of the actor and critic, weight decay, policy noise, noise clip, and policy frequency.
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
WEIGHT_DECAY = 0
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Actor model. It is a policy-based model which gives us the action given a state
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_sizes=(128, 128)):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Define the layers of the network
        self.layers = nn.ModuleList([nn.Linear(state_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:])])
        self.layers.append(nn.Linear(hidden_sizes[-1], action_size))

    def forward(self, state):
        # Define forward pass
        x = state
        # Apply ReLU activation for all layers except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        # Apply tanh activation to the last layer
        return torch.tanh(self.layers[-1](x))

# Define Critic model. It is a value-based model which gives us the Q-value given a state-action pair
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_sizes=(128, 128)):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Define the layers of the network
        self.layers = nn.ModuleList([nn.Linear(state_size + action_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:])])
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, state, action):
        # Define forward pass
        x = torch.cat((state, action), dim=1)
        # Apply ReLU activation for all layers except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        # No activation is applied to the last layer
        return self.layers[-1](x)

# Define the Ornstein-Uhlenbeck process to add noise to actions, to facilitate exploration
class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.3):
        self.mu = mu * np.ones(size)
        self.theta = theta  # The rate of mean reversion
        self.sigma = sigma  # The scale of the noise
        self.seed = random.seed(seed)
        self.size = size
        self.reset()  # Reset the internal state (= noise) to mean (mu)

    # Decrease the scale of the noise over time
    def decrease_sigma(self, delta_sigma):
        self.sigma = max(0.1, self.sigma - delta_sigma)
        print('')
        print(self.sigma)
        print('')

    # Reset the internal state (= noise) to mean (mu)
    def reset(self):
        self.state = copy.copy(self.mu)

    # Generate a new sample noise
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

# Define replay buffer for storing and sampling experience tuples
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) # Internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    # Add a new experience to memory
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    # Randomly sample a batch of experiences from memory
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # Extract tensors from the experiences
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    # Return the current size of internal memory
    def __len__(self):
        return len(self.memory)

# Define TD3 (Twin Delayed Deep Deterministic policy gradient) agent
class TD3Agent():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Initialize actor and critic networks

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Initialize two critic networks for TD3
        self.critic_local1 = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target1 = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Initialize the 2nd critic network. TD3 uses twin critics to stabilize the learning.
        self.critic_local2 = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target2 = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Initialize Ornstein-Uhlenbeck process for exploration noise in action space
        self.noise = OUNoise(action_size, random_seed)

        # Initialize the replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Learning rate schedulers for actor and critic optimizers (gamma = 1, currently set to not decrease sigma)
        self.actor_lr_scheduler = StepLR(self.actor_optimizer, step_size=1000, gamma=1)
        self.critic_lr_scheduler1 = StepLR(self.critic_optimizer1, step_size=1000, gamma=1)
        self.critic_lr_scheduler2 = StepLR(self.critic_optimizer2, step_size=1000, gamma=1)

        self.total_steps = 0 # Total number of steps taken

    # Method to manage the learning step, collect new experience, learn from the replay buffer
    def step(self, state, action, reward, next_state, done, episode, decrease_sigma_every_n_episodes, delta_sigma=0.01):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        # Decrease the scale of the noise after each episode
        if done and (episode + 1) % decrease_sigma_every_n_episodes == 0:
            self.noise.decrease_sigma(delta_sigma)

    # Returns actions for given state as per current policy
    def act(self, state, add_noise=True):
        # Reshape state to (1, state_size) and convert to tensor
        state = torch.from_numpy(state.reshape(1, -1)).float().to(device)  # Add a batch dimension
        # Set the network into evaluation mode -> Impact on Layers like Dropout, BatchNorm, etc.
        self.actor_local.eval()
        with torch.no_grad(): # Deactivates autograd, reduces memory usage and speeds up computations
            action = self.actor_local(state).cpu().data.numpy()
        # Switch back to training mode
        self.actor_local.train()

        # Add noise to the action for exploration
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1) # return action

    # Reset the OU Noise for each episode
    def reset(self):
        self.noise.reset()

    # Learn from experience by updating the policy and value parameters
    def learn(self, experiences, gamma):
        self.total_steps += 1
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            # Select next action according to the target policy
            actions_next = self.actor_target(next_states)
            # Add noise to the target actions
            noise = torch.FloatTensor(actions_next.size()).data.normal_(0, POLICY_NOISE).to(device)
            noise = torch.clamp(noise, -NOISE_CLIP, NOISE_CLIP)
            actions_next = (actions_next + noise).clamp(-1, 1)

            # Compute target Q values for both critics
            Q_targets_next1 = self.critic_target1(next_states, actions_next)
            Q_targets_next2 = self.critic_target2(next_states, actions_next)
            # Select the minimum Q value between both critics
            Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)
            # Compute the target Q values
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Update critic 1
        Q_expected1 = self.critic_local1(states, actions)
        critic_loss1 = F.mse_loss(Q_expected1, Q_targets)
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        # Update critic 2
        Q_expected2 = self.critic_local2(states, actions)
        critic_loss2 = F.mse_loss(Q_expected2, Q_targets)
        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        # Delayed policy updates
        if self.total_steps % POLICY_FREQ == 0:
            # Update actor
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local1(states, actions_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_lr_scheduler.step()

            # Update target networks
            self.soft_update(self.critic_local1, self.critic_target1, TAU)
            self.soft_update(self.critic_local2, self.critic_target2, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

        self.critic_lr_scheduler1.step()
        self.critic_lr_scheduler2.step()

    # Perform soft updates on target network parameters
    def soft_update(self, local_model, target_model, tau):
        # Update target network parameters with combination of local and target model parameters
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

